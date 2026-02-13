from typing import Any, Optional
from PIL import Image
import threading

from constants import (
    FAST_MODE,
    FAST_CONFIDENCE_SKIP,
    TEXT_LAYER_LANG_MIN_RATIO,
    TEXT_LAYER_MIN_BEN_CHARS,
    WATERMARK_FLATTEN,
    WATERMARK_CLIP_THRESHOLD,
    WATERMARK_RETRY_CONF,
    HIGH_DPI_RETRY_CONF,
    HIGH_DPI_ZOOM,
    AUTO_APPEND_ENG_FOR_BEN,
    QUALITY_MODE_DEFAULT,
    SEGMENT_RETRY_CONF,
    THIRD_PASS_SCALE,
    EASYOCR_FALLBACK_CONF,
    EASYOCR_PRIMARY_CONF,
    TESSERACT_REFINE_MIN_CHARS,
    HEADER_FOOTER_CROP_PCT,
)
from deps import EASYOCR_AVAILABLE, TESSERACT_AVAILABLE
import preprocess as preproc
import ocr_easyocr as ocr_e
import ocr_tesseract as ocr_t
from utils import (
    sanitize_tessdata_prefix,
    split_langs,
    validate_runtime_env,
    resolve_tesseract_cmd,
    normalize_text,
    bangla_ratio,
)
from performance import timer


class OcrPipeline:
    """OCR orchestrator that encapsulates EasyOCR/Tesseract strategies."""

    def __init__(self, ocr_method, ocr_lang, quality_mode, fast_mode, fast_conf_skip, tessdata_dir, log, log_error,
                 header_footer_crop_pct=HEADER_FOOTER_CROP_PCT, watermark_flatten=WATERMARK_FLATTEN, watermark_clip_threshold=WATERMARK_CLIP_THRESHOLD,
                 watermark_retry_conf=WATERMARK_RETRY_CONF, high_dpi_retry_conf=HIGH_DPI_RETRY_CONF, high_dpi_zoom=HIGH_DPI_ZOOM,
                 auto_append_eng_for_ben=AUTO_APPEND_ENG_FOR_BEN, quality_mode_default=QUALITY_MODE_DEFAULT,
                 segment_retry_conf=SEGMENT_RETRY_CONF, third_pass_scale=THIRD_PASS_SCALE,
                 easyocr_fallback_conf=EASYOCR_FALLBACK_CONF, easyocr_primary_conf=EASYOCR_PRIMARY_CONF,
                 tesseract_refine_min_chars=TESSERACT_REFINE_MIN_CHARS):
        
        self.ocr_method = ocr_method
        self.ocr_lang = ocr_lang
        self.quality_mode = quality_mode
        self.fast_mode = fast_mode
        self.fast_conf_skip = fast_conf_skip
        self.tessdata_dir = sanitize_tessdata_prefix(tessdata_dir)
        self.log = log
        self.log_error = log_error
        self.header_footer_crop_pct = header_footer_crop_pct
        self.watermark_flatten = watermark_flatten
        self.watermark_clip_threshold = watermark_clip_threshold
        self.watermark_retry_conf = watermark_retry_conf
        self.high_dpi_retry_conf = high_dpi_retry_conf
        self.high_dpi_zoom = high_dpi_zoom
        self.auto_append_eng_for_ben = auto_append_eng_for_ben
        self.quality_mode_default = quality_mode_default
        self.segment_retry_conf = segment_retry_conf
        self.third_pass_scale = third_pass_scale
        self.easyocr_fallback_conf = easyocr_fallback_conf
        self.easyocr_primary_conf = easyocr_primary_conf
        self.tesseract_refine_min_chars = tesseract_refine_min_chars
        
        self._device_logged = False
        self._engine_logged = False
        self._easyocr_lock = threading.Lock()
        self._easyocr_reader = None
        self._ensure_tesseract_cmd()
        self._verify_language_file()

    # --- shared helpers ---
    def _normalize_text(self, text):
        return normalize_text(text)

    def _maybe_split_columns(self, image):
        return preproc.maybe_split_columns(image, fast_mode=self.fast_mode)

    def _flatten_background(self, image, clip=None):
        clip_val = self.watermark_clip_threshold if clip is None else clip
        return preproc.flatten_background(image, clip=clip_val)

    def _choose_psm(self, image, segment_count):
        return preproc.choose_psm(image, segment_count)

    def _score_result(self, res):
        return ocr_t.score_result(res)

    def _log_easyocr_device_once(self):
        if self._device_logged:
            return
            
        from deps import detect_torch_device
        device_info = detect_torch_device()
        self.log(f"EasyOCR device: {device_info.get('name', 'unknown')}")
        self._device_logged = True

    def _is_noise_fragment(self, text: str, confidence: float) -> bool:
        if not text:
            return True
            
        text_len = len(text.strip())
        if text_len < 2:
            return True
            
        if confidence < 0.2 and text_len < 5:
            return True
            
        return False

    def _load_image(self, image_or_path):
        if isinstance(image_or_path, Image.Image):
            return image_or_path
            
        try:
            return Image.open(image_or_path)
        except Exception as e:
            self.log_error(f"Failed to load image: {e}")
            return None

    def _ensure_tesseract_cmd(self):
        try:
            cmd = resolve_tesseract_cmd()
            if cmd:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = cmd
        except Exception as e:
            self.log_error(f"Tesseract config error: {e}")

    def _verify_language_file(self):
        if not self.tessdata_dir:
            return
            
        try:
            from pathlib import Path
            langs = split_langs(self.ocr_lang)
            for lang in langs:
                expected_file = Path(self.tessdata_dir) / f"{lang}.traineddata"
                if not expected_file.exists():
                    self.log_error(f"Missing Tesseract language file {expected_file}")
        except Exception as e:
            self.log_error(f"Language verification error: {e}")

    def _get_easyocr_reader(self):
        self._log_easyocr_device_once()
        if self._easyocr_reader is None:
            from deps import _lazy_import_easyocr
            easyocr, easyocr_available = _lazy_import_easyocr()
            
            if not easyocr_available or easyocr is None:
                self.log_error("EasyOCR not available")
                return None
                
            try:
                with self._easyocr_lock:
                    if self._easyocr_reader is None:
                        mapped_langs = ocr_e.map_easyocr_langs(self.ocr_lang)
                        self._easyocr_reader = ocr_e.get_easyocr_reader(mapped_langs, False, self._easyocr_lock, None, self.log, self.log_error)
            except Exception as e:
                self.log_error(f"EasyOCR init: {e}")
                
        return self._easyocr_reader

    def _run_easyocr_pass(self, image):
        reader = self._get_easyocr_reader()
        if not reader:
            return None
            
        return ocr_e.run_easyocr_pass(image, self._easyocr_lock, reader, self.log, self.log_error)

    def _run_tesseract_pass(self, image, extra_config=None, extra_dilate=False, psm=None):
        return ocr_t.run_tesseract_pass(
            image, self.ocr_lang, self.quality_mode, psm=psm, extra_config=extra_config,
            extra_dilate=extra_dilate, log=self.log, log_error=self.log_error
        )

    def _tesseract_best_for_segment(self, seg, alt_seg, psm_for_seg):
        return ocr_t.tesseract_best_for_segment(
            seg, alt_seg, psm_for_seg, self.ocr_lang, self.quality_mode, self.fast_conf_skip,
            self.log, self.log_error
        )

    def preprocess_image_for_ocr(self, image_path_or_image):
        return preproc.preprocess_image_for_ocr(
            image_path_or_image, self.ocr_lang, self.fast_mode, self.quality_mode, log_fn=self.log
        )

    @timer("easyocr_pass")
    def extract_text_with_easyocr_primary(self, image_path_or_image):
        if not EASYOCR_AVAILABLE:
            return None
            
        img = self._load_image(image_path_or_image)
        if img is None:
            return None
            
        try:
            result = self._run_easyocr_pass(img)
            
            if result:
                result['method'] = 'easyocr_primary'
                
            return result
            
        except Exception as e:
            self.log_error(f"EasyOCR primary pass failed: {e}")
            return None

    @timer("tesseract_pass")
    def extract_text_with_tesseract(self, image_path_or_image):
        if not TESSERACT_AVAILABLE:
            return None
            
        img = self._load_image(image_path_or_image)
        if img is None:
            return None
            
        try:
            return self._run_tesseract_pass(img)
        except Exception as e:
            self.log_error(f"Tesseract pass failed: {e}")
            return None

    def extract_text_with_ocr(self, image_path_or_image):
        if not self._engine_logged:
            self.log(f"OCR engine: {self.ocr_method}")
            self._engine_logged = True
            
        img = self._load_image(image_path_or_image)
        if img is None:
            return None
            
        if self.ocr_method == 'easyocr':
            return self.extract_text_with_easyocr_primary(img)
        elif self.ocr_method == 'tesseract':
            return self.extract_text_with_tesseract(img)
        else:
            return None
