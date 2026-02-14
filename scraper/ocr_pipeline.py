from typing import Any, Optional
from PIL import Image
import threading
import re

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
        
        """
                 Initialize the OCR pipeline with configuration for engine selection, languages, performance modes, preprocessing, and retry/tuning parameters.
                 
                 Parameters:
                     ocr_method (str): Primary OCR engine to use, e.g. 'easyocr' or 'tesseract'.
                     ocr_lang (str): Language code(s) used for OCR (may contain multiple comma-separated codes).
                     quality_mode (str): Preferred quality preset that influences engine behavior and fallbacks.
                     fast_mode (bool): When True, prefer faster preprocessing and OCR paths.
                     fast_conf_skip (float): Confidence threshold below which fast-mode fragments may be skipped.
                     tessdata_dir (str): Path or prefix to Tesseract's tessdata directory (will be sanitized).
                     header_footer_crop_pct (float): Fractional percentage to crop from top/bottom when removing headers/footers.
                     watermark_flatten (bool): If True, attempt background flattening to reduce watermark impact.
                     watermark_clip_threshold (int): Pixel clipping threshold used when flattening watermarks.
                     watermark_retry_conf (dict): Retry configuration for attempts that handle watermarks.
                     high_dpi_retry_conf (dict): Retry configuration for high-DPI reprocessing attempts.
                     high_dpi_zoom (float): Zoom scale to apply when performing high-DPI retries.
                     auto_append_eng_for_ben (bool): If True, automatically append English language for Bengali-based heuristics.
                     quality_mode_default (str): Default quality mode to use when no explicit mode is provided.
                     segment_retry_conf (dict): Retry configuration for per-segment reprocessing attempts.
                     third_pass_scale (float): Scale factor used for an optional third-pass OCR refinement.
                     easyocr_fallback_conf (dict): Configuration controlling EasyOCR fallback behavior.
                     easyocr_primary_conf (dict): Configuration controlling the primary EasyOCR pass.
                     tesseract_refine_min_chars (int): Minimum character count required to trigger Tesseract refinement.
                 
                 Notes:
                     - Logging callbacks are provided via the `log` and `log_error` parameters (not documented here).
                     - The constructor performs initial environment validation (Tesseract command resolution and language data verification) and prepares internal state for lazy engine initialization.
                 """
        self.ocr_method = ocr_method
        self.ocr_method_effective = ocr_method
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
        """
        Normalize OCR text for consistent downstream processing.
        
        Parameters:
            text (str): Raw OCR output to normalize.
        
        Returns:
            normalized_text (str): The normalized string suitable for downstream comparison and scoring.
        """
        return normalize_text(text)

    def _maybe_split_columns(self, image):
        """
        Split a multi-column image into separate column images when appropriate.
        
        Parameters:
            image (PIL.Image.Image): The image to analyze and possibly split into columns.
        
        Returns:
            List[PIL.Image.Image]: A list of image segments corresponding to detected columns. If no column split is performed, returns a list containing the original image.
        """
        return preproc.maybe_split_columns(image, fast_mode=self.fast_mode)

    def _flatten_background(self, image, clip=None):
        """
        Flatten the image background to reduce watermark/uneven background artifacts.
        
        Parameters:
            image (PIL.Image.Image): Source image to process.
            clip (float, optional): Threshold used to clip background flattening; if omitted, the pipeline's watermark clip threshold is used.
        
        Returns:
            PIL.Image.Image: Image with a flattened background suitable for OCR.
        """
        clip_val = self.watermark_clip_threshold if clip is None else clip
        return preproc.flatten_background(image, clip=clip_val)

    def _choose_psm(self, image, segment_count):
        """
        Selects the Tesseract page segmentation mode (PSM) appropriate for the given image and detected segment count.
        
        Parameters:
            image: PIL.Image.Image
                The image to analyze when choosing a segmentation mode.
            segment_count: int
                Number of detected text segments or columns in the image.
        
        Returns:
            int: The numeric Tesseract PSM value to use for OCR.
        """
        return preproc.choose_psm(image, segment_count)

    def _score_result(self, res):
        """
        Compute a numeric quality score for an OCR result.
        
        Parameters:
            res: The OCR result object or dictionary produced by an OCR pass (contains text, confidence, and related metadata).
        
        Returns:
            score (float): A numeric score representing the quality/confidence of the OCR result; higher values indicate better quality.
        """
        return ocr_t.score_result(res)

    def _log_easyocr_device_once(self):
        """
        Log detected EasyOCR device information once.
        
        Detects the Torch device via the dependency probe, logs the device name using the instance log callback, and marks the device as logged so subsequent calls do nothing.
        """
        if self._device_logged:
            return
            
        from deps import detect_torch_device
        device_info = detect_torch_device()
        self.log(f"EasyOCR device: {device_info.get('name', 'unknown')}")
        self._device_logged = True

    def _is_noise_fragment(self, text: str, confidence: float) -> bool:
        """
        Determine whether an OCR text fragment is likely to be noise, with enhanced Bengali-specific heuristics.
        
        Parameters:
            text (str): The OCR-extracted text fragment.
            confidence (float): The OCR confidence score for the fragment (0.0–1.0).
        
        Returns:
            bool: `True` if the fragment is likely noise, `False` otherwise.
        """
        if not text:
            return True
        if not self.ocr_lang.startswith('ben'):
            return False
        tokens = re.sub(r"[\s\W_]+", "", text)
        if not tokens:
            return True
        ascii_letters = sum(1 for ch in tokens if ch.isascii())
        bengali_letters = sum(1 for ch in tokens if '\u0980' <= ch <= '\u09FF')
        length = len(tokens)
        if length <= 4 and confidence < 0.96:
            return True
        if bengali_letters == 0 and ascii_letters >= 3 and confidence < 0.93:
            return True
        ascii_ratio = ascii_letters / max(length, 1)
        if ascii_ratio > 0.65 and confidence < 0.9:
            return True
        return False

    def _load_image(self, image_or_path):
        """
        Load and return a PIL Image from either an existing Image object or a filesystem path.
        
        Parameters:
            image_or_path (PIL.Image.Image | str | os.PathLike): A PIL Image instance or a path to an image file.
        
        Returns:
            PIL.Image.Image or None: The opened PIL Image on success; logs an error and returns `None` if loading fails.
        """
        if isinstance(image_or_path, Image.Image):
            # Return a copy to prevent "Operation on closed image" errors
            return image_or_path.copy()
            
        try:
            return Image.open(image_or_path)
        except Exception as e:
            self.log_error(f"Failed to load image: {e}")
            return None

    def _ensure_tesseract_cmd(self):
        """
        Configure pytesseract to use the resolved Tesseract executable, if available.
        
        Attempts to resolve the system Tesseract command and set pytesseract.pytesseract.tesseract_cmd to that path. If resolution or assignment fails, logs a descriptive error via self.log_error.
        """
        try:
            cmd = resolve_tesseract_cmd()
            if cmd:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = cmd
        except Exception as e:
            self.log_error(f"Tesseract config error: {e}")

    def _verify_language_file(self):
        """
        Verify that Tesseract traineddata files for each language in the pipeline exist under the configured tessdata directory and log an error for any missing files or verification failures.
        
        Checks the languages derived from `self.ocr_lang`; for each language, logs an error if the corresponding `<lang>.traineddata` file is not found in `self.tessdata_dir`. Any exception raised during verification is caught and logged via `self.log_error`.
        """
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
        """
        Lazily initialize and return the EasyOCR reader configured for the pipeline's languages.
        
        Attempts to load EasyOCR if not already initialized; on success stores and returns the reader instance. If EasyOCR is unavailable or initialization fails, logs an error and returns None.
        
        Returns:
            easyocr_reader: The initialized EasyOCR reader instance, or `None` if EasyOCR is unavailable or initialization failed.
        """
        self._log_easyocr_device_once()
        if self._easyocr_reader is None:
            from deps import _lazy_import_easyocr
            easyocr = _lazy_import_easyocr()
            
            from ocr_easyocr import EASYOCR_AVAILABLE
            if not EASYOCR_AVAILABLE or easyocr is None:
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
        """
        Perform a single OCR pass using the configured EasyOCR reader.
        
        Parameters:
            image: PIL.Image or image-like object to be processed by EasyOCR.
        
        Returns:
            dict: OCR result dictionary produced by EasyOCR, or `None` if the EasyOCR reader is unavailable or the pass fails.
        """
        reader = self._get_easyocr_reader()
        if not reader:
            return None
            
        return ocr_e.run_easyocr_pass(image, self._easyocr_lock, reader, self.log, self.log_error)

    def _run_tesseract_pass(self, image, extra_config=None, extra_dilate=False, psm=None):
        """
        Perform a Tesseract OCR pass on the provided image using the pipeline's language and quality settings.
        
        Parameters:
            extra_config (str|None): Additional Tesseract configuration options to apply for this pass.
            extra_dilate (bool): If True, apply an extra dilation step to the image before running Tesseract.
            psm (int|None): Optional Tesseract Page Segmentation Mode (PSM) override for this pass.
        
        Returns:
            object: The result produced by the Tesseract OCR pass (engine-specific result structure).
        """
        return ocr_t.run_tesseract_pass(
            image, self.ocr_lang, self.quality_mode, psm=psm, extra_config=extra_config,
            extra_dilate=extra_dilate, log=self.log, log_error=self.log_error
        )

    def _tesseract_best_for_segment(self, seg, alt_seg, psm_for_seg):
        """
        Selects the best Tesseract OCR result for a given segment using an optional alternative segment and a specified page-segmentation mode.
        
        Parameters:
            seg: The primary segment (image crop or segment descriptor) to run Tesseract on.
            alt_seg: An alternative segment to compare against or use if the primary segment fails.
            psm_for_seg: Page segmentation mode (PSM) to apply for this segment; may be an int or None.
        
        Returns:
            An OCR result object containing recognized text, confidence, and associated metadata as produced by the Tesseract routine.
        """
        return ocr_t.tesseract_best_for_segment(
            seg, alt_seg, psm_for_seg, self.ocr_lang, self.quality_mode, self.fast_conf_skip,
            self.log, self.log_error
        )

    def preprocess_image_for_ocr(self, image_path_or_image):
        """
        Prepare and return an image optimized for OCR using the pipeline's language and mode settings.
        
        Parameters:
            image_path_or_image (str | PIL.Image.Image): File path or PIL Image to preprocess.
        
        Returns:
            PIL.Image.Image: A preprocessed PIL Image suitable for OCR.
        """
        return preproc.preprocess_image_for_ocr(
            image_path_or_image, self.ocr_lang, self.fast_mode, self.quality_mode, log_fn=self.log
        )

    @timer("easyocr_pass")
    def extract_text_with_easyocr_primary(self, image_path_or_image):
        if not EASYOCR_AVAILABLE:
            if self.log:
                self.log("EasyOCR not installed; falling back to Tesseract")
            return self.extract_text_with_tesseract(image_path_or_image)
        try:
            raw_img = self._load_image(image_path_or_image)
            cropped_img = preproc.crop_header_footer(raw_img, pct=self.header_footer_crop_pct)
            if cropped_img is None:
                return None
            preprocessed_img = self.preprocess_image_for_ocr(cropped_img)
            if preprocessed_img is None:
                return None

            base_segments_preview = self._maybe_split_columns(preprocessed_img)
            psm_for_seg = self._choose_psm(preprocessed_img, len(base_segments_preview))
            flattened_img = self._flatten_background(preprocessed_img) if self.watermark_flatten else None
            flat_segments = self._maybe_split_columns(flattened_img) if flattened_img is not None else None
            segments = [(seg, idx) for idx, seg in enumerate(base_segments_preview)]

            combined_text = []
            total_conf_weighted = 0.0
            total_fragments = 0
            refined = False

            if self.ocr_lang.startswith('ben'):
                try:
                    preprocessed_img = preprocessed_img.filter(ImageFilter.MaxFilter(3))
                except Exception:
                    pass
                if flattened_img is not None:
                    try:
                        flattened_img = flattened_img.filter(ImageFilter.MaxFilter(3))
                    except Exception:
                        pass

            for seg, idx in segments:
                alt_seg = flat_segments[idx] if flat_segments and idx < len(flat_segments) else None
                easy_res = self._run_easyocr_pass(seg)
                alt_easy = self._run_easyocr_pass(alt_seg) if alt_seg is not None else None
                best = easy_res
                if self._score_result(alt_easy) > self._score_result(best):
                    best = alt_easy

                needs_refine = False
                if TESSERACT_AVAILABLE:
                    if not best:
                        needs_refine = True
                    else:
                        text_len = len(best.get('text', '').strip())
                        conf = best.get('avg_confidence', 0)
                        if conf < self.easyocr_primary_conf or text_len < self.tesseract_refine_min_chars:
                            needs_refine = True

                if needs_refine:
                    self._verify_language_file()
                    tess_best = self._tesseract_best_for_segment(seg, alt_seg, psm_for_seg)
                    if self._score_result(tess_best) > self._score_result(best):
                        best = tess_best
                        refined = True

                if best:
                    conf_val = best.get('avg_confidence') or 0
                    text_val = best.get('text', '')
                    if self._is_noise_fragment(text_val, conf_val):
                        continue
                    combined_text.append(text_val)
                    total_conf_weighted += conf_val * max(best.get('fragments', 1), 1)
                    total_fragments += max(best.get('fragments', 1), 1)

            if not combined_text:
                return None

            normalized = self._normalize_text('\n\n'.join(combined_text))
            avg_conf = (total_conf_weighted / total_fragments) if total_fragments else 0.0
            self.ocr_method_effective = "easyocr+tesseract" if refined and TESSERACT_AVAILABLE else "easyocr"
            return {
                'text': normalized,
                'avg_confidence': round(avg_conf, 4),
                'fragments': total_fragments,
                'method': self.ocr_method_effective
            }
        except Exception as e:
            if self.log:
                self.log(f"EasyOCR-first error: {str(e)}")
            if self.log_error:
                self.log_error(f"EasyOCR-first error: {e}")
            return None

    @timer("tesseract_pass")
    def extract_text_with_tesseract(self, image_path_or_image):
        if not TESSERACT_AVAILABLE:
            if self.log:
                self.log("Tesseract not installed. Install tesseract-ocr and pytesseract")
            return None
        self._verify_language_file()
        try:
            raw_img = self._load_image(image_path_or_image)
            cropped_img = preproc.crop_header_footer(raw_img, pct=self.header_footer_crop_pct)
            if cropped_img is None:
                return None
            preprocessed_img = self.preprocess_image_for_ocr(cropped_img)
            if preprocessed_img is None:
                return None

            base_segments_preview = self._maybe_split_columns(preprocessed_img)
            psm_for_seg = self._choose_psm(preprocessed_img, len(base_segments_preview))
            flattened_img = self._flatten_background(preprocessed_img) if self.watermark_flatten else None

            if self.ocr_lang.startswith('ben'):
                try:
                    preprocessed_img = preprocessed_img.filter(ImageFilter.MaxFilter(3))
                except Exception:
                    pass
                if flattened_img is not None:
                    try:
                        flattened_img = flattened_img.filter(ImageFilter.MaxFilter(3))
                    except Exception:
                        pass

            base_segments = base_segments_preview
            flat_segments = self._maybe_split_columns(flattened_img) if flattened_img is not None else None
            segments = [(seg, idx) for idx, seg in enumerate(base_segments)]

            combined_text = []
            total_conf_weighted = 0.0
            total_fragments = 0

            for seg, idx in segments:
                alt_seg = flat_segments[idx] if flat_segments and idx < len(flat_segments) else None

                pass_a = self._run_tesseract_pass(seg, extra_config=None, extra_dilate=False, psm=psm_for_seg)
                pass_b = None
                if not pass_a or pass_a.get('avg_confidence', 0) < self.fast_conf_skip:
                    pass_b = self._run_tesseract_pass(seg, extra_config=["-c lstm_choice_mode=2"], extra_dilate=True, psm=psm_for_seg)

                best = pass_a if self._score_result(pass_a) >= self._score_result(pass_b) else pass_b

                if alt_seg is not None and (not best or best.get('avg_confidence', 0) < WATERMARK_RETRY_CONF):
                    alt_a = self._run_tesseract_pass(alt_seg, extra_config=None, extra_dilate=False, psm=psm_for_seg)
                    alt_b = None
                    if not alt_a or alt_a.get('avg_confidence', 0) < self.fast_conf_skip:
                        alt_b = self._run_tesseract_pass(alt_seg, extra_config=["-c lstm_choice_mode=2"], extra_dilate=True, psm=psm_for_seg)
                    alt_best = alt_a if self._score_result(alt_a) >= self._score_result(alt_b) else alt_b
                    if self._score_result(alt_best) > self._score_result(best):
                        best = alt_best

                if best is None or best.get('avg_confidence', 0) < self.segment_retry_conf or best.get('fragments', 0) < 2:
                    retry_seg = preproc.upscale_for_retry(seg, scale=THIRD_PASS_SCALE)
                    pass_c = self._run_tesseract_pass(
                        retry_seg,
                        extra_config=["-c lstm_choice_mode=2"],
                        extra_dilate=True,
                        psm=psm_for_seg,
                    )
                    if self._score_result(pass_c) > self._score_result(best):
                        best = pass_c

                if best is None or best.get('avg_confidence', 0) < self.easyocr_fallback_conf:
                    easy_res = self._run_easyocr_pass(seg)
                    if self._score_result(easy_res) > self._score_result(best):
                        best = easy_res

                if best:
                    conf_val = best.get('avg_confidence') or 0
                    text_val = best.get('text', '')
                    if self._is_noise_fragment(text_val, conf_val):
                        continue
                    combined_text.append(text_val)
                    total_conf_weighted += conf_val * max(best.get('fragments', 1), 1)
                    total_fragments += max(best.get('fragments', 1), 1)

            if not combined_text:
                return None

            normalized = self._normalize_text('\n\n'.join(combined_text))
            avg_conf = (total_conf_weighted / total_fragments) if total_fragments else 0.0
            return {
                'text': normalized,
                'avg_confidence': round(avg_conf, 4),
                'fragments': total_fragments,
                'method': 'tesseract'
            }
        except Exception as e:
            if self.log:
                self.log(f"Tesseract OCR error: {str(e)}")
            if self.log_error:
                self.log_error(f"Tesseract OCR error: {e}")
            return None

    def extract_text_with_ocr(self, image_path_or_image):
        if not self._engine_logged:
            if self.ocr_method == 'tesseract' or not EASYOCR_AVAILABLE:
                if self.log:
                    self.log("Engine: Tesseract (EasyOCR unavailable or not selected)")
            else:
                if TESSERACT_AVAILABLE:
                    if self.log:
                        self.log("Engine: EasyOCR primary; Tesseract will refine weak segments")
                else:
                    if self.log:
                        self.log("Engine: EasyOCR primary; Tesseract unavailable, refinement skipped")
            self._engine_logged = True

        if self.ocr_method == 'tesseract':
            self.ocr_method_effective = 'tesseract'
            return self.extract_text_with_tesseract(image_path_or_image)

        primary = self.extract_text_with_easyocr_primary(image_path_or_image)
        if primary is not None:
            return primary

        if TESSERACT_AVAILABLE:
            self.ocr_method_effective = 'tesseract'
            return self.extract_text_with_tesseract(image_path_or_image)

        if self.log:
            self.log("No OCR engine available")
        return None