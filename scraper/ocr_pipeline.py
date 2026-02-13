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
        Determine whether an OCR text fragment is likely to be noise.
        
        Uses simple heuristics based on trimmed text length and confidence to flag fragments that are empty, very short, or both short and low-confidence.
        
        Parameters:
            text (str): The OCR-extracted text fragment.
            confidence (float): The OCR confidence score for the fragment (0.0–1.0).
        
        Returns:
            bool: `True` if the fragment is likely noise, `False` otherwise.
        """
        if not text:
            return True
            
        text_len = len(text.strip())
        if text_len < 2:
            return True
            
        if confidence < 0.2 and text_len < 5:
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
            return image_or_path
            
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
        """
        Run a primary EasyOCR pass on the provided image.
        
        Parameters:
            image_path_or_image (str | PIL.Image.Image): File path or PIL Image to process.
        
        Returns:
            dict: OCR result dictionary annotated with `'method': 'easyocr_primary'` on success.
            None if EasyOCR is unavailable, the image cannot be loaded, or the OCR pass fails.
        """
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
        """
        Run a Tesseract OCR pass on the given image and return the OCR result.
        
        Parameters:
            image_path_or_image (str | PIL.Image.Image): File path or PIL Image to process.
        
        Returns:
            result: The OCR result object from the Tesseract pass, or `None` if Tesseract is unavailable, the image could not be loaded, or an error occurred during processing.
        """
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
        """
        Selects the configured OCR engine and extracts text from the given image.
        
        Parameters:
            image_path_or_image (str | PIL.Image.Image): File path or PIL Image to run OCR on.
        
        Returns:
            result: Engine-specific OCR result object, or `None` if the image could not be loaded or the configured OCR method is unsupported.
        """
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