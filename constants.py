from dataclasses import dataclass

DEFAULT_ZOOM = 7.0  # fixed render scale to avoid heavy DPI cost
FAST_MODE = False  # prefer accuracy over speed for Bangla-heavy docs
FAST_CONFIDENCE_SKIP = 0.92  # skip second OCR pass only when very strong
TEXT_LAYER_FIRST = True  # if True, attempt PDF text layer before OCR
TEXT_LAYER_LANG_MIN_RATIO = 0.35  # minimum Bangla ratio in text layer to accept without OCR
TEXT_LAYER_MIN_BEN_CHARS = 12  # minimum Bangla chars to trust text layer for Bangla
PDF_BYTES_CACHE_MB = 80  # cap for in-memory caching to reuse bytes for rendering
WATERMARK_FLATTEN = True  # if True, clip faint backgrounds before OCR
WATERMARK_CLIP_THRESHOLD = 245  # luminance threshold for clipping to white (Bangla: stronger flatten)
WATERMARK_RETRY_CONF = 0.82  # rerun on flattened background if below this
HIGH_DPI_RETRY_CONF = 0.92  # rerender at higher zoom when below this confidence
HIGH_DPI_ZOOM = 12.0  # higher render scale for low-confidence retries
HEADER_FOOTER_CROP_PCT = 0.12  # crop top/bottom bands to drop running headers
QUANTIZE_LEVELS = 32  # reduce grayscale to N levels to squash noise before OCR
QUANTIZE_DITHER = True  # use dithering when quantizing to preserve edges
AUTO_APPEND_ENG_FOR_BEN = False  # keep the selected language strict (no auto English)
QUALITY_MODE_DEFAULT = True  # default to accuracy-first for Bangla
SEGMENT_RETRY_CONF = 0.92  # rerun segment OCR when below this confidence
THIRD_PASS_SCALE = 1.45  # upscale factor for last-chance segment retry
EASYOCR_FALLBACK_CONF = 0.92  # fallback to EasyOCR when below this

# EasyOCR-first defaults and refinement thresholds
EASYOCR_PRIMARY_CONF = 0.94  # when below, ask Tesseract to refine segment
TESSERACT_REFINE_MIN_CHARS = 32  # if text is very short, try Tesseract to fill gaps


@dataclass
class OCRSettings:
    use_ocr: bool = True
    ocr_method: str = "easyocr"
    ocr_lang: str = "ben"
    quality_mode: bool = QUALITY_MODE_DEFAULT
