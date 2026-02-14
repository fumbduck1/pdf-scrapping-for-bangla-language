from typing import Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
import os

from constants import (
    DEFAULT_ZOOM,
    FAST_MODE,
    FAST_CONFIDENCE_SKIP,
    TEXT_LAYER_FIRST,
    TEXT_LAYER_LANG_MIN_RATIO,
    TEXT_LAYER_MIN_BEN_CHARS,
    PDF_BYTES_CACHE_MB,
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
    QUANTIZE_LEVELS,
    QUANTIZE_DITHER,
    RENDER_CACHE_MAX_ITEMS,
)
from config_manager import JobConfig
from scraper.models import PageResult, JobResult
from scraper.pdf_renderer import PdfRenderer
from scraper.ocr_pipeline import OcrPipeline
from scraper.utils import _sentence_chunks
import preprocess as preproc
import ocr_easyocr as ocr_e
import ocr_tesseract as ocr_t
from deps import (
    EASYOCR_AVAILABLE,
    pytesseract,
    TESSERACT_AVAILABLE,
    detect_poppler_path,
    detect_torch_device,
    check_pdftoppm_available,
)
from utils import (
    sanitize_tessdata_prefix,
    split_langs,
    validate_runtime_env,
    resolve_tesseract_cmd,
    normalize_text,
    bangla_ratio,
)
from performance import timer, register_metrics
import rs_correction as rs
from logger import get_logger


register_metrics()


class PDFScraper:
    def __init__(
        self,
        pdf_path,
        output_dir,
        use_ocr=True,
        ocr_method='easyocr',
        ocr_lang='ben',
        progress_callback=None,
        tessdata_dir=None,
        stop_event=None,
        quality_mode=QUALITY_MODE_DEFAULT,
        persist_renders=False,
        max_workers=None,
        fast_mode=FAST_MODE,
        fast_confidence_skip=FAST_CONFIDENCE_SKIP,
        pdf_bytes_cache_mb=PDF_BYTES_CACHE_MB,
        zoom=DEFAULT_ZOOM,
        high_dpi_zoom=HIGH_DPI_ZOOM,
        high_dpi_retry_conf=HIGH_DPI_RETRY_CONF,
        header_footer_crop_pct=HEADER_FOOTER_CROP_PCT,
        watermark_flatten=WATERMARK_FLATTEN,
        watermark_clip_threshold=WATERMARK_CLIP_THRESHOLD,
        watermark_retry_conf=WATERMARK_RETRY_CONF,
        quantize_levels=QUANTIZE_LEVELS,
        quantize_dither=QUANTIZE_DITHER,
        third_pass_scale=THIRD_PASS_SCALE,
        text_layer_first=TEXT_LAYER_FIRST,
        text_layer_lang_min_ratio=TEXT_LAYER_LANG_MIN_RATIO,
        text_layer_min_ben_chars=TEXT_LAYER_MIN_BEN_CHARS,
        render_cache_max_items=RENDER_CACHE_MAX_ITEMS,
        share_ocr_instances=True,
        ocr_pipeline_factory=None,
        auto_append_eng_for_ben=AUTO_APPEND_ENG_FOR_BEN,
        segment_retry_conf=SEGMENT_RETRY_CONF,
        easyocr_fallback_conf=EASYOCR_FALLBACK_CONF,
        easyocr_primary_conf=EASYOCR_PRIMARY_CONF,
        tesseract_refine_min_chars=TESSERACT_REFINE_MIN_CHARS,
        # Reed-Solomon error correction parameters
        rs_enabled=False,
        rs_error_correction_bytes=10,
        rs_block_size=1024,
        rs_enable_correction=True,
        rs_verify_only=False,
    ):
        """
        Create a PDFScraper configured to process a PDF and save OCR results.
        
        Initializes scraper paths, OCR and rendering options, performance tuning, Reed–Solomon parameters, and callbacks, then prepares output directories.
        
        Parameters:
            pdf_path (str): Path to the input PDF file.
            output_dir (str): Directory where outputs and intermediate renders will be written.
            use_ocr (bool): Whether to run OCR on pages.
            ocr_method (str): OCR backend to use (e.g., "easyocr", "tesseract").
            ocr_lang (str): Primary OCR language code (e.g., "ben").
            progress_callback (callable|None): Optional callback(progress:int, page:int) for progress reporting.
            tessdata_dir (str|None): Path prefix for Tesseract language data; will be sanitized.
            stop_event (object|None): Optional event-like object checked to cancel processing.
            quality_mode: Quality/performance preset used by rendering/OCR components.
            persist_renders (bool): If true, keep rendered page images on disk.
            max_workers (int|None): Max concurrent worker threads; None lets scraper choose.
            fast_mode (bool): Enable faster heuristics that may skip expensive passes.
            fast_confidence_skip (float): Confidence threshold used by fast-mode skipping logic.
            pdf_bytes_cache_mb (int): Memory cap (MB) for caching PDF bytes.
            zoom (float): Default render zoom factor for page images.
            high_dpi_zoom (float): Zoom factor used when a high-DPI retry is attempted.
            header_footer_crop_pct (float): Fraction of page height to treat as header/footer for cropping heuristics.
            watermark_flatten (bool): Whether to apply background flattening to mitigate watermarks.
            quantize_levels (int): Number of color levels for image quantization.
            quantize_dither (float): Dithering amount for quantization.
            third_pass_scale (float): Scale factor for an optional third OCR pass.
            text_layer_first (bool): Prefer extracting PDF native text layer before running OCR.
            text_layer_lang_min_ratio (float): Minimum language-character ratio to accept text-layer extraction.
            text_layer_min_ben_chars (int): Min Bengali characters required to accept text-layer extraction.
            render_cache_max_items (int): Max cached rendered pages to keep in memory.
            share_ocr_instances (bool): Reuse a single OCR pipeline instance across pages when true.
            ocr_pipeline_factory (callable|None): Factory that returns a configured OCR pipeline instance.
            auto_append_eng_for_ben (bool): Automatically append English model when Bengali selected.
            segment_retry_conf, easyocr_fallback_conf, easyocr_primary_conf: Configuration objects for segmentation and EasyOCR passes.
            tesseract_refine_min_chars (int): Minimum characters to trigger Tesseract refinement passes.
            rs_enabled (bool): Enable Reed–Solomon encoding/verification of outputs.
            rs_error_correction_bytes (int): Number of RS parity bytes for error correction.
            rs_block_size (int): Block size (bytes) used when encoding with Reed–Solomon.
            rs_enable_correction (bool): If true, attempt to correct errors when decoding RS data.
            rs_verify_only (bool): If true, only verify RS-encoded files without applying corrections.
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.use_ocr = use_ocr
        self.ocr_method = ocr_method
        self.persist_renders = bool(persist_renders)
        self.max_workers_override = max_workers
        self.user_lang = (ocr_lang or "ben").strip()
        self.auto_append_eng_for_ben = auto_append_eng_for_ben
        self.segment_retry_conf = segment_retry_conf
        self.easyocr_fallback_conf = easyocr_fallback_conf
        self.easyocr_primary_conf = easyocr_primary_conf
        self.tesseract_refine_min_chars = tesseract_refine_min_chars
        
        # Reed-Solomon error correction parameters
        self.rs_enabled = rs_enabled
        self.rs_error_correction_bytes = rs_error_correction_bytes
        self.rs_block_size = rs_block_size
        self.rs_enable_correction = rs_enable_correction
        self.rs_verify_only = rs_verify_only
        self.rs_corrector = None
        if self.rs_enabled:
            try:
                from rs_correction import RSTextCorrector
                self.rs_corrector = RSTextCorrector(rs_error_correction_bytes)
            except ImportError:
                self.log_error("Reed-Solomon library not available")
                self.rs_enabled = False
        self.ocr_lang = self.user_lang
        if self.auto_append_eng_for_ben:
            langs = split_langs(self.ocr_lang) or []
            if "ben" in langs and "eng" not in langs:
                langs.append("eng")
                self.ocr_lang = "+".join(langs)
        self.quality_mode = bool(quality_mode)
        self.fast_mode = fast_mode
        self.fast_confidence_skip = fast_confidence_skip
        self.page_render_zoom = zoom
        self.high_dpi_retry_conf = high_dpi_retry_conf
        self.high_dpi_zoom = high_dpi_zoom
        self.header_footer_crop_pct = header_footer_crop_pct
        self.watermark_flatten = watermark_flatten
        self.watermark_clip_threshold = watermark_clip_threshold
        self.watermark_retry_conf = watermark_retry_conf
        self.quantize_levels = quantize_levels
        self.quantize_dither = quantize_dither
        self.third_pass_scale = third_pass_scale
        self.text_layer_first = text_layer_first
        self.text_layer_lang_min_ratio = text_layer_lang_min_ratio
        self.text_layer_min_ben_chars = text_layer_min_ben_chars
        self.share_ocr_instances = bool(share_ocr_instances)
        self.progress_callback = progress_callback
        self.stop_event = stop_event
        self.tessdata_dir = sanitize_tessdata_prefix(tessdata_dir) if tessdata_dir else None
        self.poppler_path = os.environ.get("POPPLER_PATH") or detect_poppler_path()
        self.results = {
            'metadata': {},
            'pages': {},
            'statistics': {},
            'extraction_log': []
        }
        os.makedirs(self.output_dir, exist_ok=True)
        from logger import get_logger
        self.logger = get_logger()
        
        # Warn about potential memory issues with parallel processing and separate OCR instances
        worker_count = self._worker_count()
        if not self.share_ocr_instances and worker_count > 1:
            warning_msg = (
                "WARNING: share_ocr_instances is False and scraper is configured to run with "
                f"{worker_count} workers. This will create {worker_count} separate OCR pipeline "
                "instances (each with their own heavy EasyOCR reader), which may cause "
                "significant memory usage and potential blowups. Consider setting share_ocr_instances=True "
                "if you encounter memory issues during parallel processing."
            )
            if self.logger:
                self.logger.warning(warning_msg)
            # Also log to extraction log
            self.results['extraction_log'].append(warning_msg)
        
        # Cache guardrail: cap or disable cache; allow overrides for testing/memory constraints
        effective_cache_cap = max(int(render_cache_max_items or 0), 0)
        self.renderer = PdfRenderer(
            pdf_path=self.pdf_path,
            output_dir=self.output_dir,
            pdf_bytes_cache_mb=pdf_bytes_cache_mb,
            poppler_path=self.poppler_path,
            log=self.log,
            log_error=self.log_error,
            persist_renders=self.persist_renders,
            render_cache_max_items=effective_cache_cap,
        )
        self._ocr_factory = ocr_pipeline_factory or self._build_ocr_pipeline
        self.ocr = self._ocr_factory()

    @classmethod
    def from_job_config(cls, job_config: JobConfig, progress_callback=None, stop_event=None):
        """Construct a PDFScraper directly from a JobConfig to reduce call-site wiring."""
        return cls(
            pdf_path=job_config.input_path,
            output_dir=os.path.join(job_config.output_root, Path(job_config.input_path).stem),
            use_ocr=job_config.use_ocr,
            ocr_method=job_config.ocr.ocr_method,
            ocr_lang=job_config.ocr.ocr_lang,
            quality_mode=job_config.ocr.quality_mode,
            fast_mode=job_config.ocr.fast_mode,
            fast_confidence_skip=job_config.ocr.fast_confidence_skip,
            tessdata_dir=job_config.ocr.tessdata_dir,
            persist_renders=job_config.render.persist_renders,
            pdf_bytes_cache_mb=job_config.render.pdf_bytes_cache_mb,
            zoom=job_config.render.zoom,
            high_dpi_zoom=job_config.render.high_dpi_zoom,
            high_dpi_retry_conf=job_config.render.high_dpi_retry_conf,
            header_footer_crop_pct=job_config.preprocess.header_footer_crop_pct,
            watermark_flatten=job_config.preprocess.watermark_flatten,
            watermark_clip_threshold=job_config.preprocess.watermark_clip_threshold,
            watermark_retry_conf=job_config.preprocess.watermark_retry_conf,
            quantize_levels=job_config.preprocess.quantize_levels,
            quantize_dither=job_config.preprocess.quantize_dither,
            third_pass_scale=job_config.preprocess.third_pass_scale,
            text_layer_first=job_config.text_layer.text_layer_first,
            text_layer_lang_min_ratio=job_config.text_layer.text_layer_lang_min_ratio,
            text_layer_min_ben_chars=job_config.text_layer.text_layer_min_ben_chars,
            max_workers=job_config.max_workers,
            progress_callback=progress_callback,
            stop_event=stop_event,
            auto_append_eng_for_ben=job_config.ocr.auto_append_eng_for_ben,
            segment_retry_conf=job_config.ocr.segment_retry_conf,
            easyocr_fallback_conf=job_config.ocr.easyocr_fallback_conf,
            easyocr_primary_conf=job_config.ocr.easyocr_primary_conf,
            tesseract_refine_min_chars=job_config.ocr.tesseract_refine_min_chars,
            # Reed-Solomon error correction parameters
            rs_enabled=job_config.rs_correction.enabled,
            rs_error_correction_bytes=job_config.rs_correction.error_correction_bytes,
            rs_block_size=job_config.rs_correction.block_size,
            rs_enable_correction=job_config.rs_correction.enable_correction,
            rs_verify_only=job_config.rs_correction.verify_only,
        )

    def _build_ocr_pipeline(self):
        """
        Create an OcrPipeline configured from the scraper's OCR and preprocessing settings.
        
        Returns:
            OcrPipeline: An OcrPipeline instance initialized with the scraper's method, language, quality/fast modes, tessdata path, logging callbacks, header/footer and watermark options, and OCR-specific retry/fallback parameters.
        """
        return OcrPipeline(
            ocr_method=self.ocr_method,
            ocr_lang=self.ocr_lang,
            quality_mode=self.quality_mode,
            fast_mode=self.fast_mode,
            fast_conf_skip=self.fast_confidence_skip,
            tessdata_dir=self.tessdata_dir,
            log=self.log,
            log_error=self.log_error,
            header_footer_crop_pct=self.header_footer_crop_pct,
            watermark_flatten=self.watermark_flatten,
            watermark_clip_threshold=self.watermark_clip_threshold,
            auto_append_eng_for_ben=self.auto_append_eng_for_ben,
            segment_retry_conf=self.segment_retry_conf,
            easyocr_fallback_conf=self.easyocr_fallback_conf,
            easyocr_primary_conf=self.easyocr_primary_conf,
            tesseract_refine_min_chars=self.tesseract_refine_min_chars,
        )

    def _get_ocr_pipeline(self):
        """
        Return an OCR pipeline instance for use by the scraper, reusing a cached instance when sharing is enabled.
        
        If `share_ocr_instances` is True and a pipeline was previously created, the cached instance is returned; otherwise a new pipeline is constructed (and cached if sharing is enabled).
        
        Returns:
            The OCR pipeline instance to use for page processing. 
        """
        if self.share_ocr_instances and self.ocr:
            return self.ocr
        pipeline = self._ocr_factory()
        if self.share_ocr_instances:
            self.ocr = pipeline
        return pipeline

    def setup_directories(self):
        """
        Ensure the scraper output directory and a 'renders' subdirectory exist.
        
        Creates self.output_dir and a 'renders' subdirectory, assigns the subdirectory path to self.renders_dir, and silently ignores any filesystem errors.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.renders_dir = os.path.join(self.output_dir, 'renders')
            os.makedirs(self.renders_dir, exist_ok=True)
        except Exception:
            pass

    def _worker_count(self):
        """Optimized worker pool size for parallel OCR processing."""
        try:
            if getattr(self, "max_workers", None) is not None:
                override = self.max_workers
                if isinstance(override, (int, float)) and not isinstance(override, bool):
                    return max(1, int(override))
                elif isinstance(override, str) and override.strip().isdigit():
                    return max(1, int(override.strip()))
            cores = os.cpu_count() or 2
            langs = split_langs(self.ocr_lang) if hasattr(self, 'ocr_lang') else []
            has_ben = "ben" in langs
            if has_ben:
                max_workers = max(2, min(cores - 1, 4))
            else:
                max_workers = max(2, min(cores, 8))
            if self.quality_mode if hasattr(self, 'quality_mode') else False:
                max_workers = max(2, max_workers // 2)
            return max_workers
        except Exception:
            return 2

    def _ensure_tesseract_cmd(self):
        """Make sure pytesseract points to a real tesseract binary."""
        try:
            cmd = resolve_tesseract_cmd()
            if cmd:
                pytesseract.pytesseract.tesseract_cmd = cmd
        except Exception as e:
            self.log_error(f"Tesseract config error: {e}")

    def _verify_language_file(self):
        """Ensure the requested language data exists; log a clear error if not."""
        if not self.tessdata_dir:
            return
            
        try:
            langs = split_langs(self.ocr_lang)
            for lang in langs:
                expected_file = Path(self.tessdata_dir) / f"{lang}.traineddata"
                if not expected_file.exists():
                    self.log_error(f"Missing Tesseract language file {expected_file}")
        except Exception as e:
            self.log_error(f"Language verification error: {e}")

    def cleanup_renders(self):
        """
        Remove the renderer output directory to free disk space.
        
        Deletes the configured renders directory if it exists. Any exceptions raised during removal are suppressed.
        """
        try:
            if os.path.exists(self.renders_dir):
                import shutil
                shutil.rmtree(self.renders_dir)
        except Exception:
            pass

    def log(self, message):
        """Log message."""
        self.results['extraction_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if self.progress_callback:
            self.progress_callback(message)
        if hasattr(self, 'logger') and self.logger:
            try:
                self.logger.info(message)
            except Exception:
                pass

    def log_error(self, message):
        """Persist errors to errors.log so they remain visible after the UI advances."""
        try:
            path = os.path.join(self.output_dir, "errors.log")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {message}\n")
        except Exception:
            pass
        if hasattr(self, 'logger') and self.logger:
            try:
                self.logger.error(message)
            except Exception:
                pass

    def open_pdf(self):
        """Open PDF document."""
        ok = self.renderer.open_pdf()
        if not ok:
            return False
        self.doc = self.renderer.doc
        try:
            page_count = len(self.doc.pages) if self.doc and hasattr(self.doc, 'pages') else 0
            size_mb = round(os.path.getsize(self.pdf_path) / (1024 * 1024), 2)
        except Exception:
            page_count = 0
            size_mb = 0
        self.results['metadata'] = {
            'filename': Path(self.pdf_path).name,
            'pages': page_count,
            'creation_date': datetime.now().isoformat(),
            'file_size_mb': size_mb
        }
        self.log(f"Opened: {self.results['metadata']['filename']}")
        return True

    def preprocess_image_for_ocr(self, image_path):
        """
        Preprocess an image for OCR using the configured OCR pipeline with optimizations for Bengali and English.
        
        Parameters:
            image_path (str | os.PathLike): Path to the image file to preprocess.
        
        Returns:
            The preprocessing result produced by the OCR pipeline (for example a processed image object or a temporary file path).
        """
        return self._get_ocr_pipeline().preprocess_image_for_ocr(image_path)

    def render_page_to_image(self, page_num, zoom=None):
        """
        Render a specific PDF page to an image using the configured PDF renderer.
        
        Parameters:
            page_num (int): Page identifier passed to the renderer (as expected by the renderer).
            zoom (float | None): Zoom factor to use for rendering; when None the scraper's default zoom is used.
        
        Returns:
            tuple: `(PIL.Image.Image, str | None)` where the first element is the rendered PIL image and the second is the filesystem path where the rendered image was saved (or `None` if no file was written). Returns `None` if no renderer is available.
        """
        if self.renderer:
            if zoom is None:
                zoom = self.page_render_zoom
            return self.renderer.render_page(page_num, zoom)
        return None

    @timer("page_processing")
    def _process_page_with_ocr(self, page_num):
        """Optimized page processing with intelligent retry logic and error recovery."""
        if self.stop_event and self.stop_event.is_set():
            return page_num, None

        render_img = None
        render_path = None
        page_level_ocr = None
        ocr_engine = self._get_ocr_pipeline()
        
        try:
            render_result = self.render_page_to_image(page_num)
            if not render_result:
                return page_num, PageResult(
                    page_number=page_num,
                    content="",
                    warning='Rendering unavailable; raster OCR skipped',
                )
            if isinstance(render_result, tuple):
                render_img, render_path = render_result
            else:
                render_img = render_result
                render_path = None
            
            page_level_ocr = ocr_engine.extract_text_with_ocr(render_img)
            
            if page_level_ocr:
                confidence = page_level_ocr.get('avg_confidence', 0)
                text_length = len(page_level_ocr.get('text', '').strip())
                needs_retry = (
                    confidence < self.high_dpi_retry_conf or
                    (text_length < 50 and confidence < 0.9)
                )
                
                if needs_retry and not self.fast_mode:
                    try:
                        hi_render = self.render_page_to_image(page_num, zoom=self.high_dpi_zoom)
                        if hi_render:
                            if isinstance(hi_render, tuple):
                                hi_img, hi_path = hi_render
                            else:
                                hi_img, hi_path = hi_render, None

                            hi_ocr = ocr_engine.extract_text_with_ocr(hi_img)
                            if self._score_result(hi_ocr) > self._score_result(page_level_ocr):
                                page_level_ocr = hi_ocr
                                render_img = hi_img
                                render_path = hi_path
                                self.log(f"[Page {page_num + 1}] High-DPI retry improved results")
                            if hi_path and hi_path != render_path and os.path.exists(hi_path):
                                try:
                                    os.remove(hi_path)
                                except Exception:
                                    pass
                    except Exception as retry_err:
                        self.log(f"[Page {page_num + 1}] High-DPI retry failed: {retry_err}")

            page_text = page_level_ocr['text'] if page_level_ocr else ""
            page_data = PageResult(
                page_number=page_num,
                content=page_text,
                ocr_page_text=page_level_ocr.get('text', '') if page_level_ocr else "",
                ocr_page_confidence=page_level_ocr.get('avg_confidence', 0.0) if page_level_ocr else 0.0,
                ocr_page_fragments=page_level_ocr.get('fragments', 0) if page_level_ocr else 0,
                ocr_render=os.path.relpath(render_path, self.output_dir) if render_path else "",
            )

            return page_num, page_data
            
        except Exception as e:
            error_msg = f"Page {page_num + 1} processing error: {e}"
            self.log(error_msg)
            self.log_error(error_msg)
            
            return page_num, PageResult(
                page_number=page_num,
                content="",
                error=str(e),
            )

    def _normalize_text(self, text):
        """
        Normalize text by removing zero-width characters and collapsing/standardizing whitespace.
        
        Parameters:
            text (str): Input string to normalize.
        
        Returns:
            str: The cleaned string with zero-width characters removed and whitespace normalized.
        """
        return normalize_text(text)

    def _bangla_ratio(self, text: str):
        """
        Return the proportion of Bangla characters and their count in the given text.
        
        Returns:
            tuple: (ratio, count) where `ratio` is the fraction of characters that are Bangla (0.0–1.0) and `count` is the number of Bangla characters.
        """
        return bangla_ratio(text)

    def _flatten_background(self, image, clip=None):
        """
        Flatten the image background to reduce watermarks and uneven backgrounds.
        
        Parameters:
            image: Image to be processed.
            clip (float, optional): Threshold (typically 0–1) that controls flattening intensity; if omitted, the scraper's `watermark_clip_threshold` is used.
        
        Returns:
            The background-flattened image.
        """
        clip_val = self.watermark_clip_threshold if clip is None else clip
        return preproc.flatten_background(image, clip=clip_val)

    def _choose_psm(self, image, segment_count):
        """
        Selects an appropriate Tesseract page segmentation mode (PSM) for the given page image and detected segment count.
        
        Parameters:
            image: The page image to evaluate (e.g., PIL Image or numpy array).
            segment_count (int): Number of text/image segments detected on the page.
        
        Returns:
            int: The chosen Tesseract PSM value.
        """
        return preproc.choose_psm(image, segment_count)

    def _score_result(self, res):
        """Enhanced scoring algorithm for OCR result comparison."""
        return ocr_t.score_result(res)

    def _run_tesseract_pass(self, image, extra_config=None, extra_dilate=False, psm=None):
        """Run one Tesseract pass on a PIL image with optional extra dilation and config."""
        return ocr_t.run_tesseract_pass(
            image,
            ocr_lang=self.ocr_lang,
            quality_mode=self.quality_mode,
            psm=psm,
            extra_config=extra_config,
            extra_dilate=extra_dilate,
            log=self.log,
            log_error=self.log_error,
        )
    
    def _get_easyocr_reader(self):
        """Delegate to OCR pipeline's _get_easyocr_reader method."""
        ocr_engine = self._get_ocr_pipeline()
        return ocr_engine._get_easyocr_reader()
    
    def _run_easyocr_pass(self, image):
        """Delegate to OCR pipeline's _run_easyocr_pass method."""
        ocr_engine = self._get_ocr_pipeline()
        return ocr_engine._run_easyocr_pass(image)

    def _extract_text_layer(self, page):
        """
        Extracts and normalizes the PDF page's native text layer.
        
        Attempts to extract the page's native text and return it after normalization. Returns an empty string if no text is available or if extraction fails.
        
        Parameters:
        	page: PDF page object — the page from which to extract native text.
        
        Returns:
        	normalized_text (str): The normalized text from the page's text layer, or an empty string if unavailable or extraction fails.
        """
        try:
            text = page.extract_text()
            if text:
                return self._normalize_text(text)
        except Exception as e:
            self.log_error(f"Text layer extraction failed: {e}")
            
        return ""

    def _score_result(self, res):
        """
        Compute a numeric quality score for an OCR result.
        
        Parameters:
            res: OCR result object or dictionary containing recognized text and related metadata.
        
        Returns:
            score (float): A numeric score representing the quality or confidence of the OCR result.
        """
        return ocr_t.score_result(res)

    def _run_tesseract_pass(self, image, extra_config=None, extra_dilate=False, psm=None):
        """
        Perform a Tesseract OCR pass on a PIL image using optional config and dilation.
        
        Parameters:
            image (PIL.Image.Image): The image to run Tesseract on.
            extra_config (str | None): Additional Tesseract configuration options (raw config string) to apply for this pass.
            extra_dilate (bool): If True, apply an extra dilation preprocessing step before OCR.
            psm (int | None): Optional Tesseract Page Segmentation Mode (PSM) to use for this pass.
        
        Returns:
            object: The OCR pass result produced by the pipeline (typically includes recognized text, confidence metrics, and any fragment details).
        """
        return self._get_ocr_pipeline()._run_tesseract_pass(image, extra_config, extra_dilate, psm)

    def _get_easyocr_reader(self):
        """
        Return an EasyOCR reader instance configured for this scraper's OCR settings.
        
        Returns:
            reader: An EasyOCR `Reader` object configured with the scraper's current OCR language and options.
        """
        return self._get_ocr_pipeline()._get_easyocr_reader()

    def _run_easyocr_pass(self, image):
        """
        Run an EasyOCR recognition pass on the given image and return its result.
        
        Parameters:
            image: Image or image path to process with EasyOCR.
        
        Returns:
            ocr_result: OCR output object containing recognized text, confidence scores, and bounding boxes.
        """
        return self._get_ocr_pipeline()._run_easyocr_pass(image)

    @timer("page_processing")
    def scrape_all_pages(self):
        """Scrape all pages with optional parallel OCR per page."""
        if not self.open_pdf():
            return False
        
        try:
            if not self.doc or not hasattr(self.doc, 'pages'):
                return False
                
            total_pages = len(self.doc.pages)
            page_results = {}
            ocr_futures = []

            with ThreadPoolExecutor(max_workers=self._worker_count()) as executor:
                for page_num in range(total_pages):
                    if self.stop_event and self.stop_event.is_set():
                        self.log("Stop requested; aborting remaining pages")
                        break

                    # Calculate and report progress
                    progress = ((page_num + 1) / total_pages) * 100
                    if self.progress_callback:
                        try:
                            self.progress_callback(progress)
                        except Exception as e:
                            self.log(f"Error in progress callback: {str(e)}")
                    page_text = ""
                    page_level_ocr = None
                    render_path = None

                    try:
                        page = self.doc.pages[page_num]

                        if self.text_layer_first:
                            langs = split_langs(self.ocr_lang) or []
                            text_layer = self._extract_text_layer(page)
                            if text_layer and len(text_layer) > 5:
                                use_text_layer = False
                                if "ben" in langs:
                                    ratio, ben_count = self._bangla_ratio(text_layer)
                                    use_text_layer = (
                                        ratio >= self.text_layer_lang_min_ratio
                                        and ben_count >= self.text_layer_min_ben_chars
                                    )
                                    if not use_text_layer:
                                        self.log(
                                            f"[Page {page_num + 1}] Text layer rejected (ben ratio {ratio:.2f}, chars {ben_count}); running OCR"
                                        )
                                else:
                                    use_text_layer = True

                                if use_text_layer:
                                    page_text = text_layer
                                    self.log(f"[Page {page_num + 1}] Used PDF text layer; OCR skipped")

                        if not page_text:
                            future = executor.submit(self._process_page_with_ocr, page_num)
                            ocr_futures.append(future)
                            continue

                    except Exception as page_err:
                        self.log(f"Page {page_num + 1} OCR error: {page_err}")
                        try:
                            print(f"Page {page_num + 1} OCR error: {page_err}", file=sys.stderr)
                        except Exception:
                            pass
                        try:
                            self.log_error(f"Page {page_num + 1} OCR error: {page_err}")
                        except Exception:
                            pass
                        page_text = ""
                        page_level_ocr = None

                    preview_len = len(page_text)
                    self.log(f"[Page {page_num + 1}] OCR complete (chars: {preview_len})")

                    page_results[page_num] = PageResult(
                        page_number=page_num + 1,
                        content=page_text,
                        ocr_page_text=page_level_ocr.get('text', '') if page_level_ocr else "",
                        ocr_page_confidence=page_level_ocr.get('avg_confidence', 0.0) if page_level_ocr else 0.0,
                        ocr_page_fragments=page_level_ocr.get('fragments', 0) if page_level_ocr else 0,
                        ocr_render=os.path.relpath(render_path, self.output_dir) if render_path else "",
                    )

                for fut in as_completed(ocr_futures):
                    try:
                        page_num, page_data = fut.result()
                    except Exception as err:
                        self.log(f"Parallel OCR error: {err}")
                        continue
                    if page_data is None:
                        continue

                    page_results[page_num] = PageResult(
                        page_number=page_data.page_number + 1,
                        content=page_data.content,
                        ocr_page_text=page_data.ocr_page_text,
                        ocr_page_confidence=page_data.ocr_page_confidence,
                        ocr_page_fragments=page_data.ocr_page_fragments,
                        ocr_render=page_data.ocr_render,
                        warning=page_data.warning,
                        error=page_data.error
                    )
                    preview_len = len(page_data.content)
                    self.log(f"[Page {page_num + 1}] OCR complete (chars: {preview_len})")

            # Save results
            self._page_results = page_results
            return True

        finally:
            if self.renderer:
                self.renderer.close()

    def save_results(self):
        """
        Save the scraped page results to disk in multiple layout-preserving formats.
        
        Writes:
        - A single plain text file with page breaks (output.txt).
        - Reed–Solomon encoded continuous and structured text files when RS is enabled (output.rs.txt and output.structured.rs.txt).
        - A Reed–Solomon encoded per-sentence file when RS is enabled (sentences.rs.txt).
        """
        self._save_plain_text_files()
        self._save_rs_encoded_text()
        self._save_rs_encoded_sentences()

    def _save_plain_text_files(self):
        """
        Write all page OCR texts in page order to a single file named "output.txt" in the scraper's output directory.
        
        The file contains each page's text in order separated by two newline characters ("\n\n"). Logs the saved path on success.
        """
        ordered_pages = sorted(self._page_results.items(), key=lambda x: x[0])
        page_texts = [result.content for _, result in ordered_pages]
        
        txt_path = Path(self.output_dir) / "output.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(page_texts))
            
        self.log(f"Plain text saved to {txt_path}")

    def _save_rs_encoded_text(self):
        """
        Save Reed–Solomon encoded versions of the scraped page texts to the output directory.
        
        If RS encoding is enabled (rs_error_correction_bytes > 0) this creates an RS corrector and attempts to write two encoded files:
        - a continuous encoding (pages joined by two newlines) to "output.rs.txt"
        - a structured encoding (pages joined by a form-feed character) to "output.structured.rs.txt"
        
        Successful writes are logged. If RS encoding is disabled (rs_error_correction_bytes <= 0), no files are produced.
        """
        ordered_pages = sorted(self._page_results.items(), key=lambda x: x[0])
        page_texts = [result.content for _, result in ordered_pages]
        
        if self.rs_error_correction_bytes > 0:
            corrector = rs.create_rs_corrector(self.rs_error_correction_bytes)
            
            # Save continuous mode
            continuous_text = "\n\n".join(page_texts)
            rs_path = Path(self.output_dir) / "output.rs.txt"
            if corrector.encode_and_save(continuous_text, str(rs_path)):
                self.log(f"RS-encoded (continuous) saved to {rs_path}")
                
            # Save structured mode
            structured_text = "\f".join(page_texts)
            rs_structured_path = Path(self.output_dir) / "output.structured.rs.txt"
            if corrector.encode_and_save(structured_text, str(rs_structured_path)):
                self.log(f"RS-encoded (structured) saved to {rs_structured_path}")

    def _save_rs_encoded_sentences(self):
        """
        Save Reed–Solomon–encoded sentences to sentences.rs.txt in the output directory.
        
        If RS encoding is enabled (rs_error_correction_bytes > 0), collects sentences from all pages in page order, encodes them with the configured RS corrector, and saves the result to "sentences.rs.txt". Logs the saved path on success. If RS encoding is not enabled, no file is written.
        """
        ordered_pages = sorted(self._page_results.items(), key=lambda x: x[0])
        
        if self.rs_error_correction_bytes > 0:
            corrector = rs.create_rs_corrector(self.rs_error_correction_bytes)
            
            all_sentences = []
            for _, result in ordered_pages:
                sentences = _sentence_chunks(result.content)
                all_sentences.extend(sentences)
                
            rs_sentences_path = Path(self.output_dir) / "sentences.rs.txt"
            if corrector.encode_and_save("\n".join(all_sentences), str(rs_sentences_path)):
                self.log(f"RS-encoded sentences saved to {rs_sentences_path}")

    def _decode_rs_text_file(self, rs_filename):
        """
        Decode and verify a Reed–Solomon (RS) encoded text file.
        
        Parameters:
            rs_filename (str): Path to the RS-encoded file to decode.
        
        Returns:
            tuple: `(text, verified, errors)` where `text` is the decoded string or `None` if decoding was not performed, `verified` is `True` when the decoded data passes verification (or correction succeeded), and `errors` is the number of detected/corrected errors.
            
        Notes:
            If RS error correction is disabled (error correction bytes <= 0), the function returns `(None, False, 0)`.
        """
        if self.rs_error_correction_bytes > 0:
            corrector = rs.create_rs_corrector(self.rs_error_correction_bytes)
            return corrector.load_and_decode(rs_filename)
        return None, False, 0

    def _verify_rs_text_files(self):
        """
        Verify all Reed–Solomon (RS) encoded text files in the scraper's output directory and log verification outcomes.
        
        Checks every file matching '*.rs.txt' under the configured output directory, logs a verification message for each file that passes, logs an error for each file that fails or raises an exception, and returns an aggregate success flag.
        
        Returns:
            bool: `True` if every RS-encoded file was verified successfully, `False` if any file failed verification or an error occurred.
        """
        success = True
        
        for rs_file in Path(self.output_dir).glob("*.rs.txt"):
            try:
                text, verified, errors = self._decode_rs_text_file(str(rs_file))
                
                if verified:
                    self.log(f"Verified: {rs_file} (errors: {errors})")
                else:
                    self.log_error(f"Verification failed: {rs_file}")
                    success = False
                    
            except Exception as e:
                self.log_error(f"Error verifying {rs_file}: {e}")
                success = False
                
        return success


def run_pdf_job(job_config: JobConfig, stop_event: Optional[object], log_cb: Optional[Callable[[str], None]]) -> JobResult:
    """
    Run a single PDF scraping job configured by `job_config`.
    
    Parameters:
        job_config (JobConfig): Configuration for the PDF job (paths, OCR options, RS options, etc.).
        stop_event (Optional[object]): Optional event-like object with an `is_set()` method; if set, job may abort early.
        log_cb (Optional[Callable[[str], None]]): Optional logging callback to receive progress and error messages.
    
    Returns:
        JobResult: A dictionary with the job outcome and metrics:
            - "scrape_ok" (bool): `True` if scraping and saving completed without an uncaught exception, `False` otherwise.
            - "output_dir" (str): The job's configured output directory.
            - "num_pages" (int): Number of pages processed (0 on failure).
            - "num_errors" (int): Count of pages with errors (or 1 for a top-level failure).
            - "num_warnings" (int): Count of pages with warnings.
            - "runtime_seconds" (float): Total elapsed time for the job.
            - "errors" (List[str]): Error messages collected (page errors or a single top-level error).
            - "warnings" (List[str]): Warning messages collected.
    """
    scraper = PDFScraper.from_job_config(job_config, stop_event=stop_event)
    scraper.log = log_cb or (lambda x: None)
    
    start_time = datetime.now()
    
    try:
        scraper.scrape_all_pages()
        
        if not stop_event or not stop_event.is_set():
            scraper.save_results()
            
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        return {
            "scrape_ok": True,
            "output_dir": job_config.output_root,
            "num_pages": len(scraper._page_results),
            "num_errors": sum(1 for r in scraper._page_results.values() if r.error),
            "num_warnings": sum(1 for r in scraper._page_results.values() if r.warning),
            "runtime_seconds": runtime,
            "errors": [r.error for r in scraper._page_results.values() if r.error],
            "warnings": [r.warning for r in scraper._page_results.values() if r.warning],
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        scraper.log_error(f"Job failed: {e}\n{error_trace}")
        
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        return {
            "scrape_ok": False,
            "output_dir": job_config.output_root,
            "num_pages": 0,
            "num_errors": 1,
            "num_warnings": 0,
            "runtime_seconds": runtime,
            "errors": [str(e), error_trace],
            "warnings": [],
        }