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
        self.ocr_lang = ocr_lang
        self.tessdata_dir = sanitize_tessdata_prefix(tessdata_dir)
        self.quality_mode = quality_mode
        self.fast_mode = fast_mode
        self.fast_confidence_skip = fast_confidence_skip
        self.pdf_bytes_cache_mb = pdf_bytes_cache_mb
        self.zoom = zoom
        self.high_dpi_zoom = high_dpi_zoom
        self.high_dpi_retry_conf = high_dpi_retry_conf
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
        self.render_cache_max_items = render_cache_max_items
        self.share_ocr_instances = share_ocr_instances
        self.ocr_pipeline_factory = ocr_pipeline_factory
        self.auto_append_eng_for_ben = auto_append_eng_for_ben
        self.segment_retry_conf = segment_retry_conf
        self.easyocr_fallback_conf = easyocr_fallback_conf
        self.easyocr_primary_conf = easyocr_primary_conf
        self.tesseract_refine_min_chars = tesseract_refine_min_chars
        self.rs_enabled = rs_enabled
        self.rs_error_correction_bytes = rs_error_correction_bytes
        self.rs_block_size = rs_block_size
        self.rs_enable_correction = rs_enable_correction
        self.rs_verify_only = rs_verify_only
        
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.stop_event = stop_event
        self.log = get_logger().info
        self.log_error = get_logger().error
        self._page_results = {}
        self._page_count = 0
        self.ocr = None
        self._ocr_factory = self.ocr_pipeline_factory or self._build_ocr_pipeline
        self.setup_directories()

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
        """
        Choose an optimal number of worker processes for parallel OCR processing.
        
        Returns:
            int: Number of workers to use; equals self.max_workers when set, otherwise uses the system CPU count minus one, with a minimum of 1.
        """
        try:
            return self.max_workers or (os.cpu_count() or 4) - 1
        except Exception:
            return 1

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
        pass

    def log_error(self, message):
        """Persist errors to errors.log so they remain visible after the UI advances."""
        pass

    def open_pdf(self):
        """
        Initialize the PDF renderer for the configured PDF file and determine the document's total page count.
        
        This creates and opens a PdfRenderer assigned to `self.renderer`. If PyPDF is available and the renderer exposes a parsed document, the method sets `self._page_count` to the number of pages.
        """
        self.renderer = PdfRenderer(
            self.pdf_path, self.output_dir, self.pdf_bytes_cache_mb, detect_poppler_path(),
            self.log, self.log_error, self.persist_renders, self.render_cache_max_items
        )
        self.renderer.open_pdf()

        from deps import _lazy_import_pypdf
        PdfReader, pypdf_available = _lazy_import_pypdf()
        
        if pypdf_available and self.renderer.doc:
            self._page_count = len(self.renderer.doc.pages)

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
                zoom = self.zoom
            return self.renderer.render_page(page_num, zoom)
        return None

    @timer("page_processing")
    def _process_page_with_ocr(self, page_num):
        """
        Process a single PDF page by rendering it to an image, running OCR, and returning the structured result.
        
        On success returns a PageResult with the 1-based page_number, extracted text and OCR metadata (confidence, fragments, and method). If rendering or OCR fails, returns a PageResult with the 1-based page_number and an error message.
        """
        img = self.render_page_to_image(page_num)
        
        if img is None:
            return PageResult(
                page_number=page_num + 1,
                error="Rendering failed"
            )
            
        ocr_result = self._get_ocr_pipeline().extract_text_with_ocr(img)
        
        if ocr_result:
            page_result = PageResult(
                page_number=page_num + 1,
                content=ocr_result.get('text', ''),
                ocr_page_text=ocr_result.get('text', ''),
                ocr_page_confidence=ocr_result.get('confidence', 0.0),
                ocr_page_fragments=ocr_result.get('fragments', 0),
                ocr_render=ocr_result.get('method', 'unknown')
            )
        else:
            page_result = PageResult(
                page_number=page_num + 1,
                error="OCR failed"
            )
            
        return page_result

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
        """
        Orchestrates OCR processing for every page of the opened PDF and stores per-page results.
        
        Processes pages (possibly in parallel) and populates self._page_results with a PageResult for each page. When configured, a shared OCR pipeline is created and reused. Reports progress via self.progress_callback if provided, respects self.stop_event to cancel outstanding work, logs per-page errors and records an error PageResult for failed pages, and ensures the PDF renderer is closed when finished.
        """
        self.open_pdf()
        
        if self._page_count == 0:
            self.log_error("Could not determine page count")
            return
            
        self.log(f"Processing {self._page_count} pages")
        
        if self.share_ocr_instances:
            self.ocr = self._build_ocr_pipeline()
            
        try:
            worker_count = self._worker_count()
            
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_page = {
                    executor.submit(self._process_page_with_ocr, page_num): page_num
                    for page_num in range(self._page_count)
                }
                
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    
                    if self.stop_event and self.stop_event.is_set():
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                        
                    try:
                        result = future.result()
                        self._page_results[page_num] = result
                        
                        if self.progress_callback:
                            self.progress_callback((page_num + 1) / self._page_count * 100)
                            
                    except Exception as e:
                        self.log_error(f"Page {page_num + 1} processing failed: {e}")
                        self._page_results[page_num] = PageResult(
                            page_number=page_num + 1,
                            error=f"Processing failed: {e}"
                        )
                        
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