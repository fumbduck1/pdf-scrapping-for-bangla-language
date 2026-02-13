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
        """Initialize PDF scraper."""
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
        if self.share_ocr_instances and self.ocr:
            return self.ocr
        pipeline = self._ocr_factory()
        if self.share_ocr_instances:
            self.ocr = pipeline
        return pipeline

    def setup_directories(self):
        """Create all necessary directories upfront."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.renders_dir = os.path.join(self.output_dir, 'renders')
            os.makedirs(self.renders_dir, exist_ok=True)
        except Exception:
            pass

    def _worker_count(self):
        """Optimized worker pool size for parallel OCR processing."""
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
        """Delete renders directory after processing to save space."""
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
        """Open PDF document."""
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
        """Optimized preprocessing with advanced quantization for Bengali/English text."""
        return self._get_ocr_pipeline().preprocess_image_for_ocr(image_path)

    def render_page_to_image(self, page_num, zoom=None):
        """Render a single page using pdf2image/Poppler; returns (PIL image, saved_path|None)."""
        if self.renderer:
            if zoom is None:
                zoom = self.zoom
            return self.renderer.render_page(page_num, zoom)
        return None

    @timer("page_processing")
    def _process_page_with_ocr(self, page_num):
        """Optimized page processing with intelligent retry logic and error recovery."""
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
        """Strip zero-width characters and normalize whitespace."""
        return normalize_text(text)

    def _bangla_ratio(self, text: str):
        """Return (ratio, count) of Bangla characters in the text."""
        return bangla_ratio(text)

    def _flatten_background(self, image, clip=None):
        clip_val = self.watermark_clip_threshold if clip is None else clip
        return preproc.flatten_background(image, clip=clip_val)

    def _choose_psm(self, image, segment_count):
        return preproc.choose_psm(image, segment_count)

    def _extract_text_layer(self, page):
        """Fast path: pull native text from the PDF; returns normalized string or ''."""
        try:
            text = page.extract_text()
            if text:
                return self._normalize_text(text)
        except Exception as e:
            self.log_error(f"Text layer extraction failed: {e}")
            
        return ""

    def _score_result(self, res):
        """Enhanced scoring algorithm for OCR result comparison."""
        return ocr_t.score_result(res)

    def _run_tesseract_pass(self, image, extra_config=None, extra_dilate=False, psm=None):
        """Run one Tesseract pass on a PIL image with optional extra dilation and config."""
        return self._get_ocr_pipeline()._run_tesseract_pass(image, extra_config, extra_dilate, psm)

    def _get_easyocr_reader(self):
        """Delegate to OCR pipeline's _get_easyocr_reader method."""
        return self._get_ocr_pipeline()._get_easyocr_reader()

    def _run_easyocr_pass(self, image):
        """Delegate to OCR pipeline's _run_easyocr_pass method."""
        return self._get_ocr_pipeline()._run_easyocr_pass(image)

    @timer("page_processing")
    def scrape_all_pages(self):
        """Scrape all pages with optional parallel OCR per page."""
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
        """Save results with layout-preserving output formats."""
        self._save_plain_text_files()
        self._save_rs_encoded_text()
        self._save_rs_encoded_sentences()

    def _save_plain_text_files(self):
        """Save plain text files without Reed-Solomon correction."""
        ordered_pages = sorted(self._page_results.items(), key=lambda x: x[0])
        page_texts = [result.content for _, result in ordered_pages]
        
        txt_path = Path(self.output_dir) / "output.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(page_texts))
            
        self.log(f"Plain text saved to {txt_path}")

    def _save_rs_encoded_text(self):
        """Save RS-encoded text files."""
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
        """Save RS-encoded sentences file."""
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
        """Decode and verify an RS-encoded text file."""
        if self.rs_error_correction_bytes > 0:
            corrector = rs.create_rs_corrector(self.rs_error_correction_bytes)
            return corrector.load_and_decode(rs_filename)
        return None, False, 0

    def _verify_rs_text_files(self):
        """Verify all RS-encoded files in output directory."""
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
    """Run a single PDF job using the provided configuration."""
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
        scraper.log_error(f"Job failed: {e}")
        
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        return {
            "scrape_ok": False,
            "output_dir": job_config.output_root,
            "num_pages": 0,
            "num_errors": 1,
            "num_warnings": 0,
            "runtime_seconds": runtime,
            "errors": [str(e)],
            "warnings": [],
        }
