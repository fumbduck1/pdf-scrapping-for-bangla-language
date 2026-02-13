"""EPUB Scraper module - extract text from EPUB files with optional OCR for images."""
import os
import sys
import re
from pathlib import Path
from io import BytesIO
from datetime import datetime
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Any

from PIL import Image

from config_manager import JobConfig
from deps import _lazy_import_epublib, EPUBLIB_AVAILABLE, EASYOCR_AVAILABLE, TESSERACT_AVAILABLE

# Lazy import for ebooklib to prevent ImportError when module not installed
epub: Any = None
from scraper import PageResult
from logger import get_logger


@dataclass
class EPUBResult:
    """Result from EPUB scraping operation"""
    scrape_ok: bool
    save_ok: bool
    stats: dict
    output_dir: str


def run_epub_job(job_config: JobConfig, stop_event, log_cb) -> dict:
    """Run an EPUB scraping job using the provided configuration."""
    if not EPUBLIB_AVAILABLE:
        if log_cb:
            log_cb("EPUB support not available: ebooklib library not installed")
        return {"scrape_ok": False, "save_ok": False, "stats": {}, "output_dir": job_config.output_root}
    
    # Initialize ebooklib using lazy import
    global epub
    epub = _lazy_import_epublib()
    if epub is None:
        if log_cb:
            log_cb("EPUB support not available: ebooklib library not installed")
        return {"scrape_ok": False, "save_ok": False, "stats": {}, "output_dir": job_config.output_root}
    
    epub_name = Path(job_config.input_path).stem
    epub_output = os.path.join(job_config.output_root, epub_name)
    scraper = None
    try:
        scraper = EPUBScraper.from_job_config(job_config, progress_callback=log_cb, stop_event=stop_event)
        if log_cb:
            log_cb("Scraping EPUB...")
        scrape_ok = scraper.scrape_all_content()
        if log_cb:
            log_cb("Saving results..." if scrape_ok else "Saving partial results...")
        save_ok = scraper.save_results()
        stats = scraper.results.get('statistics', {})
        return {
            "scrape_ok": scrape_ok,
            "save_ok": save_ok,
            "stats": stats,
            "output_dir": epub_output,
        }
    except Exception as e:
        if log_cb:
            log_cb(f"Error: {e}")
        if scraper:
            scraper.log_error(f"Batch error on {job_config.input_path}: {e}")
        else:
            os.makedirs(epub_output, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(epub_output, "errors.log"), "a", encoding="utf-8") as f:
                f.write(f"[{ts}] Batch error on {job_config.input_path}: {e}\n")
        return {"scrape_ok": False, "save_ok": False, "stats": {}, "output_dir": epub_output}


class EPUBScraper:
    """Scraper for EPUB files"""
    
    def __init__(
        self,
        epub_path,
        output_dir,
        use_ocr=True,
        ocr_method='easyocr',
        ocr_lang='ben',
        progress_callback=None,
        tessdata_dir=None,
        stop_event=None,
        quality_mode=True,
        persist_renders=False,
        max_workers=None,
        fast_mode=False,
        # Reed-Solomon error correction parameters
        rs_enabled=False,
        rs_error_correction_bytes=10,
        rs_block_size=1024,
        rs_enable_correction=True,
        rs_verify_only=False,
        **kwargs
    ):
        """
        Construct an EPUBScraper configured for extracting text and optional OCR (and optional Reed–Solomon encoding).
        
        Parameters:
            epub_path (str): Path to the input EPUB file.
            output_dir (str): Directory where extraction outputs will be written.
            use_ocr (bool): Enable OCR on image resources when True.
            ocr_method (str): OCR backend identifier (e.g., 'easyocr', 'tesseract').
            ocr_lang (str): Language code used by the OCR engine.
            progress_callback (callable|None): Optional callback invoked with progress/log messages.
            tessdata_dir (str|None): Custom Tesseract tessdata directory when using Tesseract.
            stop_event (threading.Event|None): Optional event to request early cancellation.
            quality_mode (bool): Prefer higher-quality OCR/rendering when True.
            persist_renders (bool): If True, save intermediate rendered images to disk.
            max_workers (int|None): Maximum number of worker threads for parallel tasks, if supported.
            fast_mode (bool): Enable faster, lower-latency OCR/render paths when True.
        
            rs_enabled (bool): Enable Reed–Solomon error-correction encoding of saved content.
            rs_error_correction_bytes (int): Number of parity bytes used by the RS encoder.
            rs_block_size (int): Block size (in bytes) used when encoding content with RS.
            rs_enable_correction (bool): If True, encode output with error-correction parity.
            rs_verify_only (bool): If True, only verify RS encoding without writing correction data.
        
        Notes:
            - If Reed–Solomon support is requested but the RS library is unavailable, RS will be disabled and an error will be logged.
            - The constructor creates the output directory if it does not exist and initializes the OCR pipeline.
        """
        self.epub_path = epub_path
        self.output_dir = output_dir
        self.use_ocr = use_ocr
        
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
        self.ocr_method = ocr_method
        self.persist_renders = bool(persist_renders)
        self.max_workers_override = max_workers
        self.user_lang = (ocr_lang or "ben").strip()
        self.ocr_lang = self.user_lang
        self.quality_mode = bool(quality_mode)
        self.fast_mode = fast_mode
        self.progress_callback = progress_callback
        self.stop_event = stop_event
        self.tessdata_dir = tessdata_dir
        
        self.results = {
            'metadata': {},
            'pages': {},
            'statistics': {},
            'extraction_log': []
        }
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = get_logger()
        
        # Initialize OCR pipeline
        from scraper import OcrPipeline
        from constants import (
            FAST_CONFIDENCE_SKIP,
            HEADER_FOOTER_CROP_PCT,
            WATERMARK_FLATTEN,
            WATERMARK_CLIP_THRESHOLD
        )
        
        self.ocr_pipeline = OcrPipeline(
            ocr_method=self.ocr_method,
            ocr_lang=self.ocr_lang,
            quality_mode=self.quality_mode,
            fast_mode=self.fast_mode,
            fast_conf_skip=FAST_CONFIDENCE_SKIP,
            tessdata_dir=self.tessdata_dir,
            log=self.log,
            log_error=self.log_error,
            header_footer_crop_pct=HEADER_FOOTER_CROP_PCT,
            watermark_flatten=WATERMARK_FLATTEN,
            watermark_clip_threshold=WATERMARK_CLIP_THRESHOLD
        )
    
    @classmethod
    def from_job_config(cls, job_config: JobConfig, progress_callback=None, stop_event=None):
        """
        Create an EPUBScraper configured from a JobConfig.
        
        Parameters:
            job_config (JobConfig): Source configuration containing input/output paths, OCR, render, worker, and Reed–Solomon settings used to initialize the scraper.
            progress_callback (callable, optional): Callback invoked with progress/log messages; may be None.
            stop_event (threading.Event, optional): Event used to signal early cancellation; may be None.
        
        Returns:
            EPUBScraper: An instance initialized according to the provided JobConfig.
        """
        return cls(
            epub_path=job_config.input_path,
            output_dir=os.path.join(job_config.output_root, Path(job_config.input_path).stem),
            use_ocr=job_config.use_ocr,
            ocr_method=job_config.ocr.ocr_method,
            ocr_lang=job_config.ocr.ocr_lang,
            quality_mode=job_config.ocr.quality_mode,
            fast_mode=job_config.ocr.fast_mode,
            persist_renders=job_config.render.persist_renders,
            max_workers=job_config.max_workers,
            tessdata_dir=job_config.ocr.tessdata_dir,
            progress_callback=progress_callback,
            stop_event=stop_event,
            # Reed-Solomon error correction parameters
            rs_enabled=job_config.rs_correction.enabled,
            rs_error_correction_bytes=job_config.rs_correction.error_correction_bytes,
            rs_block_size=job_config.rs_correction.block_size,
            rs_enable_correction=job_config.rs_correction.enable_correction,
            rs_verify_only=job_config.rs_correction.verify_only,
        )
    
    def log(self, message):
        """
        Append a timestamped message to the extraction log, notify the progress callback, and record the message with the logger if available.
        
        This method adds a line prefixed with the current time to self.results['extraction_log'], invokes self.progress_callback(message) when a callback is set (exceptions raised by the callback are caught and logged via self.logger.error if a logger exists), and calls self.logger.info(message) when a logger is available.
        """
        self.results['extraction_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if self.progress_callback:
            try:
                self.progress_callback(message)
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.error(f"Error in progress callback: {str(e)}")
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(message)
    
    def log_error(self, message):
        """Log an error message."""
        self.results['extraction_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {message}")
        if self.progress_callback:
            try:
                self.progress_callback(f"ERROR: {message}")
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.error(f"Error in progress callback: {str(e)}")
        if hasattr(self, 'logger') and self.logger:
            self.logger.error(message)
    
    def open_epub(self):
        """Open the EPUB file and initialize parsing."""
        try:
            self.book = epub.read_epub(self.epub_path)
            self.log(f"EPUB opened successfully. Type: {self.book.get_metadata('DC', 'type')}")
            self.log(f"Title: {self.book.get_metadata('DC', 'title')}")
            self.log(f"Author: {self.book.get_metadata('DC', 'creator')}")
            return True
        except Exception as e:
            self.log_error(f"Failed to open EPUB: {e}")
            return False
    
    def scrape_all_content(self):
        """Scrape all content from the EPUB."""
        if not self.open_epub():
            return False
        
        try:
            self.extract_metadata()
            self.extract_text_content()
            self.extract_images()
            self._calculate_statistics()
            return True
        except Exception as e:
            self.log_error(f"Error during scraping: {e}")
            return False
    
    def extract_metadata(self):
        """Extract metadata from the EPUB."""
        try:
            self.results['metadata'] = {
                'title': self.book.get_metadata('DC', 'title'),
                'author': self.book.get_metadata('DC', 'creator'),
                'language': self.book.get_metadata('DC', 'language'),
                'publisher': self.book.get_metadata('DC', 'publisher'),
                'date': self.book.get_metadata('DC', 'date'),
                'format': self.book.get_metadata('DC', 'format'),
                'identifier': self.book.get_metadata('DC', 'identifier'),
            }
            self.log("Metadata extracted successfully")
        except Exception as e:
            self.log_error(f"Metadata extraction failed: {e}")
    
    def extract_text_content(self):
        """Extract text content from EPUB chapters in correct order."""
        self.log("Extracting text content...")
        
        # Get spine items (correct reading order)
        spine_items = []
        for entry in self.book.spine:
            # Extract item_id from spine entry (handle tuple format: (item_id, linear))
            item_id = entry if isinstance(entry, str) else entry[0]
            item = self.book.get_item_with_id(item_id)
            if item:
                spine_items.append(item)
                
        self.log(f"Spine items (reading order): {len(spine_items)}")
        
        # Extract text from spine items (correct order)
        page_counter = 0
        total_spine_items = max(1, len(spine_items))  # Guard against division by zero
        for idx, item in enumerate(spine_items):
            self.log(f"Spine item {idx} type: {item.get_type()}, name: {item.get_name()}")
            
            if self.stop_event and self.stop_event.is_set():
                self.log("Stop requested; aborting remaining content")
                break
                
            # Calculate and report progress
            progress = ((idx + 1) / total_spine_items) * 100
            if self.progress_callback:
                try:
                    self.progress_callback(progress)
                except Exception as e:
                    self.log(f"Error in progress callback: {str(e)}")
                
            if isinstance(item, epub.EpubHtml):
                chapter_title = self._extract_chapter_title(item)
                self.log(f"Processing chapter: {chapter_title}")
                
                text_content = self._extract_text_from_html(item.get_content())
                self.log(f"Extracted text length: {len(text_content)}")
                
                page_result = PageResult(
                    page_number=page_counter,
                    content=text_content,
                    ocr_page_text=text_content,
                    ocr_page_confidence=1.0,
                    ocr_page_fragments=1
                )
                self.results['pages'][f'page_{page_counter}'] = asdict(page_result)
                page_counter += 1
                
        # Extract remaining HTML items not in spine (fallback)
        all_items = list(self.book.get_items())
        processed_item_ids = {item.get_id() for item in spine_items}
        
        for item in all_items:
            if isinstance(item, epub.EpubHtml) and item.get_id() not in processed_item_ids:
                chapter_title = self._extract_chapter_title(item)
                self.log(f"Processing additional chapter: {chapter_title}")
                
                text_content = self._extract_text_from_html(item.get_content())
                
                page_result = PageResult(
                    page_number=page_counter,
                    content=text_content,
                    ocr_page_text=text_content,
                    ocr_page_confidence=1.0,
                    ocr_page_fragments=1
                )
                self.results['pages'][f'page_{page_counter}'] = asdict(page_result)
                page_counter += 1
    
    def _extract_chapter_title(self, item):
        """Extract chapter title from item."""
        title = item.get_name()
        try:
            # Try to extract from metadata or content
            if hasattr(item, 'title') and item.title:
                title = item.title
        except Exception:
            pass
        return title
    
    def _extract_text_from_html(self, html_content):
        """Extract plain text from HTML content preserving structure."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.extract()
                
            # Extract text with preserved structure
            # Add newlines for block elements to preserve paragraph structure
            block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'blockquote', 'pre', 'li']
            for tag in block_tags:
                for element in soup.find_all(tag):
                    element.append('\n')
            
            text = soup.get_text()
            
            # Clean up whitespace: replace multiple newlines with single, multiple spaces with single
            text = re.sub(r'\n\s*\n', '\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
            
            return text
        except ImportError:
            # Fallback to basic parsing if BeautifulSoup not available
            try:
                text = html_content.decode('utf-8', errors='ignore')
                text = re.sub(r'<[^>]+>', '', text)
                # Clean up whitespace
                text = re.sub(r'\n\s*\n', '\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
                return text
            except Exception as e:
                self.log_error(f"HTML parsing error: {e}")
                return ""
        except Exception as e:
            self.log_error(f"HTML parsing error: {e}")
            return ""
    
    def extract_images(self):
        """Extract images from EPUB and record their presence with proper numbering."""
        self.log("Extracting images...")
        items = list(self.book.get_items())
        
        # Find the highest page number currently in results
        max_page = -1
        for page_key in self.results['pages']:
            if page_key.startswith('page_'):
                page_num = int(page_key.split('_')[1])
                if page_num > max_page:
                    max_page = page_num
        
        image_counter = max_page + 1
        
        for item in items:
            if isinstance(item, epub.EpubImage):
                try:
                    self.log(f"Found image: {item.get_name()}")
                    
                    # Process image with OCR if OCR is enabled
                    ocr_text = "[Image]"
                    ocr_confidence = 0.0
                    ocr_fragments = 1
                    
                    if self.use_ocr:
                        try:
                            # Load image from EPUB item
                            from PIL import Image
                            from io import BytesIO
                            image_data = BytesIO(item.get_content())
                            img = Image.open(image_data)
                            
                            # Use OCR pipeline to extract text from image
                            ocr_result = self.ocr_pipeline.extract_text_with_ocr(img)
                            if ocr_result:
                                ocr_text = ocr_result.get('text', '[Image]')
                                ocr_confidence = ocr_result.get('avg_confidence', 0.0)
                                ocr_fragments = ocr_result.get('fragments', 1)
                            
                        except Exception as ocr_e:
                            self.log_error(f"OCR error for image {item.get_name()}: {ocr_e}")
                    
                    page_result = PageResult(
                        page_number=image_counter,
                        content=f"[Image: {item.get_name()}]",
                        ocr_page_text=ocr_text,
                        ocr_page_confidence=ocr_confidence,
                        ocr_page_fragments=ocr_fragments
                    )
                    self.results['pages'][f'page_{image_counter}'] = asdict(page_result)
                    image_counter += 1
                    
                except Exception as e:
                    self.log_error(f"Image processing error: {e}")
    
    def _calculate_statistics(self):
        """Calculate scraping statistics."""
        total_pages = len(self.results['pages'])
        total_text_length = sum(len(p.get('content', '')) for p in self.results['pages'].values())
        pages_with_ocr = sum(1 for p in self.results['pages'].values() if p.get('ocr_page_text'))
        total_ocr_chars = sum(len(p.get('ocr_page_text', '')) for p in self.results['pages'].values())
        
        self.results['statistics'] = {
            'total_pages': total_pages,
            'total_text_length': total_text_length,
            'pages_with_ocr': pages_with_ocr,
            'total_ocr_chars': total_ocr_chars
        }
    
    def save_results(self):
        """
        Persist extracted EPUB results to the scraper's output directory.
        
        Saves page contents to content.txt, metadata to metadata.txt, statistics to statistics.txt,
        and the extraction log to extraction.log. If Reed–Solomon (RS) error correction is enabled
        and an RS corrector is available, also writes an RS-encoded file content.rs using the
        corrector; in that case a plain content.txt is still written.
        
        Returns:
            bool: `True` if all files were written successfully, `False` otherwise.
        """
        try:
            # Save text content - with or without Reed-Solomon correction
            if self.rs_enabled and self.rs_corrector:
                text_output_path = os.path.join(self.output_dir, "content.txt")
                rs_output_path = os.path.join(self.output_dir, "content.rs")
                
                # Collect and write plain text file
                with open(text_output_path, "w", encoding="utf-8") as f:
                    # Sort pages by numeric order (extract integer from page key like "page_123")
                    for page_key in sorted(self.results['pages'].keys(), key=lambda k: int(k.split('_')[1])):
                        page = self.results['pages'][page_key]
                        f.write(f"=== Page {page_key} ===\n")
                        f.write(page.get('content', '') + "\n\n")
                
                # Encode and save RS-encoded version
                text_to_encode = ""
                for page_key in sorted(self.results['pages'].keys(), key=lambda k: int(k.split('_')[1])):
                    page = self.results['pages'][page_key]
                    text_to_encode += f"=== Page {page_key} ===\n"
                    text_to_encode += page.get('content', '') + "\n\n"
                
                self.rs_corrector.encode_and_save(text_to_encode, rs_output_path)
                self.log(f"Saved: content.rs (RS-encoded)")
            else:
                # Save plain text only
                text_output_path = os.path.join(self.output_dir, "content.txt")
                with open(text_output_path, "w", encoding="utf-8") as f:
                    # Sort pages by numeric order (extract integer from page key like "page_123")
                    for page_key in sorted(self.results['pages'].keys(), key=lambda k: int(k.split('_')[1])):
                        page = self.results['pages'][page_key]
                        f.write(f"=== Page {page_key} ===\n")
                        f.write(page.get('content', '') + "\n\n")
            
            # Save metadata
            metadata_output_path = os.path.join(self.output_dir, "metadata.txt")
            with open(metadata_output_path, "w", encoding="utf-8") as f:
                for key, value in self.results['metadata'].items():
                    f.write(f"{key}: {value}\n")
            
            # Save statistics
            stats_output_path = os.path.join(self.output_dir, "statistics.txt")
            with open(stats_output_path, "w", encoding="utf-8") as f:
                for key, value in self.results['statistics'].items():
                    f.write(f"{key}: {value}\n")
            
            # Save extraction log
            log_output_path = os.path.join(self.output_dir, "extraction.log")
            with open(log_output_path, "w", encoding="utf-8") as f:
                for entry in self.results['extraction_log']:
                    f.write(entry + "\n")
            
            return True
        except Exception as e:
            self.log_error(f"Failed to save results: {e}")
            return False