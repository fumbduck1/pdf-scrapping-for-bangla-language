"""
PDF Scraper Package

A modular PDF scraping library that provides:
- PDF rendering and caching
- OCR pipeline orchestration
- Text extraction and processing
- Reed-Solomon error correction
"""

from scraper.models import PageResult, JobResult
from scraper.pdf_renderer import PdfRenderer
from scraper.ocr_pipeline import OcrPipeline
from scraper.pdf_scraper import PDFScraper, run_pdf_job
from scraper.utils import _sentence_chunks
from deps import check_pdftoppm_available

__all__ = [
    'PageResult',
    'JobResult',
    'PdfRenderer',
    'OcrPipeline',
    'PDFScraper',
    'run_pdf_job',
    '_sentence_chunks',
    'check_pdftoppm_available',
]
