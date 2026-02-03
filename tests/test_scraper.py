"""Integration tests for scraper module"""
import sys
import unittest
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_manager import JobConfig, OCRConfig, RenderConfig
from scraper import PdfRenderer, OcrPipeline, PDFScraper
from deps import PDF2IMAGE_AVAILABLE, PYPDF_AVAILABLE
from PIL import Image
from unittest import mock


class TestPdfRenderer(unittest.TestCase):
    """Tests for PDF rendering"""
    
    @unittest.skipUnless(PYPDF_AVAILABLE, "pypdf not available")
    def test_pdf_renderer_init(self):
        """Test PDF renderer initialization"""
        # Create a temporary empty PDF
        test_pdf_path = Path(__file__).parent / "test_empty.pdf"
        with open(test_pdf_path, 'wb') as f:
            f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n115\n%%EOF")
        
        try:
            renderer = PdfRenderer(
                str(test_pdf_path),
                str(Path(__file__).parent),
                80,
                None,
                log=None,
                log_error=None,
                persist_renders=False
            )
            
            self.assertIsInstance(renderer, PdfRenderer)
            self.assertEqual(renderer.pdf_path, str(test_pdf_path))
            
            # Test open PDF
            self.assertTrue(renderer.open_pdf())
            self.assertIsNotNone(renderer.doc)
            
        finally:
            test_pdf_path.unlink(missing_ok=True)
    
    @unittest.skipUnless(PYPDF_AVAILABLE and PDF2IMAGE_AVAILABLE, "PDF rendering dependencies not available")
    def test_render_page(self):
        """Test page rendering (requires Poppler installation)"""
        # Create a temporary empty PDF
        test_pdf_path = Path(__file__).parent / "test_render.pdf"
        with open(test_pdf_path, 'wb') as f:
            f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n115\n%%EOF")
        
        try:
            renderer = PdfRenderer(
                str(test_pdf_path),
                str(Path(__file__).parent),
                80,
                None,
                log=None,
                log_error=None,
                persist_renders=False
            )
            
            self.assertTrue(renderer.open_pdf())
            
            # Rendering an empty PDF should return None
            result = renderer.render_page(0, 1.0)
            self.assertIsNone(result)
            
        finally:
            test_pdf_path.unlink(missing_ok=True)


class TestOcrPipeline(unittest.TestCase):
    """Tests for OCR pipeline"""
    
    def test_ocr_pipeline_init(self):
        """Test OCR pipeline initialization"""
        pipeline = OcrPipeline(
            ocr_method="easyocr",
            ocr_lang="ben",
            quality_mode=True,
            fast_mode=False,
            fast_conf_skip=0.92,
            tessdata_dir=None,
            log=None,
            log_error=None
        )
        
        self.assertIsInstance(pipeline, OcrPipeline)
        self.assertEqual(pipeline.ocr_method, "easyocr")
        self.assertEqual(pipeline.ocr_lang, "ben")
        self.assertTrue(pipeline.quality_mode)
        self.assertFalse(pipeline.fast_mode)


class TestScraperIntegration(unittest.TestCase):
    """Integration tests for scraper module"""
    
    def test_job_config(self):
        """Test PDF job configuration"""
        config = JobConfig(
            pdf_path="test.pdf",
            output_root="output",
            use_ocr=True,
            ocr=OCRConfig(
                ocr_method="easyocr",
                ocr_lang="ben",
                quality_mode=True,
                tessdata_dir=None
            ),
            render=RenderConfig(
                persist_renders=False
            ),
            max_workers=None
        )
        
        self.assertIsInstance(config, JobConfig)
        self.assertEqual(config.pdf_path, "test.pdf")
        self.assertEqual(config.output_root, "output")
        self.assertTrue(config.use_ocr)
        self.assertEqual(config.ocr.ocr_method, "easyocr")
        self.assertEqual(config.ocr.ocr_lang, "ben")
        self.assertTrue(config.ocr.quality_mode)

    def test_pdfscraper_from_job_config_output_dir(self):
        """Ensure from_job_config wires fields and derives output_dir consistently."""
        cfg = JobConfig(
            pdf_path="/tmp/sample.pdf",
            output_root="/out",
            use_ocr=True,
            ocr=OCRConfig(ocr_method="easyocr", ocr_lang="ben", quality_mode=True),
            render=RenderConfig(persist_renders=False, pdf_bytes_cache_mb=10),
        )
        scraper = PDFScraper.from_job_config(cfg)
        self.assertEqual(scraper.output_dir.replace("\\", "/"), "/out/sample")
        self.assertEqual(scraper.ocr_lang, "ben")
        self.assertEqual(scraper.ocr_method, "easyocr")


class TestRendererCache(unittest.TestCase):
    """Tests for bounded render cache behavior."""

    def test_render_cache_lru_eviction(self):
        renderer = PdfRenderer(
            pdf_path="/dev/null",
            output_dir="/tmp",
            pdf_bytes_cache_mb=1,
            poppler_path=None,
            log=None,
            log_error=None,
            persist_renders=False,
            render_cache_max_items=2,
        )

        dummy_img = Image.new("RGB", (10, 10))

        # Mock check_pdftoppm_available to return True
        # Mock subprocess.run to return dummy image data
        def mock_check_pdftoppm_available(*args, **kwargs):
            return True
            
        def mock_subprocess_run(*args, **kwargs):
            from unittest.mock import Mock
            result = Mock()
            result.returncode = 0
            # Create a minimal valid PPM file header
            ppm_header = b"P6\n100 100\n255\n"
            # Create 100x100 RGB pixel data (all black)
            ppm_data = ppm_header + b"\x00\x00\x00" * 100 * 100
            result.stdout = ppm_data
            result.stderr = b""
            return result

        renderer._pdf_bytes = b"%PDF-1.4"  # force in-memory PDF path

        with mock.patch("scraper.check_pdftoppm_available", side_effect=mock_check_pdftoppm_available), \
             mock.patch("subprocess.run", side_effect=mock_subprocess_run):
            renderer.render_page(0, 1.0)
            renderer.render_page(1, 1.0)
            renderer.render_page(2, 1.0)

        cache_keys = list(renderer._render_cache.keys())
        self.assertEqual(len(cache_keys), 2)
        self.assertEqual(cache_keys, [(1, 1.0, "png"), (2, 1.0, "png")])


class TestOcrFactorySharing(unittest.TestCase):
    """Ensure OCR factory sharing vs isolation works as configured."""

    def test_ocr_factory_share_toggle(self):
        class DummyOCR:
            def __init__(self):
                self.calls = 0

            def extract_text_with_ocr(self, img):
                self.calls += 1
                return None

        factory_calls = []

        def factory():
            inst = DummyOCR()
            factory_calls.append(inst)
            return inst

        scraper_shared = PDFScraper(
            pdf_path="/tmp/a.pdf",
            output_dir="/tmp/out",
            ocr_pipeline_factory=factory,
            share_ocr_instances=True,
        )
        o1 = scraper_shared._get_ocr_pipeline()
        o2 = scraper_shared._get_ocr_pipeline()
        self.assertIs(o1, o2)

        scraper_isolated = PDFScraper(
            pdf_path="/tmp/b.pdf",
            output_dir="/tmp/out2",
            ocr_pipeline_factory=factory,
            share_ocr_instances=False,
        )
        o3 = scraper_isolated._get_ocr_pipeline()
        o4 = scraper_isolated._get_ocr_pipeline()
        self.assertIsNot(o3, o4)
        self.assertGreaterEqual(len(factory_calls), 3)


if __name__ == "__main__":
    unittest.main()
