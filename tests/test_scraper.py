"""Integration tests for scraper module"""
import sys
import unittest
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_manager import JobConfig, OCRConfig, RenderConfig
from scraper import PdfRenderer, OcrPipeline
from deps import PDF2IMAGE_AVAILABLE, PYPDF_AVAILABLE


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


if __name__ == "__main__":
    unittest.main()
