"""Tests for the configuration manager"""
import os
import sys
import unittest
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_manager import ConfigManager, JobConfig, get_config_manager, create_job_config


class TestConfigManager(unittest.TestCase):
    """Tests for the configuration manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.config_manager = get_config_manager()
    
    def test_singleton_instance(self):
        """Test that get_config_manager returns a singleton instance"""
        instance1 = get_config_manager()
        instance2 = get_config_manager()
        self.assertIs(instance1, instance2)
    
    def test_create_job_config(self):
        """Test creating a job config with create_job_config function"""
        config = create_job_config(
            pdf_path="test.pdf",
            output_root="output",
            use_ocr=True,
            ocr_method="easyocr",
            ocr_lang="ben",
            quality_mode=True,
            fast_mode=False,
            persist_renders=True,
            max_workers=4
        )
        
        self.assertIsInstance(config, JobConfig)
        self.assertEqual(config.pdf_path, "test.pdf")
        self.assertEqual(config.output_root, "output")
        self.assertTrue(config.use_ocr)
        self.assertEqual(config.ocr.ocr_method, "easyocr")
        self.assertEqual(config.ocr.ocr_lang, "ben")
        self.assertTrue(config.ocr.quality_mode)
        self.assertFalse(config.ocr.fast_mode)
        self.assertTrue(config.render.persist_renders)
        self.assertEqual(config.max_workers, 4)
    
    def test_default_config(self):
        """Test getting default configuration"""
        config = self.config_manager.get_default_config()
        
        self.assertIsInstance(config, JobConfig)
        self.assertEqual(config.pdf_path, "")
        self.assertEqual(config.output_root, "")
        self.assertTrue(config.use_ocr)
        self.assertEqual(config.ocr.ocr_method, "easyocr")
        self.assertEqual(config.ocr.ocr_lang, "ben")
        self.assertFalse(config.render.persist_renders)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Create an invalid configuration
        config = self.config_manager.get_default_config()
        errors = self.config_manager.validate_config(config)
        self.assertGreater(len(errors), 0)
        
        # Create a valid configuration
        test_pdf_path = Path(__file__).parent / "test_valid.pdf"
        test_output_path = Path(__file__).parent / "test_output"
        
        try:
            # Create a temporary valid PDF file
            with open(test_pdf_path, 'wb') as f:
                f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n115\n%%EOF")
            
            # Create output directory
            test_output_path.mkdir(exist_ok=True)
            
            config = create_job_config(
                pdf_path=str(test_pdf_path),
                output_root=str(test_output_path)
            )
            
            errors = self.config_manager.validate_config(config)
            self.assertEqual(len(errors), 0)
        finally:
            # Clean up
            if test_pdf_path.exists():
                test_pdf_path.unlink()
            if test_output_path.exists():
                test_output_path.rmdir()
    
    def test_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            "pdf_path": "test.pdf",
            "output_root": "output",
            "use_ocr": False,
            "ocr_method": "tesseract",
            "ocr_lang": "eng",
            "quality_mode": False,
            "fast_mode": True,
            "fast_confidence_skip": 0.8,
            "tessdata_dir": "/usr/share/tessdata",
            "zoom": 10.0,
            "high_dpi_zoom": 15.0,
            "high_dpi_retry_conf": 0.9,
            "pdf_bytes_cache_mb": 100,
            "persist_renders": True,
            "header_footer_crop_pct": 0.1,
            "watermark_flatten": False,
            "watermark_clip_threshold": 240,
            "watermark_retry_conf": 0.8,
            "quantize_levels": 16,
            "quantize_dither": False,
            "third_pass_scale": 1.5,
            "text_layer_first": False,
            "text_layer_lang_min_ratio": 0.4,
            "text_layer_min_ben_chars": 10,
            "max_workers": 8
        }
        
        config = self.config_manager.from_dict(config_dict)
        
        self.assertEqual(config.pdf_path, "test.pdf")
        self.assertEqual(config.output_root, "output")
        self.assertFalse(config.use_ocr)
        self.assertEqual(config.ocr.ocr_method, "tesseract")
        self.assertEqual(config.ocr.ocr_lang, "eng")
        self.assertFalse(config.ocr.quality_mode)
        self.assertTrue(config.ocr.fast_mode)
        self.assertEqual(config.ocr.fast_confidence_skip, 0.8)
        self.assertEqual(config.ocr.tessdata_dir, "/usr/share/tessdata")
        self.assertEqual(config.render.zoom, 10.0)
        self.assertEqual(config.render.high_dpi_zoom, 15.0)
        self.assertEqual(config.render.high_dpi_retry_conf, 0.9)
        self.assertEqual(config.render.pdf_bytes_cache_mb, 100)
        self.assertTrue(config.render.persist_renders)
        self.assertEqual(config.preprocess.header_footer_crop_pct, 0.1)
        self.assertFalse(config.preprocess.watermark_flatten)
        self.assertEqual(config.preprocess.watermark_clip_threshold, 240)
        self.assertEqual(config.preprocess.watermark_retry_conf, 0.8)
        self.assertEqual(config.preprocess.quantize_levels, 16)
        self.assertFalse(config.preprocess.quantize_dither)
        self.assertEqual(config.preprocess.third_pass_scale, 1.5)
        self.assertFalse(config.text_layer.text_layer_first)
        self.assertEqual(config.text_layer.text_layer_lang_min_ratio, 0.4)
        self.assertEqual(config.text_layer.text_layer_min_ben_chars, 10)
        self.assertEqual(config.max_workers, 8)
    
    def test_from_env(self):
        """Test creating config from environment variables"""
        # Set environment variables
        os.environ["PDF_SCRAPER_PDF_PATH"] = "env_test.pdf"
        os.environ["PDF_SCRAPER_OUTPUT_ROOT"] = "env_output"
        os.environ["PDF_SCRAPER_USE_OCR"] = "false"
        os.environ["PDF_SCRAPER_OCR_METHOD"] = "tesseract"
        os.environ["PDF_SCRAPER_OCR_LANG"] = "eng+ben"
        os.environ["PDF_SCRAPER_QUALITY_MODE"] = "true"
        os.environ["PDF_SCRAPER_FAST_MODE"] = "false"
        os.environ["PDF_SCRAPER_FAST_CONFIDENCE_SKIP"] = "0.95"
        os.environ["PDF_SCRAPER_TESSDATA_DIR"] = "/env/tessdata"
        os.environ["PDF_SCRAPER_ZOOM"] = "8.0"
        os.environ["PDF_SCRAPER_HIGH_DPI_ZOOM"] = "12.0"
        os.environ["PDF_SCRAPER_HIGH_DPI_RETRY_CONF"] = "0.93"
        os.environ["PDF_SCRAPER_PDF_BYTES_CACHE_MB"] = "120"
        os.environ["PDF_SCRAPER_PERSIST_RENDERS"] = "true"
        os.environ["PDF_SCRAPER_MAX_WORKERS"] = "6"
        
        try:
            config = self.config_manager.from_env()
            
            self.assertEqual(config.pdf_path, "env_test.pdf")
            self.assertEqual(config.output_root, "env_output")
            self.assertFalse(config.use_ocr)
            self.assertEqual(config.ocr.ocr_method, "tesseract")
            self.assertEqual(config.ocr.ocr_lang, "eng+ben")
            self.assertTrue(config.ocr.quality_mode)
            self.assertFalse(config.ocr.fast_mode)
            self.assertEqual(config.ocr.fast_confidence_skip, 0.95)
            self.assertEqual(config.ocr.tessdata_dir, "/env/tessdata")
            self.assertEqual(config.render.zoom, 8.0)
            self.assertEqual(config.render.high_dpi_zoom, 12.0)
            self.assertEqual(config.render.high_dpi_retry_conf, 0.93)
            self.assertEqual(config.render.pdf_bytes_cache_mb, 120)
            self.assertTrue(config.render.persist_renders)
            self.assertEqual(config.max_workers, 6)
        finally:
            # Clean up environment variables
            for key in os.environ:
                if key.startswith("PDF_SCRAPER_"):
                    del os.environ[key]


if __name__ == "__main__":
    unittest.main()