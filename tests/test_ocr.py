"""Unit tests for OCR engine functions"""
import sys
import unittest
import warnings
from pathlib import Path
from PIL import Image

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress EasyOCR warnings about deprecated PyTorch quantization
warnings.filterwarnings("ignore", category=DeprecationWarning, module="easyocr")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.ao.quantization")

from ocr_tesseract import (
    build_tess_config,
    run_tesseract_pass,
    score_result,
    tesseract_best_for_segment,
)
from ocr_easyocr import (
    map_easyocr_langs,
    get_easyocr_reader,
    run_easyocr_pass,
)
from deps import TESSERACT_AVAILABLE, EASYOCR_AVAILABLE


class TestOCR(unittest.TestCase):
    """Tests for OCR engine functions"""
    
    def setUp(self):
        """Create test image"""
        self.test_image = Image.new('L', (800, 600), color=255)
        
    def test_map_easyocr_langs(self):
        """Test language code mapping"""
        self.assertEqual(map_easyocr_langs("ben"), ["bn"])
        self.assertEqual(map_easyocr_langs("eng"), ["en"])
        self.assertEqual(map_easyocr_langs("ben+eng"), ["bn", "en"])
        self.assertEqual(map_easyocr_langs("bn"), ["bn"])
        self.assertEqual(map_easyocr_langs("en"), ["en"])
        self.assertEqual(map_easyocr_langs(""), ["en"])
        self.assertEqual(map_easyocr_langs(None), ["en"])
    
    def test_build_tess_config(self):
        """Test Tesseract configuration builder"""
        config = build_tess_config("ben", quality_mode=True)
        self.assertIsInstance(config, str)
        self.assertIn("--psm", config)
        self.assertIn("--oem", config)
        
        config = build_tess_config("eng", quality_mode=False)
        self.assertIsInstance(config, str)
    
    def test_score_result(self):
        """Test result scoring"""
        result = {
            'text': 'Test text',
            'avg_confidence': 0.9,
            'fragments': 2
        }
        score = score_result(result)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)
        
        # Test None result
        self.assertEqual(score_result(None), 0)
    
    @unittest.skipUnless(TESSERACT_AVAILABLE, "Tesseract not available")
    def test_run_tesseract_pass(self):
        """Test Tesseract single pass (requires Tesseract installation)"""
        result = run_tesseract_pass(
            self.test_image,
            ocr_lang="eng",
            quality_mode=True,
            log=None,
            log_error=None
        )
        self.assertIsNone(result)  # Empty image should return None
    
    @unittest.skipUnless(EASYOCR_AVAILABLE, "EasyOCR not available")
    def test_get_easyocr_reader(self):
        """Test EasyOCR reader initialization (requires EasyOCR installation)"""
        import threading
        # Suppress EasyOCR warnings for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            reader = get_easyocr_reader(
                "en",
                gpu=False,
                lock=threading.Lock(),
                existing_reader=None,
                log=None,
                log_error=None
            )
        self.assertIsNotNone(reader)
    
    @unittest.skipUnless(EASYOCR_AVAILABLE, "EasyOCR not available")
    def test_run_easyocr_pass(self):
        """Test EasyOCR single pass (requires EasyOCR installation)"""
        import threading
        # Suppress EasyOCR warnings for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            reader = get_easyocr_reader(
                "en",
                gpu=False,
                lock=threading.Lock(),
                existing_reader=None,
                log=None,
                log_error=None
            )
            result = run_easyocr_pass(
                self.test_image,
                lock=threading.Lock(),
                reader=reader,
                log=None,
                log_error=None
            )
        self.assertIsNone(result)  # Empty image should return None


if __name__ == "__main__":
    unittest.main()
