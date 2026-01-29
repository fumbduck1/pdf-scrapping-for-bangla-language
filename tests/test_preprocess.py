"""Unit tests for image preprocessing functions"""
import sys
import unittest
from pathlib import Path
from PIL import Image, ImageOps

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess import (
    quantize_params,
    crop_header_footer,
    flatten_background,
    estimate_density,
    choose_psm,
    maybe_split_columns,
    upscale_for_retry,
    preprocess_image_for_ocr,
)


class TestPreprocess(unittest.TestCase):
    """Tests for image preprocessing functions"""
    
    def setUp(self):
        """Create test images"""
        self.test_image = Image.new('L', (800, 600), color=128)
        
    def test_quantize_params(self):
        """Test quantization parameter selection"""
        # Test Bangla
        levels, dither = quantize_params("ben", fast_mode=False)
        self.assertGreater(levels, 0)
        self.assertIsInstance(dither, bool)
        
        # Test English
        levels, dither = quantize_params("eng", fast_mode=True)
        self.assertGreater(levels, 0)
        self.assertIsInstance(dither, bool)
        
        # Test Bangla+English
        levels, dither = quantize_params("ben+eng", fast_mode=False)
        self.assertGreater(levels, 0)
        self.assertIsInstance(dither, bool)
    
    def test_crop_header_footer(self):
        """Test header/footer cropping"""
        cropped = crop_header_footer(self.test_image)
        self.assertIsInstance(cropped, Image.Image)
        self.assertLess(cropped.height, self.test_image.height)
    
    def test_flatten_background(self):
        """Test background flattening"""
        flattened = flatten_background(self.test_image, clip=200)
        self.assertIsInstance(flattened, Image.Image)
    
    def test_estimate_density(self):
        """Test density estimation"""
        density = estimate_density(self.test_image)
        self.assertIsInstance(density, float)
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)
    
    def test_choose_psm(self):
        """Test PSM (Page Segmentation Mode) selection"""
        psm = choose_psm(self.test_image, segment_count=1)
        self.assertIsInstance(psm, int)
        self.assertGreaterEqual(psm, 0)
        self.assertLessEqual(psm, 13)
    
    def test_maybe_split_columns(self):
        """Test column detection"""
        # Create a wide image that might be split into columns
        wide_image = Image.new('L', (2000, 800), color=255)
        
        segments = maybe_split_columns(wide_image, fast_mode=False)
        self.assertIsInstance(segments, list)
        self.assertGreaterEqual(len(segments), 1)
        
        # Fast mode should return single segment
        segments = maybe_split_columns(wide_image, fast_mode=True)
        self.assertEqual(len(segments), 1)
    
    def test_upscale_for_retry(self):
        """Test image upscale"""
        scaled = upscale_for_retry(self.test_image, scale=1.5)
        self.assertIsInstance(scaled, Image.Image)
        self.assertGreater(scaled.width, self.test_image.width)
        self.assertGreater(scaled.height, self.test_image.height)
    
    def test_preprocess_image_for_ocr(self):
        """Test complete preprocessing pipeline"""
        processed = preprocess_image_for_ocr(
            self.test_image,
            ocr_lang="ben",
            fast_mode=False,
            quality_mode=True
        )
        self.assertIsInstance(processed, Image.Image)


if __name__ == "__main__":
    unittest.main()
