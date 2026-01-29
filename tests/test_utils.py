"""Unit tests for utility functions"""
import os
import sys
import unittest
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    _sanitize_tessdata_prefix,
    _split_langs,
    resolve_tesseract_cmd,
    check_tesseract_ready,
    validate_runtime_env,
    summarize_env,
    check_poppler_ready,
)


class TestUtils(unittest.TestCase):
    """Tests for utility functions"""
    
    def test_sanitize_tessdata_prefix(self):
        """Test tessdata prefix sanitization"""
        # Test basic sanitization
        test_path = _sanitize_tessdata_prefix("test/path")
        self.assertIsInstance(test_path, str)
        self.assertTrue(test_path)
        
        self.assertEqual(_sanitize_tessdata_prefix("'test/path'"), str(Path("test/path").resolve()))
        self.assertEqual(_sanitize_tessdata_prefix('"test/path"'), str(Path("test/path").resolve()))
        self.assertEqual(_sanitize_tessdata_prefix("  test/path  "), str(Path("test/path").resolve()))
        
        # Test with home directory
        home_dir = str(Path.home())
        self.assertEqual(_sanitize_tessdata_prefix("~/test/path").startswith(home_dir), True)
        
        # Test None and empty
        self.assertIsNone(_sanitize_tessdata_prefix(None))
        self.assertIsNone(_sanitize_tessdata_prefix(""))
    
    def test_split_langs(self):
        """Test language string splitting"""
        self.assertEqual(_split_langs("ben"), ["ben"])
        self.assertEqual(_split_langs("ben+eng"), ["ben", "eng"])
        self.assertEqual(_split_langs("ben,eng"), ["ben", "eng"])
        self.assertEqual(_split_langs("ben + eng"), ["ben", "eng"])
        self.assertEqual(_split_langs(""), [])
        self.assertEqual(_split_langs(None), [])
    
    def test_validate_runtime_env(self):
        """Test runtime environment validation"""
        errors, warnings = validate_runtime_env()
        self.assertIsInstance(errors, list)
        self.assertIsInstance(warnings, list)
    
    def test_summarize_env(self):
        """Test environment summary"""
        info, warnings, errors = summarize_env()
        self.assertIsInstance(info, list)
        self.assertIsInstance(warnings, list)
        self.assertIsInstance(errors, list)
    
    def test_check_poppler_ready(self):
        """Test Poppler availability check"""
        ok, msg = check_poppler_ready()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(msg, str)


if __name__ == "__main__":
    unittest.main()
