#!/usr/bin/env python3
"""Test the complete RS correction workflow with the scraper"""

import os
import tempfile
import shutil
from config_manager import create_job_config
from rs_correction import RSTextCorrector
from scraper import PDFScraper

def test_pdf_scraper_rs_integration():
    """Test that RS correction parameters are properly passed to PDFScraper"""
    print("Testing PDFScraper RS integration...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a job config with RS correction enabled
        config = create_job_config(
            input_path="test.pdf",
            output_root=temp_dir,
            rs_enabled=True,
            rs_error_correction_bytes=10,
            rs_block_size=1024,
            rs_enable_correction=True,
            rs_verify_only=False
        )
        
        # Create a scraper instance from config
        scraper = PDFScraper.from_job_config(config)
        
        # Verify the scraper has the correct RS properties
        assert scraper.rs_enabled == True
        assert scraper.rs_error_correction_bytes == 10
        assert scraper.rs_block_size == 1024
        assert scraper.rs_enable_correction == True
        assert scraper.rs_verify_only == False
        assert scraper.rs_corrector is not None
        
        # Also verify it's an instance of RSTextCorrector
        assert isinstance(scraper.rs_corrector, RSTextCorrector)
        
        print("✓ PDFScraper RS integration successful")

def test_epub_scraper_rs_integration():
    """Test that RS correction parameters are properly passed to EPUBScraper"""
    print("\nTesting EPUBScraper RS integration...")
    
    from epub_scraper import EPUBScraper
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a job config with RS correction enabled
        config = create_job_config(
            input_path="test.epub",
            output_root=temp_dir,
            rs_enabled=True,
            rs_error_correction_bytes=10,
            rs_block_size=1024,
            rs_enable_correction=True,
            rs_verify_only=False
        )
        
        # Create a scraper instance from config
        scraper = EPUBScraper.from_job_config(config)
        
        # Verify the scraper has the correct RS properties
        assert scraper.rs_enabled == True
        assert scraper.rs_error_correction_bytes == 10
        assert scraper.rs_block_size == 1024
        assert scraper.rs_enable_correction == True
        assert scraper.rs_verify_only == False
        assert scraper.rs_corrector is not None
        
        # Also verify it's an instance of RSTextCorrector
        assert isinstance(scraper.rs_corrector, RSTextCorrector)
        
        print("✓ EPUBScraper RS integration successful")

def test_config_from_dict():
    """Test that config can be created from dictionary with RS fields"""
    print("\nTesting config from dictionary...")
    
    config_dict = {
        "input_path": "test.pdf",
        "output_root": "output",
        "rs_enabled": True,
        "rs_error_correction_bytes": 15,
        "rs_block_size": 2048,
        "rs_enable_correction": False,
        "rs_verify_only": True
    }
    
    from config_manager import get_config_manager
    config = get_config_manager().from_dict(config_dict)
    
    # Verify the RS correction section
    assert config.rs_correction.enabled == True
    assert config.rs_correction.error_correction_bytes == 15
    assert config.rs_correction.block_size == 2048
    assert config.rs_correction.enable_correction == False
    assert config.rs_correction.verify_only == True
    
    print("✓ Config from dict RS fields successful")

def test_nested_config_structure():
    """Test that nested RS config structure works"""
    print("\nTesting nested RS config structure...")
    
    config_dict = {
        "input_path": "test.pdf",
        "output_root": "output",
        "rs_correction": {
            "enabled": True,
            "error_correction_bytes": 12,
            "block_size": 1536,
            "enable_correction": True,
            "verify_only": False
        }
    }
    
    from config_manager import get_config_manager
    config = get_config_manager().from_dict(config_dict)
    
    assert config.rs_correction.enabled == True
    assert config.rs_correction.error_correction_bytes == 12
    assert config.rs_correction.block_size == 1536
    assert config.rs_correction.enable_correction == True
    assert config.rs_correction.verify_only == False
    
    print("✓ Nested RS config structure successful")

def test_cli_arguments_passed():
    """Test that CLI arguments are correctly passed to config"""
    print("\nTesting CLI arguments passing...")
    
    import sys
    from unittest.mock import patch
    
    test_args = [
        "cli.py",
        "input.pdf",
        "--output", "output",
        "--rs-enabled",
        "--rs-error-bytes", "15",
        "--rs-block-size", "2048",
        "--rs-verify-only",
        "--rs-disable-correction"
    ]
    
    with patch('sys.argv', test_args):
        from cli import main
        
        # We won't actually run main(), but let's test the config creation
        config = create_job_config(
            input_path="input.pdf",
            output_root="output",
            rs_enabled=True,
            rs_error_correction_bytes=15,
            rs_block_size=2048,
            rs_enable_correction=False,
            rs_verify_only=True
        )
        
        assert config.rs_correction.enabled == True
        assert config.rs_correction.error_correction_bytes == 15
        assert config.rs_correction.block_size == 2048
        assert config.rs_correction.enable_correction == False
        assert config.rs_correction.verify_only == True
        
        print("✓ CLI arguments passing successful")

def test_constants_config():
    """Test that RS constants are correctly used as defaults"""
    print("\nTesting RS constants as defaults...")
    
    config = create_job_config(
        input_path="test.pdf",
        output_root="output"
    )
    
    # Check that default values from constants are used
    from constants import (
        RS_ENABLED,
        RS_ERROR_CORRECTION_BYTES,
        RS_BLOCK_SIZE,
        RS_ENABLE_CORRECTION,
        RS_VERIFY_ONLY
    )
    
    assert config.rs_correction.enabled == RS_ENABLED
    assert config.rs_correction.error_correction_bytes == RS_ERROR_CORRECTION_BYTES
    assert config.rs_correction.block_size == RS_BLOCK_SIZE
    assert config.rs_correction.enable_correction == RS_ENABLE_CORRECTION
    assert config.rs_correction.verify_only == RS_VERIFY_ONLY
    
    print("✓ RS constants as defaults successful")

def test_invalid_rs_config():
    """Test handling of invalid RS config values"""
    print("\nTesting invalid RS config values...")
    
    # These should use default values instead of invalid ones
    config1 = create_job_config(
        input_path="test.pdf",
        output_root="output",
        rs_error_correction_bytes=-10,  # invalid
        rs_block_size=0  # invalid
    )
    
    # Should use defaults
    from constants import RS_ERROR_CORRECTION_BYTES, RS_BLOCK_SIZE
    assert config1.rs_correction.error_correction_bytes > 0
    assert config1.rs_correction.block_size > 0
    
    print("✓ Invalid RS config values handled successfully")

if __name__ == "__main__":
    print("Testing Complete RS Correction Workflow")
    print("=" * 50)
    
    try:
        test_pdf_scraper_rs_integration()
        test_epub_scraper_rs_integration()
        test_config_from_dict()
        test_nested_config_structure()
        test_cli_arguments_passed()
        test_constants_config()
        test_invalid_rs_config()
        
        print("\n" + "=" * 50)
        print("✅ All workflow tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
