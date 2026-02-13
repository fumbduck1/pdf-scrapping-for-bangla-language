#!/usr/bin/env python3
"""Test script to verify RS integration with the scraper"""

import os
import tempfile
import shutil
from config_manager import create_job_config
from rs_correction import RSTextCorrector
import constants

def test_scraper_rs_config():
    """Test that RS correction config is properly created"""
    print("Testing RS config creation...")
    
    # Create a job config with RS correction enabled
    config = create_job_config(
        input_path="test.pdf",
        output_root="output",
        rs_enabled=True,
        rs_error_correction_bytes=10,
        rs_block_size=1024,
        rs_enable_correction=True,
        rs_verify_only=False
    )
    
    # Verify config parameters
    assert config.rs_correction.enabled == True
    assert config.rs_correction.error_correction_bytes == 10
    assert config.rs_correction.block_size == 1024
    assert config.rs_correction.enable_correction == True
    assert config.rs_correction.verify_only == False
    
    print("✓ RS config creation successful")

def test_constants_exist():
    """Test that RS constants exist"""
    print("\nTesting RS constants...")
    
    # Check if all required constants are defined
    assert hasattr(constants, 'RS_ENABLED')
    assert hasattr(constants, 'RS_ERROR_CORRECTION_BYTES')
    assert hasattr(constants, 'RS_BLOCK_SIZE')
    assert hasattr(constants, 'RS_ENABLE_CORRECTION')
    assert hasattr(constants, 'RS_VERIFY_ONLY')
    
    print("✓ RS constants defined")

def test_corrector_initialization():
    """Test that RS corrector can be initialized from scraper config"""
    print("\nTesting RS corrector initialization...")
    
    # Create a config with different RS parameters
    config1 = create_job_config("test.pdf", "output", rs_enabled=True, rs_error_correction_bytes=10)
    config2 = create_job_config("test.pdf", "output", rs_enabled=True, rs_error_correction_bytes=20)
    
    # Initialize correctors
    corrector1 = RSTextCorrector(config1.rs_correction.error_correction_bytes)
    corrector2 = RSTextCorrector(config2.rs_correction.error_correction_bytes)
    
    # Verify they have different error correction capacities
    test_text = "Test text for comparison"
    encoded1 = corrector1.encode_text(test_text)
    encoded2 = corrector2.encode_text(test_text)
    
    # The encoded lengths should be different due to different error correction bytes
    assert len(encoded1) != len(encoded2)
    
    print("✓ RS corrector initialization successful")

def test_file_saving_scenario():
    """Test the file saving scenario with RS correction"""
    print("\nTesting RS file saving scenario...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_text = "This is a test of the file saving mechanism with RS correction"
        
        # Create a corrector
        corrector = RSTextCorrector(10)
        
        # Create a test file
        test_path = os.path.join(temp_dir, "test_file.rs")
        
        # Save the encoded file
        corrector.encode_and_save(test_text, test_path)
        
        assert os.path.exists(test_path)
        assert os.path.getsize(test_path) > 0
        
        # Verify we can load and decode it
        loaded_text, errors_corrected, _ = corrector.load_and_decode(test_path)
        
        assert test_text == loaded_text
        assert errors_corrected == 0
        
        print("✓ File saving scenario test successful")

def test_error_correction_scenario():
    """Test error correction in a realistic scenario"""
    print("\nTesting error correction scenario...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a long text similar to what OCR would produce
        test_text = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        """
        
        corrector = RSTextCorrector(20)  # More error correction for larger text
        
        # Encode and save
        test_path = os.path.join(temp_dir, "test_ocr_text.rs")
        corrector.encode_and_save(test_text, test_path)
        
        # Simulate some OCR errors by corrupting the file
        with open(test_path, 'rb') as f:
            data = bytearray(f.read())
        
        # Corrupt some bytes (OCR errors)
        corrupt_positions = [100, 150, 200]
        for pos in corrupt_positions:
            if pos < len(data):
                data[pos] = 0xFF
        
        with open(test_path, 'wb') as f:
            f.write(data)
        
        # Try to decode with errors
        loaded_text, errors_corrected, _ = corrector.load_and_decode(test_path)
        
        # Should still decode successfully (errors should be corrected)
        assert test_text.strip() == loaded_text.strip()
        assert errors_corrected > 0
        
        print(f"✓ Error correction successful - corrected {errors_corrected} errors")

if __name__ == "__main__":
    print("Testing Reed-Solomon Integration with PDF Scraper")
    print("=" * 50)
    
    try:
        test_scraper_rs_config()
        test_constants_exist()
        test_corrector_initialization()
        test_file_saving_scenario()
        test_error_correction_scenario()
        
        print("\n" + "=" * 50)
        print("✅ All integration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
