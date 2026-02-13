#!/usr/bin/env python3
"""Test script to verify Reed-Solomon error correction functionality"""

import os
import tempfile
from pathlib import Path
from rs_correction import RSTextCorrector

def test_rs_encoding_decoding():
    """Test basic encoding and decoding of text"""
    print("Testing RS encoding and decoding...")
    
    corrector = RSTextCorrector(error_correction_bytes=10)
    
    # Test text
    test_text = "This is a test of Reed-Solomon error correction for OCR text extraction."
    print(f"Test text: {test_text}")
    
    # Encode
    encoded = corrector.encode_text(test_text)
    print(f"Encoded length: {len(encoded)} bytes")
    
    # Decode
    decoded_text, errors_corrected, _ = corrector.decode_text(encoded)
    print(f"Decoded text: {decoded_text}")
    print(f"Errors corrected: {errors_corrected}")
    
    # Verify
    assert test_text == decoded_text, "Decoded text should match original"
    print("✓ Encoding and decoding successful")

def test_rs_error_correction():
    """Test error correction by corrupting some bytes"""
    print("\nTesting RS error correction...")
    
    corrector = RSTextCorrector(error_correction_bytes=10)
    
    test_text = "OCR extraction can produce errors - Reed-Solomon helps fix them!"
    encoded = corrector.encode_text(test_text)
    
    # Convert to list for modification
    encoded_bytes = bytearray(encoded)
    
    # Corrupt some bytes (simulate transmission errors)
    # Since we have 10 error correction bytes, we should be able to correct up to 5 errors
    corrupt_positions = [10, 20, 30]
    for pos in corrupt_positions:
        if pos < len(encoded_bytes):
            encoded_bytes[pos] = 0xFF  # Corrupt the byte
    
    print(f"Corrupted {len(corrupt_positions)} bytes at positions {corrupt_positions}")
    
    # Try to decode with errors
    decoded_text, errors_corrected, _ = corrector.decode_text(bytes(encoded_bytes))
    print(f"Errors corrected: {errors_corrected}")
    print(f"Decoded text: {decoded_text}")
    
    # Verify
    assert test_text == decoded_text, "Text should be corrected successfully"
    print("✓ Error correction successful")

def test_file_operations():
    """Test saving and loading encoded files"""
    print("\nTesting file operations...")
    
    corrector = RSTextCorrector(error_correction_bytes=10)
    
    test_text = "This text will be saved to a file with RS encoding."
    
    # Create temp file
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "test_encoded.rs")
        
        # Save to file
        corrector.encode_and_save(test_text, test_path)
        assert os.path.exists(test_path), "File should be created"
        print(f"File saved: {test_path}")
        
        # Load and decode
        decoded_text, errors_corrected, _ = corrector.load_and_decode(test_path)
        print(f"Errors corrected: {errors_corrected}")
        
        # Verify
        assert test_text == decoded_text, "Decoded text should match original"
        print("✓ File operations successful")

def test_verification():
    """Test text verification functionality"""
    print("\nTesting verification...")
    
    corrector = RSTextCorrector(error_correction_bytes=10)
    
    test_text = "Verification test - is this text intact?"
    encoded = corrector.encode_text(test_text)
    
    # Verify intact text
    is_intact, errors = corrector.verify_text(encoded)
    print(f"Text is intact: {is_intact}")
    print(f"Errors detected: {errors}")
    
    assert is_intact, "Text should be intact"
    
    # Corrupt a byte
    corrupted = bytearray(encoded)
    corrupted[5] = 0xFF
    
    is_intact, errors = corrector.verify_text(bytes(corrupted))
    print(f"Corrupted text is intact: {is_intact}")
    print(f"Errors detected: {errors}")
    
    assert not is_intact, "Corrupted text should not be intact"
    print("✓ Verification successful")

if __name__ == "__main__":
    print("Testing Reed-Solomon Error Correction Module")
    print("=" * 50)
    
    try:
        test_rs_encoding_decoding()
        test_rs_error_correction()
        test_file_operations()
        test_verification()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
