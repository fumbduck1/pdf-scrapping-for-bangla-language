#!/usr/bin/env python3
"""Test the reedsolo library directly to understand its API"""

from reedsolo import RSCodec

def test_rs_api():
    """Test the reedsolo library API"""
    
    # Create an RS codec with 10 error correction bytes
    rs = RSCodec(10)
    
    # Test encoding
    test_text = "This is a test"
    data = test_text.encode('utf-8')
    print(f"Original data: {data}")
    
    encoded = rs.encode(data)
    print(f"Encoded data: {encoded}")
    print(f"Length: {len(data)} bytes -> {len(encoded)} bytes")
    
    # Test decoding with no errors
    print("\n--- Decoding intact data ---")
    try:
        decoded, errors, erasures = rs.decode(encoded)
        print(f"Decoded: {decoded.decode('utf-8')}")
        print(f"Errors: {errors}")
        print(f"Erasures: {erasures}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test decoding with errors
    print("\n--- Decoding corrupted data ---")
    corrupted = bytearray(encoded)
    corrupted[5] = 0xFF  # Corrupt a byte
    
    try:
        decoded, errors, erasures = rs.decode(bytes(corrupted))
        print(f"Decoded: {decoded.decode('utf-8')}")
        print(f"Errors: {errors}")
        print(f"Erasures: {erasures}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_rs_api()
