"""
Reed-Solomon Error Correction for OCR-Extracted Text

This module provides Reed-Solomon encoding and decoding capabilities specifically
designed for OCR-extracted text. It handles UTF-8 text encoding, error detection,
and error correction using the reedsolo library.
"""

import logging
from typing import Optional, Tuple
from reedsolo import RSCodec

logger = logging.getLogger(__name__)


class RSTextCorrector:
    """
    A class to handle Reed-Solomon error correction for text data.
    
    This implementation:
    - Handles UTF-8 text encoding/decoding
    - Supports customizable error correction strength
    - Provides methods for encoding and decoding text with RS codes
    - Detects and corrects errors in OCR-extracted text
    """
    
    def __init__(self, error_correction_bytes: int = 10):
        """
        Create an RSTextCorrector configured with the specified number of Reed–Solomon error-correction bytes.
        
        Parameters:
            error_correction_bytes (int): Number of parity bytes used for Reed–Solomon encoding. Larger values increase error-correction capacity and the size of encoded output (default: 10).
        """
        self.error_correction_bytes = error_correction_bytes
        self.rs = RSCodec(error_correction_bytes)
    
    def encode_text(self, text: str) -> bytes:
        """
        Encode the given UTF-8 text and append Reed–Solomon error-correction bytes.
        
        Returns:
            bytes: Encoded bytes containing the UTF-8 payload followed by Reed–Solomon parity bytes.
        """
        try:
            # Encode text to UTF-8 bytes
            text_bytes = text.encode('utf-8')
            
            # Apply Reed-Solomon encoding
            encoded_bytes = bytes(self.rs.encode(text_bytes))
            
            logger.debug(
                "Successfully encoded %d bytes of text with %d error correction bytes",
                len(text_bytes),
                self.error_correction_bytes
            )
            
            return encoded_bytes
            
        except Exception as e:
            logger.error("Error encoding text with Reed-Solomon: %s", str(e))
            raise
    
    def decode_text(self, encoded_bytes: bytes) -> Tuple[str, bool, int]:
        """
        Decode RS-encoded bytes into UTF-8 text and report correction results.
        
        Parameters:
            encoded_bytes (bytes): RS-encoded data produced by encode_text or an equivalent encoder.
        
        Returns:
            Tuple[str, bool, int]: 
                decoded_text: The decoded UTF-8 string.
                corrected: `true` if Reed–Solomon corrected any errors, `false` if the data was already intact.
                errors_corrected: Number of byte positions that differed between the original encoded input and its re-encoded form (0 if intact).
        """
        try:
            # First, verify the data
            is_intact, errors_detected = self.verify_text(encoded_bytes)
            errors_detected = errors_detected or 0
            
            # Decode with error correction
            decoded_bytes, _, _ = self.rs.decode(encoded_bytes)
            
            # Convert back to string
            text = bytes(decoded_bytes).decode('utf-8')
            
            logger.debug(
                "Successfully decoded text, corrected %d errors",
                errors_detected
            )
            
            return text, not is_intact, errors_detected
            
        except Exception as e:
            logger.error("Error decoding text with Reed-Solomon: %s", str(e))
            raise
    
    def verify_text(self, encoded_bytes: bytes) -> Tuple[bool, Optional[int]]:
        """
        Check whether RS-encoded UTF-8 bytes are intact by re-encoding decoded data and comparing byte-wise.
        
        Parameters:
            encoded_bytes (bytes): RS-encoded UTF-8 data to verify.
        
        Returns:
            Tuple[bool, Optional[int]]: `true` if the encoded bytes exactly match the re-encoded bytes, `false` otherwise; the number of differing bytes when verification succeeds but data differs, or `None` if verification failed due to an internal error.
        """
        try:
            # To verify, we need to check if decoding would produce errors
            # The simplest way is to encode the decoded data and compare
            decoded, _, _ = self.rs.decode(encoded_bytes)
            
            # Re-encode the decoded data
            reencoded = self.rs.encode(decoded)
            
            # Compare the encoded data (should match exactly if no errors)
            is_intact = encoded_bytes == reencoded
            
            # If not intact, count the number of differing bytes
            errors = 0
            if not is_intact:
                min_len = min(len(encoded_bytes), len(reencoded))
                for i in range(min_len):
                    if encoded_bytes[i] != reencoded[i]:
                        errors += 1
                # Account for length differences
                errors += abs(len(encoded_bytes) - len(reencoded))
            
            return is_intact, errors
        except Exception as e:
            logger.error("Error verifying text: %s", str(e))
            return False, None
    
    def encode_and_save(self, text: str, output_path: str) -> bool:
        """
        Encode text with Reed–Solomon error correction and write the resulting bytes to the given file path.
        
        Parameters:
            text (str): UTF-8 text to encode.
            output_path (str): Filesystem path where encoded bytes will be written in binary mode.
        
        Returns:
            bool: `True` if the encoded bytes were written successfully, `False` otherwise.
        """
        try:
            encoded_bytes = self.encode_text(text)
            
            with open(output_path, 'wb') as f:
                f.write(encoded_bytes)
            
            logger.debug("Successfully saved encoded text to %s", output_path)
            return True
            
        except Exception as e:
            logger.error("Error saving encoded text to %s: %s", output_path, str(e))
            return False
    
    def load_and_decode(self, file_path: str) -> Tuple[Optional[str], bool, int]:
        """
        Load an encoded file from disk and decode its text using this corrector.
        
        Returns:
            Tuple[Optional[str], bool, int]: (decoded_text, corrected, errors_corrected)
                - decoded_text: Decoded UTF-8 string, or `None` if decoding failed.
                - corrected: `True` if Reed–Solomon correction was applied, `False` otherwise.
                - errors_corrected: Number of byte errors corrected.
        """
        try:
            with open(file_path, 'rb') as f:
                encoded_bytes = f.read()
            
            return self.decode_text(encoded_bytes)
            
        except Exception as e:
            logger.error("Error loading and decoding file %s: %s", file_path, str(e))
            return None, False, 0
    
    def correct_text_fragment(self, text: str, max_errors: int = 5) -> Tuple[str, int]:
        """
        Correct OCR-extracted text fragment using Reed–Solomon error correction.
        
        This method is intended to correct character-level errors common in OCR output by encoding
        the fragment with the internal Reed–Solomon codec and attempting recovery. In the current
        implementation this is a no-op: the original text is returned unchanged and the correction
        count is 0.
        
        Parameters:
            text (str): The OCR-extracted text fragment to correct.
            max_errors (int): Maximum number of byte errors to attempt to correct (default 5).
        
        Returns:
            Tuple[str, int]: A tuple of (corrected_text, errors_corrected). `errors_corrected` is
            the number of corrections applied (0 in the current implementation).
        """
        try:
            # Encode the text
            text_bytes = text.encode('utf-8')
            encoded = self.rs.encode(text_bytes)
            
            # This simulates possible corruption in OCR extraction
            # For real OCR data, the encoded bytes would be stored and
            # retrieved for verification
            
            # In practice, this method would be used with previously encoded data
            
            return text, 0
            
        except Exception as e:
            logger.error("Error correcting text fragment: %s", str(e))
            return text, 0


def create_rs_corrector(error_correction_bytes: int = 10) -> RSTextCorrector:
    """
    Create an RSTextCorrector configured with the given number of error-correction bytes.
    
    Parameters:
        error_correction_bytes (int): Number of Reed–Solomon parity (error-correction) bytes to include in encoded output; larger values increase correction capability at the cost of added size.
    
    Returns:
        RSTextCorrector: A new corrector instance configured with the specified error-correction length.
    """
    return RSTextCorrector(error_correction_bytes)


def encode_text(text: str, error_correction_bytes: int = 10) -> bytes:
    """
    Convenience function to encode text with Reed-Solomon error correction.
    
    Args:
        text: Text to encode
        error_correction_bytes: Number of error correction bytes (default: 10)
        
    Returns:
        Encoded bytes
    """
    corrector = create_rs_corrector(error_correction_bytes)
    return corrector.encode_text(text)


def decode_text(encoded_bytes: bytes, error_correction_bytes: int = 10) -> Tuple[str, bool, int]:
    """
    Decode Reed-Solomon encoded UTF-8 text using the specified error correction strength.
    
    Parameters:
        encoded_bytes (bytes): RS-encoded bytes produced by this module's encoder.
        error_correction_bytes (int): Number of Reed-Solomon error correction/parity bytes used when encoding (default 10).
    
    Returns:
        tuple: (decoded_text, errors_corrected_flag, errors_corrected_count)
            - decoded_text (str): The recovered UTF-8 text.
            - errors_corrected_flag (bool): `true` if decoding corrected any errors, `false` otherwise.
            - errors_corrected_count (int): Number of byte errors corrected during decoding.
    """
    corrector = create_rs_corrector(error_correction_bytes)
    return corrector.decode_text(encoded_bytes)


def verify_text(encoded_bytes: bytes, error_correction_bytes: int = 10) -> Tuple[bool, Optional[int]]:
    """
    Check whether Reed–Solomon encoded bytes are intact and report detected byte differences.
    
    Parameters:
        encoded_bytes (bytes): Reed–Solomon encoded payload to verify.
        error_correction_bytes (int): Number of RS parity/error-correction bytes used when the data was encoded.
    
    Returns:
        tuple (bool, Optional[int]): `True` if the provided bytes match the re-encoded data; if not intact, an integer count of differing bytes is returned; returns `None` for the error count if verification failed due to an exception.
    """
    corrector = create_rs_corrector(error_correction_bytes)
    return corrector.verify_text(encoded_bytes)