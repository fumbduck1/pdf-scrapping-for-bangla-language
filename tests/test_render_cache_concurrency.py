#!/usr/bin/env python3
"""Test to reproduce and verify the fix for _render_cache concurrency issue."""

import unittest
import threading
import time
from PIL import Image
from scraper import PdfRenderer
from unittest import mock


class TestRenderCacheConcurrency(unittest.TestCase):
    """Test to verify that _render_cache operations are thread-safe."""
    
    def test_concurrent_render_cache_access(self):
        """Test that concurrent render operations don't corrupt the cache."""
        # Create a renderer with a small cache size to force LRU behavior
        renderer = PdfRenderer(
            pdf_path="/dev/null",
            output_dir="/tmp",
            pdf_bytes_cache_mb=1,
            poppler_path=None,
            log=None,
            log_error=None,
            persist_renders=False,
            render_cache_max_items=3,
        )
        
        # Create a dummy image for rendering
        dummy_img = Image.new("RGB", (10, 10))
        
        # Mock the pdf2image functions to return dummy images
        def fake_from_bytes(*args, **kwargs):
            time.sleep(0.001)  # Add a small delay to simulate rendering time
            return [dummy_img]
        
        def fake_from_path(*args, **kwargs):
            time.sleep(0.001)  # Add a small delay to simulate rendering time
            return [dummy_img]
        
        renderer._pdf_bytes = b"%PDF-1.4"  # Force convert_from_bytes path
        
        # Number of concurrent threads and pages to render
        num_threads = 10
        num_pages = 5
        
        # Function to render multiple pages in a thread
        # Mock subprocess.run to return valid PPM data - only for pdftoppm calls
        import subprocess
        original_subprocess_run = subprocess.run
        
        def mock_subprocess_run(*args, **kwargs):
            # Only mock pdftoppm calls
            if "pdftoppm" not in str(args[0]).lower():
                return original_subprocess_run(*args, **kwargs)
                
            from unittest.mock import Mock
            result = Mock()
            result.returncode = 0
            # Create a minimal valid PPM file header
            ppm_header = b"P6\n100 100\n255\n"
            # Create 100x100 RGB pixel data (all black)
            ppm_data = ppm_header + b"\x00\x00\x00" * 100 * 100
            result.stdout = ppm_data
            result.stderr = b""
            time.sleep(0.001)  # Add a small delay to simulate rendering time
            return result
            
        def worker():
            with mock.patch("scraper.check_pdftoppm_available", return_value=True), \
                 mock.patch("subprocess.run", side_effect=mock_subprocess_run):
                for page_num in range(num_pages):
                    renderer.render_page(page_num, 1.0)
        
        # Start all worker threads
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify cache consistency - check that cache size is within bounds
        self.assertLessEqual(len(renderer._render_cache), 3)
        
        # Verify all cache keys are unique
        cache_keys = list(renderer._render_cache.keys())
        self.assertEqual(len(cache_keys), len(set(cache_keys)))
        
        print(f"Test passed. Cache size: {len(renderer._render_cache)}, Keys: {cache_keys}")


if __name__ == "__main__":
    unittest.main()
