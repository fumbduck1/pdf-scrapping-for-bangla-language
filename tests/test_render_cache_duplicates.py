#!/usr/bin/env python3
"""Test to demonstrate duplicate cache entries due to non-atomic LRU operations."""

import unittest
import threading
import time
from PIL import Image
from scraper import PdfRenderer
from unittest import mock


def run_stress_test(renderer, num_threads, iterations):
    """Run high-concurrency stress test on the same page to maximize cache contention."""
    results = []
    
    # Mock subprocess.run to return valid PPM data
    def mock_subprocess_run(*args, **kwargs):
        from unittest.mock import Mock
        result = Mock()
        result.returncode = 0
        # Create a minimal valid PPM file header
        ppm_header = b"P6\n100 100\n255\n"
        # Create 100x100 RGB pixel data (all black)
        ppm_data = ppm_header + b"\x00\x00\x00" * 100 * 100
        result.stdout = ppm_data
        result.stderr = b""
        return result
    
    def worker():
        thread_results = []
        with mock.patch("scraper.check_pdftoppm_available", return_value=True), \
             mock.patch("subprocess.run", side_effect=mock_subprocess_run):
            for _ in range(iterations):
                start_time = time.time()
                result = renderer.render_page(0, 1.0)
                end_time = time.time()
                thread_results.append(end_time - start_time)
        results.append(thread_results)
    
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    return results


class TestRenderCacheDuplicates(unittest.TestCase):
    """Test to demonstrate and verify fix for duplicate cache entries."""
    
    def test_no_duplicate_keys(self):
        """Test that no duplicate keys exist in the cache after concurrent access."""
        renderer = PdfRenderer(
            pdf_path="/dev/null",
            output_dir="/tmp",
            pdf_bytes_cache_mb=1,
            poppler_path=None,
            log=None,
            log_error=None,
            persist_renders=False,
            render_cache_max_items=1,
        )
        
        renderer._pdf_bytes = b"%PDF-1.4"
        
        # Run high-concurrency stress test on the same page
        run_stress_test(renderer, num_threads=10, iterations=100)
        
        # Check for duplicate keys
        cache_items = list(renderer._render_cache.items())
        self.assertEqual(len(cache_items), 1, "Cache should only have one entry")
        
        print(f"Cache contents: {cache_items}")
        
        # Verify all keys are unique
        keys = [key for key, value in cache_items]
        self.assertEqual(len(keys), len(set(keys)), "Cache contains duplicate keys!")
    
    def test_stress_cache_eviction(self):
        """Stress test the cache eviction mechanism with high concurrency."""
        max_cache_size = 2
        renderer = PdfRenderer(
            pdf_path="/dev/null",
            output_dir="/tmp",
            pdf_bytes_cache_mb=1,
            poppler_path=None,
            log=None,
            log_error=None,
            persist_renders=False,
            render_cache_max_items=max_cache_size,
        )
        
        renderer._pdf_bytes = b"%PDF-1.4"
        
        # Run workers accessing different pages to force cache evictions
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
            return result
            
        def worker(page_num):
            with mock.patch("scraper.check_pdftoppm_available", return_value=True), \
                 mock.patch("subprocess.run", side_effect=mock_subprocess_run):
                for _ in range(20):
                    renderer.render_page(page_num, 1.0)
        
        # Create threads for different pages to cause frequent cache evictions
        threads = []
        for page_num in range(4):
            for _ in range(3):
                t = threading.Thread(target=worker, args=(page_num,))
                t.start()
                threads.append(t)
        
        for t in threads:
            t.join()
        
        # Verify cache size and key uniqueness
        self.assertLessEqual(len(renderer._render_cache), max_cache_size)
        keys = list(renderer._render_cache.keys())
        self.assertEqual(len(keys), len(set(keys)))
        
        print(f"Cache after stress: {list(renderer._render_cache.items())}")


if __name__ == "__main__":
    unittest.main()
