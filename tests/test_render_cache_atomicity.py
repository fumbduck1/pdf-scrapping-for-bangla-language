#!/usr/bin/env python3
"""Test to demonstrate the non-atomic LRU update problem in _render_cache."""

import unittest
import threading
import time
import os
import tempfile
from PIL import Image
from scraper import PdfRenderer
from unittest import mock


def run_concurrent_renders(renderer, num_threads, num_pages, delay=0.01):
    """Run concurrent render operations on the same pages to test cache behavior."""
    results = []
    
    def worker():
        thread_results = []
        with mock.patch("scraper._lazy_import_pdf2image", 
                      return_value=(lambda *args, **kwargs: [Image.new("RGB", (10, 10))], 
                                    lambda *args, **kwargs: [Image.new("RGB", (10, 10))])):
            for page_num in range(num_pages):
                start_time = time.time()
                result = renderer.render_page(page_num, 1.0)
                end_time = time.time()
                thread_results.append((page_num, end_time - start_time))
                time.sleep(delay)  # Add delay to increase chance of race conditions
        results.append(thread_results)
    
    # Start all worker threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    return results


class TestRenderCacheAtomicity(unittest.TestCase):
    """Test to verify atomicity of LRU cache updates."""
    
    def test_cache_state_consistency(self):
        """Test that cache remains in a consistent state after concurrent operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            renderer = PdfRenderer(
                pdf_path=os.devnull,
                output_dir=temp_dir,
                pdf_bytes_cache_mb=1,
                poppler_path=None,
                log=None,
                log_error=None,
                persist_renders=False,
                render_cache_max_items=2,
            )
            
            renderer._pdf_bytes = b"%PDF-1.4"
            
            # Run concurrent renders to simulate race conditions
            run_concurrent_renders(renderer, num_threads=8, num_pages=2, delay=0.01)
            
            # Verify cache state
            self.assertIsNotNone(renderer._render_cache)
            self.assertLessEqual(len(renderer._render_cache), 2)
            
            # Print cache contents for debugging
            print(f"Cache contents: {list(renderer._render_cache.items())}")
    
    def test_cache_size_bounds(self):
        """Test that cache never exceeds maximum size even under concurrency."""
        max_cache_size = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            renderer = PdfRenderer(
                pdf_path=os.devnull,
                output_dir=temp_dir,
                pdf_bytes_cache_mb=1,
                poppler_path=None,
                log=None,
                log_error=None,
                persist_renders=False,
                render_cache_max_items=max_cache_size,
            )
            
            renderer._pdf_bytes = b"%PDF-1.4"
            
            # Run more concurrent renders to stress the cache
            run_concurrent_renders(renderer, num_threads=10, num_pages=5, delay=0.005)
            
            # Verify cache size constraints
            self.assertLessEqual(len(renderer._render_cache), max_cache_size)
            
            print(f"Cache size: {len(renderer._render_cache)}, max allowed: {max_cache_size}")
    
    def test_single_threaded_lru_behavior(self):
        """Test that LRU behavior is maintained in single-threaded scenario (deterministic)."""
        max_cache_size = 3
        with tempfile.TemporaryDirectory() as temp_dir:
            renderer = PdfRenderer(
                pdf_path=os.devnull,
                output_dir=temp_dir,
                pdf_bytes_cache_mb=1,
                poppler_path=None,
                log=None,
                log_error=None,
                persist_renders=False,
                render_cache_max_items=max_cache_size,
            )
            
            renderer._pdf_bytes = b"%PDF-1.4"
            
            # First, warm up the cache with distinct pages
            with mock.patch("scraper._lazy_import_pdf2image", 
                          return_value=(lambda *args, **kwargs: [Image.new("RGB", (10, 10))], 
                                        lambda *args, **kwargs: [Image.new("RGB", (10, 10))])):
                for page_num in range(max_cache_size):
                    renderer.render_page(page_num, 1.0)
            
            # Check initial cache contents (should be 0, 1, 2)
            initial_keys = list(renderer._render_cache.keys())
            self.assertEqual(len(initial_keys), max_cache_size)
            self.assertTrue(all(key[0] in range(max_cache_size) for key in initial_keys))
            
            # Now render newer pages in single thread
            for page_num in range(max_cache_size, max_cache_size + 2):
                renderer.render_page(page_num, 1.0)
            
            # Verify oldest keys are evicted
            new_keys = list(renderer._render_cache.keys())
            # The cache should now contain the last 3 pages (2, 3, 4)
            expected_keys = [(2, 1.0, "png"), (3, 1.0, "png"), (4, 1.0, "png")]
            
            print(f"Current cache keys: {new_keys}")
            print(f"Expected keys: {expected_keys}")
            
            # Check all expected keys are present (order might vary slightly)
            self.assertTrue(all(key in new_keys for key in expected_keys))


if __name__ == "__main__":
    unittest.main()
