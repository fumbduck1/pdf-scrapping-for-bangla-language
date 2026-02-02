#!/usr/bin/env python3
"""Test to simulate and detect race conditions in render cache operations."""

import unittest
import threading
import time
import os
import tempfile
from PIL import Image
from scraper import PdfRenderer
from unittest import mock


class TestRenderCacheRaceCondition(unittest.TestCase):
    """Test to detect race conditions in LRU cache operations."""
    
    def test_cache_race_condition(self):
        """Test that simulates a race condition when rendering the same page."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a renderer with a small cache size
            renderer = PdfRenderer(
                pdf_path=os.devnull,
                output_dir=temp_dir,
                pdf_bytes_cache_mb=1,
                poppler_path=None,
                log=None,
                log_error=None,
                persist_renders=False,
                render_cache_max_items=1,
            )
            
            renderer._pdf_bytes = b"%PDF-1.4"
            
            # Create a shared flag to track rendering in progress
            rendering_in_progress = False
            render_lock = threading.Lock()
            render_invocations = 0
            
            # Create a mock pdf2image that introduces delay and tracks rendering status
            def slow_convert_from_bytes(*args, **kwargs):
                nonlocal rendering_in_progress, render_invocations
                
                # Count actual render invocations
                with render_lock:
                    render_invocations += 1
                    rendering_in_progress = True
                
                # Introduce a delay to maximize chance of race
                time.sleep(0.1)
                
                # Mark that rendering is complete
                with render_lock:
                    rendering_in_progress = False
                
                return [Image.new("RGB", (10, 10))]
            
            def slow_convert_from_path(*args, **kwargs):
                return slow_convert_from_bytes(*args, **kwargs)
            
            # Track cache changes
            cache_changes = []
            
            # Override _render_cache to track all operations
            original_cache = renderer._render_cache
            
            class TrackedRenderCache:
                def __init__(self, original):
                    self.original = original
                    self.lock = threading.Lock()
                
                def __getattribute__(self, name):
                    if name in ['__class__', 'original', 'lock']:
                        return super().__getattribute__(name)
                    
                    value = getattr(self.original, name)
                    
                    if hasattr(value, '__call__'):
                        def tracked(*args, **kwargs):
                            thread_id = threading.current_thread().ident
                            timestamp = time.time()
                            cache_changes.append((thread_id, name, args, timestamp))
                            
                            result = value(*args, **kwargs)
                            
                            return result
                        return tracked
                    else:
                        return value
                
                def __contains__(self, key):
                    thread_id = threading.current_thread().ident
                    cache_changes.append((thread_id, '__contains__', (key,), time.time()))
                    return key in self.original
                
                def __setitem__(self, key, value):
                    thread_id = threading.current_thread().ident
                    cache_changes.append((thread_id, '__setitem__', (key, value), time.time()))
                    self.original[key] = value
                
                def __getitem__(self, key):
                    thread_id = threading.current_thread().ident
                    cache_changes.append((thread_id, '__getitem__', (key,), time.time()))
                    return self.original[key]
                
                def __len__(self):
                    thread_id = threading.current_thread().ident
                    cache_changes.append((thread_id, '__len__', (), time.time()))
                    return len(self.original)
            
            renderer._render_cache = TrackedRenderCache(renderer._render_cache)
            
            # Define worker that renders the same page repeatedly
            def worker():
                with mock.patch("scraper._lazy_import_pdf2image", 
                              return_value=(slow_convert_from_path, slow_convert_from_bytes)):
                    renderer.render_page(0, 1.0)
            
            # Start multiple workers that will all try to render the same page
            num_workers = 5
            threads = []
            
            for _ in range(num_workers):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)
            
            # Wait for all workers to complete
            for t in threads:
                t.join()
            
            # Analyze the cache changes to detect race conditions
            print("\n=== Cache Operations Analysis ===")
            
            # Count render completions and cache operations
            render_start = 0
            render_end = 0
            cache_set_operations = 0
            
            for thread_id, method, args, timestamp in cache_changes:
                if method == '__contains__' and args[0] == (0, 1.0, 'png'):
                    render_start += 1
                elif method == '__getitem__' and args[0] == (0, 1.0, 'png'):
                    render_end += 1
                elif method == '__setitem__' and args[0] == (0, 1.0, 'png'):
                    # Only count actual renders, not placeholders
                    if isinstance(args[1], tuple) and len(args[1]) == 2 and isinstance(args[1][0], Image.Image):
                        cache_set_operations += 1
            
            print(f"Number of render attempts: {render_start}")
            print(f"Number of successful renders (from cache/memoization): {render_end}")
            print(f"Number of actual render invocations: {render_invocations}")
            print(f"Number of cache set operations: {cache_set_operations}")
            
            # Assert that we have more render attempts than actual render invocations
            # (due to the placeholder mechanism preventing duplicate renders)
            self.assertGreater(render_start, render_invocations)
            
            # Assert that cache set operations match actual render invocations
            # This indicates that each render is only cached once
            self.assertEqual(render_invocations, cache_set_operations, 
                            f"Race condition detected! {render_invocations} renders completed, but {cache_set_operations} cache set operations occurred")


if __name__ == "__main__":
    unittest.main()
