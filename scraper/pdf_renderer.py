from typing import Any, Optional
from pathlib import Path
from datetime import datetime
from io import BytesIO
from collections import OrderedDict
from PIL import Image
import os
import threading
import subprocess
import sys
import shutil

from constants import RENDER_CACHE_MAX_ITEMS
from performance import timer
from deps import check_pdftoppm_available


class PdfRenderer:
    """Rendering helper to manage PDF handles and page rasterization with caching."""

    def __init__(self, pdf_path, output_dir, pdf_bytes_cache_mb, poppler_path, log, log_error, persist_renders=False, render_cache_max_items=RENDER_CACHE_MAX_ITEMS):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.renders_dir = None
        self._pdf_file_handle = None
        self._pdf_bytes = None
        self.doc = None
        self.pdf_bytes_cache_mb = pdf_bytes_cache_mb
        self.poppler_path = poppler_path
        self.log = log
        self.log_error = log_error
        self.persist_renders = persist_renders
        self.render_cache_max_items = max(int(render_cache_max_items or 0), 0)
        
        # Page cache for rendered images (bounded LRU)
        self._render_cache = OrderedDict() if self.render_cache_max_items > 0 else None
        self._render_cache_lock = threading.Lock()
        
        self.setup_directories()

    def setup_directories(self):
        if not self.persist_renders:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.renders_dir = os.path.join(self.output_dir, 'renders')
            os.makedirs(self.renders_dir, exist_ok=True)
        except Exception:
            pass

    def open_pdf(self):
        """Open the PDF into memory or file handle depending on size."""
        try:
            from pypdf import PdfReader
        except ImportError:
            self._log_missing("pypdf not installed. Install with: pip install pypdf", err_key="pypdf")
            return False
        try:
            size_bytes = os.path.getsize(self.pdf_path)
            size_mb = round(size_bytes / (1024 * 1024), 2)
            if size_mb <= self.pdf_bytes_cache_mb:
                self._pdf_bytes = Path(self.pdf_path).read_bytes()
                self.doc = PdfReader(BytesIO(self._pdf_bytes))
                self._pdf_file_handle = None
            else:
                self._pdf_file_handle = open(self.pdf_path, "rb")
                self.doc = PdfReader(self._pdf_file_handle)
            return True
        except Exception as e:
            self._log_error(f"Open failed: {e}")
            self.close()
            return False

    @timer("pdf_rendering")
    def render_page(self, page_num, zoom, fmt="png"):
        """Render a single page and return a PIL image; optionally persist to disk (with caching)."""
        # Define PENDING at method level to ensure it's in scope for all branches
        PENDING = object()
        
        # Check if we have a cached render (atomic operation)
        cache_key = (page_num, zoom, fmt)
        if self._render_cache is not None:
            with self._render_cache_lock:
                if cache_key in self._render_cache:
                    cached_val = self._render_cache.pop(cache_key)
                    self._render_cache[cache_key] = cached_val
                    return cached_val
                
                # If not in cache, add a placeholder to prevent duplicate renders
                # by other threads
                self._render_cache[cache_key] = PENDING
                
        # Check if pdftoppm is available
        if not check_pdftoppm_available(self.poppler_path):
            # Cleanup the placeholder if we failed to find pdftoppm
            if self._render_cache is not None:
                with self._render_cache_lock:
                    if self._render_cache.get(cache_key) is PENDING:
                        del self._render_cache[cache_key]
            self._log_missing("pdftoppm (Poppler) not found. Install Poppler or set POPPLER_PATH", err_key="poppler")
            return None
            
        dpi = int((zoom or 7.0) * 72)
        dpi = max(dpi, 72)
        try:
            import tempfile
            
            # Find pdftoppm executable
            if self.poppler_path:
                pdftoppm_cmd = str(Path(self.poppler_path) / ("pdftoppm.exe" if sys.platform.startswith("win") else "pdftoppm"))
            else:
                pdftoppm_cmd = shutil.which("pdftoppm")
                
            # Command arguments
            args = [
                pdftoppm_cmd,
                "-f", str(page_num + 1),
                "-l", str(page_num + 1),
                "-r", str(dpi),
                "-png"  # Always use PNG for best quality
            ]
            
            # Handle case where PDF is in memory
            input_path = None
            temp_file = None
            if self._pdf_bytes:
                # Write PDF bytes to temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                temp_file.write(self._pdf_bytes)
                temp_file.close()
                input_path = temp_file.name
            else:
                input_path = self.pdf_path
                
            # Execute pdftoppm
            result = subprocess.run(
                args + [input_path],
                capture_output=True,
                text=False
            )
            
            # Cleanup temporary file if created
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass
                    
            if result.returncode != 0:
                raise RuntimeError(f"pdftoppm failed: {result.stderr.decode('utf-8', 'ignore')}")
                
            # Convert ppm output to PIL Image
            render_img = Image.open(BytesIO(result.stdout))
            
            render_path = None
            if self.persist_renders and self.renders_dir:
                render_filename = f"page_{page_num:03d}_render.{fmt}"
                render_path = os.path.join(self.renders_dir, render_filename)
                try:
                    render_img.save(render_path, fmt.upper())
                except Exception:
                    render_path = None

            # Cache the render (atomic operation)
            if self._render_cache is not None:
                with self._render_cache_lock:
                    # Check if the entry is still our placeholder
                    if self._render_cache.get(cache_key) is PENDING:
                        # Replace placeholder with actual render
                        self._render_cache[cache_key] = (render_img, render_path)
                        if len(self._render_cache) > self.render_cache_max_items:
                            # Explicitly close the oldest image before removing from cache
                            _, old_value = self._render_cache.popitem(last=False)
                            # Only try to close if we have an actual image (not PENDING)
                            if isinstance(old_value, tuple) and len(old_value) == 2:
                                old_img, _ = old_value
                                if hasattr(old_img, 'close'):
                                    old_img.close()
                    else:
                        # If another thread already replaced the placeholder,
                        # use that instead of the one we just rendered
                        render_img.close()
                        
            return render_img
            
        except Exception as e:
            self.log_error(f"Render failed (page {page_num + 1}): {e}")
            if self._render_cache is not None:
                with self._render_cache_lock:
                    if self._render_cache.get(cache_key) is PENDING:
                        del self._render_cache[cache_key]
            return None

    def cleanup_renders(self):
        if not self.persist_renders:
            import shutil
            if self.renders_dir and Path(self.renders_dir).exists():
                try:
                    shutil.rmtree(self.renders_dir)
                except Exception as e:
                    self.log_error(f"Failed to clean up temporary renders: {e}")

    def close(self):
        try:
            if self._pdf_file_handle:
                self._pdf_file_handle.close()
        except Exception:
            pass

        self.cleanup_renders()

    def _log_missing(self, msg, err_key=None):
        if self.log_error:
            try:
                self.log_error(msg)
            except Exception:
                pass

    def _log_error(self, msg):
        if self.log_error:
            try:
                self.log_error(msg)
            except Exception:
                pass
