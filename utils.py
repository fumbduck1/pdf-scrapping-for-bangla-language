import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

from deps import (
    _lazy_import_pdf2image,
    _lazy_import_pypdf,
    TESSERACT_AVAILABLE,
    pytesseract,
    detect_poppler_path,
)


def _sanitize_tessdata_prefix(prefix: str):
    """Return a clean absolute tessdata prefix without stray quotes."""
    if not prefix:
        return None
    try:
        cleaned = str(Path(prefix).expanduser().resolve()).strip(" '\"")
        return cleaned
    except Exception:
        return prefix.strip(" '\"")


def _split_langs(lang_value: str) -> List[str]:
    """Split a Tesseract lang string like "ben+eng" or "ben,eng" into codes."""
    if not lang_value:
        return []
    parts = lang_value.replace(',', '+').split('+')
    return [p.strip() for p in parts if p.strip()]


def check_tesseract_ready() -> Tuple[bool, str]:
    """Check pytesseract import and tesseract executable availability."""
    if not TESSERACT_AVAILABLE:
        exe = sys.executable
        return False, f"pytesseract not importable in this Python. Install with: \n  {exe} -m pip install pytesseract"

    cmd = getattr(pytesseract, 'pytesseract', None)
    resolved = None
    if cmd and getattr(cmd, 'tesseract_cmd', None):
        resolved = cmd.tesseract_cmd
        if resolved and not os.path.isfile(resolved):
            resolved = None

    if resolved is None:
        which_path = shutil.which("tesseract")
        if which_path:
            resolved = which_path
            try:
                cmd.tesseract_cmd = which_path
            except Exception:
                pass

    if not resolved:
        return False, "tesseract.exe not found. Add Tesseract-OCR to PATH or set pytesseract.pytesseract.tesseract_cmd"

    try:
        cmd.tesseract_cmd = resolved
    except Exception:
        pass

    current_prefix = _sanitize_tessdata_prefix(os.environ.get("TESSDATA_PREFIX"))
    if current_prefix:
        os.environ["TESSDATA_PREFIX"] = current_prefix

    try:
        ver = pytesseract.get_tesseract_version()
        ver_str = str(ver)
        try:
            major = int(ver_str.split('.')[0])
        except Exception:
            major = None

        if major is not None and major < 4:
            return False, (
                "Tesseract is too old (" + ver_str + "). Install Tesseract 5.x and point "
                "pytesseract.pytesseract.tesseract_cmd to the new binary."
            )

        return True, f"tesseract ok ({resolved}), version {ver_str}"
    except Exception as e:
        return False, (
            "tesseract not runnable: " + str(e) + "\nIf you see 'Invalid tesseract version', "
            "install Tesseract 5.x+ and update PATH or pytesseract.pytesseract.tesseract_cmd."
        )


def validate_runtime_env():
    """Centralized dependency checks; return (errors, warnings)."""
    errors = []
    warnings = []

    if not _lazy_import_pypdf():
        errors.append("pypdf is required. Install with: pip install pypdf")
    converters = _lazy_import_pdf2image()
    if not converters or not all(converters):
        warnings.append("Raster OCR disabled: install pdf2image and Poppler if you need image-based OCR.")

    ok, msg = check_tesseract_ready()
    if not ok:
        # Tesseract missing blocks refinement but should not block EasyOCR-first flow
        warnings.append(msg)

    return errors, warnings


__all__ = [
    "_sanitize_tessdata_prefix",
    "_split_langs",
    "check_tesseract_ready",
    "validate_runtime_env",
]
