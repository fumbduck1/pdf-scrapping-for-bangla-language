import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

from deps import (
    _lazy_import_pdf2image,
    _lazy_import_pypdf,
    _lazy_import_easyocr,
    log_torch_env,
    detect_torch_device,
    TESSERACT_AVAILABLE,
    pytesseract,
    detect_poppler_path,
)


def _sanitize_tessdata_prefix(prefix: str):
    """Return a clean absolute tessdata prefix without stray quotes."""
    if not prefix:
        return None
    try:
        cleaned = prefix.strip(" '\"")
        return str(Path(cleaned).expanduser().resolve())
    except Exception:
        return prefix.strip(" '\"")


def _split_langs(lang_value: str) -> List[str]:
    """Split a Tesseract lang string like "ben+eng" or "ben,eng" into codes."""
    if not lang_value:
        return []
    parts = lang_value.replace(',', '+').split('+')
    return [p.strip() for p in parts if p.strip()]


def resolve_tesseract_cmd():
    """Resolve a usable tesseract binary path and set pytesseract if found."""
    if not TESSERACT_AVAILABLE:
        return None
    cmd = getattr(pytesseract, 'pytesseract', None)
    resolved = None
    if cmd and getattr(cmd, 'tesseract_cmd', None):
        candidate = cmd.tesseract_cmd
        if candidate and os.path.isfile(candidate):
            resolved = candidate

    if resolved is None:
        which_path = shutil.which("tesseract")
        if which_path and os.path.isfile(which_path):
            resolved = which_path

    if resolved is None:
        if sys.platform.startswith("win"):
            defaults = [
                "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
            ]
        elif sys.platform == "darwin":
            defaults = [
                "/opt/homebrew/bin/tesseract",
                "/usr/local/bin/tesseract",
            ]
        else:
            defaults = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
            ]
        for candidate in defaults:
            if os.path.isfile(candidate):
                resolved = candidate
                break

    if resolved:
        try:
            if cmd:
                cmd.tesseract_cmd = resolved
        except Exception:
            pass
    return resolved


def check_tesseract_ready() -> Tuple[bool, str]:
    """Check pytesseract import and tesseract executable availability."""
    if not TESSERACT_AVAILABLE:
        exe = sys.executable
        return False, f"pytesseract not importable in this Python. Install with: \n  {exe} -m pip install pytesseract"

    resolved = resolve_tesseract_cmd()
    if not resolved:
        return False, "tesseract not found. Add Tesseract-OCR to PATH or set pytesseract.pytesseract.tesseract_cmd"

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
    else:
        ok_poppler, poppler_msg = check_poppler_ready()
        if not ok_poppler:
            warnings.append(poppler_msg)

    ok, msg = check_tesseract_ready()
    if not ok:
        # Tesseract missing blocks refinement but should not block EasyOCR-first flow
        warnings.append(msg)

    return errors, warnings


def check_poppler_ready():
    """Check Poppler availability; return (ok, message)."""
    poppler_path = detect_poppler_path()
    if poppler_path:
        return True, f"poppler ok ({poppler_path})"

    ppm = shutil.which("pdftoppm")
    cairo = shutil.which("pdftocairo")
    if ppm or cairo:
        path_hint = Path(ppm).parent if ppm else Path(cairo).parent
        return True, f"poppler ok ({path_hint})"

    return False, "Poppler not found; raster OCR will be skipped (install poppler utilities or set POPPLER_PATH)"

def summarize_env():
    """Return (info, warnings, errors) describing the runtime environment."""
    errors, warnings = validate_runtime_env()
    info = []

    info.append(f"python: {sys.version.split()[0]} ({sys.platform})")
    info.append(f"cwd: {Path.cwd()}")

    poppler_path = detect_poppler_path()
    if poppler_path:
        info.append(f"poppler: {poppler_path}")
    else:
        info.append("poppler: not found")

    if _lazy_import_pypdf():
        info.append("pypdf: ok")
    elif not any("pypdf" in e for e in errors):
        errors.append("pypdf missing (pip install pypdf)")

    converters = _lazy_import_pdf2image()
    if converters and all(converters):
        info.append("pdf2image: ok")
    elif not any("pdf2image" in w for w in warnings):
        warnings.append("pdf2image missing; raster OCR disabled until installed")

    easyocr_mod = _lazy_import_easyocr()
    if easyocr_mod:
        version = getattr(easyocr_mod, "__version__", "unknown")
        info.append(f"easyocr: {version}")
    else:
        warnings.append("easyocr not importable (pip install easyocr)")

    try:
        device = detect_torch_device()
        if not device.get("installed"):
            warnings.append("torch not installed; EasyOCR will not run")
        else:
            info.append(f"torch device: {device.get('backend')} ({device.get('device')}) - {device.get('reason')}")
    except Exception:
        warnings.append("torch check failed; install torch/torchvision if you need EasyOCR GPU")

    return info, warnings, errors


def print_env_report():
    """Print environment diagnostics to stdout."""
    info, warnings, errors = summarize_env()
    print("Environment diagnostics:\n-----------------------")
    for line in info:
        print(f"- {line}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e}")
__all__ = [
    "_sanitize_tessdata_prefix",
    "_split_langs",
    "resolve_tesseract_cmd",
    "check_tesseract_ready",
    "validate_runtime_env",
    "summarize_env",
    "print_env_report",
    "check_poppler_ready",
]
