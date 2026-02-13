import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

ZERO_WIDTH_CHARS = ['\u200b', '\u200c', '\u200d', '\ufeff']

from deps import (
    _lazy_import_pypdf,
    _lazy_import_easyocr,
    # log_torch_env,  # unused
    detect_torch_device,
    TESSERACT_AVAILABLE,
    pytesseract,
    detect_poppler_path,
    check_pdftoppm_available,
)


def sanitize_tessdata_prefix(prefix: str | None):
    """
    Normalize and resolve a TESSDATA_PREFIX value to an absolute path.
    
    Parameters:
        prefix (str | None): A tessdata prefix value (may include surrounding quotes, whitespace, or ~), or None.
    
    Returns:
        str | None: An absolute, resolved path string with surrounding quotes and spaces removed; if prefix is falsy returns None; if resolution fails returns the stripped prefix string as a fallback.
    """
    if not prefix:
        return None
    try:
        cleaned = prefix.strip(" '\"")
        return str(Path(cleaned).expanduser().resolve())
    except Exception:
        return prefix.strip(" '\"")


def split_langs(lang_value: str) -> List[str]:
    """
    Convert a Tesseract language specifier (e.g., "ben+eng" or "ben,eng") into a list of language codes.
    
    Parameters:
        lang_value (str): Language specification string containing language codes separated by '+' or ','.
    
    Returns:
        List[str]: Ordered list of trimmed language code strings; returns an empty list for falsy input.
    """
    if not lang_value:
        return []
    parts = lang_value.replace(',', '+').split('+')
    return [p.strip() for p in parts if p.strip()]


def normalize_text(text: str) -> str:
    """Strip zero-width chars and normalize whitespace to single spaces per line."""
    if not text:
        return ""
    try:
        for zw in ZERO_WIDTH_CHARS:
            text = text.replace(zw, '')
        return '\n'.join(' '.join(line.split()) for line in text.splitlines())
    except Exception:
        return text


def bangla_ratio(text: str) -> Tuple[float, int]:
    """Return (ratio, count) of Bangla chars in the text (ignoring whitespace)."""
    if not text:
        return 0.0, 0
    try:
        tokens = [ch for ch in text if not ch.isspace()]
        ben_count = sum(1 for ch in tokens if '\u0980' <= ch <= '\u09FF')
        ratio = ben_count / max(len(tokens), 1)
        return ratio, ben_count
    except Exception:
        return 0.0, 0


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
    """
    Validate that pytesseract can be imported and that a runnable Tesseract executable is available.
    
    If the TESSDATA_PREFIX environment variable is present it will be normalized and set in the environment before probing Tesseract. The function also checks the Tesseract major version and treats versions below 4 as unsupported.
    
    Returns:
        tuple: `True` and a success message containing the resolved executable path and version if Tesseract is usable; `False` and a diagnostic message explaining the problem otherwise.
    """
    if not TESSERACT_AVAILABLE:
        exe = sys.executable
        return False, f"pytesseract not importable in this Python. Install with: \n  {exe} -m pip install pytesseract"

    resolved = resolve_tesseract_cmd()
    if not resolved:
        return False, "tesseract not found. Add Tesseract-OCR to PATH or set pytesseract.pytesseract.tesseract_cmd"

    current_prefix = sanitize_tessdata_prefix(os.environ.get("TESSDATA_PREFIX"))
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
        
    poppler_path = detect_poppler_path()
    if not check_pdftoppm_available(poppler_path):
        warnings.append("Raster OCR disabled: Poppler (pdftoppm) not found. Install Poppler or set POPPLER_PATH.")

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
        if ppm:
            path_hint = Path(ppm).parent
        elif cairo:
            path_hint = Path(cairo).parent
        else:
            return False, "Poppler not found; raster OCR will be skipped (install poppler utilities or set POPPLER_PATH)"
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

    poppler_path = detect_poppler_path()
    if check_pdftoppm_available(poppler_path):
        info.append("poppler: ok (pdftoppm available)")

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
    "sanitize_tessdata_prefix",
    "split_langs",
    "normalize_text",
    "bangla_ratio",
    "resolve_tesseract_cmd",
    "check_tesseract_ready",
    "validate_runtime_env",
    "summarize_env",
    "print_env_report",
    "check_poppler_ready",
]