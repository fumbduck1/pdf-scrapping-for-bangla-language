import os
import sys
import shutil
from pathlib import Path

TORCH_AVAILABLE = False
torch = None
torchvision = None


def _ensure_torch_imported():
    """Lazy import torch/torchvision; set TORCH_AVAILABLE flag."""
    global TORCH_AVAILABLE, torch, torchvision
    if torch is not None and torchvision is not None:
        return True
    try:
        import torch as _torch
        import torchvision as _torchvision
        torch = _torch
        torchvision = _torchvision
        TORCH_AVAILABLE = True
        return True
    except ImportError:
        TORCH_AVAILABLE = False
        return False


def detect_torch_device():
    """Return a dict describing the best-available torch device."""
    info = {
        "installed": False,
        "backend": "cpu",
        "device": "cpu",
        "reason": "torch not installed",
    }
    if not _ensure_torch_imported():
        return info

    info["installed"] = True
    try:
        if torch.cuda.is_available():
            name = None
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "cuda:0"
            info.update({
                "backend": "cuda",
                "device": "cuda:0",
                "reason": f"CUDA available ({name})",
            })
            return info
    except Exception as e:
        info["reason"] = f"cuda check failed: {e}"

    try:
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        mps_ok = False
    if mps_ok:
        info.update({
            "backend": "mps",
            "device": "mps",
            "reason": "Apple MPS available",
        })
        return info

    info["reason"] = "CPU fallback"
    return info


def log_torch_env():
    """Print torch/torchvision and accelerator status if installed."""
    if not _ensure_torch_imported():
        print("torch: not installed")
        return

    device = detect_torch_device()
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("device backend:", device.get("backend"))
    print("device:", device.get("device"))
    print("reason:", device.get("reason"))


convert_from_path = None
convert_from_bytes = None
PDF2IMAGE_AVAILABLE = False
PdfReader = None
PYPDF_AVAILABLE = False


def _lazy_import_pdf2image():
    global convert_from_path, convert_from_bytes, PDF2IMAGE_AVAILABLE
    if convert_from_path and convert_from_bytes:
        PDF2IMAGE_AVAILABLE = True
        return convert_from_path, convert_from_bytes
    try:
        from pdf2image import convert_from_path as _cfp, convert_from_bytes as _cfb
        convert_from_path, convert_from_bytes = _cfp, _cfb
        PDF2IMAGE_AVAILABLE = True
    except ImportError:
        convert_from_path = None
        convert_from_bytes = None
        PDF2IMAGE_AVAILABLE = False
    return convert_from_path, convert_from_bytes


def _lazy_import_pypdf():
    global PdfReader, PYPDF_AVAILABLE
    if PdfReader is not None:
        PYPDF_AVAILABLE = True
        return PdfReader
    try:
        from pypdf import PdfReader as _PdfReader
        PdfReader = _PdfReader
        PYPDF_AVAILABLE = True
    except ImportError:
        PdfReader = None
        PYPDF_AVAILABLE = False
    return PdfReader


np = None
EASYOCR_AVAILABLE = False
easyocr = None


def _lazy_import_numpy():
    global np
    if np is not None:
        return np
    try:
        import numpy as _np
        np = _np
    except ImportError:
        np = None
    return np


def _lazy_import_easyocr():
    global easyocr, EASYOCR_AVAILABLE
    if easyocr is not None:
        return easyocr
    try:
        import easyocr as _easyocr
        easyocr = _easyocr
        EASYOCR_AVAILABLE = True
    except ImportError:
        easyocr = None
        EASYOCR_AVAILABLE = False
    return easyocr


# Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False


def _bootstrap_tesseract_default_paths():
    """Set default tesseract binary if installed in common locations."""
    if not TESSERACT_AVAILABLE or not hasattr(pytesseract, "pytesseract"):
        return
    cmd = pytesseract.pytesseract
    if getattr(cmd, "tesseract_cmd", None) and Path(cmd.tesseract_cmd).is_file():
        return

    if sys.platform.startswith("win"):
        default_paths = [
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        ]
    elif sys.platform == "darwin":
        default_paths = [
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
    else:
        default_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]

    for candidate in default_paths:
        if os.path.isfile(candidate):
            cmd.tesseract_cmd = candidate
            break


_bootstrap_tesseract_default_paths()


def _poppler_bins_exist(path: Path):
    return any((path / exe).exists() for exe in ["pdftoppm", "pdftocairo"])


def detect_poppler_path():
    """Best-effort detection of poppler binaries across platforms."""
    env_path = os.environ.get("POPPLER_PATH")
    if env_path:
        env_candidate = Path(env_path)
        if env_candidate.exists() and _poppler_bins_exist(env_candidate):
            return str(env_candidate)

    candidates = []
    if sys.platform.startswith("win"):
        pf = os.environ.get("PROGRAMFILES", r"C:\\Program Files")
        pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\\Program Files (x86)")
        for base in [pf, pf86]:
            try:
                base_path = Path(base)
                if base_path.exists():
                    candidates.extend(base_path.glob("poppler*\\Library\\bin"))
            except Exception:
                pass
        candidates.append(Path(r"C:\\poppler\\bin"))
    elif sys.platform == "darwin":
        candidates = [
            Path("/opt/homebrew/opt/poppler/bin"),
            Path("/usr/local/opt/poppler/bin"),
        ]
    else:
        candidates = [Path("/usr/bin"), Path("/usr/local/bin")]

    # Also consider PATH entries
    for exe in ("pdftoppm", "pdftocairo"):
        found = shutil.which(exe)
        if found:
            candidates.append(Path(found).parent)

    for cand in candidates:
        try:
            if cand and cand.exists() and _poppler_bins_exist(cand):
                return str(cand)
        except Exception:
            continue
    return None

__all__ = [
    "convert_from_path",
    "convert_from_bytes",
    "PdfReader",
    "PDF2IMAGE_AVAILABLE",
    "PYPDF_AVAILABLE",
    "np",
    "easyocr",
    "EASYOCR_AVAILABLE",
    "pytesseract",
    "TESSERACT_AVAILABLE",
    "log_torch_env",
    "detect_torch_device",
    "detect_poppler_path",
]
