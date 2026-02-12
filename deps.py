import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Any

torch: Optional[Any] = None
torchvision: Optional[Any] = None


def _ensure_torch_imported():
    """Lazy import torch/torchvision; return True when both modules are present."""
    global torch, torchvision
    if torch is not None and torchvision is not None:
        return True
    try:
        import torch as _torch
        import torchvision as _torchvision
        torch = _torch
        torchvision = _torchvision
        return True
    except ImportError:
        return False


from typing import TypedDict

class TorchDeviceInfo(TypedDict):
    installed: bool
    backend: str
    device: str
    reason: str

def detect_torch_device() -> TorchDeviceInfo:
    """Return a dict describing the best-available torch device."""
    info: TorchDeviceInfo = {
        "installed": False,
        "backend": "cpu",
        "device": "cpu",
        "reason": "torch not installed",
    }
    if not _ensure_torch_imported():
        return info

    info["installed"] = True
    try:
        if torch.cuda.is_available():  # type: ignore
            name = None
            try:
                name = torch.cuda.get_device_name(0)  # type: ignore
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
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore
    except Exception:
        mps_ok = False
    if mps_ok:
        info.update({
            "backend": "mps",
            "device": "mps",
            "reason": "Apple MPS available",
        })
        return info

    if "cuda check failed" in info.get("reason", ""):
        info["reason"] = f"{info['reason']}; CPU fallback"
    else:
        info["reason"] = "CPU fallback"
    return info


def log_torch_env():
    """Print torch/torchvision and accelerator status if installed."""
    if not _ensure_torch_imported():
        print("torch: not installed")
        return

    device = detect_torch_device()
    print("torch:", torch.__version__)  # type: ignore
    print("torchvision:", torchvision.__version__)  # type: ignore
    print("device backend:", device.get("backend"))
    print("device:", device.get("device"))
    print("reason:", device.get("reason"))
    
PdfReader: Optional[Any] = None
pypdf_available = False

# Compatibility alias for existing code/test compatibility
PDF2IMAGE_AVAILABLE = False

# Poppler commands availability
pdftoppm_available = False


def _lazy_import_pdf2image():
    """Compatibility function for existing code - returns None values."""
    return None, None


def check_pdftoppm_available(poppler_path: Optional[str] = None) -> bool:
    """Check if pdftoppm command is available."""
    global pdftoppm_available
    
    import shutil
    
    if pdftoppm_available:
        return True
        
    # Try to find pdftoppm in poppler path or system PATH
    if poppler_path:
        pdftoppm_path = Path(poppler_path) / ("pdftoppm.exe" if sys.platform.startswith("win") else "pdftoppm")
        if pdftoppm_path.exists() and pdftoppm_path.is_file():
            pdftoppm_available = True
            return True
            
    # Check system PATH
    if shutil.which("pdftoppm"):
        pdftoppm_available = True
        return True
        
    return False


def _lazy_import_pypdf():
    global PdfReader, pypdf_available
    if PdfReader is not None:
        pypdf_available = True
        return PdfReader
    try:
        from pypdf import PdfReader as _PdfReader
        PdfReader = _PdfReader
        pypdf_available = True
    except ImportError:
        PdfReader = None
        pypdf_available = False
    return PdfReader


np: Optional[Any] = None
easyocr_available = False
easyocr: Optional[Any] = None


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
    global easyocr, easyocr_available
    if easyocr is not None:
        return easyocr
    try:
        import easyocr as _easyocr
        easyocr = _easyocr
        easyocr_available = True
    except ImportError:
        easyocr = None
        easyocr_available = False
    return easyocr


# Tesseract
try:
    import pytesseract
    tesseract_available = True
except ImportError:
    pytesseract: Optional[Any] = None
    tesseract_available = False


def _bootstrap_tesseract_default_paths():
    """Set default tesseract binary if installed in common locations."""
    if not tesseract_available or pytesseract is None or not hasattr(pytesseract, "pytesseract"):
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
    exe_names = ["pdftoppm", "pdftocairo", "pdftoppm.exe", "pdftocairo.exe"]
    return any((path / exe).exists() for exe in exe_names)


def detect_poppler_path():
    """Best-effort detection of poppler binaries across platforms."""
    env_path = os.environ.get("POPPLER_PATH")
    if env_path:
        env_candidate = Path(env_path)
        if env_candidate.exists() and _poppler_bins_exist(env_candidate):
            return str(env_candidate)

    candidates: list[Path] = []
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

# EPUB support
epub_lib: Optional[Any] = None
epublib_available = False

def _lazy_import_epublib():
    global epub_lib, epublib_available
    if epub_lib is not None:
        return epub_lib
    try:
        from ebooklib import epub
        epub_lib = epub
        epublib_available = True
    except ImportError as e:
        print(f"EPUB library import error: {e}")
        epub_lib = None
        epublib_available = False
    return epub_lib

# Initialize on module import
_lazy_import_epublib()

# Backward compatibility aliases
PYPDF_AVAILABLE = pypdf_available
PDFTOPPM_AVAILABLE = pdftoppm_available
EASYOCR_AVAILABLE = easyocr_available
TESSERACT_AVAILABLE = tesseract_available
EPUBLIB_AVAILABLE = epublib_available

__all__ = [
    "PdfReader",
    "pypdf_available",
    "PYPDF_AVAILABLE",
    "pdftoppm_available",
    "PDFTOPPM_AVAILABLE",
    "check_pdftoppm_available",
    "np",
    "easyocr",
    "easyocr_available",
    "EASYOCR_AVAILABLE",
    "pytesseract",
    "tesseract_available",
    "TESSERACT_AVAILABLE",
    "log_torch_env",
    "detect_torch_device",
    "detect_poppler_path",
    "epub_lib",
    "epublib_available",
    "EPUBLIB_AVAILABLE",
    "_lazy_import_epublib",
    "_lazy_import_pypdf",
    "_lazy_import_numpy",
    "_lazy_import_easyocr",
    "_lazy_import_pdf2image",
]
