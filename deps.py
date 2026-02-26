import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Any

torch: Optional[Any] = None
torchvision: Optional[Any] = None


def _ensure_torch_imported():
    """
    Ensure `torch` and `torchvision` are imported into the module globals if available.
    
    Returns:
        bool: `True` if both `torch` and `torchvision` are present or were successfully imported, `False` otherwise.
    """
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
    """
    Detects the best available torch device and returns a mapping describing it.
    
    The returned mapping reports whether torch is installed and which compute backend/device is selected, along with a brief reason string explaining the choice or any detection failure.
    
    Returns:
        TorchDeviceInfo: A mapping with the following keys:
            installed (bool): True if torch was successfully imported, False otherwise.
            backend (str): Chosen backend name: "cuda", "mps", or "cpu".
            device (str): Device identifier (e.g., "cuda:0", "mps", "cpu").
            reason (str): Human-readable explanation of the selected device or why detection failed.
    """
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
    """
    Compatibility shim that preserves the pdf2image import API while indicating the library is unavailable.
    
    Returns:
        tuple: A pair of `None` values serving as placeholders for the pdf2image module and its primary import/function.
    """
    return None, None


def check_pdftoppm_available(poppler_path: Optional[str] = None) -> bool:
    """
    Determine whether the `pdftoppm` executable is available on the system.
    
    If `poppler_path` is provided, the function first checks for `pdftoppm` inside that directory (uses `pdftoppm.exe` on Windows). If not found there, it checks the system PATH.
    
    Parameters:
        poppler_path (Optional[str]): Path to a Poppler binaries directory to check for the `pdftoppm` executable.
    
    Returns:
        bool: `True` if `pdftoppm` is found, `False` otherwise.
    """
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
    """
    Lazily import pypdf's `PdfReader`, cache it in the module, and update the availability flag.
    
    If `pypdf` is importable this sets the module-level `PdfReader` to the imported class and `pypdf_available` to `True`; otherwise sets `PdfReader` to `None` and `pypdf_available` to `False`. The result is cached so subsequent calls return the previously resolved value.
    
    Returns:
        tuple: (PdfReader or None, bool) - The `PdfReader` class from `pypdf` if available, None otherwise; and a boolean indicating availability.
    """
    global PdfReader, pypdf_available
    if PdfReader is not None:
        pypdf_available = True
        return PdfReader, pypdf_available
    try:
        from pypdf import PdfReader as _PdfReader
        PdfReader = _PdfReader
        pypdf_available = True
    except ImportError:
        PdfReader = None
        pypdf_available = False
    return PdfReader, pypdf_available


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
    """
    Attempt to import and cache the easyocr library.
    
    If the import succeeds, stores the module in the module-level variable `easyocr`
    and sets `easyocr_available` to True. If the import fails, sets `easyocr` to
    None and `easyocr_available` to False.
    
    Returns:
        The imported `easyocr` module if available, `None` otherwise.
    """
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
    # Update the exposed EASYOCR_AVAILABLE variable
    global EASYOCR_AVAILABLE
    EASYOCR_AVAILABLE = easyocr_available
    return easyocr

# Initialize EASYOCR_AVAILABLE
EASYOCR_AVAILABLE = False


# Tesseract
try:
    import pytesseract
    tesseract_available = True
except ImportError:
    pytesseract: Optional[Any] = None
    tesseract_available = False


def _bootstrap_tesseract_default_paths():
    """
    Configure pytesseract's tesseract_cmd to a common system path when the tesseract binary is present.
    
    If pytesseract is unavailable or already configured with an existing file path, this function does nothing. Otherwise it checks a small set of platform-appropriate candidate locations and sets `pytesseract.pytesseract.tesseract_cmd` to the first path where a tesseract executable is found.
    """
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
    """
    Finds a filesystem path that contains Poppler command-line binaries.
    
    Checks the POPPLER_PATH environment variable, common OS-specific installation locations, and entries on the system PATH for directories containing Poppler executables. 
    
    Returns:
        str: Path to a directory containing Poppler binaries (e.g., containing `pdftoppm`/`pdftocairo`), or `None` if no suitable directory is found.
    """
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
    """
    Attempt to import and cache the EPUB library from ebooklib.epub.
    
    If the import succeeds, caches the module in the `epub_lib` global and sets
    `epublib_available` to True. If the import fails, sets `epub_lib` to None and
    `epublib_available` to False.
    
    Returns:
        The imported `ebooklib.epub` module if available, `None` otherwise.
    """
    global epub_lib, epublib_available
    if epub_lib is not None:
        # Keep the exported availability flag in sync after first successful import
        globals()["EPUBLIB_AVAILABLE"] = True
        return epub_lib
    try:
        from ebooklib import epub
        epub_lib = epub
        epublib_available = True
        globals()["EPUBLIB_AVAILABLE"] = True
    except ImportError as e:
        print(f"EPUB library import error: {e}")
        epub_lib = None
        epublib_available = False
        globals()["EPUBLIB_AVAILABLE"] = False
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