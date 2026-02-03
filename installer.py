#!/usr/bin/env python3
"""
PDF Scraper - Lightweight Installer
Installs all necessary dependencies and configures the environment.
Runs completely on the device and prompts for NVIDIA CUDA installation if not detected.
"""

import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Optional


def print_step(msg: str):
    """Print step information with formatting."""
    print(f"\033[1;34m→\033[0m {msg}")


def print_success(msg: str):
    """Print success message with formatting."""
    print(f"\033[1;32m✓\033[0m {msg}")


def print_error(msg: str):
    """Print error message with formatting."""
    print(f"\033[1;31m✗\033[0m {msg}")


def print_warning(msg: str):
    """Print warning message with formatting."""
    print(f"\033[1;33m⚠\033[0m {msg}")


import shlex
from typing import Union, Sequence

def run_command(cmd: Union[str, Sequence[str]], check: bool = True, capture_output: bool = False) -> Optional[subprocess.CompletedProcess]:
    """Run a command with optional error checking and output capture."""
    try:
        # Convert string command to list using shlex.split for safe parsing
        if isinstance(cmd, str):
            cmd_list = shlex.split(cmd)
        else:
            cmd_list = list(cmd)
            
        if capture_output:
            result = subprocess.run(
                cmd_list,
                shell=False,
                check=check,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
        else:
            result = subprocess.run(cmd_list, shell=False, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {cmd}")
        # Check available error information
        error_msg = None
        if e.stderr:
            error_msg = e.stderr.strip()
        elif e.output:
            error_msg = e.output.strip()
        elif e.stdout:
            error_msg = e.stdout.strip()
        else:
            error_msg = str(e)
        print_error(f"Error: {error_msg}")
        return None
    except Exception as e:
        print_error(f"Command failed: {cmd}")
        print_error(f"Error: {e}")
        return None


def check_python_version():
    """Check if Python version meets requirements (3.8+)."""
    print_step("Checking Python version...")
    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required")
        sys.exit(1)
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")


def install_python_deps():
    """Install Python dependencies from requirements.txt."""
    print_step("Installing Python dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("requirements.txt not found")
        sys.exit(1)
    
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    result = run_command(cmd, check=False, capture_output=True)
    
    if not result or result.returncode != 0:
        print_error("Failed to install Python dependencies")
        if result and result.stderr:
            print_error(f"Error output: {result.stderr}")
        sys.exit(1)
    
    print_success("Python dependencies installed successfully")


def is_tesseract_installed() -> bool:
    """Check if Tesseract OCR is installed."""
    return shutil.which("tesseract") is not None


def install_tesseract():
    """Install Tesseract OCR based on OS."""
    print_step("Installing Tesseract OCR...")
    
    system = platform.system()
    
    if system == "Windows":
        print_warning("Tesseract OCR installation on Windows requires manual download")
        print_warning("Please download from: https://github.com/tesseract-ocr/tesseract")
        print_warning("Install to C:\\Program Files\\Tesseract-OCR or C:\\Program Files (x86)\\Tesseract-OCR")
        input("Press Enter after installation is complete...")
    
    elif system == "Darwin":  # macOS
        cmd = ["brew", "install", "tesseract"]
        result = run_command(cmd, capture_output=True, check=False)
        if not result or result.returncode != 0:
            print_error("Failed to install Tesseract OCR")
            if result and result.stderr:
                print_error(f"Error output: {result.stderr}")
            return False
    
    else:  # Linux
        if shutil.which("apt-get"):
            # For apt-get, we need to run update first then install
            result1 = run_command(["sudo", "apt-get", "update"], capture_output=True, check=False)
            if result1 and result1.returncode == 0:
                result = run_command(["sudo", "apt-get", "install", "-y", "tesseract-ocr"], capture_output=True, check=False)
            else:
                result = result1
        elif shutil.which("yum"):
            result = run_command(["sudo", "yum", "install", "-y", "tesseract"], capture_output=True, check=False)
        elif shutil.which("dnf"):
            result = run_command(["sudo", "dnf", "install", "-y", "tesseract"], capture_output=True, check=False)
        else:
            print_error("Unsupported package manager. Please install tesseract manually.")
            return False
        if not result or result.returncode != 0:
            print_error("Failed to install Tesseract OCR")
            if result and result.stderr:
                print_error(f"Error output: {result.stderr}")
            return False
    
    if is_tesseract_installed():
        print_success("Tesseract OCR installed successfully")
        return True
    else:
        print_error("Tesseract OCR installation verification failed")
        return False


def is_poppler_installed() -> bool:
    """Check if Poppler tools are installed."""
    return shutil.which("pdftoppm") is not None and shutil.which("pdftocairo") is not None


def install_poppler():
    """Install Poppler tools based on OS."""
    print_step("Installing Poppler tools...")
    
    system = platform.system()
    
    if system == "Windows":
        print_warning("Poppler installation on Windows requires manual download")
        print_warning("Please download from: https://github.com/oschwartz10612/poppler-windows/releases")
        print_warning("Extract to C:\\poppler or add to system PATH")
        input("Press Enter after installation is complete...")
    
    elif system == "Darwin":  # macOS
        cmd = ["brew", "install", "poppler"]
        result = run_command(cmd, capture_output=True, check=False)
        if not result or result.returncode != 0:
            print_error("Failed to install Poppler tools")
            if result and result.stderr:
                print_error(f"Error output: {result.stderr}")
            return False
    
    else:  # Linux
        if shutil.which("apt-get"):
            # For apt-get, we need to run update first then install
            result1 = run_command(["sudo", "apt-get", "update"], capture_output=True, check=False)
            if result1 and result1.returncode == 0:
                result = run_command(["sudo", "apt-get", "install", "-y", "poppler-utils"], capture_output=True, check=False)
            else:
                result = result1
        elif shutil.which("yum"):
            result = run_command(["sudo", "yum", "install", "-y", "poppler-utils"], capture_output=True, check=False)
        elif shutil.which("dnf"):
            result = run_command(["sudo", "dnf", "install", "-y", "poppler-utils"], capture_output=True, check=False)
        else:
            print_error("Unsupported package manager. Please install poppler-utils manually.")
            return False
        if not result or result.returncode != 0:
            print_error("Failed to install Poppler tools")
            if result and result.stderr:
                print_error(f"Error output: {result.stderr}")
            return False
    
    if is_poppler_installed():
        print_success("Poppler tools installed successfully")
        return True
    else:
        print_error("Poppler tools installation verification failed")
        return False


def is_cuda_available() -> bool:
    """Check if NVIDIA CUDA is available."""
    print_step("Checking for NVIDIA CUDA...")
    
    system = platform.system()
    
    if system == "Windows":
        # Check if CUDA is installed via registry or command
        try:
            # Check NVIDIA-smi
            result = run_command(["nvidia-smi"], check=False, capture_output=True)
            if result and result.returncode == 0:
                return True
            
            # Check CUDA installation directory
            cuda_paths = [
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
                "C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA"
            ]
            
            for path in cuda_paths:
                if Path(path).exists() and any(Path(path).glob("v*")):
                    return True
                    
            return False
            
        except Exception as e:
            print_error(f"CUDA detection error: {e}")
            return False
    
    elif system == "Darwin":  # macOS
        # CUDA not available on macOS (Apple Silicon uses Metal)
        print_warning("CUDA not available on macOS (Apple Silicon uses Metal)")
        return False
    
    else:  # Linux
        try:
            # Check NVIDIA-smi
            result = run_command(["nvidia-smi"], check=False, capture_output=True)
            if result and result.returncode == 0:
                return True
                
            # Check CUDA installation
            cuda_paths = ["/usr/local/cuda", "/usr/cuda"]
            for path in cuda_paths:
                if Path(path).exists():
                    return True
                    
            return False
            
        except Exception as e:
            print_error(f"CUDA detection error: {e}")
            return False


def prompt_cuda_installation():
    """Prompt user to install CUDA if not available."""
    if is_cuda_available():
        print_success("NVIDIA CUDA is available")
        return True
    
    print_warning("NVIDIA CUDA not detected")
    
    # Check if NVIDIA GPU is present
    has_nvidia_gpu = False
    system = platform.system()
    
    if system == "Windows":
        try:
            import wmi
        except ImportError:
            print_step("Installing wmi library for GPU detection...")
            run_command([sys.executable, "-m", "pip", "install", "wmi"], check=False)
            try:
                import wmi
            except ImportError:
                print_warning("Failed to install wmi library - skipping GPU detection")
                return has_nvidia_gpu
        
        try:
            w = wmi.WMI()
            for gpu in w.Win32_VideoController():
                if "NVIDIA" in gpu.Name:
                    has_nvidia_gpu = True
                    break
        except Exception as e:
            print_warning(f"GPU detection error: {e}")
    
    elif system == "Linux":
        try:
            # For pipeline commands, we need to handle them separately
            # First run lspci, then grep the output
            result_lspci = run_command(["lspci"], check=False, capture_output=True)
            if result_lspci and result_lspci.returncode == 0 and "NVIDIA" in result_lspci.stdout:
                has_nvidia_gpu = True
        except Exception as e:
            print_warning(f"GPU detection error: {e}")
    
    if has_nvidia_gpu:
        print_warning("NVIDIA GPU detected but CUDA not installed")
        response = input("Would you like to install NVIDIA CUDA? (y/N): ").strip().lower()
        
        if response == "y":
            print_step("Installing NVIDIA CUDA...")
            if system == "Windows":
                print_warning("Please download CUDA from: https://developer.nvidia.com/cuda-toolkit")
                print_warning("Recommended version: CUDA 11.8 or 12.x")
                input("Press Enter after installation is complete...")
                
                # Verify installation
                if is_cuda_available():
                    print_success("NVIDIA CUDA installed successfully")
                    return True
                else:
                    print_warning("CUDA installation not verified")
                    return False
                    
            elif system == "Linux":
                print_warning("Please follow NVIDIA CUDA installation instructions for your Linux distribution")
                print_warning("URL: https://developer.nvidia.com/cuda-toolkit-archive")
                input("Press Enter after installation is complete...")
                
                if is_cuda_available():
                    print_success("NVIDIA CUDA installed successfully")
                    return True
                else:
                    print_warning("CUDA installation not verified")
                    return False
                    
        else:
            print_warning("CUDA installation skipped. The scraper will run on CPU.")
            return False
    
    else:
        print_warning("No NVIDIA GPU detected. Running on CPU.")
        return False


def validate_environment():
    """Validate the installed environment."""
    print_step("Validating environment...")
    
    # Check Python deps
    try:
        import easyocr
        import pytesseract
        import pypdf
        import PIL
        import numpy
        print_success("All Python dependencies imported successfully")
    except ImportError as e:
        print_error(f"Python dependency import failed: {e}")
        return False
    
    # Check system deps
    if not is_tesseract_installed():
        print_error("Tesseract OCR not installed or not in PATH")
        return False
        
    if not is_poppler_installed():
        print_error("Poppler tools not installed or not in PATH")
        return False
    
    print_success("Environment validation complete")
    return True


def run_smoke_test():
    """Run a quick smoke test to verify the scraper works."""
    print_step("Running smoke test...")
    
    # Try to import the main module
    try:
        import deps
        import scraper
        import preprocess
        
        # Test dependency detection
        deps.log_torch_env()
        device_info = deps.detect_torch_device()
        print(f"Device: {device_info['device']} ({device_info['reason']})")
        
        poppler_path = deps.detect_poppler_path()
        if poppler_path:
            print(f"Poppler path: {poppler_path}")
        
        print_success("Smoke test passed")
        return True
        
    except Exception as e:
        print_error(f"Smoke test failed: {e}")
        print_error("Please check your installation")
        import traceback
        print_error(f"Stack trace:\n{traceback.format_exc()}")
        return False


def main():
    """Main installation process."""
    print("\033[1;36m" + "="*50 + "\033[0m")
    print("\033[1;36mPDF Scraper - Lightweight Installer\033[0m")
    print("\033[1;36m" + "="*50 + "\033[0m")
    
    # Check Python version
    check_python_version()
    
    # Install Python dependencies
    install_python_deps()
    
    # Install system dependencies
    if not is_tesseract_installed():
        install_tesseract()
    
    if not is_poppler_installed():
        install_poppler()
    
    # Check CUDA
    prompt_cuda_installation()
    
    # Validate environment
    if validate_environment():
        print_success("\n✅ Installation complete!")
        
        # Run smoke test
        if run_smoke_test():
            print_success("\n🎉 The PDF scraper is ready to use!")
            print("\n\033[1;34mUsage options:\033[0m")
            print("  - GUI: python gui.py")
            print("  - CLI: python cli.py --help")
            print("  - Check env: python cli.py --check-env")
        else:
            print_warning("\n⚠️ Installation completed but smoke test failed")
            print_warning("Please check for errors above")
    else:
        print_error("\n❌ Installation failed - please fix the errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
