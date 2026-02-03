# PDF Scraper - OCR Extractor

## Overview

This project is a comprehensive PDF scraping tool that uses OCR (Optical Character Recognition) to extract text from PDF documents. It supports both EasyOCR and Tesseract OCR engines, with intelligent retry logic and performance monitoring.

## Features

- **OCR Engine Support**: EasyOCR as primary, Tesseract as fallback/refinement
- **Performance Monitoring**: Detailed metrics for each phase of processing
- **Configuration Management**: Centralized configuration system
- **Logging**: Comprehensive logging with file and console outputs
- **Multi-threading**: Parallel processing for faster document processing
- **GUI and CLI**: Both graphical and command-line interfaces available
- **Configurable**: Support for environment variables, config files, and command-line parameters

## Installation

### Quick Installation (Recommended)

Run the lightweight installer that automatically installs all dependencies and configures the environment:

```bash
python installer.py
```

The installer will:
- Check and install Python dependencies
- Install system dependencies (Tesseract OCR, Poppler tools)
- Detect NVIDIA CUDA and prompt installation if needed
- Validate the environment
- Run a smoke test

### Manual Installation

If you prefer manual installation:

1. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Install Tesseract OCR (required for Tesseract engine)
    - Windows: Download from https://github.com/tesseract-ocr/tesseract
    - Linux: `sudo apt-get install tesseract-ocr`
    - macOS: `brew install tesseract`

3. Install Poppler tools (required for PDF rendering)
    - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
    - Linux: `sudo apt-get install poppler-utils`
    - macOS: `brew install poppler`

4. Optional: Install NVIDIA CUDA (for GPU acceleration)
    - Download from: https://developer.nvidia.com/cuda-toolkit
    - Recommended version: CUDA 11.8 or 12.x

## Usage

### Command Line Interface (CLI)

```bash
python cli.py --help
```

Basic usage:
```bash
python cli.py input.pdf --output output_dir --lang ben
```

Options:
- `--lang`: OCR language (default: ben)
- `--quality`: Enable quality mode (slower, cleaner)
- `--fast`: Enable fast mode (skip extra retries)
- `--tessdata-dir`: Custom tessdata directory for Tesseract
- `--persist-renders`: Save rendered page images for debugging
- `--max-workers`: Override worker pool size
- `--check-env`: Run environment diagnostics and exit

### Graphical User Interface (GUI)

```bash
python gui.py
```

### Environment Variables

You can configure the scraper using environment variables:

```bash
PDF_SCRAPER_PDF_PATH=input.pdf
PDF_SCRAPER_OUTPUT_ROOT=output_dir
PDF_SCRAPER_OCR_LANG=ben+eng
PDF_SCRAPER_QUALITY_MODE=true
PDF_SCRAPER_PERSIST_RENDERS=false
PDF_SCRAPER_MAX_WORKERS=4
```

### Configuration File

You can also use a JSON or YAML configuration file:

**config.json:**
```json
{
  "pdf_path": "input.pdf",
  "output_root": "output_dir",
  "use_ocr": true,
  "ocr": {
    "ocr_method": "easyocr",
    "ocr_lang": "ben+eng",
    "quality_mode": true,
    "fast_mode": false,
    "fast_confidence_skip": 0.92,
    "tessdata_dir": null
  },
  "render": {
    "zoom": 7.0,
    "high_dpi_zoom": 12.0,
    "high_dpi_retry_conf": 0.92,
    "pdf_bytes_cache_mb": 80,
    "persist_renders": false
  },
  "preprocess": {
    "header_footer_crop_pct": 0.12,
    "watermark_flatten": true,
    "watermark_clip_threshold": 245,
    "watermark_retry_conf": 0.82,
    "quantize_levels": 32,
    "quantize_dither": true,
    "third_pass_scale": 1.45
  },
  "text_layer": {
    "text_layer_first": true,
    "text_layer_lang_min_ratio": 0.35,
    "text_layer_min_ben_chars": 12
  },
  "max_workers": 4
}
```
## Troubleshooting Guide

### Common Issues

#### 1. OCR Engine Errors
- **EasyOCR not available**: Make sure you have EasyOCR installed: `pip install easyocr`
- **Tesseract not available**: Make sure you have Tesseract OCR installed and in your PATH.
- **Language file not found**: For Tesseract, make sure the language file is in the tessdata directory.

#### 2. Performance Issues
- **Slow processing**: Try using fast mode or increasing worker count.
- **High memory usage**: Reduce the cache size or worker count.

#### 3. Rendering Issues
- **Poppler not found**: Make sure Poppler is installed and in your PATH.
- **Rendering errors**: Try adjusting the zoom level.

### Logging and Debugging

- Check the `extraction.log` file in the output directory for detailed logs
- Use `--persist-renders` option to save rendered images for debugging
- Check the console output for error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes
5. Submit a pull request

## License

MIT License

## Credits

- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **Tesseract**: https://github.com/tesseract-ocr/tesseract
- **pdf2image**: https://github.com/Belval/pdf2image
- **PyPDF2**: https://github.com/py-pdf/pypdf2
