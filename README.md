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

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Optional: Install Tesseract OCR (required for Tesseract engine)
   - Windows: Download from https://github.com/tesseract-ocr/tesseract
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

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

## API Documentation

### Configuration Classes

#### `JobConfig`
Complete job configuration dataclass.

**Attributes:**
- `pdf_path`: Path to the PDF file
- `output_root`: Output directory for results
- `use_ocr`: Whether to use OCR (default: True)
- `ocr`: OCR configuration (see `OCRConfig`)
- `render`: Render configuration (see `RenderConfig`)
- `preprocess`: Preprocess configuration (see `PreprocessConfig`)
- `text_layer`: Text layer configuration (see `TextLayerConfig`)
- `max_workers`: Maximum number of workers for parallel processing (default: None)

#### `OCRConfig`
OCR engine configuration.

**Attributes:**
- `ocr_method`: OCR method (easyocr or tesseract, default: easyocr)
- `ocr_lang`: OCR language (default: ben)
- `quality_mode`: Whether to use quality mode (default: QUALITY_MODE_DEFAULT)
- `fast_mode`: Whether to use fast mode (default: FAST_MODE)
- `fast_confidence_skip`: Confidence threshold for skipping second OCR pass (default: FAST_CONFIDENCE_SKIP)
- `tessdata_dir`: Path to Tesseract data directory (default: None)

#### `RenderConfig`
PDF rendering configuration.

**Attributes:**
- `zoom`: Rendering zoom level (default: DEFAULT_ZOOM)
- `high_dpi_zoom`: High DPI rendering zoom level (default: HIGH_DPI_ZOOM)
- `high_dpi_retry_conf`: Confidence threshold for high DPI retry (default: HIGH_DPI_RETRY_CONF)
- `pdf_bytes_cache_mb`: Cache size for PDF bytes (default: PDF_BYTES_CACHE_MB)
- `persist_renders`: Whether to save rendered images (default: False)

#### `PreprocessConfig`
Image preprocessing configuration.

**Attributes:**
- `header_footer_crop_pct`: Header/footer crop percentage (default: HEADER_FOOTER_CROP_PCT)
- `watermark_flatten`: Whether to flatten watermarks (default: WATERMARK_FLATTEN)
- `watermark_clip_threshold`: Watermark clip threshold (default: WATERMARK_CLIP_THRESHOLD)
- `watermark_retry_conf`: Watermark retry confidence (default: WATERMARK_RETRY_CONF)
- `quantize_levels`: Quantization levels (default: QUANTIZE_LEVELS)
- `quantize_dither`: Whether to use dithering (default: QUANTIZE_DITHER)
- `third_pass_scale`: Third pass scale (default: THIRD_PASS_SCALE)

#### `TextLayerConfig`
Text layer extraction configuration.

**Attributes:**
- `text_layer_first`: Whether to try text layer first (default: TEXT_LAYER_FIRST)
- `text_layer_lang_min_ratio`: Minimum language ratio (default: TEXT_LAYER_LANG_MIN_RATIO)
- `text_layer_min_ben_chars`: Minimum Bengali characters (default: TEXT_LAYER_MIN_BEN_CHARS)

### Configuration Manager

#### `ConfigManager`
Centralized configuration manager.

**Methods:**
- `from_dict(config_dict)`: Create a JobConfig from a dictionary
- `from_env()`: Create a JobConfig from environment variables
- `from_file(config_path)`: Load configuration from JSON or YAML file
- `validate_config(config)`: Validate configuration
- `get_default_config()`: Get default configuration

**Usage:**
```python
from config_manager import get_config_manager

config_manager = get_config_manager()
config = config_manager.from_dict({
    "pdf_path": "input.pdf",
    "output_root": "output",
    "ocr_lang": "ben+eng",
    "quality_mode": True
})
errors = config_manager.validate_config(config)
if errors:
    print("Validation errors:", errors)
else:
    print("Configuration is valid")
```

### Performance Monitoring

#### `PerformanceMonitor`
Performance monitoring and profiling class.

**Methods:**
- `register_metric(name, units, description)`: Register a new metric
- `add_metric_value(name, value, units)`: Add a value to a metric
- `get_metric(name)`: Get a registered metric
- `get_all_metrics()`: Get all registered metrics
- `get_summary()`: Get summary of all metrics and total time
- `start_profiling(name)`: Start profiling a section
- `stop_profiling(name)`: Stop profiling a section and record time
- `clear()`: Clear all metrics and reset timer
- `print_summary()`: Print a formatted summary

**Decorator Usage:**
```python
from performance import timer, profile

@timer("test_function")
def test_function():
    # Function to be timed
    pass

# Context manager usage
with profile("test_section"):
    # Code to be profiled
    pass
```

### Scraper Classes

#### `PDFScraper`
Main scraper class for PDF processing.

**Methods:**
- `scrape_all_pages()`: Scrape all pages with optional parallel OCR per page
- `save_results()`: Save results to output directory
- `cleanup_renders()`: Clean up temporary render files

**Usage:**
```python
from scraper import PDFScraper
from config_manager import create_job_config

config = create_job_config("input.pdf", "output")
scraper = PDFScraper(
    config.pdf_path,
    config.output_root,
    use_ocr=config.use_ocr,
    ocr_method=config.ocr.ocr_method,
    ocr_lang=config.ocr.ocr_lang,
    quality_mode=config.ocr.quality_mode,
    fast_mode=config.ocr.fast_mode,
    fast_confidence_skip=config.ocr.fast_confidence_skip,
    tessdata_dir=config.ocr.tessdata_dir,
    persist_renders=config.render.persist_renders,
    pdf_bytes_cache_mb=config.render.pdf_bytes_cache_mb,
    zoom=config.render.zoom,
    high_dpi_zoom=config.render.high_dpi_zoom,
    high_dpi_retry_conf=config.render.high_dpi_retry_conf,
    header_footer_crop_pct=config.preprocess.header_footer_crop_pct,
    watermark_flatten=config.preprocess.watermark_flatten,
    watermark_clip_threshold=config.preprocess.watermark_clip_threshold,
    watermark_retry_conf=config.preprocess.watermark_retry_conf,
    quantize_levels=config.preprocess.quantize_levels,
    quantize_dither=config.preprocess.quantize_dither,
    third_pass_scale=config.preprocess.third_pass_scale,
    text_layer_first=config.text_layer.text_layer_first,
    text_layer_lang_min_ratio=config.text_layer.text_layer_lang_min_ratio,
    text_layer_min_ben_chars=config.text_layer.text_layer_min_ben_chars,
    max_workers=config.max_workers
)
scraper.scrape_all_pages()
scraper.save_results()
```

## Performance Optimization Tips

1. **Enable GPU acceleration**: EasyOCR runs much faster with CUDA. Make sure you have CUDA and PyTorch installed.

2. **Use fast mode for quick processing**:
   ```bash
   python cli.py input.pdf --output output --fast
   ```

3. **Adjust worker count**: Increase max workers to use more CPU cores:
   ```bash
   python cli.py input.pdf --output output --max-workers 8
   ```

4. **Reduce quality mode for speed**:
   ```bash
   python cli.py input.pdf --output output --fast
   ```

5. **Increase cache size**: For large PDFs, increase the cache size:
   ```bash
   python cli.py input.pdf --output output --config config.json
   ```
   where config.json contains:
   ```json
   {"pdf_bytes_cache_mb": 200}
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
