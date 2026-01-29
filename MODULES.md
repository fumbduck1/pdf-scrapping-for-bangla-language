# PDF Scraper - New Modules Documentation

## Overview

This document provides detailed documentation for the new modules introduced to enhance the PDF scraper application:

1. **Configuration System** (`config_manager.py`) - Centralized configuration management
2. **Performance Monitoring** (`performance.py`) - Performance tracking and profiling
3. **Logging System** (`logger.py`) - Enhanced logging framework

## Configuration System (`config_manager.py`)

### Overview

The configuration system provides a centralized, type-safe way to manage all application settings. It supports:

- Type-safe configuration with dataclasses
- Multiple configuration sources (dict, environment variables, files)
- Configuration validation
- Default values for all settings

### Key Components

#### `JobConfig`

The main configuration class representing a complete job configuration:

```python
@dataclass
class JobConfig:
    pdf_path: str
    output_root: str
    use_ocr: bool = True
    ocr: OCRConfig = field(default_factory=OCRConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    text_layer: TextLayerConfig = field(default_factory=TextLayerConfig)
    max_workers: Optional[int] = None
```

#### `OCRConfig`

OCR engine configuration:

```python
@dataclass
class OCRConfig:
    ocr_method: str = "easyocr"  # or "tesseract"
    ocr_lang: str = "ben"
    quality_mode: bool = True
    fast_mode: bool = False
    fast_confidence_skip: float = 0.92
    tessdata_dir: Optional[str] = None
```

#### `RenderConfig`

PDF rendering configuration:

```python
@dataclass
class RenderConfig:
    zoom: float = 7.0
    high_dpi_zoom: float = 12.0
    high_dpi_retry_conf: float = 0.92
    pdf_bytes_cache_mb: int = 80
    persist_renders: bool = False
```

#### `PreprocessConfig`

Image preprocessing configuration:

```python
@dataclass
class PreprocessConfig:
    header_footer_crop_pct: float = 0.12
    watermark_flatten: bool = True
    watermark_clip_threshold: int = 245
    watermark_retry_conf: float = 0.82
    quantize_levels: int = 32
    quantize_dither: bool = True
    third_pass_scale: float = 1.45
```

#### `TextLayerConfig`

PDF text layer extraction configuration:

```python
@dataclass
class TextLayerConfig:
    text_layer_first: bool = True
    text_layer_lang_min_ratio: float = 0.35
    text_layer_min_ben_chars: int = 12
```

### Usage Examples

#### Basic Configuration

```python
from config_manager import create_job_config

# Create a basic job configuration
config = create_job_config(
    pdf_path="document.pdf",
    output_root="output",
    use_ocr=True
)

print(config.ocr.ocr_lang)  # Output: "ben"
print(config.render.zoom)  # Output: 7.0
```

#### Custom Configuration

```python
from config_manager import create_job_config

# Create a custom job configuration
config = create_job_config(
    "document.pdf", "output",
    use_ocr=True,
    ocr={
        "ocr_method": "tesseract",
        "ocr_lang": "eng",
        "quality_mode": False
    },
    render={
        "zoom": 10.0,
        "persist_renders": True
    },
    max_workers=4
)

print(config.ocr.ocr_method)  # Output: "tesseract"
print(config.ocr.quality_mode)  # Output: False
```

#### Loading from Environment Variables

```python
import os
from config_manager import get_config_manager

# Set environment variables
os.environ["PDF_SCRAPER_PDF_PATH"] = "document.pdf"
os.environ["PDF_SCRAPER_OUTPUT_ROOT"] = "output"
os.environ["PDF_SCRAPER_OCR_METHOD"] = "easyocr"
os.environ["PDF_SCRAPER_OCR_LANG"] = "en"
os.environ["PDF_SCRAPER_QUALITY_MODE"] = "false"

# Load from environment variables
config_manager = get_config_manager()
config = config_manager.from_env()

print(config.ocr.ocr_lang)  # Output: "en"
print(config.ocr.quality_mode)  # Output: False
```

#### Configuration Validation

```python
from config_manager import get_config_manager, create_job_config

config_manager = get_config_manager()
config = create_job_config("nonexistent.pdf", "output")

# Validate configuration
errors = config_manager.validate_config(config)
if errors:
    for error in errors:
        print(f"Validation error: {error}")
```

## Performance Monitoring (`performance.py`)

### Overview

The performance monitoring system provides detailed metrics about the application's execution. It supports:

- Decorator-based timing of functions
- Context manager for profiling sections
- Comprehensive performance reports
- Multiple metrics collection

### Key Components

#### `PerformanceMonitor`

The central performance monitoring class:

```python
monitor = get_monitor()  # Singleton instance
```

#### `@timer` Decorator

Timing functions:

```python
@timer("function_name")
def process_document():
    # Function implementation
    pass
```

#### `profile` Context Manager

Profiling sections of code:

```python
with profile("section_name"):
    # Code to profile
    pass
```

### Usage Examples

#### Basic Performance Tracking

```python
from performance import get_monitor, timer, profile, register_metrics, print_performance_report
import time

register_metrics()  # Register common metrics

@timer("processing")
def process_task():
    with profile("subtask1"):
        time.sleep(0.1)
    
    with profile("subtask2"):
        time.sleep(0.2)

# Run multiple times
for _ in range(3):
    process_task()

# Print performance report
print_performance_report()
```

#### Custom Metrics

```python
from performance import get_monitor, register_metrics, print_performance_report
import time

monitor = get_monitor()
monitor.register_metric("custom_task", "seconds", "My custom task timing")

with profile("custom_task"):
    time.sleep(0.3)

print_performance_report()
```

### Performance Report

The performance report includes:

- Total execution time
- Metric statistics (count, min, max, avg, median, total)
- Performance tips and recommendations

Example output:
```
Performance Summary (19:32:06)
==================================================
Total time: 9.65 seconds
--------------------------------------------------

pdf_rendering (seconds):
  Count: 3
  Min: 0.80
  Max: 0.80
  Avg: 0.80
  Median: 0.80
  Total: 2.40

...

==================================================
Performance Tips
==================================================
- PDF rendering is taking > 0.5 seconds per page. Check Poppler installation.
```

## Logging System (`logger.py`)

### Overview

The logging system provides a centralized logging framework with:

- Singleton logger instance
- File and console output
- Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Custom formatting
- Log rotation support

### Usage Examples

#### Basic Logging

```python
from logger import debug, info, warning, error, critical

debug("Debug message")
info("Information message")
warning("Warning message")
error("Error message")
critical("Critical message")
```

#### Log Configuration

```python
from logger import get_logger
import logging

logger = get_logger()
logger.setLevel(logging.DEBUG)

# Log to a file
logger.add_file_handler("app.log")
```

#### Using LogCallback

```python
from logger import create_log_callback

# Create a callback for integration with existing systems
log_callback = create_log_callback()
log_callback("Message from callback")

# Or with a specific level
error_callback = create_log_callback(logging.ERROR)
error_callback("Error message")
```

## Integration with Existing Code

### Updating Existing Configuration

Current code uses `config.py`. To migrate:

```python
# Before
from config import PdfJobConfig

job = PdfJobConfig(
    pdf_path="document.pdf",
    output_root="output",
    quality_mode=True
)

# After
from config_manager import create_job_config

job = create_job_config(
    "document.pdf",
    "output",
    ocr={"quality_mode": True}
)
```

### Adding Performance Monitoring

```python
from performance import timer, register_metrics
from scraper import run_pdf_job

register_metrics()

@timer("pdf_processing")
def process_pdf(config):
    return run_pdf_job(config)

# Usage
result = process_pdf(job_config)
```

### Enhanced Logging

```python
from logger import info, error
from scraper import run_pdf_job

try:
    info("Starting PDF processing")
    result = run_pdf_job(job_config, log_cb=info)
    info(f"Processing complete: {result}")
except Exception as e:
    error(f"Processing failed: {e}")
```

## Testing

All new modules include comprehensive tests:

```bash
pytest tests/test_utils.py -v  # Configuration and logging tests
pytest tests/test_ocr.py -v    # OCR engine tests
pytest tests/test_preprocess.py -v  # Image processing tests
pytest tests/test_scraper.py -v  # Scraper integration tests
```

## Requirements

The new modules require:

- Python 3.8+
- No additional dependencies beyond existing requirements (uses standard library for most features)

## Summary

These new modules significantly enhance the PDF scraper application by:

1. **Centralizing configuration** - Making it easier to manage and validate settings
2. **Providing performance insights** - Helping identify bottlenecks and optimize performance
3. **Enhancing logging** - Providing a more robust and flexible logging framework
4. **Improving maintainability** - Clear, type-safe APIs and comprehensive documentation

The modules are designed to work seamlessly with existing code while providing modern development practices and improved observability.
