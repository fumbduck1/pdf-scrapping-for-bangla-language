# Scraper.py Restructuring Plan

## Current Problem
The `scraper.py` file is a monolithic 74,000+ character file containing all PDF scraping logic, making it hard to maintain, test, and extend.

## Solution: Modular Architecture

### New Directory Structure
```
scraper/
├── __init__.py          # Main entry point exposing all public APIs
├── models.py            # Data classes (PageResult, JobResult)
├── pdf_renderer.py      # PdfRenderer class for PDF rendering
├── ocr_pipeline.py      # OcrPipeline class for OCR orchestration
├── pdf_scraper.py       # PDFScraper class for main scraping logic
└── utils.py             # Utility functions (_sentence_chunks, etc.)
```

### Restructuring Steps

1. **Create the scraper directory**
2. **Move data classes to models.py**
3. **Extract PdfRenderer to pdf_renderer.py**
4. **Extract OcrPipeline to ocr_pipeline.py**  
5. **Extract PDFScraper to pdf_scraper.py**
6. **Extract utility functions to utils.py**
7. **Create __init__.py to expose public APIs**
8. **Update all imports in dependent files**
9. **Test the restructuring**

### Key Changes

#### models.py
- Contains `@dataclass` PageResult
- Contains `TypedDict` JobResult
- Dependencies: typing, dataclasses

#### pdf_renderer.py
- Contains `class PdfRenderer`
- Handles PDF opening, rendering, and caching
- Dependencies: pathlib, PIL, deps, constants, performance, utils

#### ocr_pipeline.py
- Contains `class OcrPipeline`
- Orchestrates EasyOCR and Tesseract strategies
- Dependencies: ocr_easyocr, ocr_tesseract, preprocess, utils, constants, performance

#### pdf_scraper.py
- Contains `class PDFScraper`
- Main scraping orchestrator
- Contains `run_pdf_job()` function
- Dependencies: All other scraper modules, config_manager, rs_correction, performance

#### utils.py
- Contains `_sentence_chunks()` helper
- Can include additional scraper-specific utilities

### Benefits of Restructuring
1. **Improved maintainability** - Each module focuses on a single responsibility
2. **Easier testing** - Individual components can be tested in isolation
3. **Better reusability** - Components can be reused across different scraping implementations
4. **Clearer architecture** - Separation of concerns makes the system easier to understand
5. **Faster development** - Parallel development on separate components

### Testing Strategy
- Run existing unit tests to ensure functionality remains intact
- Test each module independently
- Verify integration between modules
- Test edge cases and error handling
