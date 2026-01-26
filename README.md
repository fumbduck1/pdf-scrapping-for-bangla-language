# PDF Scraper (EasyOCR + Tesseract)

Local OCR app with GUI (Tkinter) and CLI. Renders PDFs with Poppler, runs EasyOCR first, then Tesseract refinements. Defaults favor quality for Bangla; a speed toggle is available.

## Clone
```bash
git clone <repo-url>
cd pdf-scrapper-ongoing
```

## Prerequisites
- Python 3.10+
- Poppler utilities (for raster OCR). If absent, the app can still extract text layers but will skip image OCR.
- Tesseract OCR + language data
- pip packages: see requirements.txt (EasyOCR pulls PyTorch; CPU is fine)

### Install Poppler (only if you need raster OCR)
- **Windows**: Install Poppler for Windows; set `POPPLER_PATH` to the `bin` folder (e.g., `C:\poppler\Library\bin`).
- **macOS**: `brew install poppler`
- **Linux**: `sudo apt-get install poppler-utils` (or distro equivalent)

### Install Tesseract
- **Windows**: Install Tesseract 5.x; ensure `tesseract.exe` is on PATH or set `pytesseract.pytesseract.tesseract_cmd`.
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

### Python deps
```bash
pip install -r requirements.txt
```

If you want CPU-only PyTorch/EasyOCR on Windows/macOS/Linux to avoid large GPU wheels, install like:
```bash
pip install easyocr --no-deps
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --no-deps
```

## Running
### GUI
```bash
python main.py
```
Select PDFs and output folder. Toggles:
- **Quality mode**: on by default; slower, cleaner.
- **Speed mode**: disables quality mode and skips extra retries.
- **Save renders**: persist rendered pages for debugging.
- **Environment Check**: button in the GUI to verify Poppler/Tesseract/EasyOCR/torch setup.

### CLI
```bash
python cli.py input1.pdf [input2.pdf ...] -o out_dir --lang ben --quality  # quality mode
python cli.py input1.pdf -o out_dir --fast                                # speed-first
python cli.py sample.pdf --check-env                                      # environment diagnostics
```
Optional: `--tessdata-dir /path/to/tessdata_best`, `--persist-renders` to save page images, `--max-workers N` to cap parallelism.

## Smoke test
Provide a small sample PDF of your own and run:
```bash
python cli.py your_sample.pdf -o out_test --fast
```