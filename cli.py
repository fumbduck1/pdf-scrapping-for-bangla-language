import argparse
import os
from pathlib import Path

from config import PdfJobConfig
from scraper import run_pdf_job
from utils import validate_runtime_env, print_env_report

def main():
    parser = argparse.ArgumentParser(description="PDF OCR scraper (EasyOCR + Tesseract)")
    parser.add_argument("pdf", nargs="+", help="PDF file(s) to process")
    parser.add_argument("--output", "-o", default="output", help="Output root directory")
    parser.add_argument("--lang", default="ben", help="OCR language, e.g., ben, eng, ben+eng")
    parser.add_argument("--quality", action="store_true", help="Enable quality mode (slower, cleaner)")
    parser.add_argument("--fast", action="store_true", help="Prefer speed (disables quality mode)")
    parser.add_argument("--tessdata-dir", help="Custom tessdata directory for Tesseract")
    parser.add_argument("--persist-renders", action="store_true", help="Save rendered page images for debugging")
    parser.add_argument("--max-workers", type=int, help="Override worker pool size")
    parser.add_argument("--check-env", action="store_true", help="Run environment diagnostics and exit")
    args = parser.parse_args()

    if args.check_env:
        print_env_report()
        return

    errors, warnings = validate_runtime_env()
    if errors:
        raise SystemExit("\n".join(errors))
    for w in warnings:
        print(f"Warning: {w}")

    quality_mode = args.quality and not args.fast

    os.makedirs(args.output, exist_ok=True)

    for pdf_path in args.pdf:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"Skip missing file: {pdf_path}")
            continue
        job = PdfJobConfig(
            pdf_path=str(pdf_path),
            output_root=args.output,
            use_ocr=True,
            ocr_method="easyocr",
            ocr_lang=args.lang,
            quality_mode=quality_mode,
            tessdata_dir=args.tessdata_dir,
            persist_renders=args.persist_renders,
            max_workers=args.max_workers,
        )
        print(f"Processing {pdf_path.name} ...")
        result = run_pdf_job(job, stop_event=None, log_cb=print)
        status = "ok" if result.get("save_ok") else "failed"
        print(f"Done: {pdf_path.name} [{status}] -> {result.get('output_dir')}")


if __name__ == "__main__":
    main()
