"""Simple smoke test runner.

Usage:
    python smoke_test.py --pdf /path/to/sample.pdf --output smoke_out --fast
"""
import argparse
from pathlib import Path

from config import PdfJobConfig
from scraper import run_pdf_job
from utils import validate_runtime_env

def main():
    parser = argparse.ArgumentParser(description="Run a smoke test OCR on one PDF (provide your own small sample)")
    parser.add_argument("--pdf", required=True, help="Path to a small sample PDF")
    parser.add_argument("--output", default="smoke_out", help="Output root directory")
    parser.add_argument("--fast", action="store_true", help="Run in speed-first mode")
    parser.add_argument("--lang", default="ben", help="OCR language")
    args = parser.parse_args()

    errors, warnings = validate_runtime_env()
    if errors:
        raise SystemExit("\n".join(errors))
    for w in warnings:
        print(f"Warning: {w}")

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"Sample PDF not found: {pdf}")

    job = PdfJobConfig(
        pdf_path=str(pdf),
        output_root=args.output,
        ocr_lang=args.lang,
        quality_mode=not args.fast,
    )

    print(f"Running smoke test on {pdf} ...")
    result = run_pdf_job(job, stop_event=None, log_cb=print)
    status = "ok" if result.get("save_ok") else "failed"
    print(f"Smoke test {status}; output: {result.get('output_dir')}")


if __name__ == "__main__":
    main()
