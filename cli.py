import argparse
import os
from pathlib import Path

from config_manager import create_job_config
from scraper_old import run_pdf_job
from utils import validate_runtime_env, print_env_report

def main():
    """
    Parse CLI arguments and run OCR scraping jobs for PDF or EPUB files.
    
    This is the command-line entry point: it parses options, validates the runtime environment (or prints an environment report when requested), and either loads a job configuration from a JSON/YAML file or builds per-file job configurations. Supported options include output directory, OCR language and engine settings, quality vs. fast modes, custom tessdata directory, render persistence, and worker pool size. For each input file the function selects the appropriate scraper (PDF or EPUB), executes the job, and prints a concise success/failure status and output location.
    """
    parser = argparse.ArgumentParser(description="PDF/EPUB OCR scraper (EasyOCR + Tesseract)")
    parser.add_argument("files", nargs="*", help="PDF or EPUB file(s) to process")
    parser.add_argument("--output", "-o", default="output", help="Output root directory")
    parser.add_argument("--lang", default="ben", help="OCR language, e.g., ben, eng, ben+eng")
    parser.add_argument("--quality", action="store_true", help="Enable quality mode (slower, cleaner)")
    parser.add_argument("--fast", action="store_true", help="Prefer speed (disables quality mode)")
    parser.add_argument("--tessdata-dir", help="Custom tessdata directory for Tesseract")
    parser.add_argument("--persist-renders", action="store_true", help="Save rendered page images for debugging")
    parser.add_argument("--max-workers", type=int, help="Override worker pool size")
    parser.add_argument("--preset", choices=["default", "lite"], default="default", help="Preset: default (full) or lite (consumer-friendly, EasyOCR-only)")
    parser.add_argument("--check-env", action="store_true", help="Run environment diagnostics and exit")
    parser.add_argument("--config", "-c", help="Load configuration from JSON or YAML file")
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
    fast_mode = args.fast and not args.quality

    os.makedirs(args.output, exist_ok=True)

    # Load configuration from file if specified
    if args.config:
        from config_manager import get_config_manager
        config_manager = get_config_manager()
        config = config_manager.from_file(args.config)
        
        # Override config with command-line arguments if specified
        if args.files:
            config.input_path = args.files[0]
        
        if args.output:
            config.output_root = args.output
        
        if args.lang:
            config.ocr.ocr_lang = args.lang
        
        if args.quality:
            config.ocr.quality_mode = quality_mode
        
        if args.fast:
            config.ocr.fast_mode = fast_mode
        
        if args.tessdata_dir:
            config.ocr.tessdata_dir = args.tessdata_dir
        
        if args.persist_renders:
            config.render.persist_renders = args.persist_renders

        if args.preset:
            config.preset = args.preset
            config = config_manager.apply_preset(config)
        
        # Validate configuration
        errors = config_manager.validate_config(config)
        if errors:
            raise SystemExit("\n".join(errors))
        
        # Determine file type and run appropriate scraper
        file_ext = Path(config.input_path).suffix.lower()
        if file_ext == '.pdf':
            from scraper_old import run_pdf_job
            result = run_pdf_job(config, stop_event=None, log_cb=print)
        elif file_ext == '.epub':
            from epub_scraper import run_epub_job
            result = run_epub_job(config, stop_event=None, log_cb=print)
        else:
            print(f"Unsupported file type: {file_ext}")
            return
            
        status = "ok" if result.get("save_ok") else "failed"
        print(f"Done: {Path(config.input_path).name} [{status}] -> {result.get('output_dir')}")
    else:
        # Process each file
        for file_path in args.files:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"Skip missing file: {file_path}")
                continue
            job = create_job_config(
                input_path=str(file_path),
                output_root=args.output,
                preset=args.preset,
                use_ocr=True,
                ocr_method="easyocr",
                ocr_lang=args.lang,
                quality_mode=quality_mode,
                fast_mode=fast_mode,
                tessdata_dir=args.tessdata_dir,
                persist_renders=args.persist_renders,
                max_workers=args.max_workers,
            )
            print(f"Processing {file_path.name} ...")
            
            # Determine file type and run appropriate scraper
            file_ext = file_path.suffix.lower()
            if file_ext == '.pdf':
                from scraper_old import run_pdf_job
                result = run_pdf_job(job, stop_event=None, log_cb=print)
            elif file_ext == '.epub':
                from epub_scraper import run_epub_job
                result = run_epub_job(job, stop_event=None, log_cb=print)
            else:
                print(f"Unsupported file type: {file_ext}")
                continue
                
            status = "ok" if result.get("save_ok") else "failed"
            print(f"Done: {file_path.name} [{status}] -> {result.get('output_dir')}")


if __name__ == "__main__":
    main()