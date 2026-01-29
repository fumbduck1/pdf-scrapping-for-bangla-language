import queue
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config_manager import create_job_config
from constants import QUALITY_MODE_DEFAULT, OCRSettings
from utils import validate_runtime_env, check_tesseract_ready, summarize_env
from deps import EASYOCR_AVAILABLE, TESSERACT_AVAILABLE
from scraper import run_pdf_job
class MinimalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Scraper - OCR")
        self.root.geometry("640x560")
        self.root.minsize(640, 560)
        self.root.resizable(True, True)
        self.style = ttk.Style()
        
        self.pdf_files = []
        self.output_dir = None
        self.is_processing = False
        self.stop_event = threading.Event()
        self.lang_var = tk.StringVar(value="ben")
        self.quality_var = tk.BooleanVar(value=QUALITY_MODE_DEFAULT)
        self.speed_var = tk.BooleanVar(value=False)
        self.persist_var = tk.BooleanVar(value=False)
        self.ui_scale = None
        self.event_queue = queue.Queue()
        
        self.setup_ui()
        self.apply_ui_scale()
        self.auto_size_window()
        self.root.after(100, self._drain_events)

    def apply_ui_scale(self, *_):
        """Adjust widget sizing/fonts based on screen-based scale (no slider)."""
        try:
            h = max(1, self.root.winfo_screenheight())
            scale = h / 1080.0
        except Exception:
            scale = 1.0
        scale = max(0.8, min(scale, 1.4))
        self.ui_scale = scale
        try:
            self.root.tk.call("tk", "scaling", scale)
        except Exception:
            pass
        self.auto_size_window()

    def auto_size_window(self):
        """Resize window to fit current content so sections don't get clipped."""
        try:
            self.root.update_idletasks()
            w = self.root.winfo_reqwidth()
            h = self.root.winfo_reqheight()
            self.root.minsize(w, h)
            self.root.geometry(f"{w}x{h}")
        except Exception:
            pass
    
    def setup_ui(self):
        """Setup minimal UI."""
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)
        self.title_label = ttk.Label(main, text="PDF Scraper", font=("Arial", 14, "bold"))
        self.title_label.pack(pady=(0, 5))
    
        gpu_status = "🖥 EasyOCR (GPU/CPU) + Tesseract refine"
        self.subtitle_label = ttk.Label(main, text=gpu_status, font=("Arial", 9), foreground="orange")
        self.subtitle_label.pack(pady=(0, 5))
        self.perf_label = ttk.Label(main, text="EasyOCR first; Tesseract refines weak text", font=("Arial", 9), foreground="gray")
        self.perf_label.pack(pady=(0, 10))

        step1 = ttk.LabelFrame(main, text="1. Select PDFs", padding="10")
        step1.pack(fill=tk.X, pady=5)
        ttk.Button(step1, text="Choose PDF Files", command=self.select_pdfs).pack(side=tk.LEFT, padx=5)
        self.pdf_label = ttk.Label(step1, text="No files selected", foreground="gray")
        self.pdf_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        step2 = ttk.LabelFrame(main, text="2. Select Output Folder", padding="10")
        step2.pack(fill=tk.X, pady=5)
        ttk.Button(step2, text="Choose Folder", command=self.select_output).pack(side=tk.LEFT, padx=5)
        self.output_label = ttk.Label(step2, text="No folder selected", foreground="gray")
        self.output_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        step3 = ttk.LabelFrame(main, text="3. OCR & Process", padding="10")
        step3.pack(fill=tk.X, pady=5)
        ocr_frame = ttk.Frame(step3)
        ocr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ocr_frame, text="OCR:").pack(side=tk.LEFT)
        ttk.Label(ocr_frame, text="EasyOCR + Tesseract refine", foreground="blue").pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(ocr_frame, text="Quality mode (slower, cleaner)", variable=self.quality_var, command=self._on_quality_toggle).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(ocr_frame, text="Speed mode (skip extra retries)", variable=self.speed_var, command=self._on_speed_toggle).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(ocr_frame, text="Save renders (debug)", variable=self.persist_var).pack(side=tk.LEFT, padx=8)
        lang_frame = ttk.Frame(step3)
        lang_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lang_frame, text="Language:").pack(side=tk.LEFT)
        self.lang_select = ttk.Combobox(lang_frame, textvariable=self.lang_var, state="readonly", width=12)
        self.lang_select['values'] = ("ben", "eng", "ben+eng")
        self.lang_select.pack(side=tk.LEFT, padx=5)
        self.lang_select.current(0)

        btn_frame = ttk.Frame(step3)
        btn_frame.pack(fill=tk.X, pady=5)
        self.process_btn = ttk.Button(btn_frame, text="Start Processing", command=self.start_processing)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.env_btn = ttk.Button(btn_frame, text="Environment Check", command=self.run_env_check)
        self.env_btn.pack(side=tk.LEFT, padx=5)
        
        progress_frame = ttk.LabelFrame(main, text="Progress", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        self.status_var = tk.StringVar(value="Idle")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var, font=("Arial", 9))
        self.status_label.pack(fill=tk.X)

    def default_ocr_missing(self):
        """Return error message if selected OCR engine is missing."""
        ok, msg = check_tesseract_ready()
        if ok:
            return None
        return msg
    
    def select_pdfs(self):
        """Select PDF files."""
        files = filedialog.askopenfilenames(title="Select PDF files", filetypes=[("PDF", "*.pdf")])
        if files:
            self.pdf_files = list(files)
            self.pdf_label.config(text=f"{len(files)} file(s) selected", foreground="black")

    def select_output(self):
        """Select output directory."""
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir = folder
            self.output_label.config(text=folder, foreground="black")

    def log(self, message):
        """Log message."""
        try:
            self.event_queue.put({"kind": "status", "message": message})
        except Exception:
            pass

    def _emit_progress(self, value):
        try:
            self.event_queue.put({"kind": "progress", "value": value})
        except Exception:
            pass

    def _emit_done(self, completed=True):
        try:
            self.event_queue.put({
                "kind": "done",
                "completed": completed,
                "output_dir": self.output_dir,
                "message": "Processing complete!" if completed else "Stopped",
            })
        except Exception:
            pass

    def _drain_events(self):
        try:
            while True:
                event = self.event_queue.get_nowait()
                kind = event.get("kind")
                if kind == "status":
                    self.status_var.set(event.get("message", ""))
                elif kind == "progress":
                    self.progress_var.set(event.get("value", 0))
                elif kind == "done":
                    self.process_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.is_processing = False
                    if event.get("completed"):
                        self.progress_var.set(100)
                    if event.get("output_dir") and event.get("completed"):
                        messagebox.showinfo("Done", f"Processing finished!\n\n{event.get('output_dir')}")
                    self.status_var.set(event.get("message", ""))
                elif kind == "stopped":
                    self.status_var.set(event.get("message", ""))
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _on_speed_toggle(self):
        if self.speed_var.get():
            self.quality_var.set(False)

    def _on_quality_toggle(self):
        if self.quality_var.get():
            self.speed_var.set(False)

    def run_env_check(self):
        """Run environment diagnostics and show a dialog."""
        try:
            info, warnings, errors = summarize_env()
        except Exception as exc:
            messagebox.showerror("Environment Check", f"Environment check failed: {exc}")
            return

        lines = []
        lines.extend(f"- {item}" for item in info)
        if warnings:
            lines.append("\nWarnings:")
            lines.extend(f"- {w}" for w in warnings)
        if errors:
            lines.append("\nErrors:")
            lines.extend(f"- {e}" for e in errors)
        msg = "\n".join(lines)

        if errors:
            messagebox.showerror("Environment Check", msg)
        elif warnings:
            messagebox.showwarning("Environment Check", msg)
        else:
            messagebox.showinfo("Environment Check", msg)

        self.log("Environment check completed")
        if warnings:
            self.log(f"Warnings: {len(warnings)}")
        if errors:
            self.log(f"Errors: {len(errors)}")
    
    def start_processing(self):
        """Start processing."""
        if not self.pdf_files:
            messagebox.showerror("Error", "Select PDF files first")
            return
        if not self.output_dir:
            messagebox.showerror("Error", "Select output folder first")
            return

        errors, warnings = validate_runtime_env()
        if errors:
            messagebox.showerror("Error", "\n".join(errors))
            self.log(errors[0])
            return
        for w in warnings:
            # Non-blocking warnings (e.g., Tesseract missing for refinement)
            messagebox.showwarning("Warning", w)
            self.log(w)
        ok, msg = check_tesseract_ready()
        if ok:
            self.log("Engine: EasyOCR primary; Tesseract refines weak text")
            self.log(msg)
        else:
            self.log("Engine: EasyOCR primary; Tesseract unavailable, skipping refinement")
        self.log(f"EasyOCR available: {EASYOCR_AVAILABLE}")
        self.log(f"Python: {sys.executable}")

        self.is_processing = True
        self.stop_event.clear()
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.log("Starting...")
        
        thread = threading.Thread(target=self.process_batch)
        thread.daemon = True
        thread.start()
    
    def process_batch(self):
        """Process PDFs in background."""
        quality_choice = self.quality_var.get()
        if self.speed_var.get():
            quality_choice = False

        settings = OCRSettings(
            use_ocr=True,
            ocr_method='easyocr',
            ocr_lang=self.lang_var.get() or 'ben',
            quality_mode=quality_choice,
        )
        self.log(f"PDFs selected: {len(self.pdf_files)}")
        self.log(f"Output dir: {self.output_dir}")
        self.log(f"Language: {self.lang_var.get()}")

        total = len(self.pdf_files)
        if total == 0:
            self.log("No PDFs to process; aborting.")
            return

        for idx, pdf_file in enumerate(self.pdf_files):
            if not self.is_processing or (self.stop_event and self.stop_event.is_set()):
                break
            self.log(f"[{idx + 1}/{total}] {Path(pdf_file).name}")
            job_config = create_job_config(
                pdf_path=pdf_file,
                output_root=self.output_dir,
                use_ocr=settings.use_ocr,
                ocr_method=settings.ocr_method,
                ocr_lang=settings.ocr_lang,
                quality_mode=settings.quality_mode,
                fast_mode=self.speed_var.get(),
                persist_renders=self.persist_var.get(),
            )
            result = run_pdf_job(job_config, self.stop_event, self.log)

            if result.get("save_ok"):
                stats = result.get("stats", {})
                total_pages = stats.get('total_pages', 0)
                pages_with_text = stats.get('pages_with_ocr_text', 0)
                total_chars = stats.get('total_ocr_characters', 0)
                self.log("Done!" if result.get("scrape_ok") else "Partial data saved")
                self.log(f"Pages processed: {pages_with_text}/{total_pages}")
                self.log(f"OCR Characters: {total_chars}")
            else:
                self.log("Save failed")

            self._emit_progress((idx + 1) / total * 100)

        self._emit_done(completed=not (self.stop_event and self.stop_event.is_set()))

    def stop_processing(self):
        """Stop processing."""
        self.is_processing = False
        if self.stop_event:
            self.stop_event.set()
        self.log("⏹  Stop requested; finishing current page...")
        self.event_queue.put({"kind": "stopped", "message": "Stop requested; finishing current page..."})
