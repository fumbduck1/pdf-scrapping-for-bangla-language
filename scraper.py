import logging
import re
import os
import sys
import threading
import shutil
from pathlib import Path
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = 500_000_000

from constants import (
    DEFAULT_ZOOM,
    FAST_MODE,
    FAST_CONFIDENCE_SKIP,
    TEXT_LAYER_FIRST,
    TEXT_LAYER_LANG_MIN_RATIO,
    TEXT_LAYER_MIN_BEN_CHARS,
    PDF_BYTES_CACHE_MB,
    WATERMARK_FLATTEN,
    WATERMARK_CLIP_THRESHOLD,
    WATERMARK_RETRY_CONF,
    HIGH_DPI_RETRY_CONF,
    HIGH_DPI_ZOOM,
    AUTO_APPEND_ENG_FOR_BEN,
    QUALITY_MODE_DEFAULT,
    SEGMENT_RETRY_CONF,
    THIRD_PASS_SCALE,
    EASYOCR_FALLBACK_CONF,
    EASYOCR_PRIMARY_CONF,
    TESSERACT_REFINE_MIN_CHARS,
)
from config import PdfJobConfig
import preprocess as preproc
import ocr_easyocr as ocr_e
import ocr_tesseract as ocr_t
from deps import (
    _lazy_import_pdf2image,
    _lazy_import_pypdf,
    EASYOCR_AVAILABLE,
    pytesseract,
    TESSERACT_AVAILABLE,
    detect_poppler_path,
    detect_torch_device,
)
from utils import _sanitize_tessdata_prefix, _split_langs, validate_runtime_env, resolve_tesseract_cmd


class PdfRenderer:
    """Rendering helper to manage PDF handles and page rasterization."""

    def __init__(self, pdf_path, output_dir, pdf_bytes_cache_mb, poppler_path, log, log_error, persist_renders=False):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.renders_dir = None
        self._pdf_file_handle = None
        self._pdf_bytes = None
        self.doc = None
        self.pdf_bytes_cache_mb = pdf_bytes_cache_mb
        self.poppler_path = poppler_path
        self.log = log
        self.log_error = log_error
        self.persist_renders = persist_renders
        self.setup_directories()

    def setup_directories(self):
        if not self.persist_renders:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.renders_dir = os.path.join(self.output_dir, 'renders')
            os.makedirs(self.renders_dir, exist_ok=True)
        except Exception:
            pass

    def open_pdf(self):
        """Open the PDF into memory or file handle depending on size."""
        PdfReader = _lazy_import_pypdf()
        if not PdfReader:
            self._log_missing("pypdf not installed. Install with: pip install pypdf", err_key="pypdf")
            return False
        try:
            size_bytes = os.path.getsize(self.pdf_path)
            size_mb = round(size_bytes / (1024 * 1024), 2)
            if size_mb <= self.pdf_bytes_cache_mb:
                self._pdf_bytes = Path(self.pdf_path).read_bytes()
                self.doc = PdfReader(BytesIO(self._pdf_bytes))
                self._pdf_file_handle = None
            else:
                self._pdf_file_handle = open(self.pdf_path, "rb")
                self.doc = PdfReader(self._pdf_file_handle)
            return True
        except Exception as e:
            self._log_error(f"Open failed: {e}")
            self.close()
            return False

    def render_page(self, page_num, zoom, fmt="png"):
        """Render a single page and return a PIL image; optionally persist to disk."""
        convert_from_path, convert_from_bytes = _lazy_import_pdf2image()
        if not convert_from_path or not convert_from_bytes:
            self._log_missing("pdf2image not installed. Install with: pip install pdf2image", err_key="pdf2image")
            return None
        dpi = int((zoom or DEFAULT_ZOOM) * 72)
        dpi = max(dpi, 72)
        try:
            if self._pdf_bytes:
                images = convert_from_bytes(
                    self._pdf_bytes,
                    dpi=dpi,
                    first_page=page_num + 1,
                    last_page=page_num + 1,
                    fmt=fmt,
                    poppler_path=self.poppler_path or None,
                )
            else:
                images = convert_from_path(
                    self.pdf_path,
                    dpi=dpi,
                    first_page=page_num + 1,
                    last_page=page_num + 1,
                    fmt=fmt,
                    poppler_path=self.poppler_path or None,
                )
            if not images:
                raise RuntimeError("No image returned from pdf2image")

            render_img = images[0]
            render_path = None
            if self.persist_renders:
                render_filename = f"page_{page_num:03d}_render.{fmt}"
                render_path = os.path.join(self.renders_dir, render_filename)
                try:
                    render_img.save(render_path, fmt.upper())
                except Exception:
                    render_path = None

            return render_img, render_path
        except Exception as e:
            self._log_error(f"Render error (page {page_num + 1}): {e}")
            return None

    def cleanup_renders(self):
        if not self.persist_renders:
            return
        try:
            if self.renders_dir and os.path.isdir(self.renders_dir):
                shutil.rmtree(self.renders_dir, ignore_errors=True)
        except Exception:
            pass

    def close(self):
        try:
            if self._pdf_file_handle:
                self._pdf_file_handle.close()
        except Exception:
            pass
        self._pdf_file_handle = None
        self._pdf_bytes = None
        self.doc = None

    def _log_missing(self, msg, err_key=None):
        try:
            if self.log:
                self.log(msg)
            if self.log_error:
                self.log_error(f"Missing dependency: {err_key or msg}")
        except Exception:
            pass

    def _log_error(self, msg):
        try:
            if self.log:
                self.log(msg)
            if self.log_error:
                self.log_error(msg)
        except Exception:
            pass


def _sentence_chunks(text: str):
    """Split text into rough sentences/clauses using Bangla/English punctuation."""
    if not text:
        return []
    # Normalize whitespace
    cleaned = '\n'.join(' '.join(line.split()) for line in text.splitlines())
    # Split on Bangla danda or common sentence enders
    parts = re.split(r"(?<=[।!?])\s+", cleaned)
    return [p.strip() for p in parts if p and p.strip()]


class OcrPipeline:
    """OCR orchestrator that encapsulates EasyOCR/Tesseract strategies."""

    def __init__(self, ocr_method, ocr_lang, quality_mode, fast_mode, fast_conf_skip, tessdata_dir, log, log_error):
        self.ocr_method = ocr_method
        self.ocr_method_effective = ocr_method
        self.ocr_lang = ocr_lang
        self.quality_mode = quality_mode
        self.fast_mode = fast_mode
        self.fast_conf_skip = fast_conf_skip
        self.tessdata_dir = _sanitize_tessdata_prefix(tessdata_dir)
        self.log = log
        self.log_error = log_error
        self._force_easyocr_cpu = str(os.environ.get("EASYOCR_FORCE_CPU", "")).lower() in ("1", "true", "yes", "on")
        self._torch_device = detect_torch_device()
        self._easyocr_gpu = (
            not self._force_easyocr_cpu
            and self._torch_device.get("installed")
            and self._torch_device.get("backend") == "cuda"
        )
        self._easyocr_reader = None
        self._easyocr_lock = threading.Lock()
        self._engine_logged = False
        self._device_logged = False
        self._ensure_tesseract_cmd()

    # --- shared helpers ---
    def _normalize_text(self, text):
        if not text:
            return ""
        zero_width = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for zw in zero_width:
            text = text.replace(zw, '')
        return '\n'.join(' '.join(line.split()) for line in text.splitlines())

    def _maybe_split_columns(self, image):
        return preproc.maybe_split_columns(image, fast_mode=self.fast_mode)

    def _flatten_background(self, image, clip=WATERMARK_CLIP_THRESHOLD):
        return preproc.flatten_background(image, clip=clip)

    def _choose_psm(self, image, segment_count):
        return preproc.choose_psm(image, segment_count)

    def _score_result(self, res):
        return ocr_t.score_result(res)

    def _map_easyocr_langs(self):
        return ocr_e.map_easyocr_langs(self.ocr_lang)

    def _log_easyocr_device_once(self):
        if self._device_logged:
            return
        self._device_logged = True
        if self._force_easyocr_cpu:
            if self.log:
                self.log("EasyOCR: CPU forced via EASYOCR_FORCE_CPU")
            return
        if not self._torch_device.get("installed"):
            if self.log:
                self.log("EasyOCR: torch not installed; CPU mode")
            return
        backend = self._torch_device.get("backend")
        reason = self._torch_device.get("reason")
        if backend == "cuda" and self._easyocr_gpu:
            if self.log:
                self.log(f"EasyOCR: using CUDA ({reason})")
        elif backend == "mps":
            if self.log:
                self.log("EasyOCR: MPS detected; running CPU because EasyOCR expects CUDA")
        else:
            if self.log:
                self.log(f"EasyOCR: running on CPU ({reason})")

    def _is_noise_fragment(self, text: str, confidence: float) -> bool:
        if not text:
            return True
        if not self.ocr_lang.startswith('ben'):
            return False
        tokens = re.sub(r"[\s\W_]+", "", text)
        if not tokens:
            return True
        ascii_letters = sum(1 for ch in tokens if ch.isascii())
        bengali_letters = sum(1 for ch in tokens if '\u0980' <= ch <= '\u09FF')
        length = len(tokens)
        if length <= 4 and confidence < 0.96:
            return True
        if bengali_letters == 0 and ascii_letters >= 3 and confidence < 0.93:
            return True
        ascii_ratio = ascii_letters / max(length, 1)
        if ascii_ratio > 0.65 and confidence < 0.9:
            return True
        return False

    def _load_image(self, image_or_path):
        if isinstance(image_or_path, Image.Image):
            return image_or_path
        return Image.open(image_or_path)

    def _ensure_tesseract_cmd(self):
        try:
            cmd_obj = getattr(pytesseract, 'pytesseract', None)
            current = getattr(cmd_obj, 'tesseract_cmd', None) if cmd_obj else None
            if current and Path(current).is_file():
                return
            default_paths = [
                Path(r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"),
                Path(r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"),
            ]
            for candidate in default_paths:
                if candidate.is_file() and cmd_obj:
                    cmd_obj.tesseract_cmd = str(candidate)
                    break
        except Exception:
            pass

    def _verify_language_file(self):
        if not self.tessdata_dir:
            return
        langs = _split_langs(self.ocr_lang) or ["ben"]
        missing = []
        for code in langs:
            lang_file = Path(self.tessdata_dir) / f"{code}.traineddata"
            if not lang_file.is_file():
                missing.append(str(lang_file))
        if missing:
            if self.log_error:
                self.log_error(f"Missing language file(s): {', '.join(missing)}")
            return
        os.environ["TESSDATA_PREFIX"] = _sanitize_tessdata_prefix(self.tessdata_dir) or self.tessdata_dir

    def _get_easyocr_reader(self):
        reader = ocr_e.get_easyocr_reader(
            self.ocr_lang,
            gpu=True,
            lock=self._easyocr_lock,
            existing_reader=self._easyocr_reader,
            log=self.log,
            log_error=self.log_error,
        )
        if reader is not None:
            self._easyocr_reader = reader
        return reader

    def _run_easyocr_pass(self, image):
        reader = self._get_easyocr_reader()
        res = ocr_e.run_easyocr_pass(image, lock=self._easyocr_lock, reader=reader, log=self.log, log_error=self.log_error)
        if res and 'text' in res:
            res['text'] = self._normalize_text(res.get('text', ''))
        return res

    def _run_tesseract_pass(self, image, extra_config=None, extra_dilate=False, psm=None):
        return ocr_t.run_tesseract_pass(
            image,
            ocr_lang=self.ocr_lang,
            quality_mode=self.quality_mode,
            psm=psm,
            extra_config=extra_config,
            extra_dilate=extra_dilate,
            log=self.log,
            log_error=self.log_error,
        )

    def _tesseract_best_for_segment(self, seg, alt_seg, psm_for_seg):
        return ocr_t.tesseract_best_for_segment(
            seg,
            alt_seg,
            psm_for_seg,
            ocr_lang=self.ocr_lang,
            quality_mode=self.quality_mode,
            fast_conf_skip=self.fast_conf_skip,
            log=self.log,
            log_error=self.log_error,
        )

    def preprocess_image_for_ocr(self, image_path_or_image):
        return preproc.preprocess_image_for_ocr(
            image_path_or_image,
            ocr_lang=self.ocr_lang,
            fast_mode=self.fast_mode,
            quality_mode=self.quality_mode,
            log_fn=self.log,
        )

    def extract_text_with_easyocr_primary(self, image_path_or_image):
        if not EASYOCR_AVAILABLE:
            if self.log:
                self.log("EasyOCR not installed; falling back to Tesseract")
            return self.extract_text_with_tesseract(image_path_or_image)
        try:
            raw_img = self._load_image(image_path_or_image)
            cropped_img = preproc.crop_header_footer(raw_img)
            preprocessed_img = self.preprocess_image_for_ocr(cropped_img)

            base_segments_preview = self._maybe_split_columns(preprocessed_img)
            psm_for_seg = self._choose_psm(preprocessed_img, len(base_segments_preview))
            flattened_img = self._flatten_background(preprocessed_img) if WATERMARK_FLATTEN else None
            flat_segments = self._maybe_split_columns(flattened_img) if flattened_img is not None else None
            segments = [(seg, idx) for idx, seg in enumerate(base_segments_preview)]

            combined_text = []
            total_conf_weighted = 0.0
            total_fragments = 0
            refined = False

            if self.ocr_lang.startswith('ben'):
                try:
                    preprocessed_img = preprocessed_img.filter(ImageFilter.MaxFilter(3))
                except Exception:
                    pass
                if flattened_img is not None:
                    try:
                        flattened_img = flattened_img.filter(ImageFilter.MaxFilter(3))
                    except Exception:
                        pass

            for seg, idx in segments:
                alt_seg = flat_segments[idx] if flat_segments and idx < len(flat_segments) else None
                easy_res = self._run_easyocr_pass(seg)
                alt_easy = self._run_easyocr_pass(alt_seg) if alt_seg is not None else None
                best = easy_res
                if self._score_result(alt_easy) > self._score_result(best):
                    best = alt_easy

                needs_refine = False
                if TESSERACT_AVAILABLE:
                    if not best:
                        needs_refine = True
                    else:
                        text_len = len(best.get('text', '').strip())
                        conf = best.get('avg_confidence', 0)
                        if conf < EASYOCR_PRIMARY_CONF or text_len < TESSERACT_REFINE_MIN_CHARS:
                            needs_refine = True

                if needs_refine:
                    self._verify_language_file()
                    tess_best = self._tesseract_best_for_segment(seg, alt_seg, psm_for_seg)
                    if self._score_result(tess_best) > self._score_result(best):
                        best = tess_best
                        refined = True

                if best:
                    conf_val = best.get('avg_confidence') or 0
                    text_val = best.get('text', '')
                    if self._is_noise_fragment(text_val, conf_val):
                        continue
                    combined_text.append(text_val)
                    total_conf_weighted += conf_val * max(best.get('fragments', 1), 1)
                    total_fragments += max(best.get('fragments', 1), 1)

            if not combined_text:
                return None

            normalized = self._normalize_text('\n\n'.join(combined_text))
            avg_conf = (total_conf_weighted / total_fragments) if total_fragments else 0.0
            self.ocr_method_effective = "easyocr+tesseract" if refined and TESSERACT_AVAILABLE else "easyocr"
            return {
                'text': normalized,
                'avg_confidence': round(avg_conf, 4),
                'fragments': total_fragments
            }
        except Exception as e:
            if self.log:
                self.log(f"EasyOCR-first error: {str(e)}")
            if self.log_error:
                self.log_error(f"EasyOCR-first error: {e}")
            return None

    def extract_text_with_tesseract(self, image_path_or_image):
        if not TESSERACT_AVAILABLE:
            if self.log:
                self.log("Tesseract not installed. Install tesseract-ocr and pytesseract")
            return None
        self._verify_language_file()
        try:
            raw_img = self._load_image(image_path_or_image)
            cropped_img = preproc.crop_header_footer(raw_img)
            preprocessed_img = self.preprocess_image_for_ocr(cropped_img)

            base_segments_preview = self._maybe_split_columns(preprocessed_img)
            psm_for_seg = self._choose_psm(preprocessed_img, len(base_segments_preview))
            flattened_img = self._flatten_background(preprocessed_img) if WATERMARK_FLATTEN else None

            if self.ocr_lang.startswith('ben'):
                try:
                    preprocessed_img = preprocessed_img.filter(ImageFilter.MaxFilter(3))
                except Exception:
                    pass
                if flattened_img is not None:
                    try:
                        flattened_img = flattened_img.filter(ImageFilter.MaxFilter(3))
                    except Exception:
                        pass

            base_segments = base_segments_preview
            flat_segments = self._maybe_split_columns(flattened_img) if flattened_img is not None else None
            segments = [(seg, idx) for idx, seg in enumerate(base_segments)]

            combined_text = []
            total_conf_weighted = 0.0
            total_fragments = 0

            for seg, idx in segments:
                alt_seg = flat_segments[idx] if flat_segments and idx < len(flat_segments) else None

                pass_a = self._run_tesseract_pass(seg, extra_config=None, extra_dilate=False, psm=psm_for_seg)
                pass_b = None
                if not pass_a or pass_a.get('avg_confidence', 0) < self.fast_conf_skip:
                    pass_b = self._run_tesseract_pass(seg, extra_config=["-c lstm_choice_mode=2"], extra_dilate=True, psm=psm_for_seg)

                best = pass_a if self._score_result(pass_a) >= self._score_result(pass_b) else pass_b

                if alt_seg is not None and (not best or best.get('avg_confidence', 0) < WATERMARK_RETRY_CONF):
                    alt_a = self._run_tesseract_pass(alt_seg, extra_config=None, extra_dilate=False, psm=psm_for_seg)
                    alt_b = None
                    if not alt_a or alt_a.get('avg_confidence', 0) < self.fast_conf_skip:
                        alt_b = self._run_tesseract_pass(alt_seg, extra_config=["-c lstm_choice_mode=2"], extra_dilate=True, psm=psm_for_seg)
                    alt_best = alt_a if self._score_result(alt_a) >= self._score_result(alt_b) else alt_b
                    if self._score_result(alt_best) > self._score_result(best):
                        best = alt_best

                if best is None or best.get('avg_confidence', 0) < SEGMENT_RETRY_CONF or best.get('fragments', 0) < 2:
                    retry_seg = preproc.upscale_for_retry(seg, scale=THIRD_PASS_SCALE)
                    pass_c = self._run_tesseract_pass(
                        retry_seg,
                        extra_config=["-c lstm_choice_mode=2"],
                        extra_dilate=True,
                        psm=psm_for_seg,
                    )
                    if self._score_result(pass_c) > self._score_result(best):
                        best = pass_c

                if best is None or best.get('avg_confidence', 0) < EASYOCR_FALLBACK_CONF:
                    easy_res = self._run_easyocr_pass(seg)
                    if self._score_result(easy_res) > self._score_result(best):
                        best = easy_res

                if best:
                    conf_val = best.get('avg_confidence') or 0
                    text_val = best.get('text', '')
                    if self._is_noise_fragment(text_val, conf_val):
                        continue
                    combined_text.append(text_val)
                    total_conf_weighted += conf_val * max(best.get('fragments', 1), 1)
                    total_fragments += max(best.get('fragments', 1), 1)

            if not combined_text:
                return None

            normalized = self._normalize_text('\n\n'.join(combined_text))
            avg_conf = (total_conf_weighted / total_fragments) if total_fragments else 0.0
            return {
                'text': normalized,
                'avg_confidence': round(avg_conf, 4),
                'fragments': total_fragments
            }
        except Exception as e:
            if self.log:
                self.log(f"Tesseract OCR error: {str(e)}")
            if self.log_error:
                self.log_error(f"Tesseract OCR error: {e}")
            return None

    def extract_text_with_ocr(self, image_path_or_image):
        if not self._engine_logged:
            if self.ocr_method == 'tesseract' or not EASYOCR_AVAILABLE:
                if self.log:
                    self.log("Engine: Tesseract (EasyOCR unavailable or not selected)")
            else:
                if TESSERACT_AVAILABLE:
                    if self.log:
                        self.log("Engine: EasyOCR primary; Tesseract will refine weak segments")
                else:
                    if self.log:
                        self.log("Engine: EasyOCR primary; Tesseract unavailable, refinement skipped")
            self._engine_logged = True

        if self.ocr_method == 'tesseract':
            self.ocr_method_effective = 'tesseract'
            return self.extract_text_with_tesseract(image_path_or_image)

        primary = self.extract_text_with_easyocr_primary(image_path_or_image)
        if primary is not None:
            return primary

        if TESSERACT_AVAILABLE:
            self.ocr_method_effective = 'tesseract'
            return self.extract_text_with_tesseract(image_path_or_image)

        if self.log:
            self.log("No OCR engine available")
        return None


class PDFScraper:
    def __init__(
        self,
        pdf_path,
        output_dir,
        use_ocr=True,
        ocr_method='easyocr',
        ocr_lang='ben',
        progress_callback=None,
        tessdata_dir=None,
        stop_event=None,
        quality_mode=QUALITY_MODE_DEFAULT,
        persist_renders=False,
        max_workers=None,
    ):
        """Initialize PDF scraper."""
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.use_ocr = use_ocr
        self.ocr_method = ocr_method
        self.persist_renders = bool(persist_renders)
        self.max_workers_override = max_workers
        self.user_lang = (ocr_lang or "ben").strip()
        self.ocr_lang = self.user_lang
        if AUTO_APPEND_ENG_FOR_BEN:
            langs = _split_langs(self.ocr_lang) or []
            if "ben" in langs and "eng" not in langs:
                langs.append("eng")
                self.ocr_lang = "+".join(langs)
        self.quality_mode = bool(quality_mode)
        self.fast_mode = False if self.quality_mode else FAST_MODE
        self.fast_conf_skip = 0.75 if self.quality_mode else FAST_CONFIDENCE_SKIP
        self.line_merge_y_tolerance = 12
        self.page_render_zoom = DEFAULT_ZOOM
        self.high_dpi_retry_conf = HIGH_DPI_RETRY_CONF
        self.high_dpi_zoom = HIGH_DPI_ZOOM
        self.progress_callback = progress_callback
        self.stop_event = stop_event
        self.tessdata_dir = _sanitize_tessdata_prefix(tessdata_dir) if tessdata_dir else None
        self.poppler_path = os.environ.get("POPPLER_PATH") or detect_poppler_path()
        self.results = {
            'metadata': {},
            'pages': {},
            'statistics': {},
            'extraction_log': []
        }
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(f"pdfscraper.{Path(self.pdf_path).stem}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            try:
                fh = logging.FileHandler(os.path.join(self.output_dir, "extraction.log"), encoding="utf-8")
                fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
                self.logger.addHandler(fh)
            except Exception:
                pass
        self.renderer = PdfRenderer(
            pdf_path=self.pdf_path,
            output_dir=self.output_dir,
            pdf_bytes_cache_mb=PDF_BYTES_CACHE_MB,
            poppler_path=self.poppler_path,
            log=self.log,
            log_error=self.log_error,
            persist_renders=self.persist_renders,
        )
        self.ocr = OcrPipeline(
            ocr_method=self.ocr_method,
            ocr_lang=self.ocr_lang,
            quality_mode=self.quality_mode,
            fast_mode=self.fast_mode,
            fast_conf_skip=self.fast_conf_skip,
            tessdata_dir=self.tessdata_dir,
            log=self.log,
            log_error=self.log_error,
        )
    
    def setup_directories(self):
        """Create all necessary directories upfront."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.renders_dir = os.path.join(self.output_dir, 'renders')
            os.makedirs(self.renders_dir, exist_ok=True)
        except Exception:
            pass

    def _resolve_tessdata_dir(self, provided):
        """Pick a tessdata folder, preferring explicit/env/known install paths."""
        if provided and Path(provided).is_dir():
            return str(Path(provided))

        env_dir = os.environ.get("TESSDATA_PREFIX")
        if env_dir and Path(env_dir).is_dir():
            return str(Path(env_dir))

        try:
            cmd = getattr(pytesseract, 'pytesseract', None)
            if cmd and getattr(cmd, 'tesseract_cmd', None):
                root = Path(cmd.tesseract_cmd).parent
                candidates = [root / "tessdata_best", root / "tessdata"]
                for cand in candidates:
                    if cand.is_dir():
                        return str(cand)
        except Exception:
            pass

        win_candidates = [
            Path(r"C:\\Program Files\\Tesseract-OCR\\tessdata"),
            Path(r"C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"),
        ]
        for cand in win_candidates:
            if cand.is_dir():
                return str(cand)

        return None

    def _worker_count(self):
        """Optimized worker pool size for parallel OCR processing."""
        try:
            if getattr(self, "max_workers_override", None):
                return max(1, int(self.max_workers_override))
            cores = os.cpu_count() or 2
            langs = _split_langs(self.ocr_lang) if hasattr(self, 'ocr_lang') else []
            has_ben = "ben" in langs
            if has_ben:
                max_workers = max(2, min(cores - 1, 4))
            else:
                max_workers = max(2, min(cores, 8))
            if self.quality_mode if hasattr(self, 'quality_mode') else False:
                max_workers = max(2, max_workers // 2)
            return max_workers
        except Exception:
            return 2

    def _ensure_tesseract_cmd(self):
        """Make sure pytesseract points to a real tesseract binary."""
        try:
            resolved = resolve_tesseract_cmd()
            if resolved and self.log:
                self.log(f"tesseract_cmd resolved to {resolved}")
        except Exception:
            pass

    def _map_easyocr_langs(self):
        """Convert Tesseract-style lang string to EasyOCR list."""
        return ocr_e.map_easyocr_langs(getattr(self, 'ocr_lang', '') or '')

    def _get_easyocr_reader(self):
        """Lazy-init a shared EasyOCR reader; fallback to None on failure."""
        self._log_easyocr_device_once()
        gpu_flag = bool(self._easyocr_gpu)
        reader = ocr_e.get_easyocr_reader(
            self.ocr_lang,
            gpu=gpu_flag,
            lock=self._easyocr_lock,
            existing_reader=self._easyocr_reader,
            log=self.log,
            log_error=self.log_error,
        )
        if reader is not None:
            self._easyocr_reader = reader
        return reader

    def _verify_language_file(self):
        """Ensure the requested language data exists; log a clear error if not."""
        if not self.tessdata_dir:
            self.log("Warning: tessdata directory not found; set TESSDATA_PREFIX or pass tessdata_dir")
            return

        langs = _split_langs(self.ocr_lang) or ["ben"]
        missing = []
        for code in langs:
            lang_file = Path(self.tessdata_dir) / f"{code}.traineddata"
            if not lang_file.is_file():
                missing.append(str(lang_file))

        if missing:
            self.log_error(f"Missing language file(s): {', '.join(missing)}")
            self.log("Tesseract language data not found; install the file or update tessdata path")
            return

        os.environ["TESSDATA_PREFIX"] = _sanitize_tessdata_prefix(self.tessdata_dir) or self.tessdata_dir
        self.log(f"Using tessdata: {self.tessdata_dir}; languages: {', '.join(langs)}")

        try:
            cmd_obj = getattr(pytesseract, 'pytesseract', None)
            cmd_path = getattr(cmd_obj, 'tesseract_cmd', None) if cmd_obj else None
            self.log(f"tesseract_cmd: {cmd_path}")
        except Exception:
            pass

    def cleanup_renders(self):
        """Delete renders directory after processing to save space."""
        try:
            self.renderer.cleanup_renders()
        except Exception:
            pass
    
    def log(self, message):
        """Log message."""
        self.results['extraction_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if self.progress_callback:
            self.progress_callback(message)
        if hasattr(self, 'logger') and self.logger:
            try:
                self.logger.info(message)
            except Exception:
                pass

    def log_error(self, message):
        """Persist errors to errors.log so they remain visible after the UI advances."""
        try:
            path = os.path.join(self.output_dir, "errors.log")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {message}\n")
        except Exception:
            pass
        if hasattr(self, 'logger') and self.logger:
            try:
                self.logger.error(message)
            except Exception:
                pass
    
    def open_pdf(self):
        """Open PDF document."""
        ok = self.renderer.open_pdf()
        if not ok:
            return False
        self.doc = self.renderer.doc
        try:
            page_count = len(self.doc.pages)
            size_mb = round(os.path.getsize(self.pdf_path) / (1024 * 1024), 2)
        except Exception:
            page_count = 0
            size_mb = 0
        self.results['metadata'] = {
            'filename': Path(self.pdf_path).name,
            'pages': page_count,
            'creation_date': datetime.now().isoformat(),
            'file_size_mb': size_mb
        }
        self.log(f"Opened: {self.results['metadata']['filename']}")
        return True
    
    def preprocess_image_for_ocr(self, image_path):
        """Optimized preprocessing with advanced quantization for Bengali/English text."""
        return preproc.preprocess_image_for_ocr(
            image_path,
            ocr_lang=self.ocr_lang,
            fast_mode=self.fast_mode,
            quality_mode=self.quality_mode,
            log_fn=self.log,
        )

    def render_page_to_image(self, page_num, zoom=None):
        """Render a single page using pdf2image/Poppler; returns (PIL image, saved_path|None)."""
        return self.renderer.render_page(page_num, zoom or self.page_render_zoom, fmt="png")

    def _process_page_with_ocr(self, page_num):
        """Optimized page processing with intelligent retry logic and error recovery."""
        if self.stop_event and self.stop_event.is_set():
            return page_num, None

        render_img = None
        render_path = None
        page_level_ocr = None
        
        try:
            render_result = self.render_page_to_image(page_num)
            if not render_result:
                return page_num, {
                    'page_number': page_num,
                    'content': "",
                    'warning': 'Rendering unavailable; raster OCR skipped',
                    'ocr_page_confidence': 0.0,
                    'ocr_page_fragments': 0,
                }
            if isinstance(render_result, tuple):
                render_img, render_path = render_result
            else:
                render_img = render_result
                render_path = None
            
            page_level_ocr = self.ocr.extract_text_with_ocr(render_img)
            
            if page_level_ocr:
                confidence = page_level_ocr.get('avg_confidence', 0)
                text_length = len(page_level_ocr.get('text', '').strip())
                needs_retry = (
                    confidence < self.high_dpi_retry_conf or
                    (text_length < 50 and confidence < 0.9)
                )
                
                if needs_retry and not self.fast_mode:
                    try:
                        hi_render = self.render_page_to_image(page_num, zoom=self.high_dpi_zoom)
                        if hi_render:
                            if isinstance(hi_render, tuple):
                                hi_img, hi_path = hi_render
                            else:
                                hi_img, hi_path = hi_render, None

                            hi_ocr = self.ocr.extract_text_with_ocr(hi_img)
                            if self._score_result(hi_ocr) > self._score_result(page_level_ocr):
                                page_level_ocr = hi_ocr
                                render_img = hi_img
                                render_path = hi_path
                                self.log(f"[Page {page_num + 1}] High-DPI retry improved results")
                            if hi_path and hi_path != render_path and os.path.exists(hi_path):
                                try:
                                    os.remove(hi_path)
                                except Exception:
                                    pass
                    except Exception as retry_err:
                        self.log(f"[Page {page_num + 1}] High-DPI retry failed: {retry_err}")

            page_text = page_level_ocr['text'] if page_level_ocr else ""
            page_data = {
                'page_number': page_num,
                'content': page_text,
            }

            if page_level_ocr:
                page_data['ocr_page_text'] = page_level_ocr['text']
                page_data['ocr_page_confidence'] = page_level_ocr['avg_confidence']
                page_data['ocr_page_fragments'] = page_level_ocr['fragments']
                page_data['ocr_render'] = os.path.relpath(render_path, self.output_dir) if render_path else None

            return page_num, page_data
            
        except Exception as e:
            error_msg = f"Page {page_num + 1} processing error: {e}"
            self.log(error_msg)
            self.log_error(error_msg)
            
            return page_num, {
                'page_number': page_num,
                'content': "",
                'error': str(e),
                'ocr_page_confidence': 0.0,
                'ocr_page_fragments': 0
            }

    def _normalize_text(self, text):
        """Strip zero-width characters and normalize whitespace."""
        if not text:
            return ""
        zero_width = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for zw in zero_width:
            text = text.replace(zw, '')
        return '\n'.join(' '.join(line.split()) for line in text.splitlines())

    def _flatten_background(self, image, clip=WATERMARK_CLIP_THRESHOLD):
        return preproc.flatten_background(image, clip=clip)

    def _choose_psm(self, image, segment_count):
        return preproc.choose_psm(image, segment_count)

    def _build_tess_config(self, extra=None, psm=None):
        """Build optimized Tesseract config for Bengali/English processing."""
        langs = _split_langs(self.ocr_lang) if hasattr(self, 'ocr_lang') else []
        has_ben = "ben" in langs
        has_eng = "eng" in langs
        mode = psm or 6
        parts = [
            f"--psm {mode}",
            "--oem 1",
        ]
        if has_ben:
            parts.extend([
                "-c preserve_interword_spaces=1",
                "-c chop_enable=0",
                "-c use_new_state_cost=F",
                "-c segment_penalty_garbage=1.5",
                "-c wordrec_enable_assoc=1",
                "-c language_model_penalty_non_freq_dict_word=0.1",
                "-c language_model_penalty_non_dict_word=0.15",
            ])
            if has_eng:
                parts.extend([
                    "-c textord_really_old_xheight=1",
                    "-c textord_min_xheight=12",
                ])
        else:
            parts.extend([
                "-c preserve_interword_spaces=1",
                "-c segment_penalty_garbage=0.5",
                "-c wordrec_enable_assoc=0",
                "-c language_model_penalty_non_dict_word=0.5",
            ])
        if self.quality_mode if hasattr(self, 'quality_mode') else False:
            parts.extend([
                "-c tessedit_char_blacklist=",
                "-c textord_noise_normratio=2",
            ])
        else:
            parts.extend([
                "-c tessedit_char_blacklist=|[]{}",
                "-c textord_noise_normratio=1",
            ])
        if extra:
            parts.extend(extra)
        return ' '.join(parts)

    def _extract_text_layer(self, page):
        """Fast path: pull native text from the PDF; returns normalized string or ''."""
        try:
            raw = page.extract_text() or ""
            normalized = self._normalize_text(raw)
            return normalized
        except Exception:
            return ""

    def _score_result(self, res):
        """Enhanced scoring algorithm for OCR result comparison."""
        return ocr_t.score_result(res)

    def _run_tesseract_pass(self, image, extra_config=None, extra_dilate=False, psm=None):
        """Run one Tesseract pass on a PIL image with optional extra dilation and config."""
        return ocr_t.run_tesseract_pass(
            image,
            ocr_lang=self.ocr_lang,
            quality_mode=self.quality_mode,
            psm=psm,
            extra_config=extra_config,
            extra_dilate=extra_dilate,
            log=self.log,
            log_error=self.log_error,
        )

    def scrape_all_pages(self):
        """Scrape all pages with optional parallel OCR per page."""
        if not self.open_pdf():
            return False
        
        try:
            total_pages = len(self.doc.pages)
            page_results = {}
            ocr_futures = []

            with ThreadPoolExecutor(max_workers=self._worker_count()) as executor:
                for page_num in range(total_pages):
                    if self.stop_event and self.stop_event.is_set():
                        self.log("Stop requested; aborting remaining pages")
                        break

                    self.log(f"Processing {page_num + 1}/{total_pages}")
                    page_text = ""
                    page_level_ocr = None
                    render_path = None

                    try:
                        page = self.doc.pages[page_num]

                        if TEXT_LAYER_FIRST:
                            langs = _split_langs(self.ocr_lang) or []
                            text_layer = None
                            # Only trust PDF text layer for non-Bengali jobs (or English-only).
                            if "ben" not in langs:
                                text_layer = self._extract_text_layer(page)
                            if text_layer and len(text_layer) > 5:
                                page_text = text_layer
                                self.log(f"[Page {page_num + 1}] Used PDF text layer; OCR skipped")

                        if not page_text:
                            future = executor.submit(self._process_page_with_ocr, page_num)
                            ocr_futures.append(future)
                            continue

                    except Exception as page_err:
                        self.log(f"Page {page_num + 1} OCR error: {page_err}")
                        try:
                            print(f"Page {page_num + 1} OCR error: {page_err}", file=sys.stderr)
                        except Exception:
                            pass
                        try:
                            self.log_error(f"Page {page_num + 1} OCR error: {page_err}")
                        except Exception:
                            pass
                        page_text = ""
                        page_level_ocr = None

                    preview_len = len(page_text)
                    self.log(f"[Page {page_num + 1}] OCR complete (chars: {preview_len})")

                    page_data = {
                        'page_number': page_num,
                        'content': page_text,
                    }

                    if page_level_ocr:
                        page_data['ocr_page_text'] = page_level_ocr['text']
                        page_data['ocr_page_confidence'] = page_level_ocr['avg_confidence']
                        page_data['ocr_page_fragments'] = page_level_ocr['fragments']
                        page_data['ocr_render'] = os.path.relpath(render_path, self.output_dir) if render_path else None

                    page_results[page_num] = page_data

                for fut in as_completed(ocr_futures):
                    try:
                        page_num, page_data = fut.result()
                    except Exception as err:
                        self.log(f"Parallel OCR error: {err}")
                        continue
                    if page_data is None:
                        continue
                    page_results[page_num] = page_data

            for page_num in sorted(page_results.keys()):
                page_data = page_results[page_num]
                self.results['pages'][f'page_{page_num}'] = page_data
                preview_len = len(page_data.get('content', '') or '')
                self.log(f"[Page {page_num + 1}] OCR merged (chars: {preview_len})")
            
            total_page_ocr = sum(len(p.get('ocr_page_text', '')) for p in self.results['pages'].values())
            pages_with_ocr = sum(1 for p in self.results['pages'].values() if 'ocr_page_text' in p)
            
            self.results['statistics'] = {
                'total_pages': total_pages,
                'total_text_length': sum(len(p.get('content', '')) for p in self.results['pages'].values()),
                'ocr_enabled': self.use_ocr,
                'pages_with_ocr_text': pages_with_ocr,
                'total_ocr_characters': total_page_ocr,
                'ocr_method': (getattr(self.ocr, 'ocr_method_effective', None) or self.ocr_method) if self.use_ocr else None,
                'preprocessing_applied': True,
            }
            
            return True
        except Exception as e:
            self.log(f"Error: {str(e)}")
            try:
                self.log_error(f"Scrape failure: {e}")
            except Exception:
                pass
            return False
        finally:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.doc = None
    
    def save_results(self):
        """Save results with layout-preserving output formats."""
        try:
            metadata = self.results.get('metadata') or {}
            if 'filename' not in metadata:
                metadata['filename'] = Path(self.pdf_path).name
            if 'pages' not in metadata:
                metadata['pages'] = len(self.results.get('pages', {}))
            if 'creation_date' not in metadata:
                metadata['creation_date'] = datetime.now().isoformat()
            if 'file_size_mb' not in metadata:
                if os.path.exists(self.pdf_path):
                    metadata['file_size_mb'] = round(os.path.getsize(self.pdf_path) / (1024 * 1024), 2)
                else:
                    metadata['file_size_mb'] = 0.0
            self.results['metadata'] = metadata
            display_lang = getattr(self, "user_lang", None) or self.ocr_lang

            ordered_pages = sorted(
                self.results['pages'].values(),
                key=lambda p: p.get('page_number', 0)
            )

            with open(os.path.join(self.output_dir, 'extracted_text.txt'), 'w', encoding='utf-8') as f:
                f.write(f"File: {self.results['metadata']['filename']}\n")
                f.write(f"Pages: {len(self.results['pages'])}\n")
                f.write(f"OCR: {self.ocr_method}\n")
                f.write(f"Language: {display_lang}\n")
                f.write("=" * 80 + "\n")

                for page_data in ordered_pages:
                    page_num = page_data['page_number']
                    f.write(f"\n----- PAGE {page_num + 1} -----\n")
                    content = page_data.get('content', '') or ''
                    f.write(content)
                    f.write("\n")

            with open(os.path.join(self.output_dir, 'extracted_text_continuous.txt'), 'w', encoding='utf-8') as f:
                f.write(f"# {self.results['metadata']['filename']}\n")
                f.write(f"# Extracted: {metadata.get('creation_date', '')[:19]}\n")
                f.write(f"# Language: {display_lang} | Pages: {len(self.results['pages'])}\n\n")
                for i, page_data in enumerate(ordered_pages):
                    content = page_data.get('content', '') or ''
                    if content.strip():
                        if i > 0:
                            f.write('\n\n')
                        f.write(content.strip())

            with open(os.path.join(self.output_dir, 'extracted_text_structured.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Document: {self.results['metadata']['filename']}\n")
                f.write(f"Language: {display_lang}\n")
                f.write(f"Total Pages: {len(self.results['pages'])}\n")
                f.write(f"Processing Date: {metadata.get('creation_date', '')[:19]}\n")
                f.write("\n" + "=" * 100 + "\n\n")
                
                for page_data in ordered_pages:
                    page_num = page_data['page_number']
                    content = page_data.get('content', '') or ''
                    confidence = page_data.get('ocr_page_confidence', 0)
                    fragments = page_data.get('ocr_page_fragments', 0)
                    
                    f.write(f"PAGE {page_num + 1}")
                    if confidence > 0:
                        f.write(f" [Confidence: {confidence:.3f}, Fragments: {fragments}]")
                    f.write(f"\n{'-' * 50}\n")
                    
                    if content.strip():
                        f.write(content)
                        f.write("\n\n")
                    else:
                        f.write("[No text detected on this page]\n\n")

            sentences_out = os.path.join(self.output_dir, 'extracted_text_sentences.txt')
            with open(sentences_out, 'w', encoding='utf-8') as f:
                f.write(f"Document: {self.results['metadata']['filename']}\n")
                f.write(f"Language: {display_lang}\n")
                f.write(f"Sentences/Clauses\n")
                f.write(f"{'-' * 40}\n")
                all_sentences = []
                for page_data in ordered_pages:
                    content = page_data.get('content', '') or ''
                    all_sentences.extend(_sentence_chunks(content))
                for sent in all_sentences:
                    f.write(sent + "\n")

            with open(os.path.join(self.output_dir, 'extraction_report.txt'), 'w', encoding='utf-8') as f:
                stats = self.results.get('statistics', {})
                f.write(f"PDF EXTRACTION REPORT\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"File: {metadata.get('filename', 'Unknown')}\n")
                f.write(f"Size: {metadata.get('file_size_mb', 0):.2f} MB\n")
                f.write(f"Total Pages: {stats.get('total_pages', 0)}\n")
                f.write(f"Pages with OCR: {stats.get('pages_with_ocr_text', 0)}\n")
                f.write(f"Total Characters: {stats.get('total_ocr_characters', 0):,}\n")
                f.write(f"OCR Engine: {stats.get('ocr_method', 'Unknown')}\n")
                f.write(f"Language: {display_lang}\n")
                f.write(f"Quality Mode: {'Yes' if self.quality_mode else 'No'}\n")
                f.write(f"Processing Time: {metadata.get('creation_date', '')}\n\n")
                
                f.write("PAGE QUALITY METRICS\n")
                f.write("-" * 30 + "\n")
                for page_data in ordered_pages:
                    page_num = page_data['page_number'] + 1
                    confidence = page_data.get('ocr_page_confidence', 0)
                    char_count = len(page_data.get('content', ''))
                    fragments = page_data.get('ocr_page_fragments', 0)
                    status = "Good" if confidence > 0.8 else "Fair" if confidence > 0.6 else "Poor"
                    f.write(f"Page {page_num:3d}: {confidence:.3f} confidence, {char_count:4d} chars, {fragments:2d} fragments [{status}]\n")

            self.log("Saved: extracted_text.txt (standard)")
            self.log("Saved: extracted_text_continuous.txt (layout-preserving)")
            self.log("Saved: extracted_text_structured.txt (detailed)")
            self.log("Saved: extracted_text_sentences.txt (sentences/clauses)")
            self.log("Saved: extraction_report.txt (quality metrics)")
            
            return True
        except Exception as e:
            self.log(f"Save error: {str(e)}")
            self.log_error(f"Save error: {e}")
            return False


def run_pdf_job(job_config: PdfJobConfig, stop_event, log_cb):
    """Run a single PDF job using the provided configuration."""
    errors, warnings = validate_runtime_env()
    if errors:
        if log_cb:
            for err in errors:
                log_cb(err)
        return {"scrape_ok": False, "save_ok": False, "stats": {}, "output_dir": job_config.output_root}
    if log_cb:
        for w in warnings:
            log_cb(f"Warning: {w}")

    pdf_name = Path(job_config.pdf_path).stem
    pdf_output = os.path.join(job_config.output_root, pdf_name)
    scraper = None
    try:
        scraper = PDFScraper(
            job_config.pdf_path,
            pdf_output,
            use_ocr=job_config.use_ocr,
            ocr_method=job_config.ocr_method,
            ocr_lang=job_config.ocr_lang,
            quality_mode=job_config.quality_mode,
            tessdata_dir=job_config.tessdata_dir,
            persist_renders=getattr(job_config, "persist_renders", False),
            max_workers=getattr(job_config, "max_workers", None),
            progress_callback=log_cb,
            stop_event=stop_event,
        )
        if log_cb:
            log_cb("Scraping PDF...")
        scrape_ok = scraper.scrape_all_pages()
        if log_cb:
            log_cb("Saving results..." if scrape_ok else "Saving partial results...")
        save_ok = scraper.save_results()
        try:
            scraper.cleanup_renders()
        except Exception:
            pass
        stats = scraper.results.get('statistics', {})
        return {
            "scrape_ok": scrape_ok,
            "save_ok": save_ok,
            "stats": stats,
            "output_dir": pdf_output,
        }
    except Exception as e:
        if log_cb:
            log_cb(f"Error: {e}")
        if scraper:
            scraper.log_error(f"Batch error on {job_config.pdf_path}: {e}")
        else:
            os.makedirs(pdf_output, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(pdf_output, "errors.log"), "a", encoding="utf-8") as f:
                f.write(f"[{ts}] Batch error on {job_config.pdf_path}: {e}\n")
        return {"scrape_ok": False, "save_ok": False, "stats": {}, "output_dir": pdf_output}
