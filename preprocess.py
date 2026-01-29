"""Image preprocessing utilities for OCR pipelines."""
import os
from typing import List, Tuple

from PIL import Image, ImageEnhance, ImageFilter

from constants import (
    HEADER_FOOTER_CROP_PCT,
    WATERMARK_CLIP_THRESHOLD,
    QUANTIZE_LEVELS,
    QUANTIZE_DITHER,
    THIRD_PASS_SCALE,
    MAX_OCR_PIXELS,
)
from deps import _lazy_import_numpy
np = _lazy_import_numpy()
from utils import _split_langs

# Allow large images; defer overall cap to caller if needed.
Image.MAX_IMAGE_PIXELS = 500_000_000
_ = Image.MAX_IMAGE_PIXELS  # keep side-effect assignment visible to linters


def quantize_params(ocr_lang: str, fast_mode: bool) -> Tuple[int, bool]:
    """Pick quantization levels/dither based on language mix and speed mode."""
    try:
        langs = _split_langs(ocr_lang)
    except Exception:
        langs = []
    has_ben = "ben" in langs
    has_eng = "eng" in langs
    if has_ben and has_eng:
        levels = 56 if not fast_mode else 40
        dither = True
    elif has_ben and not has_eng:
        levels = 40 if not fast_mode else 28
        dither = True
    elif has_eng and not has_ben:
        levels = 72 if not fast_mode else 48
        dither = False
    else:
        levels = 48
        dither = True
    return levels, dither


def crop_header_footer(image, pct: float = HEADER_FOOTER_CROP_PCT):
    """Remove top/bottom bands to drop running headers/footers."""
    try:
        if pct <= 0:
            return image
        img = image if isinstance(image, Image.Image) else Image.open(image)
        w, h = img.size
        band = int(h * pct)
        top = band
        bottom = h - band
        if bottom <= top:
            return img
        return img.crop((0, top, w, bottom))
    except Exception:
        return image


def flatten_background(image, clip=WATERMARK_CLIP_THRESHOLD):
    """Clip near-white pixels to pure white to reduce faint watermarks/backgrounds."""
    try:
        img = image.convert("L") if image.mode != "L" else image
        clip = max(0, min(int(clip), 255))
        return img.point(lambda p: 255 if p >= clip else p)
    except Exception:
        return image


def estimate_density(image) -> float:
    """Estimate dark-pixel density (0-1) to pick a better PSM."""
    try:
        if np is not None:
            arr = np.array(image.convert('L'))
            total = arr.size or 1
            return float((arr < 240).sum()) / float(total)
        hist = image.convert('L').histogram()
        total = sum(hist) or 1
        dark = sum(hist[:240])
        return float(dark) / float(total)
    except Exception:
        return 0.12


def choose_psm(image, segment_count: int) -> int:
    """Pick a page segmentation mode based on layout and density."""
    density = estimate_density(image)
    if segment_count > 1:
        return 4
    if density < 0.06:
        return 11  # very sparse ink; favor single-column sparse mode
    if density < 0.12:
        return 6   # moderately sparse; allow block segmentation
    return 6


def maybe_split_columns(image, fast_mode: bool):
    """Lightweight two-column check; skipped in fast mode to save time."""
    if fast_mode or not (np is not None):
        return [image]
    try:
        arr = np.array(image.convert('L'))
        h, w = arr.shape
        if w < 1400 or w / max(h, 1) < 0.9:
            return [image]
        proj = (arr < 245).sum(axis=0)
        if proj.max() == 0:
            return [image]
        norm = proj / proj.max()
        mid = w // 2
        window = max(int(w * 0.05), 30)
        gap_slice = norm[mid - window:mid + window]
        if len(gap_slice) == 0:
            return [image]
        gap_idx = np.argmin(gap_slice) + (mid - window)
        gap_val = gap_slice.min()
        left_dense = norm[:mid].mean()
        right_dense = norm[mid:].mean()
        if gap_val < 0.08 and min(left_dense, right_dense) > 0.15:
            pad = 10
            left_box = (0, 0, max(gap_idx - pad, 0), h)
            right_box = (min(gap_idx + pad, w), 0, w, h)
            return [image.crop(left_box), image.crop(right_box)]
    except Exception:
        return [image]
    return [image]


def upscale_for_retry(image, scale=THIRD_PASS_SCALE):
    """Resize image up for a last-chance OCR pass."""
    try:
        scale = max(scale, 1.0)
        w, h = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        if MAX_OCR_PIXELS and (new_w * new_h) > MAX_OCR_PIXELS:
            # Respect global pixel cap to avoid Tesseract/Leptonica OOMs
            safe_scale = (MAX_OCR_PIXELS / float(w * h)) ** 0.5
            new_w = max(1, int(w * safe_scale))
            new_h = max(1, int(h * safe_scale))
        new_size = (new_w, new_h)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    except Exception:
        return image


def preprocess_image_for_ocr(image_or_path, ocr_lang: str, fast_mode: bool, quality_mode: bool, log_fn=None):
    """Full preprocessing pipeline reused by both OCR engines."""
    try:
        img = image_or_path if isinstance(image_or_path, Image.Image) else Image.open(image_or_path)
        langs = _split_langs(ocr_lang)
        has_ben = "ben" in langs
        has_eng = "eng" in langs

        # Downscale very large renders before any heavy processing to prevent RAM exhaustion.
        try:
            w, h = img.size
            pixels = w * h
            if MAX_OCR_PIXELS and pixels > MAX_OCR_PIXELS:
                scale = (MAX_OCR_PIXELS / float(pixels)) ** 0.5
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                if log_fn:
                    log_fn(f"Downscaled large render from {w}x{h} to {new_size[0]}x{new_size[1]} to stay under memory cap")
        except Exception:
            pass

        if img.width < 1200:
            if has_ben:
                scale_factor = max(2, 1200 // img.width)
            else:
                scale_factor = max(2, 900 // img.width)
            new_size = (img.width * scale_factor, img.height * scale_factor)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        img = img.convert('L')

        if np is not None and not fast_mode:
            arr = np.array(img)
            brightness = float(arr.mean())
            contrast_std = float(arr.std())
        else:
            brightness = 128.0
            contrast_std = 40.0

        if brightness < 100:
            brightness_boost = 1.3 if has_ben else 1.2
            contrast_boost = 2.5 if contrast_std < 30 else 2.0
        elif brightness > 180:
            brightness_boost = 0.85 if has_ben else 0.9
            contrast_boost = 2.8 if contrast_std < 20 else 2.2
        else:
            brightness_boost = 1.1 if has_ben else 1.05
            contrast_boost = 2.0 if contrast_std < 25 else 1.8

        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(contrast_boost)
        brightness_enhancer = ImageEnhance.Brightness(img)
        img = brightness_enhancer.enhance(brightness_boost)

        if not fast_mode:
            if has_ben:
                img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
            else:
                img = img.filter(ImageFilter.SHARPEN)

        q_levels, q_dither = quantize_params(ocr_lang, fast_mode)
        if q_levels and q_levels > 0:
            try:
                method = getattr(Image, "MEDIANCUT", 1)
                dither = Image.FLOYDSTEINBERG if q_dither else Image.NONE
                quantized = img.quantize(colors=q_levels, method=method, dither=dither)
                img = quantized.convert('L')
                if has_ben and not fast_mode:
                    img = img.filter(ImageFilter.MedianFilter(size=3))
            except Exception:
                try:
                    method = getattr(Image, "FASTOCTREE", 2)
                    dither = Image.FLOYDSTEINBERG if q_dither else Image.NONE
                    img = img.quantize(colors=q_levels, method=method, dither=dither).convert('L')
                except Exception:
                    pass

        return img
    except Exception as e:
        if log_fn:
            try:
                log_fn(f"Preprocessing error: {e}")
            except Exception:
                pass
        return Image.open(image_or_path) if not isinstance(image_or_path, Image.Image) else image_or_path


__all__ = [
    "preprocess_image_for_ocr",
    "crop_header_footer",
    "flatten_background",
    "maybe_split_columns",
    "estimate_density",
    "choose_psm",
    "quantize_params",
    "upscale_for_retry",
]
