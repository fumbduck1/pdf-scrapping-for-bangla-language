"""Image preprocessing utilities for OCR pipelines."""
from typing import Tuple, Optional

from PIL import Image, ImageEnhance, ImageFilter

from logger import warning

from constants import (
    HEADER_FOOTER_CROP_PCT,
    WATERMARK_CLIP_THRESHOLD,
    THIRD_PASS_SCALE,
    MAX_OCR_PIXELS,
)
from deps import _lazy_import_numpy
np = _lazy_import_numpy()

# Optional import for enhanced watermark detection
try:
    import cv2
except ImportError:
    cv2 = None

from utils import split_langs

# Allow large images; defer overall cap to caller if needed.
Image.MAX_IMAGE_PIXELS = 500_000_000
_ = Image.MAX_IMAGE_PIXELS  # keep side-effect assignment visible to linters


def quantize_params(ocr_lang: str, fast_mode: bool) -> Tuple[int, bool]:
    """
    Determine quantization color levels and whether to apply dithering based on the OCR language mix and speed mode.
    
    If the language parsing fails, no languages are assumed. When both Bengali ('ben') and English ('eng') are present the function favors mid-range levels and enables dithering; Bengali-only favors lower levels with dithering; English-only favors higher levels without dithering; otherwise a moderate level with dithering is chosen.
    
    Parameters:
        ocr_lang (str): Language specifier string passed to split_langs.
        fast_mode (bool): If True, choose lower-quality (faster) quantization levels.
    
    Returns:
        Tuple[int, bool]: A pair (levels, dither) where `levels` is the number of quantization levels and `dither` is `True` if dithering should be applied, `False` otherwise.
    """
    try:
        langs = split_langs(ocr_lang)
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
        img = image if isinstance(image, Image.Image) else Image.open(image)
        if pct <= 0:
            return img
        w, h = img.size
        band = int(h * pct)
        top = band
        bottom = h - band
        if bottom <= top:
            return img
        return img.crop((0, top, w, bottom))
    except Exception:
        return image if isinstance(image, Image.Image) else None


def flatten_background(image, clip=WATERMARK_CLIP_THRESHOLD):
    """Clip near-white pixels to pure white to reduce faint watermarks/backgrounds."""
    try:
        img = image.convert("L") if image.mode != "L" else image
        clip = max(0, min(int(clip), 255))
        return img.point(lambda p: 255 if p >= clip else p)
    except Exception:
        return image


def detect_watermark_regions(image, watermark_confidence_threshold: float = 0.6) -> Optional:
    """
    Detect potential watermark regions using edge detection and frequency analysis.

    Returns a PIL Image mask where watermark-likely regions are white (255).
    Returns None if detection is not available or fails.
    """
    try:
        if np is None:
            return None

        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            arr = np.array(image.convert('L'))
        else:
            arr = np.array(image)
            if len(arr.shape) == 3:
                arr = (arr[:,:,0] * 0.3 + arr[:,:,1] * 0.59 + arr[:,:,2] * 0.11).astype(np.uint8)

        # Try to use cv2 for edge detection, fallback if not available
        try:
            # Canny edge detection to find text-like structures
            if cv2 is not None:
                blurred = cv2.GaussianBlur(arr, (3, 3), 0)
                edges = cv2.Canny(blurred, 50, 150)
            else:
                raise ImportError("cv2 not available")
        except (ImportError, AttributeError, Exception):
            # Fallback: use numpy-based edge detection (Sobel)
            try:
                from scipy import ndimage
                edges = ndimage.sobel(arr).astype(np.uint8)
                edges = (edges > 0).astype(np.uint8) * 255
            except ImportError:
                # No edge detection available
                return None

        # Frequency analysis: detect repetitive patterns
        # If same regions appear at regular intervals, likely watermark
        h, w = edges.shape

        # Horizontal projection: find columns with consistent edges
        h_proj = (edges > 0).sum(axis=0)

        # Look for regular spacing (watermark repeats)
        if h_proj.max() > 0:
            normalized_proj = h_proj / h_proj.max()

            # Simple peak detection - if you see consistent peaks, likely watermark pattern
            peaks = (normalized_proj > 0.3).astype(np.uint8)
            peak_diff = np.diff(peaks)

            # Count transitions (from low to high)
            transitions = np.sum(np.abs(peak_diff)) // 2

            # If >5 regular transitions, likely a watermark pattern
            watermark_likelihood = min(1.0, transitions / 5.0)
        else:
            watermark_likelihood = 0.0

        # Create mask: regions with high edge density + watermark pattern
        mask = edges.copy()

        # Apply morphological closing to connect nearby edges
        try:
            if cv2 is not None:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            else:
                raise ImportError("cv2 not available")
        except (Exception, AttributeError):
            # Fallback: use scipy
            try:
                from scipy import ndimage
                mask = ndimage.binary_closing(mask > 0, structure=np.ones((3,3))).astype(np.uint8) * 255
            except ImportError:
                pass

        # Threshold: only high-confidence watermark regions
        if watermark_likelihood >= watermark_confidence_threshold:
            return Image.fromarray(mask)

        return None

    except Exception as e:
        warning(f"Watermark detection failed: {e}")
        return None


def apply_adaptive_watermark_removal(image, watermark_mask: Optional = None, flip_background_threshold: int = 200) -> Image.Image:
    """
    Apply adaptive watermark removal using bilateral filtering on detected watermark regions.

    For watermarked regions: use bilateral filter (smooths while preserving edges)
    For content regions: preserve detail with sharpening
    """
    try:
        if watermark_mask is None:
            # If no mask, apply standard flatten
            return flatten_background(image, clip=flip_background_threshold)

        # Convert to array
        if isinstance(image, Image.Image):
            arr = np.array(image.convert('L'))
        else:
            arr = np.array(image)
            if len(arr.shape) == 3:
                arr = (arr[:,:,0] * 0.3 + arr[:,:,1] * 0.59 + arr[:,:,2] * 0.11).astype(np.uint8)

        # Convert mask to array
        mask_arr = np.array(watermark_mask.convert('L')) if isinstance(watermark_mask, Image.Image) else np.array(watermark_mask)
        mask_arr = (mask_arr > 127).astype(np.uint8)  # Binary mask

        # Apply bilateral filtering to watermark regions (soften + preserve edges)
        try:
            if cv2 is not None:
                filtered = cv2.bilateralFilter(arr, 9, 75, 75)
            else:
                raise ImportError("cv2 not available")
        except (ImportError, AttributeError):
            # Fallback: use median filter
            from PIL import ImageFilter as IF
            filtered = np.array(Image.fromarray(arr).filter(IF.MedianFilter(size=3)))

        # Blend: weighted combination of original and filtered
        result = arr.copy()
        result[mask_arr > 0] = 0.4 * arr[mask_arr > 0] + 0.6 * filtered[mask_arr > 0]
        result = result.astype(np.uint8)

        # Apply background flattening
        result = result.copy()
        result[result >= flip_background_threshold] = 255

        return Image.fromarray(result)

    except Exception as e:
        warning(f"Adaptive watermark removal failed: {e}")
        return flatten_background(image, clip=flip_background_threshold)



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


def preprocess_image_for_ocr(image_or_path, ocr_lang: str, fast_mode: bool, quality_mode: bool, log_fn=None, lite_mode: bool = False):
    """
    Run a full preprocessing pipeline on an image to prepare it for OCR.
    
    Performs optional downscaling to respect a global pixel cap, upscales small inputs, converts to grayscale, adjusts contrast and brightness based on image statistics and detected languages (e.g., Bengali "ben" or English "eng"), applies optional sharpening, and performs color quantization with optional dithering.
    
    Parameters:
        image_or_path (PIL.Image.Image | str): A PIL Image or a filesystem path to an image.
        ocr_lang (str): Language tags used to adjust processing (expects language codes such as "ben" and "eng").
        fast_mode (bool): If true, skip heavier operations and use conservative defaults to speed processing.
        quality_mode (bool): Optional flag for quality-oriented workflows (accepted but not required by all adjustments).
        log_fn (callable|None): Optional function that accepts a single string for logging status and error messages.
    
    Returns:
        PIL.Image.Image | None: The preprocessed grayscale PIL Image on success; if preprocessing fails and the input was a path, returns `None`; if preprocessing fails and the input was already a PIL Image, returns the original Image.
    """
    try:
        img = image_or_path if isinstance(image_or_path, Image.Image) else Image.open(image_or_path)
        langs = split_langs(ocr_lang)
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
            if lite_mode:
                scale_factor = min(scale_factor, 2)
            new_size = (img.width * scale_factor, img.height * scale_factor)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        img = img.convert('L')

        # Layer 1: Detect potential watermark regions for better handling
        watermark_mask = None
        if quality_mode and not fast_mode and not lite_mode:
            watermark_mask = detect_watermark_regions(img)
            if watermark_mask and log_fn:
                log_fn("Detected potential watermark regions")

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

        # Layer 2: Apply adaptive watermark removal for detected regions
        # Use lower threshold (200) for dark watermarks instead of 245
        if watermark_mask is not None:
            img = apply_adaptive_watermark_removal(img, watermark_mask, flip_background_threshold=200)
        else:
            # Standard watermark flattening if no mask detected
            img = flatten_background(img, clip=WATERMARK_CLIP_THRESHOLD)

        if not fast_mode and not lite_mode:
            if has_ben:
                img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
            else:
                img = img.filter(ImageFilter.SHARPEN)

        if not lite_mode:
            q_levels, q_dither = quantize_params(ocr_lang, fast_mode)
            if q_levels and q_levels > 0:
                try:
                    method = getattr(Image, "MEDIANCUT", 0)
                    dither = Image.Dither.FLOYDSTEINBERG if q_dither else Image.Dither.NONE
                    quantized = img.quantize(colors=q_levels, method=method, dither=dither)
                    img = quantized.convert('L')
                    if has_ben and not fast_mode:
                        img = img.filter(ImageFilter.MedianFilter(size=3))
                except Exception as e:
                    warning(f"Failed to quantize image with MEDIANCUT method: {e}")
                    try:
                        method = getattr(Image, "FASTOCTREE", 2)
                        dither = Image.Dither.FLOYDSTEINBERG if q_dither else Image.Dither.NONE
                        img = img.quantize(colors=q_levels, method=method, dither=dither).convert('L')
                    except Exception as e2:
                        warning(f"Failed to quantize image with FASTOCTREE fallback method: {e2}")

        return img
    except Exception as e:
        if log_fn:
            try:
                log_fn(f"Preprocessing error: {e}")
            except Exception:
                pass
        return None if not isinstance(image_or_path, Image.Image) else image_or_path


__all__ = [
    "preprocess_image_for_ocr",
    "crop_header_footer",
    "flatten_background",
    "detect_watermark_regions",
    "apply_adaptive_watermark_removal",
    "maybe_split_columns",
    "estimate_density",
    "choose_psm",
    "quantize_params",
    "upscale_for_retry",
]