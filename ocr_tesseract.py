"""Tesseract engine helpers and multi-pass strategy."""

from constants import (
    SEGMENT_RETRY_CONF,
    THIRD_PASS_SCALE,
    WATERMARK_RETRY_CONF,
)
from deps import pytesseract
from preprocess import upscale_for_retry
from utils import _split_langs
from PIL import ImageFilter

def build_tess_config(ocr_lang: str, quality_mode: bool, extra=None, psm=None):
    langs = _split_langs(ocr_lang) if ocr_lang else []
    has_ben = "ben" in langs
    has_eng = "eng" in langs
    mode = psm or 6
    parts = [f"--psm {mode}", "--oem 1"]
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
    if quality_mode:
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


def run_tesseract_pass(image, ocr_lang: str, quality_mode: bool, psm=None, extra_config=None, extra_dilate=False, log=None, log_error=None):
    try:
        work_img = image
        if extra_dilate:
            try:
                work_img = work_img.filter(ImageFilter.MaxFilter(3))
            except Exception:
                pass

        config = build_tess_config(ocr_lang, quality_mode, extra=extra_config, psm=psm)
        data = pytesseract.image_to_data(
            work_img,
            lang=ocr_lang,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        if not data or 'text' not in data:
            return None

        entries = []
        confidences = []
        n_items = len(data['text'])
        for i in range(n_items):
            raw = data['text'][i] or ''
            txt = raw.strip()
            try:
                conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
            except Exception:
                conf = 0.0
            if not txt:
                continue
            try:
                y = int(data.get('top', [0])[i])
                x = int(data.get('left', [0])[i])
            except Exception:
                y, x = 0, 0
            block = int(data.get('block_num', [0])[i] or 0)
            par = int(data.get('par_num', [0])[i] or 0)
            line = int(data.get('line_num', [0])[i] or 0)
            entries.append((block, par, line, y, x, txt, conf))
            confidences.append(conf)

        if not entries:
            return None

        line_map = {}
        line_tops = {}
        for block, par, line, y, x, txt, conf in entries:
            key = (block, par, line)
            line_map.setdefault(key, []).append((x, txt))
            if key not in line_tops:
                line_tops[key] = y
            else:
                line_tops[key] = min(line_tops[key], y)

        ordered_keys = sorted(
            line_map.keys(),
            key=lambda key: (key[0], key[1], key[2], line_tops.get(key, 0))
        )

        lines = []
        for key in ordered_keys:
            words = sorted(line_map[key], key=lambda item: item[0])
            line_text = ' '.join(word for _, word in words).strip()
            if line_text:
                lines.append(line_text)

        if not lines:
            return None

        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        return {
            'text': '\n'.join(lines),
            'avg_confidence': round(avg_conf / 100, 4),
            'fragments': len(lines)
        }
    except Exception as e:
        if log:
            try:
                log(f"Tesseract OCR error: {e}")
            except Exception:
                pass
        if log_error:
            try:
                log_error(f"Tesseract OCR error: {e}")
            except Exception:
                pass
        return None


def score_result(res):
    if not res:
        return 0
    confidence = res.get('avg_confidence', 0)
    text_length = len(res.get('text', ''))
    fragments = res.get('fragments', 0)
    base_score = confidence * 100
    length_bonus = min(text_length * 0.1, 50)
    fragment_bonus = min(fragments * 2, 20)
    if text_length < 10 and confidence > 0.9:
        base_score *= 0.7
    total_score = base_score + length_bonus + fragment_bonus
    return total_score


def tesseract_best_for_segment(seg, alt_seg, psm_for_seg, ocr_lang, quality_mode, fast_conf_skip, log, log_error):
    pass_a = run_tesseract_pass(seg, ocr_lang, quality_mode, psm=psm_for_seg, log=log, log_error=log_error)
    pass_b = None
    if not pass_a or pass_a.get('avg_confidence', 0) < fast_conf_skip:
        pass_b = run_tesseract_pass(seg, ocr_lang, quality_mode, psm=psm_for_seg, extra_config=["-c lstm_choice_mode=2"], extra_dilate=True, log=log, log_error=log_error)

    best = pass_a if score_result(pass_a) >= score_result(pass_b) else pass_b

    if alt_seg is not None and (not best or best.get('avg_confidence', 0) < WATERMARK_RETRY_CONF):
        alt_a = run_tesseract_pass(alt_seg, ocr_lang, quality_mode, psm=psm_for_seg, log=log, log_error=log_error)
        alt_b = None
        if not alt_a or alt_a.get('avg_confidence', 0) < fast_conf_skip:
            alt_b = run_tesseract_pass(alt_seg, ocr_lang, quality_mode, psm=psm_for_seg, extra_config=["-c lstm_choice_mode=2"], extra_dilate=True, log=log, log_error=log_error)
        alt_best = alt_a if score_result(alt_a) >= score_result(alt_b) else alt_b
        if score_result(alt_best) > score_result(best):
            best = alt_best

    if best is None or best.get('avg_confidence', 0) < SEGMENT_RETRY_CONF or best.get('fragments', 0) < 2:
        retry_seg = upscale_for_retry(seg, scale=THIRD_PASS_SCALE)
        pass_c = run_tesseract_pass(
            retry_seg,
            ocr_lang,
            quality_mode,
            psm=psm_for_seg,
            extra_config=["-c lstm_choice_mode=2"],
            extra_dilate=True,
            log=log,
            log_error=log_error,
        )
        if score_result(pass_c) > score_result(best):
            best = pass_c

    return best


__all__ = [
    "build_tess_config",
    "run_tesseract_pass",
    "tesseract_best_for_segment",
    "score_result",
]
