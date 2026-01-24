"""EasyOCR engine helpers."""
import threading

from deps import _lazy_import_easyocr, _lazy_import_numpy

easyocr = _lazy_import_easyocr()
EASYOCR_AVAILABLE = easyocr is not None
np = _lazy_import_numpy()
from utils import _split_langs


def map_easyocr_langs(ocr_lang: str):
    codes = []
    for code in _split_langs(ocr_lang or ""):
        if code.lower() in ("ben", "bn"):
            codes.append("bn")
        elif code.lower() in ("eng", "en"):
            codes.append("en")
        else:
            codes.append(code.lower())
    return codes or ["en"]


def get_easyocr_reader(ocr_lang: str, gpu: bool, lock: threading.Lock, existing_reader, log, log_error):
    if not EASYOCR_AVAILABLE:
        return None
    try:
        with lock:
            if existing_reader is not None:
                return existing_reader
            langs = map_easyocr_langs(ocr_lang)
            reader = easyocr.Reader(langs, gpu=gpu)
            if log:
                try:
                    log(f"EasyOCR initialized (langs: {','.join(langs)})")
                except Exception:
                    pass
            return reader
    except Exception as e:
        if log:
            try:
                log(f"EasyOCR init failed: {e}")
            except Exception:
                pass
        if log_error:
            try:
                log_error(f"EasyOCR init failed: {e}")
            except Exception:
                pass
        return None


def run_easyocr_pass(image, lock: threading.Lock, reader, log, log_error):
    """Run EasyOCR on a PIL image; returns dict or None."""
    if not reader:
        return None
    try:
        arr = np.array(image.convert('RGB')) if np is not None else None
        if arr is None:
            return None
        with lock:
            try:
                results = reader.readtext(arr, detail=1, paragraph=True)
            except ValueError as ve:
                if log:
                    try:
                        log(f"EasyOCR detail parse fallback: {ve}")
                    except Exception:
                        pass
                if log_error:
                    try:
                        log_error(f"EasyOCR detail parse fallback: {ve}")
                    except Exception:
                        pass
                fallback = reader.readtext(arr, detail=0, paragraph=True)
                results = [(None, txt, 0.0) for txt in fallback]
        if not results:
            return None
        texts = []
        confs = []
        for item in results:
            try:
                text = None
                conf = 0.0

                if isinstance(item, dict):
                    text = item.get('text') or item.get('value')
                    conf = item.get('confidence', item.get('prob', 0.0))
                elif isinstance(item, (list, tuple)):
                    if len(item) >= 3:
                        text = item[1]
                        conf = item[2]
                    elif len(item) == 2:
                        text = item[0]
                        conf = item[1]
                    elif len(item) == 1:
                        text = item[0]
                else:
                    text = item

                if isinstance(text, (list, tuple)):
                    text_value = ' '.join(str(t) for t in text if str(t).strip())
                else:
                    text_value = str(text or '')

                cleaned = text_value.strip()
                if not cleaned:
                    continue
                texts.append(cleaned)
                try:
                    confs.append(float(conf))
                except Exception:
                    confs.append(0.0)
            except Exception:
                continue
        if not texts:
            return None
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return {
            'text': '\n'.join(texts),
            'avg_confidence': round(avg_conf, 4),
            'fragments': len(texts),
        }
    except Exception as e:
        if log:
            try:
                log(f"EasyOCR error: {e}")
            except Exception:
                pass
        if log_error:
            try:
                log_error(f"EasyOCR error: {e}")
            except Exception:
                pass
        return None


__all__ = [
    "map_easyocr_langs",
    "get_easyocr_reader",
    "run_easyocr_pass",
]
