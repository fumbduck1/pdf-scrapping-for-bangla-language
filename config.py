from dataclasses import dataclass
from constants import QUALITY_MODE_DEFAULT
@dataclass
class PdfJobConfig:
    pdf_path: str
    output_root: str
    use_ocr: bool = True
    ocr_method: str = "easyocr"
    ocr_lang: str = "ben"
    quality_mode: bool = QUALITY_MODE_DEFAULT
    tessdata_dir: str | None = None
    persist_renders: bool = False
    max_workers: int | None = None