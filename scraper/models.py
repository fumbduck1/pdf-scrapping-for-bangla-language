from typing import Any, Optional, Callable
from dataclasses import dataclass
from typing_extensions import TypedDict


@dataclass
class PageResult:
    page_number: int
    content: str = ""
    ocr_page_text: str = ""
    ocr_page_confidence: float = 0.0
    ocr_page_fragments: int = 0
    ocr_render: str = ""
    warning: str = ""
    error: str = ""


class JobResult(TypedDict):
    scrape_ok: bool
    output_dir: str
    num_pages: int
    num_errors: int
    num_warnings: int
    runtime_seconds: float
    errors: list
    warnings: list
