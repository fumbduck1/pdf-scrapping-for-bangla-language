"""Centralized configuration manager for PDF scraper"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

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
    HEADER_FOOTER_CROP_PCT,
    QUANTIZE_LEVELS,
    QUANTIZE_DITHER,
    AUTO_APPEND_ENG_FOR_BEN,
    QUALITY_MODE_DEFAULT,
    SEGMENT_RETRY_CONF,
    THIRD_PASS_SCALE,
    EASYOCR_FALLBACK_CONF,
    EASYOCR_PRIMARY_CONF,
    TESSERACT_REFINE_MIN_CHARS,
)


@dataclass
class OCRConfig:
    """OCR engine configuration"""
    ocr_method: str = "easyocr"
    ocr_lang: str = "ben"
    quality_mode: bool = QUALITY_MODE_DEFAULT
    fast_mode: bool = FAST_MODE
    fast_confidence_skip: float = FAST_CONFIDENCE_SKIP
    tessdata_dir: Optional[str] = None


@dataclass
class RenderConfig:
    """PDF rendering configuration"""
    zoom: float = DEFAULT_ZOOM
    high_dpi_zoom: float = HIGH_DPI_ZOOM
    high_dpi_retry_conf: float = HIGH_DPI_RETRY_CONF
    pdf_bytes_cache_mb: int = PDF_BYTES_CACHE_MB
    persist_renders: bool = False


@dataclass
class PreprocessConfig:
    """Image preprocessing configuration"""
    header_footer_crop_pct: float = HEADER_FOOTER_CROP_PCT
    watermark_flatten: bool = WATERMARK_FLATTEN
    watermark_clip_threshold: int = WATERMARK_CLIP_THRESHOLD
    watermark_retry_conf: float = WATERMARK_RETRY_CONF
    quantize_levels: int = QUANTIZE_LEVELS
    quantize_dither: bool = QUANTIZE_DITHER
    third_pass_scale: float = THIRD_PASS_SCALE


@dataclass
class TextLayerConfig:
    """PDF text layer extraction configuration"""
    text_layer_first: bool = TEXT_LAYER_FIRST
    text_layer_lang_min_ratio: float = TEXT_LAYER_LANG_MIN_RATIO
    text_layer_min_ben_chars: int = TEXT_LAYER_MIN_BEN_CHARS


@dataclass
class JobConfig:
    """Complete job configuration"""
    pdf_path: str
    output_root: str
    use_ocr: bool = True
    ocr: OCRConfig = field(default_factory=OCRConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    text_layer: TextLayerConfig = field(default_factory=TextLayerConfig)
    max_workers: Optional[int] = None


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self):
        self._config: Optional[JobConfig] = None
        self._env_vars = self._load_env_vars()
    
    def _load_env_vars(self) -> Dict[str, str]:
        """Load configuration from environment variables"""
        env_vars = {}
        prefix = "PDF_SCRAPER_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                env_vars[key[len(prefix):].lower()] = value
        
        return env_vars
    
    def from_dict(self, config_dict: Dict[str, Any]) -> JobConfig:
        """Create job config from dictionary. Accepts flat keys or nested sections (ocr/render/preprocess/text_layer)."""
        ocr_section = config_dict.get("ocr", {}) if isinstance(config_dict.get("ocr", {}), dict) else {}
        render_section = config_dict.get("render", {}) if isinstance(config_dict.get("render", {}), dict) else {}
        preprocess_section = config_dict.get("preprocess", {}) if isinstance(config_dict.get("preprocess", {}), dict) else {}
        text_layer_section = config_dict.get("text_layer", {}) if isinstance(config_dict.get("text_layer", {}), dict) else {}

        ocr_config = OCRConfig(
            ocr_method=config_dict.get("ocr_method", ocr_section.get("ocr_method", "easyocr")),
            ocr_lang=config_dict.get("ocr_lang", ocr_section.get("ocr_lang", "ben")),
            quality_mode=config_dict.get("quality_mode", ocr_section.get("quality_mode", QUALITY_MODE_DEFAULT)),
            fast_mode=config_dict.get("fast_mode", ocr_section.get("fast_mode", FAST_MODE)),
            fast_confidence_skip=config_dict.get("fast_confidence_skip", ocr_section.get("fast_confidence_skip", FAST_CONFIDENCE_SKIP)),
            tessdata_dir=config_dict.get("tessdata_dir", ocr_section.get("tessdata_dir")),
        )
        
        render_config = RenderConfig(
            zoom=config_dict.get("zoom", render_section.get("zoom", DEFAULT_ZOOM)),
            high_dpi_zoom=config_dict.get("high_dpi_zoom", render_section.get("high_dpi_zoom", HIGH_DPI_ZOOM)),
            high_dpi_retry_conf=config_dict.get("high_dpi_retry_conf", render_section.get("high_dpi_retry_conf", HIGH_DPI_RETRY_CONF)),
            pdf_bytes_cache_mb=config_dict.get("pdf_bytes_cache_mb", render_section.get("pdf_bytes_cache_mb", PDF_BYTES_CACHE_MB)),
            persist_renders=config_dict.get("persist_renders", render_section.get("persist_renders", False)),
        )
        
        preprocess_config = PreprocessConfig(
            header_footer_crop_pct=config_dict.get("header_footer_crop_pct", preprocess_section.get("header_footer_crop_pct", HEADER_FOOTER_CROP_PCT)),
            watermark_flatten=config_dict.get("watermark_flatten", preprocess_section.get("watermark_flatten", WATERMARK_FLATTEN)),
            watermark_clip_threshold=config_dict.get("watermark_clip_threshold", preprocess_section.get("watermark_clip_threshold", WATERMARK_CLIP_THRESHOLD)),
            watermark_retry_conf=config_dict.get("watermark_retry_conf", preprocess_section.get("watermark_retry_conf", WATERMARK_RETRY_CONF)),
            quantize_levels=config_dict.get("quantize_levels", preprocess_section.get("quantize_levels", QUANTIZE_LEVELS)),
            quantize_dither=config_dict.get("quantize_dither", preprocess_section.get("quantize_dither", QUANTIZE_DITHER)),
            third_pass_scale=config_dict.get("third_pass_scale", preprocess_section.get("third_pass_scale", THIRD_PASS_SCALE)),
        )
        
        text_layer_config = TextLayerConfig(
            text_layer_first=config_dict.get("text_layer_first", text_layer_section.get("text_layer_first", TEXT_LAYER_FIRST)),
            text_layer_lang_min_ratio=config_dict.get("text_layer_lang_min_ratio", text_layer_section.get("text_layer_lang_min_ratio", TEXT_LAYER_LANG_MIN_RATIO)),
            text_layer_min_ben_chars=config_dict.get("text_layer_min_ben_chars", text_layer_section.get("text_layer_min_ben_chars", TEXT_LAYER_MIN_BEN_CHARS)),
        )
        
        return JobConfig(
            pdf_path=config_dict["pdf_path"],
            output_root=config_dict["output_root"],
            use_ocr=config_dict.get("use_ocr", True),
            ocr=ocr_config,
            render=render_config,
            preprocess=preprocess_config,
            text_layer=text_layer_config,
            max_workers=config_dict.get("max_workers"),
        )
    
    def from_env(self) -> JobConfig:
        """Create job config from environment variables"""
        # Reload environment variables to get the latest values
        self._env_vars = self._load_env_vars()
        
        config_dict = {}
        
        if "pdf_path" in self._env_vars:
            config_dict["pdf_path"] = self._env_vars["pdf_path"]
        
        if "output_root" in self._env_vars:
            config_dict["output_root"] = self._env_vars["output_root"]
        
        if "use_ocr" in self._env_vars:
            config_dict["use_ocr"] = self._env_vars["use_ocr"].lower() in ("true", "1", "yes")
        
        if "ocr_method" in self._env_vars:
            config_dict["ocr_method"] = self._env_vars["ocr_method"]
        
        if "ocr_lang" in self._env_vars:
            config_dict["ocr_lang"] = self._env_vars["ocr_lang"]
        
        if "quality_mode" in self._env_vars:
            config_dict["quality_mode"] = self._env_vars["quality_mode"].lower() in ("true", "1", "yes")
        
        if "fast_mode" in self._env_vars:
            config_dict["fast_mode"] = self._env_vars["fast_mode"].lower() in ("true", "1", "yes")
        
        if "fast_confidence_skip" in self._env_vars:
            try:
                config_dict["fast_confidence_skip"] = float(self._env_vars["fast_confidence_skip"])
            except ValueError:
                pass
        
        if "tessdata_dir" in self._env_vars:
            config_dict["tessdata_dir"] = self._env_vars["tessdata_dir"]
        
        if "zoom" in self._env_vars:
            try:
                config_dict["zoom"] = float(self._env_vars["zoom"])
            except ValueError:
                pass
        
        if "high_dpi_zoom" in self._env_vars:
            try:
                config_dict["high_dpi_zoom"] = float(self._env_vars["high_dpi_zoom"])
            except ValueError:
                pass
        
        if "high_dpi_retry_conf" in self._env_vars:
            try:
                config_dict["high_dpi_retry_conf"] = float(self._env_vars["high_dpi_retry_conf"])
            except ValueError:
                pass
        
        if "pdf_bytes_cache_mb" in self._env_vars:
            try:
                config_dict["pdf_bytes_cache_mb"] = int(self._env_vars["pdf_bytes_cache_mb"])
            except ValueError:
                pass
        
        if "persist_renders" in self._env_vars:
            config_dict["persist_renders"] = self._env_vars["persist_renders"].lower() in ("true", "1", "yes")
        
        if "max_workers" in self._env_vars:
            try:
                config_dict["max_workers"] = int(self._env_vars["max_workers"])
            except ValueError:
                pass
        
        # Set default values for required fields if not provided
        if "pdf_path" not in config_dict:
            config_dict["pdf_path"] = ""
        
        if "output_root" not in config_dict:
            config_dict["output_root"] = ""
        
        return self.from_dict(config_dict)
    
    def from_file(self, config_path: str) -> JobConfig:
        """Load configuration from file (JSON or YAML)"""
        config_path = Path(config_path)
        
        if config_path.suffix == ".json":
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        
        elif config_path.suffix in (".yaml", ".yml"):
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return self.from_dict(config_dict)
    
    def validate_config(self, config: JobConfig) -> List[str]:
        """Validate configuration and return a list of error messages."""
        errors: List[str] = []
        
        # Validate paths
        if not config.pdf_path:
            errors.append("PDF path must be provided")
        else:
            pdf_path = Path(config.pdf_path)
            if not pdf_path.exists():
                errors.append(f"PDF file not found: {config.pdf_path}")
            if pdf_path.suffix.lower() != ".pdf":
                errors.append(f"File must be a PDF: {config.pdf_path}")
        
        if not config.output_root:
            errors.append("Output directory must be provided")
        else:
            output_path = Path(config.output_root)
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create output directory: {e}")
        
        # Validate OCR configuration
        if config.use_ocr:
            if config.ocr.ocr_method not in ("easyocr", "tesseract"):
                errors.append(f"Invalid OCR method: {config.ocr.ocr_method}")
            
            if not config.ocr.ocr_lang:
                errors.append("OCR language must be specified")
        
        # Validate numerical values
        if config.render.zoom <= 0:
            errors.append("Zoom level must be positive")
        
        if config.render.high_dpi_zoom <= config.render.zoom:
            errors.append("High DPI zoom must be greater than default zoom")
        
        if not (0.0 <= config.render.high_dpi_retry_conf <= 1.0):
            errors.append("High DPI retry confidence must be between 0 and 1")
        
        if config.render.pdf_bytes_cache_mb <= 0:
            errors.append("PDF bytes cache size must be positive")
        
        return errors
    
    def get_default_config(self) -> JobConfig:
        """Get default configuration"""
        return JobConfig(
            pdf_path="",
            output_root="",
            use_ocr=True,
            ocr=OCRConfig(),
            render=RenderConfig(),
            preprocess=PreprocessConfig(),
            text_layer=TextLayerConfig(),
            max_workers=None,
        )


config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get singleton config manager instance"""
    return config_manager


def create_job_config(pdf_path: str, output_root: str, **kwargs) -> JobConfig:
    """Create a new job configuration"""
    config_dict = {
        "pdf_path": pdf_path,
        "output_root": output_root,
        **kwargs
    }
    return config_manager.from_dict(config_dict)
