"""Centralized configuration manager for PDF scraper"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

from constants import (
    DEFAULT_ZOOM,
    FAST_MODE,
    FAST_CONFIDENCE_SKIP,
    LITE_PRESET,
    TEXT_LAYER_FIRST,
    TEXT_LAYER_LANG_MIN_RATIO,
    TEXT_LAYER_MIN_BEN_CHARS,
    PDF_BYTES_CACHE_MB,
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
    RENDER_CACHE_MAX_ITEMS,
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
    auto_append_eng_for_ben: bool = AUTO_APPEND_ENG_FOR_BEN
    segment_retry_conf: float = SEGMENT_RETRY_CONF
    easyocr_fallback_conf: float = EASYOCR_FALLBACK_CONF
    easyocr_primary_conf: float = EASYOCR_PRIMARY_CONF
    tesseract_refine_min_chars: int = TESSERACT_REFINE_MIN_CHARS


@dataclass
class RenderConfig:
    """PDF rendering configuration"""
    zoom: float = DEFAULT_ZOOM
    high_dpi_zoom: float = HIGH_DPI_ZOOM
    high_dpi_retry_conf: float = HIGH_DPI_RETRY_CONF
    pdf_bytes_cache_mb: int = PDF_BYTES_CACHE_MB
    persist_renders: bool = False
    render_cache_max_items: int = RENDER_CACHE_MAX_ITEMS


@dataclass
class PreprocessConfig:
    """Image preprocessing configuration"""
    header_footer_crop_pct: float = HEADER_FOOTER_CROP_PCT
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
    input_path: str
    output_root: str
    use_ocr: bool = True
    preset: str = "default"
    lite_mode: bool = False
    ocr: OCRConfig = field(default_factory=OCRConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    text_layer: TextLayerConfig = field(default_factory=TextLayerConfig)
    max_workers: Optional[int] = None
    
    @property
    def file_type(self) -> str:
        """Determine file type from extension."""
        ext = Path(self.input_path).suffix.lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext == '.epub':
            return 'epub'
        return 'unknown'


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self):
        self._env_vars = self._load_env_vars()
    
    def _load_env_vars(self) -> Dict[str, str]:
        """
        Collect environment variables that start with the PDF_SCRAPER_ prefix and return them keyed by the name with the prefix removed and converted to lowercase.
        
        Returns:
            Dict[str, str]: Mapping from environment variable name (prefix removed and lowercased) to its string value.
        """
        env_vars: Dict[str, str] = {}
        prefix = "PDF_SCRAPER_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                env_vars[key[len(prefix):].lower()] = value
        
        return env_vars
    
    def from_dict(self, config_dict: Dict[str, Any]) -> JobConfig:
        """
        Builds a JobConfig from a flat or nested configuration dictionary.
        
        Accepts either top-level keys or nested sections ("ocr", "render", "preprocess", "text_layer"). Recognizes legacy "pdf_path" as an alias for "input_path" and coerces the resulting input path to a string. Requires "output_root" to be present in the provided dictionary.
        
        Parameters:
            config_dict (Dict[str, Any]): Configuration values; section keys may contain the same option names as top-level keys (top-level keys take precedence).
        
        Returns:
            JobConfig: A fully populated JobConfig with nested OCR, render, preprocess, text-layer, and RS-correction subconfigs. Numeric RS fields are clamped: `error_correction_bytes` is at least 1 and `block_size` is at least 64. The `use_ocr` field defaults to `True` when not provided.
        """
        ocr_section: Dict[str, Any] = config_dict.get("ocr", {}) if isinstance(config_dict.get("ocr", {}), dict) else {}
        render_section: Dict[str, Any] = config_dict.get("render", {}) if isinstance(config_dict.get("render", {}), dict) else {}
        preprocess_section: Dict[str, Any] = config_dict.get("preprocess", {}) if isinstance(config_dict.get("preprocess", {}), dict) else {}
        text_layer_section: Dict[str, Any] = config_dict.get("text_layer", {}) if isinstance(config_dict.get("text_layer", {}), dict) else {}
        preset_name = (config_dict.get("preset") or "default").lower()

        ocr_config = OCRConfig(
            ocr_method=config_dict.get("ocr_method", ocr_section.get("ocr_method", "easyocr")),
            ocr_lang=config_dict.get("ocr_lang", ocr_section.get("ocr_lang", "ben")),
            quality_mode=config_dict.get("quality_mode", ocr_section.get("quality_mode", QUALITY_MODE_DEFAULT)),
            fast_mode=config_dict.get("fast_mode", ocr_section.get("fast_mode", FAST_MODE)),
            fast_confidence_skip=config_dict.get("fast_confidence_skip", ocr_section.get("fast_confidence_skip", FAST_CONFIDENCE_SKIP)),
            tessdata_dir=config_dict.get("tessdata_dir", ocr_section.get("tessdata_dir")),
            auto_append_eng_for_ben=config_dict.get("auto_append_eng_for_ben", ocr_section.get("auto_append_eng_for_ben", AUTO_APPEND_ENG_FOR_BEN)),
            segment_retry_conf=config_dict.get("segment_retry_conf", ocr_section.get("segment_retry_conf", SEGMENT_RETRY_CONF)),
            easyocr_fallback_conf=config_dict.get("easyocr_fallback_conf", ocr_section.get("easyocr_fallback_conf", EASYOCR_FALLBACK_CONF)),
            easyocr_primary_conf=config_dict.get("easyocr_primary_conf", ocr_section.get("easyocr_primary_conf", EASYOCR_PRIMARY_CONF)),
            tesseract_refine_min_chars=config_dict.get("tesseract_refine_min_chars", ocr_section.get("tesseract_refine_min_chars", TESSERACT_REFINE_MIN_CHARS)),
        )
        
        render_config = RenderConfig(
            zoom=config_dict.get("zoom", render_section.get("zoom", DEFAULT_ZOOM)),
            high_dpi_zoom=config_dict.get("high_dpi_zoom", render_section.get("high_dpi_zoom", HIGH_DPI_ZOOM)),
            high_dpi_retry_conf=config_dict.get("high_dpi_retry_conf", render_section.get("high_dpi_retry_conf", HIGH_DPI_RETRY_CONF)),
            pdf_bytes_cache_mb=config_dict.get("pdf_bytes_cache_mb", render_section.get("pdf_bytes_cache_mb", PDF_BYTES_CACHE_MB)),
            persist_renders=config_dict.get("persist_renders", render_section.get("persist_renders", False)),
            render_cache_max_items=config_dict.get("render_cache_max_items", render_section.get("render_cache_max_items", RENDER_CACHE_MAX_ITEMS)),
        )
        
        preprocess_config = PreprocessConfig(
            header_footer_crop_pct=config_dict.get("header_footer_crop_pct", preprocess_section.get("header_footer_crop_pct", HEADER_FOOTER_CROP_PCT)),
            quantize_levels=config_dict.get("quantize_levels", preprocess_section.get("quantize_levels", QUANTIZE_LEVELS)),
            quantize_dither=config_dict.get("quantize_dither", preprocess_section.get("quantize_dither", QUANTIZE_DITHER)),
            third_pass_scale=config_dict.get("third_pass_scale", preprocess_section.get("third_pass_scale", THIRD_PASS_SCALE)),
        )
        
        text_layer_config = TextLayerConfig(
            text_layer_first=config_dict.get("text_layer_first", text_layer_section.get("text_layer_first", TEXT_LAYER_FIRST)),
            text_layer_lang_min_ratio=config_dict.get("text_layer_lang_min_ratio", text_layer_section.get("text_layer_lang_min_ratio", TEXT_LAYER_LANG_MIN_RATIO)),
            text_layer_min_ben_chars=config_dict.get("text_layer_min_ben_chars", text_layer_section.get("text_layer_min_ben_chars", TEXT_LAYER_MIN_BEN_CHARS)),
        )
        
        # RS correction configuration
         # Support both input_path and pdf_path for backward compatibility
        input_path = config_dict.get("input_path", config_dict.get("pdf_path"))
        if input_path is None:
            input_path = ""
            
        # Ensure input_path is a string
        if not isinstance(input_path, str):
            input_path = str(input_path)
            
        job = JobConfig(
            input_path=input_path,
            output_root=config_dict["output_root"],
            use_ocr=config_dict.get("use_ocr", True),
            preset=preset_name,
            ocr=ocr_config,
            render=render_config,
            preprocess=preprocess_config,
            text_layer=text_layer_config,
            max_workers=config_dict.get("max_workers"),
            lite_mode=preset_name == "lite",
        )
        return self.apply_preset(job)
    
    def from_env(self) -> JobConfig:
        """
        Build a JobConfig populated from current PDF_SCRAPER_* environment variables.
        
        Parses recognized environment keys (including either `input_path` or `pdf_path`) and converts boolean, integer, and float string values to their respective types. Unparsable numeric values are ignored. Ensures `pdf_path` and `output_root` are present in the resulting configuration (set to empty string if absent).
        
        Returns:
            JobConfig: A JobConfig populated from the environment variables.
        """
        # Reload environment variables to get the latest values
        self._env_vars = self._load_env_vars()
        
        config_dict: Dict[str, Any] = {}
        
        if "input_path" in self._env_vars:
            config_dict["input_path"] = self._env_vars["input_path"]
        elif "pdf_path" in self._env_vars:
            config_dict["input_path"] = self._env_vars["pdf_path"]
        
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
        
        if "auto_append_eng_for_ben" in self._env_vars:
            config_dict["auto_append_eng_for_ben"] = self._env_vars["auto_append_eng_for_ben"].lower() in ("true", "1", "yes")
        
        if "segment_retry_conf" in self._env_vars:
            try:
                config_dict["segment_retry_conf"] = float(self._env_vars["segment_retry_conf"])
            except ValueError:
                pass
        
        if "easyocr_fallback_conf" in self._env_vars:
            try:
                config_dict["easyocr_fallback_conf"] = float(self._env_vars["easyocr_fallback_conf"])
            except ValueError:
                pass
        
        if "easyocr_primary_conf" in self._env_vars:
            try:
                config_dict["easyocr_primary_conf"] = float(self._env_vars["easyocr_primary_conf"])
            except ValueError:
                pass
        
        if "tesseract_refine_min_chars" in self._env_vars:
            try:
                config_dict["tesseract_refine_min_chars"] = int(self._env_vars["tesseract_refine_min_chars"])
            except ValueError:
                pass
        
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

        if "preset" in self._env_vars:
            config_dict["preset"] = self._env_vars["preset"]
        
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

    def apply_preset(self, config: JobConfig) -> JobConfig:
        """Apply named presets to a JobConfig instance."""
        preset = (config.preset or "default").lower()
        if preset != "lite":
            return config

        # Force lightweight settings for consumer devices
        config.lite_mode = True
        config.ocr.ocr_method = "easyocr"
        config.ocr.fast_mode = True
        config.ocr.quality_mode = True
        config.ocr.fast_confidence_skip = max(config.ocr.fast_confidence_skip, 0.9)
        config.ocr.segment_retry_conf = 0.0  # disable heavy per-segment retries
        config.ocr.easyocr_primary_conf = min(config.ocr.easyocr_primary_conf, 0.92)
        config.ocr.easyocr_fallback_conf = 0.0
        config.ocr.tesseract_refine_min_chars = 9999  # effectively disable tesseract refine paths

        config.render.zoom = LITE_PRESET["zoom"]
        config.render.high_dpi_zoom = LITE_PRESET["high_dpi_zoom"]
        config.render.high_dpi_retry_conf = LITE_PRESET["high_dpi_retry_conf"]
        config.render.persist_renders = False
        config.render.pdf_bytes_cache_mb = min(config.render.pdf_bytes_cache_mb, 64)
        config.render.render_cache_max_items = LITE_PRESET["render_cache_max_items"]

        config.preprocess.quantize_levels = LITE_PRESET["quantize_levels"]
        config.preprocess.quantize_dither = LITE_PRESET["quantize_dither"]
        config.preprocess.third_pass_scale = LITE_PRESET["third_pass_scale"]

        config.text_layer.text_layer_first = True
        config.text_layer.text_layer_lang_min_ratio = LITE_PRESET["text_layer_lang_min_ratio"]
        config.text_layer.text_layer_min_ben_chars = LITE_PRESET["text_layer_min_ben_chars"]

        # Bound worker count and render cache size
        config.max_workers = config.max_workers or LITE_PRESET["max_workers"]
        return config
    
    def from_file(self, config_path: str) -> JobConfig:
        """Load configuration from file (JSON or YAML)"""
        path_obj = Path(config_path)
        
        if path_obj.suffix == ".json":
            import json
            with open(path_obj, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        
        elif path_obj.suffix in (".yaml", ".yml"):
            import yaml
            with open(path_obj, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        
        else:
            raise ValueError(f"Unsupported config file format: {path_obj.suffix}")
        
        return self.from_dict(config_dict)
    
    def validate_config(self, config: JobConfig) -> List[str]:
        """Validate configuration and return a list of error messages."""
        errors: List[str] = []
        
        # Validate paths
        if not config.input_path:
            errors.append("Input path must be provided")
        else:
            input_path = Path(config.input_path)
            if not input_path.exists():
                errors.append(f"Input file not found: {config.input_path}")
            if input_path.suffix.lower() not in (".pdf", ".epub"):
                errors.append(f"File must be a PDF or EPUB: {config.input_path}")
        
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

        if config.render.render_cache_max_items < 0:
            errors.append("Render cache size cannot be negative")
        
        return errors
    
    def get_default_config(self) -> JobConfig:
        """Get default configuration"""
        return JobConfig(
            input_path="",
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
    """
    Provide access to the module-level ConfigManager singleton.
    
    Returns:
        config_manager (ConfigManager): The shared ConfigManager instance.
    """
    return config_manager


def create_job_config(input_path: str, output_root: str | None = None, **kwargs: Any) -> JobConfig:
    """
    Create a JobConfig by merging the provided paths with additional configuration options.
    
    Parameters:
        input_path (str): Path to the input file (PDF, EPUB, etc.).
        output_root (str | None): Destination root for outputs; empty string used if None.
        **kwargs: Additional flat or nested configuration values forwarded to the configuration loader.
    
    Returns:
        JobConfig: A populated JobConfig built from the merged values.
    """
    config_dict: Dict[str, Any] = {
        "input_path": input_path,
        "output_root": output_root or "",
        **kwargs
    }
    return config_manager.from_dict(config_dict)