"""
Example script demonstrating the usage of the new performance module and configuration system.
This script shows how to integrate performance monitoring with the existing PDF scraper functionality.
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from performance import (
    get_monitor, timer, profile, register_metrics, print_performance_report
)
from logger import info, debug, warning, error, exception
from config_manager import get_config_manager, create_job_config


@timer("example_task")
def simulate_pdf_processing():
    """Simulate PDF processing tasks to demonstrate performance monitoring"""
    import time
    
    info("Starting PDF processing simulation")
    
    with profile("pdf_rendering"):
        debug("Rendering PDF pages...")
        time.sleep(0.8)
    
    with profile("image_preprocessing"):
        debug("Preprocessing images...")
        time.sleep(0.3)
    
    with profile("easyocr_pass"):
        debug("Running EasyOCR pass...")
        time.sleep(1.2)
    
    with profile("tesseract_pass"):
        debug("Running Tesseract pass...")
        time.sleep(0.7)
    
    with profile("text_saving"):
        debug("Saving text outputs...")
        time.sleep(0.2)
    
    info("PDF processing simulation complete")
    return True


def test_config_manager():
    """Test the configuration system"""
    try:
        info("Testing configuration system")
        
        config_manager = get_config_manager()
        
        # Get default config
        default_config = config_manager.get_default_config()
        debug(f"Default config OCR method: {default_config.ocr.ocr_method}")
        debug(f"Default config quality mode: {default_config.ocr.quality_mode}")
        
        # Create custom config
        custom_config = create_job_config(
            "test.pdf", "output", 
            ocr={"ocr_lang": "es", "quality_mode": False},
            max_workers=4
        )
        debug(f"Custom config OCR lang: {custom_config.ocr.ocr_lang}")
        debug(f"Custom config quality mode: {custom_config.ocr.quality_mode}")
        debug(f"Custom config max workers: {custom_config.max_workers}")
        
        # Validate config
        errors = config_manager.validate_config(custom_config)
        if errors:
            for err in errors:
                warning(f"Config validation error: {err}")
        else:
            debug("Config validation passed")
        
        info("Configuration system test complete")
        return True
        
    except Exception as e:
        error(f"Configuration system test failed: {e}")
        return False


def main():
    """Main demonstration function"""
    try:
        info("Starting performance module and configuration system demonstration")
        
        # Initialize performance monitor
        monitor = get_monitor()
        register_metrics()
        
        # Test configuration system
        config_test_result = test_config_manager()
        
        # Test performance monitoring
        processing_result = simulate_pdf_processing()
        
        # Run simulation 3 times to collect metrics
        for i in range(2):
            info(f"Running simulation iteration {i+2}")
            simulate_pdf_processing()
        
        # Print performance report
        print_performance_report()
        
        if config_test_result and processing_result:
            info("All demonstrations completed successfully!")
        else:
            error("Some demonstrations failed")
            
    except Exception as e:
        exception(f"Fatal error in demonstration: {e}")
        return False


if __name__ == "__main__":
    main()
