"""Performance benchmarks and tests for the scraper"""
import time
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from config_manager import create_job_config
from scraper import run_pdf_job
from performance import get_monitor, print_performance_report


@pytest.mark.skip(reason="Performance tests are only run manually")
def test_performance_benchmark():
    """Benchmark the scraper on a sample PDF file"""
    # Note: This test is not run by default
    # You should provide your own test PDF file
    test_pdf_path = r"C:\Users\akram\Downloads\sample.pdf"
    output_dir = r"c:\Users\akram\Desktop\pdf scrapper ongoing\benchmark_output"
    
    # Create a job configuration
    config = create_job_config(
        pdf_path=test_pdf_path,
        output_root=output_dir,
        use_ocr=True,
        ocr_method="easyocr",
        ocr_lang="ben+eng",
        quality_mode=True,
        fast_mode=False,
        persist_renders=False,
        max_workers=4
    )
    
    # Run the job and measure time
    start_time = time.time()
    result = run_pdf_job(config, stop_event=None, log_cb=print)
    total_time = time.time() - start_time
    
    print(f"\n\n=== Performance Benchmark ===")
    print(f"Total time: {total_time:.2f} seconds")
    
    if result:
        print(f"Pages processed: {result.get('stats', {}).get('total_pages', 0)}")
        print(f"Characters OCR'd: {result.get('stats', {}).get('total_ocr_characters', 0)}")
        print(f"Save status: {'OK' if result.get('save_ok') else 'Failed'}")
    
    # Print performance metrics
    print("\n=== Performance Metrics ===")
    print_performance_report()
    
    # Assert basic performance expectations
    assert result.get("save_ok"), "Job failed to save results"
    
    pages = result.get('stats', {}).get('total_pages', 0)
    if pages > 0:
        avg_time_per_page = total_time / pages
        assert avg_time_per_page < 30, f"Average time per page is too high: {avg_time_per_page:.2f}s"
    
    # Check if at least some pages were processed
    assert result.get('stats', {}).get('pages_with_ocr_text', 0) > 0, "No pages processed"


def test_performance_monitor():
    """Test the performance monitor functionality"""
    # Reset the monitor
    monitor = get_monitor()
    monitor.clear()
    
    # Create some test metrics
    monitor.register_metric("test_metric", "seconds", "Test metric")
    monitor.add_metric_value("test_metric", 1.2)
    monitor.add_metric_value("test_metric", 1.5)
    monitor.add_metric_value("test_metric", 1.3)
    
    # Verify the monitor has metrics
    metrics = monitor.get_all_metrics()
    assert "test_metric" in metrics
    
    # Verify metric values
    test_metric = metrics["test_metric"]
    assert len(test_metric.values) == 3
    assert test_metric.values == [1.2, 1.5, 1.3]
    
    # Verify summary statistics
    summary = monitor.get_summary()
    assert summary["total_time"] > 0
    assert "test_metric" in summary["metrics"]
    
    metric_summary = summary["metrics"]["test_metric"]
    assert metric_summary["count"] == 3
    assert metric_summary["min"] == 1.2
    assert metric_summary["max"] == 1.5
    assert metric_summary["avg"] == (1.2 + 1.5 + 1.3) / 3
    assert metric_summary["total"] == 1.2 + 1.5 + 1.3
    
    print("Performance monitor test passed")


def test_config_validation():
    """Test various configuration validation scenarios"""
    from config_manager import get_config_manager
    
    config_manager = get_config_manager()
    
    # Test empty config
    config = config_manager.get_default_config()
    errors = config_manager.validate_config(config)
    print("Validation errors for empty config:")
    for error in errors:
        print(f"- {error}")
    
    # Test config without PDF path
    config = config_manager.get_default_config()
    config.pdf_path = ""
    config.output_root = "output"
    errors = config_manager.validate_config(config)
    assert "PDF path must be provided" in errors
    
    # Test config without output root
    config = config_manager.get_default_config()
    config.pdf_path = "test.pdf"
    config.output_root = ""
    errors = config_manager.validate_config(config)
    assert "Output directory must be provided" in errors
    
    # Test with invalid OCR method
    config = config_manager.get_default_config()
    config.pdf_path = "test.pdf"
    config.output_root = "output"
    config.ocr.ocr_method = "invalid"
    errors = config_manager.validate_config(config)
    assert any("invalid ocr method" in e.lower() for e in errors)
    
    print("\nConfiguration validation test passed")


if __name__ == "__main__":
    print("Running performance monitor test...")
    test_performance_monitor()
    print("\nRunning configuration validation test...")
    test_config_validation()
    
    # Uncomment to run the actual performance benchmark
    # print("\nRunning performance benchmark...")
    # test_performance_benchmark()