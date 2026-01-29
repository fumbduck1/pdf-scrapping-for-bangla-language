"""Analyze performance bottlenecks using the new monitoring system"""
import time
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import create_job_config
from scraper import run_pdf_job
from performance import get_monitor, print_performance_report


def analyze_performance():
    """Analyze performance bottlenecks on a test PDF"""
    # Test PDF path - replace with your own test file
    test_pdf_path = r"C:\Users\akram\Downloads\sample.pdf"
    output_dir = r"c:\Users\akram\Desktop\pdf scrapper ongoing\performance_analysis"
    
    print("Starting performance analysis...")
    print(f"PDF Path: {test_pdf_path}")
    print(f"Output Dir: {output_dir}")
    print("-" * 50)
    
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
    
    # Reset the performance monitor
    monitor = get_monitor()
    monitor.clear()
    
    # Run the job
    start_time = time.time()
    result = run_pdf_job(config, stop_event=None, log_cb=print)
    total_time = time.time() - start_time
    
    print("\n" + "-" * 50)
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if result:
        print(f"Pages processed: {result.get('stats', {}).get('total_pages', 0)}")
        print(f"Characters OCR'd: {result.get('stats', {}).get('total_ocr_characters', 0)}")
        print(f"Save status: {'OK' if result.get('save_ok') else 'Failed'}")
    
    # Print detailed performance report
    print("\n" + "-" * 50)
    print_performance_report()
    
    # Analyze bottlenecks
    print("\n" + "-" * 50)
    print("Bottleneck Analysis:")
    summary = monitor.get_summary()
    metrics = summary.get("metrics", {})
    
    avg_easyocr = metrics.get("easyocr_pass", {}).get("avg", 0)
    avg_render = metrics.get("pdf_rendering", {}).get("avg", 0)
    avg_page = metrics.get("page_processing", {}).get("avg", 0)
    
    if "easyocr_pass" in metrics and avg_easyocr > 2.0:
        print(f"⚠️  EasyOCR passes are taking {avg_easyocr:.2f} seconds on average - consider GPU acceleration")
    
    if "pdf_rendering" in metrics and avg_render > 0.5:
        print(f"⚠️  PDF rendering is taking {avg_render:.2f} seconds on average - check Poppler installation")
    
    if "page_processing" in metrics and avg_page > 5.0:
        print(f"⚠️  Page processing is taking {avg_page:.2f} seconds on average - consider fast mode")
    
    # Print suggestions for optimization
    print("\n" + "-" * 50)
    print("Optimization Suggestions:")
    
    if avg_easyocr > 2.0:
        print("- Enable GPU acceleration for EasyOCR")
        print("- Try using Tesseract as the primary OCR engine")
        print("- Reduce OCR language complexity (e.g., use 'ben' instead of 'ben+eng')")
    
    if avg_render > 0.5:
        print("- Check Poppler installation and performance")
        print("- Try reducing the render zoom level")
        print("- Increase the PDF bytes cache size")
    
    if avg_page > 5.0:
        print("- Enable fast mode")
        print("- Reduce quality mode")
        print("- Increase the number of workers")
        print("- Check if your system has enough resources")
    
    return True


if __name__ == "__main__":
    # Check if user provided a test PDF path
    if len(sys.argv) > 1:
        test_pdf_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "performance_analysis"
        
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
        
        monitor = get_monitor()
        monitor.clear()
        
        start_time = time.time()
        result = run_pdf_job(config, stop_event=None, log_cb=print)
        total_time = time.time() - start_time
        
        print_performance_report()
    else:
        # Run with default test file
        analyze_performance()