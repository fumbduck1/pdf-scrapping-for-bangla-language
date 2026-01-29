"""Performance monitoring and profiling module"""
import time
import functools
import statistics
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PerformanceMetric:
    """Data class for performance metric"""
    name: str
    values: list = field(default_factory=list)
    units: str = "seconds"
    description: str = ""

    def add_value(self, value: float):
        """Add a new value to the metric"""
        self.values.append(value)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the metric"""
        if not self.values:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "median": None,
                "total": None
            }

        return {
            "count": len(self.values),
            "min": min(self.values),
            "max": max(self.values),
            "avg": statistics.mean(self.values),
            "median": statistics.median(self.values),
            "total": sum(self.values)
        }


class PerformanceMonitor:
    """Performance monitoring and profiling class"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._metrics: Dict[str, PerformanceMetric] = {}
        self._start_time = time.time()
        self._profiling_stack = []
        self._current_profile = None

    def register_metric(self, name: str, units: str = "seconds", description: str = ""):
        """Register a new performance metric"""
        if name not in self._metrics:
            self._metrics[name] = PerformanceMetric(name, units=units, description=description)

    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get a registered metric"""
        return self._metrics.get(name)

    def add_metric_value(self, name: str, value: float, units: str = "seconds"):
        """Add a value to a metric (create if doesn't exist)"""
        if name not in self._metrics:
            self.register_metric(name, units=units)
        self._metrics[name].add_value(value)

    def get_all_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get all registered metrics"""
        return self._metrics.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics and total time"""
        total_time = time.time() - self._start_time
        summary = {
            "total_time": total_time,
            "metrics": {}
        }

        for name, metric in self._metrics.items():
            summary["metrics"][name] = metric.get_summary()
            summary["metrics"][name]["units"] = metric.units
            summary["metrics"][name]["description"] = metric.description

        return summary

    def start_profiling(self, name: str):
        """Start profiling a section"""
        self._profiling_stack.append({
            "name": name,
            "start": time.time()
        })

    def stop_profiling(self, name: str) -> Optional[float]:
        """Stop profiling a section and record the time"""
        if not self._profiling_stack:
            return None

        # Find the matching profiling entry
        index = None
        for i, entry in enumerate(reversed(self._profiling_stack)):
            if entry["name"] == name:
                index = len(self._profiling_stack) - 1 - i
                break

        if index is None:
            return None

        entry = self._profiling_stack.pop(index)
        duration = time.time() - entry["start"]
        self.add_metric_value(name, duration)
        return duration

    def clear(self):
        """Clear all metrics and reset timer"""
        self._metrics.clear()
        self._start_time = time.time()
        self._profiling_stack.clear()
        self._current_profile = None

    def print_summary(self):
        """Print a formatted summary of performance metrics"""
        summary = self.get_summary()
        print(f"Performance Summary ({datetime.now().strftime('%H:%M:%S')})")
        print("=" * 50)
        print(f"Total time: {summary['total_time']:.2f} seconds")
        print("-" * 50)

        for name, metric in summary["metrics"].items():
            print(f"\n{name} ({metric['units']}):")
            print(f"  Count: {metric['count']}")
            print(f"  Min: {metric['min']:.2f}" if metric['min'] is not None else "  Min: -")
            print(f"  Max: {metric['max']:.2f}" if metric['max'] is not None else "  Max: -")
            print(f"  Avg: {metric['avg']:.2f}" if metric['avg'] is not None else "  Avg: -")
            print(f"  Median: {metric['median']:.2f}" if metric['median'] is not None else "  Median: -")
            print(f"  Total: {metric['total']:.2f}" if metric['total'] is not None else "  Total: -")


# Global instance
monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """Get singleton performance monitor instance"""
    return monitor


def timer(name: str = None, units: str = "seconds"):
    """Decorator for timing function calls"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metric_name = name or func.__name__
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                monitor.add_metric_value(metric_name, duration, units=units)
                return result
            except Exception as e:
                duration = time.time() - start
                monitor.add_metric_value(metric_name, duration, units=units)
                raise e

        return wrapper
    return decorator


def profile(section_name: str):
    """Context manager for profiling sections of code"""
    class ProfileContext:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            monitor.start_profiling(self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            monitor.stop_profiling(self.name)

    return ProfileContext(section_name)


def register_metrics():
    """Register common metrics used in the application"""
    monitor.register_metric("pdf_rendering", "seconds", "PDF page rendering time")
    monitor.register_metric("image_preprocessing", "seconds", "Image preprocessing time")
    monitor.register_metric("easyocr_pass", "seconds", "EasyOCR pass time")
    monitor.register_metric("tesseract_pass", "seconds", "Tesseract OCR pass time")
    monitor.register_metric("page_processing", "seconds", "Complete page processing time")
    monitor.register_metric("text_saving", "seconds", "Text file saving time")


def print_performance_report():
    """Print a comprehensive performance report"""
    monitor.print_summary()

    # Print detailed performance tips
    print("\n" + "=" * 50)
    print("Performance Tips")
    print("=" * 50)

    summary = monitor.get_summary()
    metrics = summary.get("metrics", {})

    if "easyocr_pass" in metrics and metrics["easyocr_pass"]["avg"] is not None and metrics["easyocr_pass"]["avg"] > 2.0:
        print("- EasyOCR passes are taking > 2 seconds. Consider GPU acceleration.")

    if "pdf_rendering" in metrics and metrics["pdf_rendering"]["avg"] is not None and metrics["pdf_rendering"]["avg"] > 0.5:
        print("- PDF rendering is taking > 0.5 seconds per page. Check Poppler installation.")

    if "page_processing" in metrics and metrics["page_processing"]["avg"] is not None and metrics["page_processing"]["avg"] > 5.0:
        print("- Page processing is taking > 5 seconds. Consider fast mode.")


if __name__ == "__main__":
    # Test the performance monitor
    register_metrics()

    with profile("test_section"):
        time.sleep(0.1)

    @timer("test_function")
    def test_func():
        time.sleep(0.05)

    for _ in range(5):
        test_func()

    print_performance_report()
