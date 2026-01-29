"""Centralized logging system for PDF scraper"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class Logger:
    """Centralized logging manager"""
    
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
        self._logger = logging.getLogger("pdf_scraper")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        self._logger.addHandler(console_handler)
        
        # File handler
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"pdf_scraper_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)
    
    def get_logger(self):
        """Get the configured logger instance"""
        return self._logger
    
    def set_level(self, level: int):
        """Set the logging level"""
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)
    
    def set_console_level(self, level: int):
        """Set the console handler level"""
        for handler in self._logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)
    
    def set_file_level(self, level: int):
        """Set the file handler level"""
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
    
    def add_file_handler(self, file_path: str):
        """Add an additional file handler"""
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)


# Global logger instance
logger = Logger()


def get_logger():
    """Get the singleton logger instance"""
    return logger.get_logger()


def debug(msg: str, *args, **kwargs):
    """Log debug message"""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """Log info message"""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """Log warning message"""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """Log error message"""
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """Log critical message"""
    get_logger().critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs):
    """Log exception message"""
    get_logger().exception(msg, *args, **kwargs)


class LogCallback:
    """Wrapper class to adapt logging to callback interface"""
    
    def __init__(self, level: int = logging.INFO):
        self.level = level
    
    def __call__(self, message: str):
        """Call the logger with the appropriate level"""
        if self.level == logging.DEBUG:
            debug(message)
        elif self.level == logging.INFO:
            info(message)
        elif self.level == logging.WARNING:
            warning(message)
        elif self.level == logging.ERROR:
            error(message)
        elif self.level == logging.CRITICAL:
            critical(message)
        else:
            info(message)


def create_log_callback(level: int = logging.INFO) -> LogCallback:
    """Create a logging callback function"""
    return LogCallback(level)
