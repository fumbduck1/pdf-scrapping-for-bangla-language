"""Centralized logging system for PDF scraper"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

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


# Global logger instance
logger = Logger()


def get_logger():
    """
    Return the configured singleton logger for the PDF scraper.
    
    Returns:
        logging.Logger: The configured singleton logger instance named "pdf_scraper".
    """
    return logger.get_logger()


def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a message with DEBUG severity using the module's configured logger.
    
    The message and any positional or keyword formatting arguments are forwarded to the underlying logger.
    """
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a message at INFO level using the module's configured logger.
    
    Parameters:
        msg (str): Message or format string to be logged.
        *args: Positional arguments used for formatting `msg`.
        **kwargs: Keyword arguments forwarded to the logger (for example, `exc_info`, `stacklevel`, or `extra`).
    """
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a warning-level message using the module's configured logger.
    
    Parameters:
        msg (str): Message format string or message to log.
        *args: Positional arguments used for %-style formatting of `msg`.
        **kwargs: Keyword arguments forwarded to the underlying logger.
    """
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a message at the ERROR level using the module's singleton logger.
    
    Parameters:
        msg (str): The log message or format string. Positional and keyword arguments are forwarded to the logger for formatting.
    """
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Emit a critical-level log message using the configured PDF scraper logger.
    
    Passes `msg` and any formatting or keyword arguments through to the logger's `critical` method.
    """
    get_logger().critical(msg, *args, **kwargs)


def exception(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a message at error level including the current exception's traceback.
    
    Parameters:
        msg (str): Message format string to be logged.
        *args: Positional arguments applied to the message format.
        **kwargs: Keyword arguments forwarded to the underlying logger; if `exc_info` is not provided, the current exception traceback is included.
    """
    get_logger().exception(msg, *args, **kwargs)