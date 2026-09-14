"""
Centralized logging configuration for the RAG application.
Provides structured, consistent console and rotating file logging.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_is_configured = False


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Initialize application-wide logging handlers.
    Safe to call multiple times (idempotent).
    """
    global _is_configured
    if _is_configured:
        return

    level_str = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, level_str, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if re-initialized
    root_logger.handlers.clear()

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. Rotating File Handler
    file_path_str = log_file or os.getenv("LOG_FILE", "logs/app.log")
    try:
        log_path = Path(file_path_str).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to console if file logging cannot be set up
        root_logger.warning(
            f"Failed to set up file logging at {file_path_str}: {e}"
        )

    # Suppress overly chatty third-party loggers
    for noisy_lib in ["urllib3", "httpx", "httpcore", "multipart"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    _is_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Factory function to get a named logger with configured handlers.
    """
    if not _is_configured:
        setup_logging()
    return logging.getLogger(name)
