"""
Centralized logging configuration for the backend.

This module provides a consistent logging setup across the entire application,
with proper formatting, rotation, and environment-aware configuration.
"""
from __future__ import annotations

import os
import logging
import logging.handlers
import sys
from typing import Optional
from pathlib import Path

from core.paths import get_logs_dir


def setup_logging(
    name: str = "refiner",
    level: Optional[int] = None,
    enable_console: Optional[bool] = None
) -> logging.Logger:
    """
    Set up and configure application logging.
    
    Args:
        name: Logger name (default: 'refiner')
        level: Log level override (None = auto-detect from environment)
        enable_console: Force console output (None = auto-detect)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Determine log level
    if level is None:
        debug_env = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
        level = logging.DEBUG if debug_env else logging.INFO
    
    logger.setLevel(level)
    
    # Suppress noisy third-party loggers
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"
    )
    
    # File handler with rotation (only in non-serverless environments)
    if not os.getenv("VERCEL"):  # Vercel doesn't support file writes
        try:
            log_dir = get_logs_dir()
            log_path = log_dir / "refiner.log"
            file_handler = logging.handlers.RotatingFileHandler(
                str(log_path),
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3,
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # File logging failed, continue with console only
            pass
    
    # Console handler
    if enable_console is None:
        enable_console = os.getenv("DEBUG", "").lower() in ("1", "true", "yes") or os.getenv("VERCEL")
    
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "refiner") -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger


