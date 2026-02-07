"""
SentinXFL Logging Configuration
================================

Centralized logging using Loguru with JSON formatting support.

Author: Anshuman Bakshi
"""

import sys
from pathlib import Path

from loguru import logger

from sentinxfl.core.config import settings


def setup_logging() -> None:
    """Configure application logging with Loguru."""
    # Remove default handler
    logger.remove()

    # Console handler
    log_format_console = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format_console,
        level=settings.log_level,
        colorize=True,
    )

    # File handler with rotation
    log_file = settings.get_absolute_path(settings.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if settings.log_format == "json":
        logger.add(
            str(log_file),
            format="{message}",
            level=settings.log_level,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            serialize=True,
            compression="gz",
        )
    else:
        log_format_file = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        )
        logger.add(
            str(log_file),
            format=log_format_file,
            level=settings.log_level,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            compression="gz",
        )

    logger.info(
        f"Logging configured: level={settings.log_level}, "
        f"format={settings.log_format}, file={log_file}"
    )


def get_logger(name: str) -> "logger":
    """Get a contextualized logger instance."""
    return logger.bind(name=name)


# Module-level logger
log = get_logger("sentinxfl")
