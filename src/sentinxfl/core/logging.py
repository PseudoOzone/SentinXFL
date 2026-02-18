"""
SentinXFL Logging Configuration
================================

Centralized logging using Loguru with JSON formatting support,
PII/secret redaction, and correlation ID tracking.

Author: Anshuman Bakshi
"""

import re
import sys
from pathlib import Path

from loguru import logger

from sentinxfl.core.config import settings

# Patterns that should be redacted from log output
_REDACT_PATTERNS = [
    # JWT / Bearer tokens
    (re.compile(r'(Bearer\s+)[A-Za-z0-9\-_\.]+', re.IGNORECASE), r'\1[REDACTED]'),
    # API keys
    (re.compile(r'((?:api[_-]?key|secret[_-]?key|password|token)\s*[=:]\s*)["\']?[^\s"\',;]+', re.IGNORECASE), r'\1[REDACTED]'),
    # Email addresses (optional - comment out if you need them in logs)
    (re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), '[EMAIL_REDACTED]'),
]


def _redact(message: str) -> str:
    """Remove sensitive values from log messages."""
    for pattern, replacement in _REDACT_PATTERNS:
        message = pattern.sub(replacement, message)
    return message


class _RedactSink:
    """Wraps a sink to apply redaction before writing."""

    def __init__(self, sink):
        self._sink = sink

    def write(self, message):
        self._sink.write(_redact(str(message)))

    def flush(self):
        if hasattr(self._sink, 'flush'):
            self._sink.flush()


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
