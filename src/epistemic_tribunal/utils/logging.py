"""Structured logging setup for Epistemic Tribunal.

Uses the ``rich`` library for console output when available, falling back to
the stdlib ``logging`` module with a plain formatter.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def get_logger(name: str = "epistemic_tribunal", level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger instance.

    Parameters
    ----------
    name:
        Logger name (usually ``__name__`` of the caller).
    level:
        Override the log level (default: reads ``LOG_LEVEL`` env var or ``INFO``).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    _level = level or os.environ.get("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, _level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    handler.setLevel(numeric_level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
