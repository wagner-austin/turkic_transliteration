"""
Centralized logging configuration.

Policy:
- Configure logging once per entrypoint (CLI/web), not at import-time.
- Level is controlled via TURKIC_LOG_LEVEL (DEBUG, INFO, ...); default INFO.
- Uses Rich for colorized output when available; falls back to stdlib.
"""

import logging
import os
import sys
from functools import lru_cache

from .error_service import CorrelationFilter, init_error_service


def _env_level() -> str:
    """Return desired log level from env (default: INFO)."""
    return (os.environ.get("TURKIC_LOG_LEVEL") or "INFO").upper()


@lru_cache(maxsize=1)
def setup() -> logging.Logger:
    """
    Set up logging with Rich if available, with fallback to stdlib logging.
    Uses TURKIC_LOG_LEVEL environment variable or defaults to INFO.

    Uses @lru_cache to ensure this is only run once.
    """
    root_logger = logging.getLogger()

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set the log level based on environment variable
    lvl_str = _env_level()
    log_level = getattr(logging, lvl_str, logging.INFO)
    root_logger.setLevel(log_level)

    # Choose structured JSON logging when available; fallback to Rich or stdlib
    fmt_pref = (os.environ.get("TURKIC_LOG_FORMAT") or "json").lower()
    handler = logging.StreamHandler(sys.stderr)
    formatter: logging.Formatter
    if fmt_pref == "json":
        try:
            from pythonjsonlogger import jsonlogger

            formatter = jsonlogger.JsonFormatter(  # type: ignore[attr-defined]
                "%(asctime)s %(name)s %(levelname)s %(message)s %(correlation_id)s",
                rename_fields={
                    "levelname": "level",
                    "asctime": "time",
                    "name": "logger",
                },
            )
        except Exception:
            try:
                from rich.logging import RichHandler

                handler = RichHandler(
                    rich_tracebacks=True, markup=True, show_time=False, show_path=False
                )
                formatter = logging.Formatter("%(message)s")
            except Exception:
                formatter = logging.Formatter("%(levelname)s: %(message)s")
    else:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                rich_tracebacks=True, markup=True, show_time=False, show_path=False
            )
            formatter = logging.Formatter("%(message)s")
        except Exception:
            formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Configure and add the handler
    handler.setFormatter(formatter)
    handler.addFilter(CorrelationFilter())
    root_logger.addHandler(handler)

    # Initialise optional error backend (e.g., Sentry) if configured
    init_error_service()

    logger = logging.getLogger("turkic_translit")
    logger.debug(f"Logging initialized at level {lvl_str}")

    return logger
