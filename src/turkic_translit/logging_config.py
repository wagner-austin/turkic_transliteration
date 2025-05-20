"""
Centralized logging configuration module.
Uses Rich for colorized output if available with fallback to standard library.
"""

import logging
import os
import sys
from functools import lru_cache

# Get log level from environment or default to INFO
LOG_LEVEL = os.environ.get("TURKIC_LOG_LEVEL", "INFO").upper()


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
    log_level = getattr(logging, LOG_LEVEL, logging.INFO)
    root_logger.setLevel(log_level)

    # Try to use Rich for pretty, colorized output
    try:
        from rich.logging import RichHandler

        # Configure Rich handler with appropriate settings
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=False,
            show_path=False,
        )
        formatter = logging.Formatter("%(message)s")

    except ImportError:
        # Fall back to standard logging if Rich is not available
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Configure and add the handler
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    logger = logging.getLogger("turkic_translit")
    logger.debug(f"Logging initialized at level {LOG_LEVEL}")

    return logger
