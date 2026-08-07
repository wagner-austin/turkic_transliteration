"""Centralised logging configuration.

Policy:

- Configure logging once per entry point (CLI or web), never at import.
- ``TURKIC_LOG_LEVEL`` sets the level; default INFO.
- ``TURKIC_LOG_FORMAT`` selects ``json`` (default) or ``rich``.

Both formatters are declared runtime dependencies, so neither is
optional and neither is wrapped in a try/except. The previous version
had a three-deep fallback ladder — JSON, then Rich, then stdlib — for
imports that cannot fail, which meant a genuine formatter error would
have been silently downgraded to plain text instead of reported.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

from pythonjsonlogger.json import JsonFormatter
from rich.logging import RichHandler

from turkic_translit.error_service import CorrelationFilter, init_error_service

JSON_FORMAT = "json"
_JSON_FIELDS = "%(asctime)s %(name)s %(levelname)s %(message)s %(correlation_id)s"
_JSON_RENAMES = {"levelname": "level", "asctime": "time", "name": "logger"}


def _env_level() -> str:
    """Return the desired log level from the environment.

    Returns:
        The level name, upper-cased; INFO when unset.
    """
    return (os.environ.get("TURKIC_LOG_LEVEL") or "INFO").upper()


def _json_handler() -> logging.Handler:
    """Build a handler emitting one JSON object per record.

    Returns:
        A stderr handler with the structured formatter attached.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(_JSON_FIELDS, rename_fields=_JSON_RENAMES))
    return handler


def _rich_handler() -> logging.Handler:
    """Build a handler emitting colourised human-readable output.

    Returns:
        A Rich handler with a bare message formatter attached.
    """
    handler = RichHandler(rich_tracebacks=True, markup=True, show_time=False, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


@lru_cache(maxsize=1)
def setup() -> logging.Logger:
    """Configure the root logger once for this process.

    Returns:
        The project logger, ``turkic_translit``.
    """
    root_logger = logging.getLogger()
    for existing in root_logger.handlers[:]:
        root_logger.removeHandler(existing)

    level_name = _env_level()
    root_logger.setLevel(getattr(logging, level_name, logging.INFO))

    wants_json = (os.environ.get("TURKIC_LOG_FORMAT") or JSON_FORMAT).lower() == JSON_FORMAT
    handler = _json_handler() if wants_json else _rich_handler()
    handler.addFilter(CorrelationFilter())
    root_logger.addHandler(handler)

    init_error_service()

    logger = logging.getLogger("turkic_translit")
    logger.debug("Logging initialised at level %s", level_name)
    return logger


__all__ = ["JSON_FORMAT", "setup"]
