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
import sys
from functools import lru_cache

from pythonjsonlogger.json import JsonFormatter
from rich.logging import RichHandler

from turkic_translit import _test_hooks
from turkic_translit.error_service import CorrelationFilter, init_error_service

JSON_FORMAT = "json"
_JSON_FIELDS = "%(asctime)s %(name)s %(levelname)s %(message)s %(correlation_id)s"
_JSON_RENAMES = {"levelname": "level", "asctime": "time", "name": "logger"}


def default_level() -> str:
    """Return the log level the environment asks for.

    Args:
        None.

    Returns:
        The value of ``TURKIC_LOG_LEVEL`` upper-cased, or ``INFO``.
    """
    return (_test_hooks.environment.get("TURKIC_LOG_LEVEL") or "INFO").upper()


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
def setup(level: str) -> logging.Logger:
    """Configure the root logger once for this process.

    The level is passed in rather than read here, so a command with a
    ``--log-level`` flag does not have to write to the environment for
    this function to see it. Callers with no flag of their own pass
    :func:`default_level`.

    Args:
        level: Level name, e.g. ``DEBUG``. Anything unrecognised by the
            logging module falls to ``INFO``.

    Returns:
        The project logger, ``turkic_translit``.
    """
    root_logger = logging.getLogger()
    for existing in root_logger.handlers[:]:
        root_logger.removeHandler(existing)

    root_logger.setLevel(getattr(logging, level, logging.INFO))

    requested = _test_hooks.environment.get("TURKIC_LOG_FORMAT") or JSON_FORMAT
    handler = _json_handler() if requested.lower() == JSON_FORMAT else _rich_handler()
    handler.addFilter(CorrelationFilter())
    root_logger.addHandler(handler)

    init_error_service()

    logger = logging.getLogger("turkic_translit")
    logger.debug("Logging initialised at level %s", level)
    return logger


__all__ = ["JSON_FORMAT", "default_level", "setup"]
