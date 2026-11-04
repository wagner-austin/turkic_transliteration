"""The log panel in the browser: a buffering handler and the filter it applies.

This was part of ``web/web_utils.py`` until 2026-11-04, when it moved here
unchanged so that each concern of the web helpers stays under the
600-line limit.
"""

from __future__ import annotations

import logging


class GradioLogHandler(logging.Handler):
    """Buffers log records so UI callbacks can flush them into the browser."""

    def __init__(self) -> None:
        """Start with an empty buffer."""
        super().__init__()
        self.buffer: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Add one formatted record to the buffer.

        Args:
            record: The record being logged.
        """
        self.buffer.append(self.format(record))

    def dump(self) -> str:
        """Take everything buffered so far, leaving the buffer empty.

        Returns:
            The buffered records as one newline-separated block.
        """
        out, self.buffer = "\n".join(self.buffer), []
        return out


class UiPrettyLogFilter(logging.Filter):
    """Skip verbose HTTP and housekeeping messages for UI logs."""

    _SKIP_PHRASES = (
        "HTTP Request:",
        "turkic_model.model not found",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether a record belongs in the browser's log.

        Args:
            record: The record being logged.

        Returns:
            False for the housekeeping messages that would bury the ones
            a user is actually watching for.
        """
        msg = record.getMessage()
        return not any(p in msg for p in self._SKIP_PHRASES)


_UI_LOG_HANDLER: GradioLogHandler | None = None


def get_ui_log_handler() -> GradioLogHandler:
    """Return the shared handler that feeds the browser's log panel.

    Returns:
        The handler, attached to the project logger on first call and
        reused afterwards so records are buffered in one place.
    """
    global _UI_LOG_HANDLER
    if _UI_LOG_HANDLER is None:
        h = GradioLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        h.addFilter(UiPrettyLogFilter())
        logging.getLogger("turkic_translit").addHandler(h)
        _UI_LOG_HANDLER = h
    return _UI_LOG_HANDLER


__all__ = ["GradioLogHandler", "UiPrettyLogFilter", "get_ui_log_handler"]
