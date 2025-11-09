"""
Error service utilities: correlation IDs, structured error responses, and
optional Sentry integration.

Usage:
- Call `init_error_service()` from entrypoints to enable Sentry (if DSN provided).
- Use `set_correlation_id()` at the start of each request/CLI run.
- Use `error_response(...)` to build standard error payloads.
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
import uuid
from typing import Any

# Per-execution/request correlation ID
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

# Optional request context payload (e.g., route, lang, user)
_request_ctx: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "request_ctx", default=None
)


def set_correlation_id(value: str | None = None) -> str:
    """Set correlation ID for the current context; returns the ID used."""
    cid = value or str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    cid = _correlation_id.get("")
    return cid or ""


def set_request_context(**fields: Any) -> None:
    base = _request_ctx.get(None) or {}
    ctx = base.copy()
    ctx.update(fields)
    _request_ctx.set(ctx)


def get_request_context() -> dict[str, Any]:
    ctx = _request_ctx.get(None) or {}
    return ctx.copy()


class CorrelationFilter(logging.Filter):
    """Inject correlation ID and request context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            record.correlation_id = get_correlation_id() or None
            # flatten selected context keys for convenience
            ctx = get_request_context()
            for k, v in ctx.items():
                # Avoid clobbering built-in LogRecord attributes
                if not hasattr(record, k):
                    setattr(record, k, v)
        except Exception:
            # Logging must never raise
            pass
        return True


def init_error_service() -> None:
    """Initialise external error backend (Sentry) if configured via env."""
    dsn = os.getenv("TURKIC_SENTRY_DSN")
    if not dsn:
        return
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            environment=os.getenv("TURKIC_ENV", "local"),
            traces_sample_rate=float(os.getenv("TURKIC_SENTRY_TRACES", "0")),
            release=os.getenv("TURKIC_RELEASE"),
        )
        logging.getLogger(__name__).info("Sentry initialised")
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("Sentry init failed: %s", exc)


def error_response(
    message: str,
    *,
    status: int = 500,
    code: str = "internal_error",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Standardised error payload for UI/HTTP-style responses."""
    return {
        "timestamp": int(time.time()),
        "status": status,
        "code": code,
        "message": message,
        "correlationId": get_correlation_id() or None,
        "details": details or {},
    }


def error_markdown(payload: dict[str, Any]) -> str:
    """Render a standard error payload as Markdown for the web UI."""
    cor = payload.get("correlationId") or ""
    det = payload.get("details") or {}
    lines = [
        "### ❌ Error",
        f"- Status: `{payload.get('status')}`",
        f"- Code: `{payload.get('code')}`",
        f"- Message: {payload.get('message')}",
    ]
    if cor:
        lines.append(f"- Correlation ID: `{cor}`")
    if det:
        lines.append(f"- Details: `{det}`")
    return "\n".join(lines)
