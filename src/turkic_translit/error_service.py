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
import time
import uuid

from turkic_translit import _test_hooks

logger = logging.getLogger(__name__)

# Per-execution/request correlation ID
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")

# Optional request context payload (e.g., route, lang, user)
_request_ctx: contextvars.ContextVar[
    dict[str, str | int | float | bool | None | list[str]] | None
] = contextvars.ContextVar("request_ctx", default=None)


def set_correlation_id(value: str | None = None) -> str:
    """Set the correlation ID for the current context.

    Args:
        value: Identifier to adopt. ``None`` or an empty string mints a
            fresh one, so a caller that has nothing to propagate still
            gets a run it can be identified by.

    Returns:
        The identifier now in force.
    """
    cid = value or str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Return the correlation ID in force for this context.

    Returns:
        The identifier, or an empty string when none was set.
    """
    cid = _correlation_id.get("")
    return cid or ""


def set_request_context(**fields: str | int | float | bool | None | list[str]) -> None:
    """Add fields to the context attached to every later log record.

    Fields accumulate rather than replace, so one action can describe
    itself in stages as it learns more.

    Args:
        **fields: Values to record, by name.
    """
    base = _request_ctx.get(None) or {}
    ctx = base.copy()
    ctx.update(fields)
    _request_ctx.set(ctx)


def get_request_context() -> dict[str, str | int | float | bool | None | list[str]]:
    """Return the context fields recorded for this action.

    Returns:
        A copy of the fields, so a caller cannot alter what later log
        records will carry.
    """
    ctx = _request_ctx.get(None) or {}
    return ctx.copy()


class CorrelationFilter(logging.Filter):
    """Inject correlation ID and request context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach the correlation ID and request context to ``record``.

        Args:
            record: The record being emitted.

        Returns:
            True always; this filter annotates rather than excludes.
        """
        record.correlation_id = get_correlation_id() or None
        for key, value in get_request_context().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def init_error_service() -> None:
    """Initialise the external error backend when one is configured.

    No DSN means no backend was asked for, which is the ordinary case and
    not a failure. A DSN that is set is a request, and it is carried out
    or it raises: the previous version caught the missing-package case
    and warned, which left an operator who had configured reporting with
    a log line and no reporting.

    Raises:
        ImportError: If a DSN is configured but the backend's package is
            not installed. Install the ``sentry`` extra, or unset
            ``TURKIC_SENTRY_DSN``.
    """
    dsn = _test_hooks.environment.get("TURKIC_SENTRY_DSN")
    if not dsn:
        return

    _test_hooks.reporter.initialise(
        dsn=dsn,
        environment=_test_hooks.environment.get("TURKIC_ENV") or "local",
        traces_sample_rate=float(_test_hooks.environment.get("TURKIC_SENTRY_TRACES") or "0"),
        release=_test_hooks.environment.get("TURKIC_RELEASE"),
    )
    logger.info("error reporting initialised")


def error_response(
    message: str,
    *,
    status: int = 500,
    code: str = "internal_error",
    details: dict[str, str | int | float | bool | None | list[str]] | None = None,
) -> dict[
    str,
    str
    | int
    | float
    | bool
    | None
    | list[str]
    | dict[str, str | int | float | bool | None | list[str]],
]:
    """Build the error payload every UI surface reports failures with.

    Args:
        message: Human-readable description of what went wrong.
        status: HTTP-style status code.
        code: Stable machine-readable identifier for the failure.
        details: Extra fields describing the failure, or ``None``.

    Returns:
        The payload, carrying the correlation ID of the run that
        produced it so a report can be traced back to its logs.
    """
    return {
        "timestamp": int(time.time()),
        "status": status,
        "code": code,
        "message": message,
        "correlationId": get_correlation_id() or None,
        "details": details or {},
    }


def error_markdown(
    payload: dict[
        str,
        str
        | int
        | float
        | bool
        | None
        | list[str]
        | dict[str, str | int | float | bool | None | list[str]],
    ],
) -> str:
    """Render an error payload as Markdown for the web UI.

    Args:
        payload: A payload from :func:`error_response`.

    Returns:
        The Markdown block. The correlation ID and the details are
        omitted when absent rather than rendered empty.
    """
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
