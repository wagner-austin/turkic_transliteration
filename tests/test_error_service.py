"""Tests for correlation IDs, request context and error rendering.

The external reporting backend is bound to a real implementation of the
production protocol that records its configuration, and the real Sentry
reporter is exercised separately against a DSN that resolves nowhere.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from turkic_translit import _test_hooks
from turkic_translit.error_service import (
    CorrelationFilter,
    error_markdown,
    error_response,
    get_correlation_id,
    get_request_context,
    init_error_service,
    set_correlation_id,
    set_request_context,
)

INERT_DSN = "https://0123456789abcdef0123456789abcdef@o0.ingest.example.invalid/0"


@pytest.fixture
def reporter() -> Iterator[_test_hooks.RecordingErrorReporter]:
    """Bind a reporter that records its configuration and starts nothing.

    Yields:
        The reporter the code under test configured.
    """
    previous = _test_hooks.reporter
    recording = _test_hooks.RecordingErrorReporter()
    _test_hooks.reporter = recording
    yield recording
    _test_hooks.reporter = previous


def bind_environment(**values: str) -> _test_hooks.Environment:
    """Bind an environment defining exactly the given variables.

    Args:
        **values: Variables the environment reports.

    Returns:
        The environment that was replaced, so a test can restore it.
    """
    previous = _test_hooks.environment
    _test_hooks.environment = _test_hooks.MappingEnvironment(values)
    return previous


def record(**fields: str) -> logging.LogRecord:
    """Build a log record carrying the given attributes.

    Args:
        **fields: Attributes to set on the record.

    Returns:
        A record ready to be passed through a filter.
    """
    built = logging.LogRecord("t", logging.INFO, "p", 1, "m", None, None)
    for key, value in fields.items():
        setattr(built, key, value)
    return built


def test_an_explicit_correlation_id_is_used_as_given() -> None:
    """A caller that supplies an ID gets that ID back."""
    assert set_correlation_id("abc-123") == "abc-123"
    assert get_correlation_id() == "abc-123"


def test_an_absent_correlation_id_is_generated() -> None:
    """With nothing supplied a fresh identifier is minted."""
    first = set_correlation_id(None)
    second = set_correlation_id(None)

    assert first != second
    assert len(first) == 36


def test_an_empty_correlation_id_is_generated() -> None:
    """An empty string is treated as no identifier, not as one."""
    assert set_correlation_id("") != ""


def test_request_context_accumulates_across_calls() -> None:
    """Later fields join earlier ones rather than replacing them."""
    set_request_context(action="download")
    set_request_context(lang="kk")

    context = get_request_context()

    assert context["action"] == "download"
    assert context["lang"] == "kk"


def test_request_context_is_returned_as_a_copy() -> None:
    """Mutating the returned mapping does not alter the stored one."""
    set_request_context(action="download")

    taken = get_request_context()
    taken["action"] = "tampered"

    assert get_request_context()["action"] == "download"


def test_the_filter_attaches_the_correlation_id() -> None:
    """Every record carries the ID in force when it was emitted."""
    set_correlation_id("cid-1")
    emitted = record()

    assert CorrelationFilter().filter(emitted) is True
    assert vars(emitted)["correlation_id"] == "cid-1"


def test_the_filter_attaches_the_request_context() -> None:
    """Context fields become record attributes."""
    set_correlation_id("cid-2")
    set_request_context(action="mask_russian", thr=0.5)
    emitted = record()

    CorrelationFilter().filter(emitted)

    assert vars(emitted)["action"] == "mask_russian"
    assert vars(emitted)["thr"] == 0.5


def test_the_filter_leaves_an_existing_attribute_alone() -> None:
    """A record that already names a field keeps its own value."""
    set_correlation_id("cid-3")
    set_request_context(action="from_context")
    emitted = record(action="from_record")

    CorrelationFilter().filter(emitted)

    assert vars(emitted)["action"] == "from_record"


def test_no_dsn_starts_no_reporting(reporter: _test_hooks.RecordingErrorReporter) -> None:
    """Reporting is opt-in, so an unset DSN configures nothing.

    Args:
        reporter: The bound reporter, which must stay untouched.
    """
    previous = bind_environment()
    try:
        init_error_service()
    finally:
        _test_hooks.environment = previous

    assert reporter.initialised == []


def test_an_empty_dsn_starts_no_reporting(reporter: _test_hooks.RecordingErrorReporter) -> None:
    """A DSN set to the empty string asks for nothing.

    Args:
        reporter: The bound reporter, which must stay untouched.
    """
    previous = bind_environment(TURKIC_SENTRY_DSN="")
    try:
        init_error_service()
    finally:
        _test_hooks.environment = previous

    assert reporter.initialised == []


def test_a_dsn_configures_the_reporter(reporter: _test_hooks.RecordingErrorReporter) -> None:
    """Every configured value reaches the backend unchanged.

    Args:
        reporter: The bound reporter.
    """
    previous = bind_environment(
        TURKIC_SENTRY_DSN=INERT_DSN,
        TURKIC_ENV="staging",
        TURKIC_SENTRY_TRACES="0.25",
        TURKIC_RELEASE="0.3.9",
    )
    try:
        init_error_service()
    finally:
        _test_hooks.environment = previous

    assert reporter.initialised == [(INERT_DSN, "staging", 0.25, "0.3.9")]


def test_the_unset_reporting_fields_take_their_defaults(
    reporter: _test_hooks.RecordingErrorReporter,
) -> None:
    """A bare DSN reports the local environment and traces nothing.

    Args:
        reporter: The bound reporter.
    """
    previous = bind_environment(TURKIC_SENTRY_DSN=INERT_DSN)
    try:
        init_error_service()
    finally:
        _test_hooks.environment = previous

    assert reporter.initialised == [(INERT_DSN, "local", 0.0, None)]


def test_the_production_reporter_initialises_sentry() -> None:
    """The reporter production binds starts the real SDK."""
    sentry = __import__("sentry_sdk")

    _test_hooks.SentryReporter().initialise(INERT_DSN, "test", 0.0, None)

    assert sentry.get_client().is_active() is True
    sentry.get_client().close()


def test_an_error_payload_carries_the_correlation_id() -> None:
    """The payload names the run that produced it."""
    set_correlation_id("cid-4")

    payload = error_response("boom", status=503, code="unavailable", details={"retry": 5})

    assert payload["status"] == 503
    assert payload["code"] == "unavailable"
    assert payload["message"] == "boom"
    assert payload["correlationId"] == "cid-4"
    assert payload["details"] == {"retry": 5}


def test_an_error_payload_takes_its_defaults() -> None:
    """An unclassified failure is a 500 with no details."""
    set_correlation_id("cid-5")

    payload = error_response("boom")

    assert payload["status"] == 500
    assert payload["code"] == "internal_error"
    assert payload["details"] == {}


def test_an_error_payload_reports_an_absent_correlation_id_as_null() -> None:
    """A run with no ID reports null rather than an empty string."""
    set_correlation_id("")
    from turkic_translit import error_service

    error_service._correlation_id.set("")

    payload = error_response("boom")

    assert payload["correlationId"] is None


def test_the_rendered_error_names_every_field() -> None:
    """A fully populated payload renders all five lines."""
    set_correlation_id("cid-6")
    rendered = error_markdown(error_response("boom", status=400, code="bad", details={"x": 1}))

    assert "- Status: `400`" in rendered
    assert "- Code: `bad`" in rendered
    assert "- Message: boom" in rendered
    assert "- Correlation ID: `cid-6`" in rendered
    assert "- Details: `{'x': 1}`" in rendered


def test_the_rendered_error_omits_what_is_absent() -> None:
    """A payload with no ID and no details renders neither line."""
    rendered = error_markdown(
        {"status": 500, "code": "internal_error", "message": "boom", "correlationId": None}
    )

    assert "Correlation ID" not in rendered
    assert "Details" not in rendered
    assert rendered.startswith("### ❌ Error")
