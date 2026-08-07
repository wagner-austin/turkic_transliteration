"""Outbound HTTP identity for every request this project makes.

Wikimedia's User-Agent policy returns HTTP 403 to the default
``python-urllib/<version>`` and ``python-requests/<version>`` agents, so
every outbound request must identify the project and a contact route.
Building requests through :func:`build_request` is what makes that
unconditional rather than remembered per call site.

See https://meta.wikimedia.org/wiki/User-Agent_policy.
"""

from __future__ import annotations

from typing import Final
from urllib.request import Request

USER_AGENT: Final[str] = (
    "turkic-translit/0.3.9 "
    "(+https://github.com/wagner-austin/turkic-transliteration; "
    "austinwagner@msn.com)"
)

DEFAULT_HEADERS: Final[dict[str, str]] = {"User-Agent": USER_AGENT}


def build_request(url: str, method: str) -> Request:
    """Build a request carrying this project's User-Agent.

    Args:
        url: Fully-qualified URL to request. Any scheme ``urllib``
            handles is accepted, including ``file://``, which is how the
            adapters built on this are exercised without a network.
        method: HTTP method, e.g. ``GET`` or ``HEAD``. Ignored by
            non-HTTP handlers.

    Returns:
        A request with :data:`DEFAULT_HEADERS` applied.
    """
    return Request(url, headers=DEFAULT_HEADERS, method=method)


__all__ = ["DEFAULT_HEADERS", "USER_AGENT", "build_request"]
