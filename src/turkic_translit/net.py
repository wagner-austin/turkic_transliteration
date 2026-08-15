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

from turkic_translit import __version__

# Read from the installed package rather than written out here. The
# version used to be a literal in this string, which meant a release bump
# left the agent announcing the previous one, and nothing could notice.
USER_AGENT: Final[str] = (
    f"turkic-translit/{__version__} "
    "(+https://github.com/wagner-austin/turkic-transliteration; "
    "austinwagner@msn.com)"
)

DEFAULT_HEADERS: Final[dict[str, str]] = {"User-Agent": USER_AGENT}


def build_request(url: str, method: str, token: str | None = None) -> Request:
    """Build a request carrying this project's User-Agent.

    Args:
        url: Fully-qualified URL to request. Any scheme ``urllib``
            handles is accepted, including ``file://``, which is how the
            adapters built on this are exercised without a network.
        method: HTTP method, e.g. ``GET`` or ``HEAD``. Ignored by
            non-HTTP handlers.
        token: Bearer credential to present, or ``None`` for the
            unauthenticated request that public data needs. OSCAR is
            gated, so its shards are refused without one, while the file
            listing naming those shards is served without one.

    Returns:
        A request with :data:`DEFAULT_HEADERS` applied, and an
        ``Authorization`` header when a token was given.
    """
    headers = dict(DEFAULT_HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return Request(url, headers=headers, method=method)


__all__ = ["DEFAULT_HEADERS", "USER_AGENT", "build_request"]
