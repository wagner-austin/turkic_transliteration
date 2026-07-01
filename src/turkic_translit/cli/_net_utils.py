"""Networking helpers shared by the CLI download commands.

Centralises the outbound HTTP User-Agent so every direct ``requests`` call
made by this package identifies itself to remote hosts. This is required by
Wikimedia's User-Agent policy (https://meta.wikimedia.org/wiki/User-Agent_policy),
which returns HTTP 403 for the default ``python-requests/<version>`` UA.
"""

from __future__ import annotations

import logging
from typing import Final

# Single source of truth for the outbound User-Agent. Any module in this
# package that hits an external HTTP endpoint MUST import and pass
# ``DEFAULT_HEADERS`` (or at least ``USER_AGENT``) so that policy-enforcing
# hosts do not reject the request.
USER_AGENT: Final[str] = (
    "turkic-translit/0.3.9 "
    "(+https://github.com/wagner-austin/turkic-transliteration; "
    "austinwagner@msn.com)"
)

DEFAULT_HEADERS: Final[dict[str, str]] = {"User-Agent": USER_AGENT}

_logger = logging.getLogger(__name__)


def url_ok(url: str, timeout: float = 5.0) -> bool:
    """Probe ``url`` and report whether the host answers with a 2xx/3xx.

    Sends an identifying User-Agent so hosts that enforce a UA policy
    (notably Wikimedia dumps) do not respond with HTTP 403. Falls back
    from HEAD to a streaming GET for hosts that reject HEAD (e.g. some
    Leipzig corpus mirrors).

    Args:
        url: Fully-qualified URL to probe.
        timeout: Per-request timeout in seconds.

    Returns:
        True when the host answered with a status code below 400;
        False when the host answered with an error code or the request
        raised a ``requests.RequestException`` (connection error,
        timeout, DNS failure, etc.).
    """
    import requests

    try:
        head_response = requests.head(
            url, allow_redirects=True, timeout=timeout, headers=DEFAULT_HEADERS
        )
    except requests.RequestException as exc:
        _logger.debug("HEAD %s failed: %s", url, exc)
        return False

    if head_response.status_code < 400:
        return True

    _logger.debug(
        "HEAD %s returned %d; retrying with streaming GET",
        url,
        head_response.status_code,
    )
    try:
        get_response = requests.get(
            url, stream=True, timeout=timeout, headers=DEFAULT_HEADERS
        )
    except requests.RequestException as exc:
        _logger.debug("GET %s failed: %s", url, exc)
        return False

    return get_response.status_code < 400
