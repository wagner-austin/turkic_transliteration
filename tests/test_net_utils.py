"""Tests for :mod:`turkic_translit.cli._net_utils`.

Two categories:

* module-level constant checks — the User-Agent must actually identify
  the package (empty or generic ``python-requests`` UAs would violate
  Wikimedia's policy);
* live network probes — real HTTP requests against ``dumps.wikimedia.org``
  confirming that the shipped User-Agent is not on Wikimedia's blocklist.
  These are marked ``@pytest.mark.network`` so ``pytest -m "not network"``
  skips them when the test host is offline.
"""

from __future__ import annotations

import re

import pytest
import requests

from turkic_translit.cli._net_utils import DEFAULT_HEADERS, USER_AGENT, url_ok

_WIKIMEDIA_DUMPS_URL = "https://dumps.wikimedia.org/"


def test_user_agent_identifies_the_package() -> None:
    """The User-Agent must contain the project name and contact info.

    Wikimedia's User-Agent policy requires an identifying string. A UA
    that merely says ``python-requests/2.x`` is on their blocklist and
    returns HTTP 403. The regex below asserts both a project label and
    a contact hint (URL or email) are present.
    """
    assert re.match(r"^turkic-translit/", USER_AGENT), (
        f"User-Agent does not begin with the project name: {USER_AGENT!r}"
    )
    assert "@" in USER_AGENT or "github.com" in USER_AGENT, (
        f"User-Agent lacks a contact hint (email or URL): {USER_AGENT!r}"
    )


def test_default_headers_contain_the_user_agent() -> None:
    """``DEFAULT_HEADERS`` is the single-source dict callers must pass on."""
    assert DEFAULT_HEADERS.get("User-Agent") == USER_AGENT


@pytest.mark.network
def test_url_ok_true_for_wikimedia_dumps_with_ua() -> None:
    """``url_ok`` returns True for a Wikimedia URL because of the new UA.

    Regression guard: before this fix, ``url_ok`` used the default
    ``python-requests`` UA and Wikimedia returned HTTP 403, so this
    assertion would have failed.
    """
    assert url_ok(_WIKIMEDIA_DUMPS_URL) is True


@pytest.mark.network
def test_wikimedia_returns_200_when_ua_is_set() -> None:
    """Concretely: a HEAD request with ``DEFAULT_HEADERS`` gets HTTP 200.

    Distinct from the ``url_ok`` test above: this one exercises the
    ``requests`` call directly so a future refactor of ``url_ok``
    cannot mask a broken UA by short-circuiting on some other code
    path.
    """
    response = requests.head(
        _WIKIMEDIA_DUMPS_URL,
        allow_redirects=True,
        timeout=10.0,
        headers=DEFAULT_HEADERS,
    )
    assert response.status_code == 200, (
        f"Wikimedia rejected the request even with UA {USER_AGENT!r}; "
        f"status={response.status_code}"
    )


@pytest.mark.network
def test_wikimedia_blocks_default_python_requests_ua() -> None:
    """Companion negative test: bare ``python-requests`` UA is still blocked.

    Documents the invariant that motivated ``DEFAULT_HEADERS`` in the
    first place. If Wikimedia ever removes the block this test will
    fail and remind us to reconsider whether the UA header is still
    load-bearing.
    """
    response = requests.head(_WIKIMEDIA_DUMPS_URL, allow_redirects=True, timeout=10.0)
    assert response.status_code == 403, (
        "Wikimedia no longer blocks the default python-requests UA; "
        "DEFAULT_HEADERS may no longer be required."
    )


def test_url_ok_false_for_unresolvable_host() -> None:
    """``url_ok`` returns False when the host cannot be resolved.

    Exercises the ``RequestException`` branch without depending on the
    network — an invalid TLD returns immediately from the DNS layer.
    """
    assert url_ok("https://this-host-must-not-exist.invalid/") is False
