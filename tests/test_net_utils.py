"""Tests for the project's outbound HTTP identity.

Wikimedia answers HTTP 403 to the default ``python-urllib`` agent, so the
User-Agent is not cosmetic: without it the Wikipedia driver cannot fetch
a dump at all. The offline tests pin the header onto every request; the
network-marked test confirms the live policy still accepts it.
"""

from __future__ import annotations

import re
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from turkic_translit.net import DEFAULT_HEADERS, USER_AGENT, build_request

WIKIMEDIA_DUMPS_URL = "https://dumps.wikimedia.org/"


def test_user_agent_names_the_project_and_a_contact() -> None:
    """The agent identifies the project and how to reach its author."""
    assert re.match(r"^turkic-translit/", USER_AGENT)
    assert "github.com" in USER_AGENT


def test_default_headers_are_the_single_source_of_the_agent() -> None:
    """Callers pass one dict rather than each spelling the agent out."""
    assert DEFAULT_HEADERS == {"User-Agent": USER_AGENT}


def test_build_request_applies_the_agent_and_the_method() -> None:
    """Every request built here carries the agent and the method asked for."""
    request = build_request("https://example.invalid/dump.bz2", "HEAD")
    assert request.get_header("User-agent") == USER_AGENT
    assert request.get_method() == "HEAD"


def test_build_request_preserves_the_url_unchanged() -> None:
    """The URL is passed through so query strings survive intact."""
    url = "https://meta.wikimedia.org/w/api.php?action=sitematrix&format=json"
    assert build_request(url, "GET").full_url == url


@pytest.mark.network
def test_wikimedia_accepts_the_project_agent() -> None:
    """The live dump host answers a request carrying this agent.

    Regression guard: with the default urllib agent this host answers
    403, which surfaced as an empty corpus rather than as an error.
    """
    with urlopen(build_request(WIKIMEDIA_DUMPS_URL, "HEAD"), timeout=30) as response:
        assert response.status == 200


@pytest.mark.network
def test_wikimedia_still_rejects_an_unidentified_agent() -> None:
    """The policy that motivated the agent is still being enforced.

    If this ever stops raising, the User-Agent requirement has been
    relaxed upstream and the header is no longer load-bearing.
    """
    from urllib.request import Request

    with pytest.raises(HTTPError) as excinfo:
        urlopen(Request(WIKIMEDIA_DUMPS_URL, method="HEAD"), timeout=30)
    assert excinfo.value.code == 403
