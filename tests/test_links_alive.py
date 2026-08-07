"""Network-marked check that every registered corpus source is being served.

This is the only test that reaches the real hosts. It uses the same
health-check URL and the same probe the ``doctor`` command uses, so a
failure here and a red ``doctor`` mean the same thing.
"""

from __future__ import annotations

import pytest

from turkic_translit.corpus._test_hooks import UrlReachabilityProbe
from turkic_translit.corpus.catalogue import health_check_url
from turkic_translit.corpus.sources import SOURCE_REGISTRY


@pytest.mark.network
@pytest.mark.parametrize("source_id", list(SOURCE_REGISTRY))
def test_registered_source_is_reachable(source_id: str) -> None:
    """The host behind each registered source answers a HEAD request."""
    url = health_check_url(SOURCE_REGISTRY[source_id])
    assert UrlReachabilityProbe().reachable(url) is True, f"{source_id} at {url}"
