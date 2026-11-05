"""Answering "what can I ask for" and "is it up" per source.

Both questions are driver-specific and both reach the network, so they
live behind the same seam as the drivers themselves. Keeping them here
rather than in the command-line layer means the interface never touches
the private hook module, and the dispatch on ``driver`` exists in exactly
one place per question.
"""

from __future__ import annotations

from turkic_translit.corpus import _test_hooks, hub
from turkic_translit.corpus.drivers import WIKIPEDIA_DUMP_HOST
from turkic_translit.corpus.sources import OscarSourceSpec, WikipediaSourceSpec


def available_languages(
    spec: OscarSourceSpec | WikipediaSourceSpec,
) -> tuple[str, ...]:
    """List the language codes a source can be asked for.

    Args:
        spec: The source to enumerate.

    Returns:
        Language codes, sorted. For a Hugging Face dataset these are its
        configuration names; for Wikipedia they are the editions that are
        open.
    """
    if spec["driver"] == "oscar":
        return _test_hooks.languages.oscar_configurations(spec["hf_name"])
    return _test_hooks.languages.wikipedia_editions()


def health_check_url(spec: OscarSourceSpec | WikipediaSourceSpec) -> str:
    """Name the URL that shows whether a source is being served.

    Args:
        spec: The source to probe.

    Returns:
        A URL that answers cheaply, without downloading corpus data.
    """
    if spec["driver"] == "oscar":
        return hub.dataset_api_url(spec["hf_name"])
    return WIKIPEDIA_DUMP_HOST


def unreachable_reason(spec: OscarSourceSpec | WikipediaSourceSpec) -> str | None:
    """Report why a source's host is not answering, if it is not.

    Args:
        spec: The source to probe.

    Returns:
        ``None`` when the host answered the health-check request, else
        the reason it did not.
    """
    return _test_hooks.reachability.unreachable_reason(health_check_url(spec))


__all__ = [
    "available_languages",
    "health_check_url",
    "unreachable_reason",
]
