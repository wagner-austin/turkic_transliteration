"""Traceable error types for corpus acquisition.

Every failure raised by this package carries a stable, greppable code, so
a caller can tell "you named a source that does not exist" apart from
"the source exists but its host would not answer" without reading prose.
Codes are never reused or renumbered.
"""

from __future__ import annotations

from typing import Final

ERR_UNKNOWN_SOURCE: Final = "TURKIC_CORPUS_001_UNKNOWN_SOURCE"
ERR_STREAM_FAILED: Final = "TURKIC_CORPUS_002_STREAM_FAILED"


class CorpusError(Exception):
    """Base class for corpus-acquisition failures.

    Args:
        code: Stable error code from this module.
        message: Human-readable description naming the offending value.
    """

    def __init__(self, code: str, message: str) -> None:
        """Store the code and render ``code: message`` as the string form."""
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class UnknownCorpusSourceError(CorpusError):
    """Raised when a source id is not present in the registry."""

    def __init__(self, source_id: str, known: tuple[str, ...]) -> None:
        """Name the rejected id and list every id that would have worked.

        Args:
            source_id: The identifier the caller supplied.
            known: Every registered identifier, in registry order.
        """
        super().__init__(
            ERR_UNKNOWN_SOURCE,
            f"no corpus source registered under {source_id!r}; "
            f"known sources are {', '.join(known)}",
        )
        self.source_id = source_id
        self.known = known


class CorpusStreamError(CorpusError):
    """Raised when a source's host could not be read.

    Args:
        url: The location that failed.
        detail: The underlying transport failure, rendered as text.
    """

    def __init__(self, url: str, detail: str) -> None:
        """Name the URL and the transport failure behind it."""
        super().__init__(ERR_STREAM_FAILED, f"could not read {url}: {detail}")
        self.url = url
        self.detail = detail


__all__ = [
    "ERR_STREAM_FAILED",
    "ERR_UNKNOWN_SOURCE",
    "CorpusError",
    "CorpusStreamError",
    "UnknownCorpusSourceError",
]
