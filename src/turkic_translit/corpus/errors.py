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
ERR_SYMBOL_MAP_MALFORMED: Final = "TURKIC_CORPUS_003_SYMBOL_MAP_MALFORMED"
ERR_NO_CORPORA: Final = "TURKIC_CORPUS_005_NO_CORPORA"


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


class SymbolMapMalformedError(CorpusError):
    """Raised when a symbol map cannot be read as a table of decisions.

    Args:
        origin: Name of the CSV.
        detail: What about it could not be read.
    """

    def __init__(self, origin: str, detail: str) -> None:
        """Name the file and the way it is malformed."""
        super().__init__(
            ERR_SYMBOL_MAP_MALFORMED,
            f"{origin} is not a readable symbol map: {detail}",
        )
        self.origin = origin
        self.detail = detail


class NoCorporaError(CorpusError):
    """Raised when a cleaning run is given nothing to clean.

    Args:
        directory: Where corpora were looked for.
    """

    def __init__(self, directory: str) -> None:
        """Name the directory that held no corpus."""
        super().__init__(
            ERR_NO_CORPORA,
            f"no corpus to clean in {directory}; equalising a corpus against "
            f"an empty set of others has no meaning",
        )
        self.directory = directory


__all__ = [
    "ERR_NO_CORPORA",
    "ERR_STREAM_FAILED",
    "ERR_SYMBOL_MAP_MALFORMED",
    "ERR_UNKNOWN_SOURCE",
    "CorpusError",
    "CorpusStreamError",
    "NoCorporaError",
    "SymbolMapMalformedError",
    "UnknownCorpusSourceError",
]
