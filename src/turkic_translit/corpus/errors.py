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
ERR_RECORD_MISSING: Final = "TURKIC_CORPUS_006_TRANSLITERATION_RECORD_MISSING"
ERR_RECORD_LANGUAGE: Final = "TURKIC_CORPUS_007_TRANSLITERATION_RECORD_LANGUAGE"
ERR_RECORD_RULES: Final = "TURKIC_CORPUS_008_TRANSLITERATED_BY_OTHER_RULES"
ERR_RECORD_OUTPUT: Final = "TURKIC_CORPUS_009_TRANSLITERATION_OUTPUT_CHANGED"
ERR_FILTER_KEPT_NOTHING: Final = "TURKIC_CORPUS_010_FILTER_KEPT_NOTHING"


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


class TransliterationRecordMissingError(CorpusError):
    """Raised when an IPA file carries no record of the rules that wrote it.

    Args:
        corpus: The IPA file.
        record: Where its record should have been.
    """

    def __init__(self, corpus: str, record: str) -> None:
        """Name the file and the record it lacks."""
        super().__init__(
            ERR_RECORD_MISSING,
            f"{corpus} has no transliteration record at {record}; nothing says which "
            f"rules wrote it, so text the installed rules cannot produce would be "
            f"dropped as foreign. Transliterate the source text with "
            f"'turkic-translit translit --out-ipa <file>', which writes the record",
        )
        self.corpus = corpus
        self.record = record


class TransliterationRecordLanguageError(CorpusError):
    """Raised when a record names another language than its file's name.

    Args:
        corpus: The IPA file.
        named: The language the file's name carries.
        recorded: The language its record names.
    """

    def __init__(self, corpus: str, named: str, recorded: str) -> None:
        """Name the file and both languages."""
        super().__init__(
            ERR_RECORD_LANGUAGE,
            f"{corpus} is named as {named} but its record says the {recorded} rules wrote it",
        )
        self.corpus = corpus
        self.named = named
        self.recorded = recorded


class TransliteratedByOtherRulesError(CorpusError):
    """Raised when an IPA file was written by rules other than the installed ones.

    Args:
        corpus: The IPA file.
        recorded: The rule file and digest its record names.
        installed: The installed rule file and its digest.
    """

    def __init__(self, corpus: str, recorded: str, installed: str) -> None:
        """Name the file and both rule files with their digests."""
        super().__init__(
            ERR_RECORD_RULES,
            f"{corpus} was written by {recorded}, and the installed rules are "
            f"{installed}; cleaning it against these rules would drop every word "
            f"the other rules spelled differently. Transliterate the source text "
            f"again with the installed rules",
        )
        self.corpus = corpus
        self.recorded = recorded
        self.installed = installed


class TransliterationOutputChangedError(CorpusError):
    """Raised when an IPA file is not the bytes its record describes.

    Args:
        corpus: The IPA file.
        recorded: The SHA-256 its record names.
        found: The SHA-256 of the file as it is.
    """

    def __init__(self, corpus: str, recorded: str, found: str) -> None:
        """Name the file and both digests."""
        super().__init__(
            ERR_RECORD_OUTPUT,
            f"{corpus} has SHA-256 {found}, and its transliteration record "
            f"describes {recorded}; the file changed after the rules wrote it",
        )
        self.corpus = corpus
        self.recorded = recorded
        self.found = found


class FilterKeptNothingError(CorpusError):
    """Raised when a language filter keeps none of the lines a source yielded.

    Args:
        source_id: Registry key of the source streamed.
        language: The language the filter required.
        model_id: The classifier that decided.
        lines_seen: Lines the source yielded, every one of them rejected.
        output: The corpus path, removed because it would be empty.
    """

    def __init__(
        self, source_id: str, language: str, model_id: str, lines_seen: int, output: str
    ) -> None:
        """Name the source, the filter and how many lines it rejected."""
        super().__init__(
            ERR_FILTER_KEPT_NOTHING,
            f"{model_id} kept none of {lines_seen:,} lines from {source_id} as "
            f"{language!r}; an empty corpus is not a download, so {output} was "
            f"removed and no manifest written",
        )
        self.source_id = source_id
        self.language = language
        self.model_id = model_id
        self.lines_seen = lines_seen
        self.output = output


__all__ = [
    "ERR_FILTER_KEPT_NOTHING",
    "ERR_NO_CORPORA",
    "ERR_RECORD_LANGUAGE",
    "ERR_RECORD_MISSING",
    "ERR_RECORD_OUTPUT",
    "ERR_RECORD_RULES",
    "ERR_STREAM_FAILED",
    "ERR_SYMBOL_MAP_MALFORMED",
    "ERR_UNKNOWN_SOURCE",
    "CorpusError",
    "CorpusStreamError",
    "FilterKeptNothingError",
    "NoCorporaError",
    "SymbolMapMalformedError",
    "TransliteratedByOtherRulesError",
    "TransliterationOutputChangedError",
    "TransliterationRecordLanguageError",
    "TransliterationRecordMissingError",
    "UnknownCorpusSourceError",
]
