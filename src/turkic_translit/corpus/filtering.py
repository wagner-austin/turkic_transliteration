"""Deciding which lines a corpus run keeps.

The decision is expressed as a :class:`LineFilter` with two real
implementations, rather than as an optional classifier the download loop
has to test for on every line. "No filter requested" is then a filter
that keeps everything, not a null to branch on, which keeps the loop free
of a second meaning for ``None``.

Requesting a filter is a boundary crossing: the model id, the language,
and the threshold arrive from a command line as loose values. They are
therefore decoded and validated into a :class:`LidFilterRequest` before
any weights are loaded, so a mistyped threshold fails before a gigabyte
of model is read from disk.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypedDict

from turkic_translit.lid.classifier import LidClassifier
from turkic_translit.lid.factory import LidRunRecord, build_classifier
from turkic_translit.lid.locations import (
    default_destination_dir,
    default_search_dirs,
)
from turkic_translit.validation import (
    require_non_empty_str,
    require_present,
    require_probability,
)


class LidFilterRequest(TypedDict):
    """A request to keep only lines a classifier assigns to one language.

    Attributes:
        language: Label the classifier must return, matched as a prefix
            so that ``uzn`` accepts the script-aware ``uzn_Latn``.
        model_id: Registry key of the classifier to use, e.g.
            ``lid218e``.
        threshold: Minimum probability a line must reach to be kept.
    """

    language: str
    model_id: str
    threshold: float


def encode_lid_filter_request(request: LidFilterRequest) -> dict[str, str | float]:
    """Render a filter request as a plain mapping.

    Args:
        request: The request to encode.

    Returns:
        A mapping carrying exactly the three request fields.
    """
    return {
        "language": request["language"],
        "model_id": request["model_id"],
        "threshold": request["threshold"],
    }


def decode_lid_filter_request(
    source: Mapping[str, str | int | float | bool],
) -> LidFilterRequest:
    """Validate a loosely-typed mapping into a :class:`LidFilterRequest`.

    Args:
        source: Mapping holding the three request fields, ordinarily
            assembled from command-line options.

    Returns:
        A fully validated request.

    Raises:
        FieldError: If a field is missing, of the wrong type, empty, or —
            for the threshold — outside the unit interval.
    """
    return LidFilterRequest(
        language=require_non_empty_str("language", require_present("language", source)),
        model_id=require_non_empty_str("model_id", require_present("model_id", source)),
        threshold=require_probability("threshold", require_present("threshold", source)),
    )


class LineFilter(Protocol):
    """A decision about whether one line belongs in the output."""

    def keeps(self, text: str) -> bool:
        """Report whether ``text`` should be written.

        Args:
            text: A normalised, non-empty line.

        Returns:
            True when the line belongs in the corpus.
        """
        ...


class KeepEveryLine:
    """The filter used when no language filter was requested.

    This is the honest representation of an unfiltered run: the manifest
    records ``filter_language`` as null, and the download loop still
    consults exactly one filter object.
    """

    def keeps(self, text: str) -> bool:
        """Keep the line.

        Args:
            text: A normalised, non-empty line.

        Returns:
            Always True.
        """
        return True


class LanguageLineFilter:
    """A filter binding a classifier to one language and threshold.

    Args:
        classifier: The loaded classifier making the decision.
        language: Label the classifier must return, matched as a prefix.
        threshold: Minimum probability required to keep a line.
    """

    def __init__(self, classifier: LidClassifier, language: str, threshold: float) -> None:
        """Bind the classifier to the language and threshold it applies."""
        self._classifier = classifier
        self._language = language
        self._threshold = threshold

    def keeps(self, text: str) -> bool:
        """Report whether the classifier assigns ``text`` to the language.

        Args:
            text: A normalised, non-empty line.

        Returns:
            True when the top label starts with the requested language
            and its probability meets the threshold.
        """
        return self._classifier.accepts(text, self._language, self._threshold)


def build_line_filter(
    request: LidFilterRequest | None,
) -> tuple[LineFilter, LidRunRecord | None]:
    """Build the filter a run will apply, and the record naming it.

    Args:
        request: The filter to apply, or ``None`` for an unfiltered run.

    Returns:
        The filter, and the record identifying the weights behind it —
        ``None`` when no classifier was loaded, because there is then no
        model identity to record.

    Raises:
        UnknownLidModelError: If the request names an unregistered model.
        LidModelFileEmptyError: If the model's weights are zero bytes.
    """
    if request is None:
        return KeepEveryLine(), None
    classifier, record = build_classifier(
        request["model_id"],
        default_search_dirs(),
        default_destination_dir(),
        request["threshold"],
    )
    return (
        LanguageLineFilter(classifier, request["language"], request["threshold"]),
        record,
    )


__all__ = [
    "KeepEveryLine",
    "LanguageLineFilter",
    "LidFilterRequest",
    "LineFilter",
    "build_line_filter",
    "decode_lid_filter_request",
    "encode_lid_filter_request",
]
