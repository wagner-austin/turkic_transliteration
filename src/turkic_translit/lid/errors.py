"""Traceable error types for language-identification model resolution.

Every failure in this package carries a stable, greppable code so that a
caller can distinguish "you named a model that does not exist" from "the
model you named is not on disk" without parsing prose. Codes are never
reused or renumbered.

Codes 004 through 006 covered field validation while this package
validated its own specifications. That job now belongs to
:mod:`turkic_translit.validation`, which reports ``TURKIC_FIELD_*``
codes, so the three numbers are retired rather than reassigned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

ERR_UNKNOWN_MODEL: Final = "TURKIC_LID_001_UNKNOWN_MODEL"
ERR_MODEL_FILE_MISSING: Final = "TURKIC_LID_002_MODEL_FILE_MISSING"
ERR_MODEL_FILE_EMPTY: Final = "TURKIC_LID_003_MODEL_FILE_EMPTY"
ERR_LABEL_MALFORMED: Final = "TURKIC_LID_007_LABEL_MALFORMED"
ERR_EMPTY_TEXT: Final = "TURKIC_LID_008_EMPTY_TEXT"
ERR_MULTILINE_TEXT: Final = "TURKIC_LID_009_MULTILINE_TEXT"


class LidError(Exception):
    """Base class for language-identification failures.

    Args:
        code: Stable error code from this module.
        message: Human-readable description naming the offending value.
    """

    def __init__(self, code: str, message: str) -> None:
        """Store the code and render ``code: message`` as the string form."""
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class UnknownLidModelError(LidError):
    """Raised when a model id is not present in the registry."""

    def __init__(self, model_id: str, known: tuple[str, ...]) -> None:
        """Name the rejected id and list every id that would have worked.

        Args:
            model_id: The identifier the caller supplied.
            known: Every registered identifier, in registry order.
        """
        super().__init__(
            ERR_UNKNOWN_MODEL,
            f"no language-identification model registered under {model_id!r}; "
            f"known models are {', '.join(known)}",
        )
        self.model_id = model_id
        self.known = known


class LidModelFileMissingError(LidError):
    """Raised when a registered model's weights are absent from disk."""

    def __init__(self, model_id: str, path: Path) -> None:
        """Name the model and the exact path that was checked.

        Args:
            model_id: The registered identifier being resolved.
            path: Absolute path where the weights were expected.
        """
        super().__init__(
            ERR_MODEL_FILE_MISSING,
            f"weights for model {model_id!r} are not present at {path}",
        )
        self.model_id = model_id
        self.path = path


class LidModelFileEmptyError(LidError):
    """Raised when a model file exists but holds no bytes."""

    def __init__(self, model_id: str, path: Path) -> None:
        """Name the model and the zero-length path.

        Args:
            model_id: The registered identifier being resolved.
            path: Absolute path of the empty file.
        """
        super().__init__(
            ERR_MODEL_FILE_EMPTY,
            f"weights for model {model_id!r} at {path} are zero bytes",
        )
        self.model_id = model_id
        self.path = path


class LidLabelError(LidError):
    """Raised when a classifier emits a label the model spec cannot parse."""

    def __init__(self, model_id: str, label: str, expected_prefix: str) -> None:
        """Name the model, the label, and the prefix that was required.

        Args:
            model_id: Registered identifier of the model that emitted it.
            label: The raw label string returned by the classifier.
            expected_prefix: Prefix the model's labels are declared to carry.
        """
        super().__init__(
            ERR_LABEL_MALFORMED,
            f"model {model_id!r} emitted label {label!r} which does not begin "
            f"with the declared prefix {expected_prefix!r}",
        )
        self.model_id = model_id
        self.label = label
        self.expected_prefix = expected_prefix


class EmptyClassificationTextError(LidError):
    """Raised when classification is attempted on text with no content."""

    def __init__(self) -> None:
        """Report that the caller must filter empty lines itself."""
        super().__init__(
            ERR_EMPTY_TEXT,
            "cannot classify text that is empty after stripping; the caller "
            "decides what an empty line means, so this is not defaulted to a "
            "sentinel language",
        )


class MultilineClassificationTextError(LidError):
    """Raised when classification is attempted on text spanning lines."""

    def __init__(self) -> None:
        """Report that a classifier reads exactly one line."""
        super().__init__(
            ERR_MULTILINE_TEXT,
            "cannot classify text containing a newline; fastText reads one "
            "line and would silently discard the rest, so the caller must "
            "split the text itself",
        )


__all__ = [
    "ERR_EMPTY_TEXT",
    "ERR_LABEL_MALFORMED",
    "ERR_MODEL_FILE_EMPTY",
    "ERR_MODEL_FILE_MISSING",
    "ERR_MULTILINE_TEXT",
    "ERR_UNKNOWN_MODEL",
    "EmptyClassificationTextError",
    "LidError",
    "LidLabelError",
    "LidModelFileEmptyError",
    "LidModelFileMissingError",
    "MultilineClassificationTextError",
    "UnknownLidModelError",
]
