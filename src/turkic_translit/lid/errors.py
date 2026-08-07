"""Traceable error types for language-identification model resolution.

Every failure in this package carries a stable, greppable code so that a
caller can distinguish "you named a model that does not exist" from "the
model you named is not on disk" without parsing prose. Codes are never
reused or renumbered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

ERR_UNKNOWN_MODEL: Final = "TURKIC_LID_001_UNKNOWN_MODEL"
ERR_MODEL_FILE_MISSING: Final = "TURKIC_LID_002_MODEL_FILE_MISSING"
ERR_MODEL_FILE_EMPTY: Final = "TURKIC_LID_003_MODEL_FILE_EMPTY"
ERR_SPEC_FIELD_MISSING: Final = "TURKIC_LID_004_SPEC_FIELD_MISSING"
ERR_SPEC_FIELD_TYPE: Final = "TURKIC_LID_005_SPEC_FIELD_TYPE"
ERR_SPEC_FIELD_EMPTY: Final = "TURKIC_LID_006_SPEC_FIELD_EMPTY"
ERR_LABEL_MALFORMED: Final = "TURKIC_LID_007_LABEL_MALFORMED"
ERR_EMPTY_TEXT: Final = "TURKIC_LID_008_EMPTY_TEXT"


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


class LidSpecFieldError(LidError):
    """Raised when a decoded specification field fails validation."""

    def __init__(self, code: str, field: str, detail: str) -> None:
        """Name the field and why it was rejected.

        Args:
            code: One of the ``ERR_SPEC_FIELD_*`` codes.
            field: Field name within the specification mapping.
            detail: Why the value was rejected.
        """
        super().__init__(code, f"specification field {field!r}: {detail}")
        self.field = field


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


__all__ = [
    "ERR_EMPTY_TEXT",
    "ERR_LABEL_MALFORMED",
    "ERR_MODEL_FILE_EMPTY",
    "ERR_MODEL_FILE_MISSING",
    "ERR_SPEC_FIELD_EMPTY",
    "ERR_SPEC_FIELD_MISSING",
    "ERR_SPEC_FIELD_TYPE",
    "ERR_UNKNOWN_MODEL",
    "EmptyClassificationTextError",
    "LidError",
    "LidLabelError",
    "LidModelFileEmptyError",
    "LidModelFileMissingError",
    "LidSpecFieldError",
    "UnknownLidModelError",
]
