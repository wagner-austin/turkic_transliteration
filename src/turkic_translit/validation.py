"""Boundary validation for loosely-typed mappings.

Every decoder in this package funnels through these checks, so a value
read from YAML, from JSON, or from a command line is proven to hold the
type its ``TypedDict`` declares before anything downstream sees it. There
is one implementation of each check: a package that needs a new field
type adds a function here rather than repeating an ``isinstance`` test at
its own boundary.

The parameter type is deliberately the widest value a decoded document in
this project can hold. Narrower mappings are accepted without adaptation
because ``Mapping`` is covariant in its value type, so a caller decoding
``Mapping[str, str | bool]`` passes these functions unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

ERR_FIELD_MISSING: Final = "TURKIC_FIELD_001_MISSING"
ERR_FIELD_TYPE: Final = "TURKIC_FIELD_002_TYPE"
ERR_FIELD_EMPTY: Final = "TURKIC_FIELD_003_EMPTY"
ERR_FIELD_RANGE: Final = "TURKIC_FIELD_004_RANGE"


class FieldError(Exception):
    """Raised when a field of a decoded mapping fails validation.

    Args:
        code: One of the ``ERR_FIELD_*`` codes in this module.
        field: Name of the offending field within its mapping.
        detail: Why the value was rejected, naming what was expected.
    """

    def __init__(self, code: str, field: str, detail: str) -> None:
        """Store the code and field, rendering ``code: field: detail``."""
        super().__init__(f"{code}: field {field!r}: {detail}")
        self.code = code
        self.field = field
        self.detail = detail


def require_present(
    field: str,
    source: Mapping[str, str | int | float | bool | None | Mapping[str, str | int | float | bool]],
) -> str | int | float | bool | None | Mapping[str, str | int | float | bool]:
    """Return ``source[field]`` or raise when the key is absent.

    A key holding ``None`` is present; only a missing key is an error.
    Optional fields are therefore explicit in the encoded document rather
    than inferred from absence.

    Args:
        field: Field name to look up.
        source: Loosely-typed mapping being decoded.

    Returns:
        The raw value stored under ``field``.

    Raises:
        FieldError: If the key is not present in the mapping.
    """
    if field not in source:
        raise FieldError(ERR_FIELD_MISSING, field, "is required")
    return source[field]


def require_str(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> str:
    """Return ``value`` as a ``str`` or raise.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated string, which may be empty.

    Raises:
        FieldError: If the value is not a string.
    """
    if not isinstance(value, str):
        raise FieldError(ERR_FIELD_TYPE, field, f"expected str, got {type(value).__name__}")
    return value


def require_non_empty_str(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> str:
    """Return ``value`` as a ``str`` with content, or raise.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated string.

    Raises:
        FieldError: If the value is not a string, or is empty or
            whitespace-only.
    """
    text = require_str(field, value)
    if text.strip() == "":
        raise FieldError(ERR_FIELD_EMPTY, field, "must not be empty")
    return text


def require_optional_non_empty_str(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> str | None:
    """Return ``value`` as a non-empty ``str``, passing ``None`` through.

    Args:
        field: Field name, used in the error message.
        value: Candidate value, permitted to be ``None``.

    Returns:
        The validated string, or ``None`` when the field is null.

    Raises:
        FieldError: If the value is neither ``None`` nor a non-empty
            string.
    """
    if value is None:
        return None
    return require_non_empty_str(field, value)


def require_bool(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> bool:
    """Return ``value`` as a ``bool`` or raise.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated boolean.

    Raises:
        FieldError: If the value is not a ``bool``.
    """
    if not isinstance(value, bool):
        raise FieldError(ERR_FIELD_TYPE, field, f"expected bool, got {type(value).__name__}")
    return value


def require_int(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> int:
    """Return ``value`` as an ``int`` or raise.

    ``bool`` is rejected even though it is a subclass of ``int``, because
    a boolean landing in a count field means the document is wrong.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated integer.

    Raises:
        FieldError: If the value is a ``bool`` or is not an ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise FieldError(ERR_FIELD_TYPE, field, f"expected int, got {type(value).__name__}")
    return value


def require_non_negative_int(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> int:
    """Return ``value`` as an ``int`` of zero or more, or raise.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated non-negative integer.

    Raises:
        FieldError: If the value is not an ``int``, or is negative.
    """
    number = require_int(field, value)
    if number < 0:
        raise FieldError(ERR_FIELD_RANGE, field, f"must not be negative, got {number}")
    return number


def require_float(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> float:
    """Return ``value`` as a ``float`` or raise.

    An ``int`` is accepted and widened, because a threshold written as
    ``1`` in JSON is the same number as ``1.0``. ``bool`` is rejected.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated value as a ``float``.

    Raises:
        FieldError: If the value is a ``bool`` or is neither ``int`` nor
            ``float``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FieldError(ERR_FIELD_TYPE, field, f"expected float, got {type(value).__name__}")
    return float(value)


def require_probability(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> float:
    """Return ``value`` as a ``float`` within the unit interval, or raise.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated probability, from 0.0 to 1.0 inclusive.

    Raises:
        FieldError: If the value is not a number, or lies outside the
            unit interval.
    """
    number = require_float(field, value)
    if number < 0.0 or number > 1.0:
        raise FieldError(ERR_FIELD_RANGE, field, f"must lie between 0.0 and 1.0, got {number}")
    return number


def require_one_of(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
    permitted: tuple[str, ...],
) -> str:
    """Return ``value`` as a ``str`` drawn from ``permitted``, or raise.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.
        permitted: Every accepted value, listed in the error message so a
            caller learns the whole vocabulary from one failure.

    Returns:
        The validated string.

    Raises:
        FieldError: If the value is not a string, or is not one of the
            permitted values.
    """
    text = require_str(field, value)
    if text not in permitted:
        raise FieldError(
            ERR_FIELD_TYPE,
            field,
            f"expected one of {', '.join(permitted)}, got {text!r}",
        )
    return text


def require_mapping(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> Mapping[str, str | int | float | bool]:
    """Return ``value`` as a nested mapping or raise.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated nested mapping, ready to be decoded in turn.

    Raises:
        FieldError: If the value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise FieldError(ERR_FIELD_TYPE, field, f"expected a mapping, got {type(value).__name__}")
    return value


def require_optional_mapping(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> Mapping[str, str | int | float | bool] | None:
    """Return ``value`` as a nested mapping, passing ``None`` through.

    Args:
        field: Field name, used in the error message.
        value: Candidate value, permitted to be ``None``.

    Returns:
        The validated nested mapping, or ``None`` when the field is null.

    Raises:
        FieldError: If the value is neither ``None`` nor a mapping.
    """
    if value is None:
        return None
    return require_mapping(field, value)


def require_absent(
    field: str,
    source: Mapping[str, str | int | float | bool | None | Mapping[str, str | int | float | bool]],
    reason: str,
) -> None:
    """Raise when ``field`` is present, reporting why it must not be.

    Used where one field's value forbids another, so a document carrying
    a contradictory pair fails at the boundary instead of having the
    surplus field silently ignored.

    Args:
        field: Field name that must not appear.
        source: Loosely-typed mapping being decoded.
        reason: Why this field is forbidden in this document.

    Raises:
        FieldError: If the key is present in the mapping.
    """
    if field in source:
        raise FieldError(ERR_FIELD_TYPE, field, f"must not be present: {reason}")


__all__ = [
    "ERR_FIELD_EMPTY",
    "ERR_FIELD_MISSING",
    "ERR_FIELD_RANGE",
    "ERR_FIELD_TYPE",
    "FieldError",
    "require_absent",
    "require_bool",
    "require_float",
    "require_int",
    "require_mapping",
    "require_non_empty_str",
    "require_non_negative_int",
    "require_one_of",
    "require_optional_mapping",
    "require_optional_non_empty_str",
    "require_present",
    "require_probability",
    "require_str",
]
