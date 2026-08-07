"""Typed specification of a language-identification model.

A :class:`LidModelSpec` is the whole truth about one classifier: what it
is called, what file holds its weights, where that file comes from, how
its labels are shaped, and whether those labels distinguish script. The
mapping is treated as immutable; nothing in this package mutates a spec
after construction.

Decoding is total and strict. :func:`decode_lid_model_spec` accepts an
untyped mapping only after every field has cleared a ``require_*`` check,
so a malformed registry entry fails at the boundary rather than surfacing
later as a confusing classification result.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from turkic_translit.lid.errors import (
    ERR_SPEC_FIELD_EMPTY,
    ERR_SPEC_FIELD_MISSING,
    ERR_SPEC_FIELD_TYPE,
    LidLabelError,
    LidSpecFieldError,
)


class LidModelSpec(TypedDict):
    """Everything needed to locate, fetch, and interpret one LID model.

    Attributes:
        model_id: Stable registry key, e.g. ``lid.176`` or ``lid218e``.
        filename: Basename of the weights file on disk.
        url: Canonical download location for the weights.
        label_prefix: Prefix every emitted label carries, e.g.
            ``__label__``.
        script_aware: Whether labels encode script as well as language,
            e.g. ``kaz_Cyrl`` rather than ``kk``.
    """

    model_id: str
    filename: str
    url: str
    label_prefix: str
    script_aware: bool


def require_non_empty_str(field: str, value: str | bool) -> str:
    """Return ``value`` as a non-empty ``str`` or raise.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated string.

    Raises:
        LidSpecFieldError: If the value is not a string, or is empty or
            whitespace-only.
    """
    if not isinstance(value, str):
        raise LidSpecFieldError(
            ERR_SPEC_FIELD_TYPE, field, f"expected str, got {type(value).__name__}"
        )
    if value.strip() == "":
        raise LidSpecFieldError(ERR_SPEC_FIELD_EMPTY, field, "must not be empty")
    return value


def require_bool(field: str, value: str | bool) -> bool:
    """Return ``value`` as a ``bool`` or raise.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated boolean.

    Raises:
        LidSpecFieldError: If the value is not a ``bool``.
    """
    if not isinstance(value, bool):
        raise LidSpecFieldError(
            ERR_SPEC_FIELD_TYPE, field, f"expected bool, got {type(value).__name__}"
        )
    return value


def require_present(field: str, source: Mapping[str, str | bool]) -> str | bool:
    """Return ``source[field]`` or raise when the key is absent.

    Args:
        field: Field name to look up.
        source: Loosely-typed mapping being decoded.

    Returns:
        The raw value stored under ``field``.

    Raises:
        LidSpecFieldError: If the key is not present.
    """
    if field not in source:
        raise LidSpecFieldError(ERR_SPEC_FIELD_MISSING, field, "is required")
    return source[field]


def decode_lid_model_spec(source: Mapping[str, str | bool]) -> LidModelSpec:
    """Validate a loosely-typed mapping into a :class:`LidModelSpec`.

    Args:
        source: Mapping holding the five specification fields.

    Returns:
        A fully validated specification.

    Raises:
        LidSpecFieldError: If any field is missing, of the wrong type, or
            empty.
    """
    return LidModelSpec(
        model_id=require_non_empty_str("model_id", require_present("model_id", source)),
        filename=require_non_empty_str("filename", require_present("filename", source)),
        url=require_non_empty_str("url", require_present("url", source)),
        label_prefix=require_non_empty_str("label_prefix", require_present("label_prefix", source)),
        script_aware=require_bool("script_aware", require_present("script_aware", source)),
    )


def encode_lid_model_spec(spec: LidModelSpec) -> dict[str, str | bool]:
    """Render a specification back to a plain mapping.

    The inverse of :func:`decode_lid_model_spec`, used for manifest
    writing and for round-trip assertions in tests.

    Args:
        spec: The specification to encode.

    Returns:
        A mapping carrying exactly the five specification fields.
    """
    return {
        "model_id": spec["model_id"],
        "filename": spec["filename"],
        "url": spec["url"],
        "label_prefix": spec["label_prefix"],
        "script_aware": spec["script_aware"],
    }


def strip_label_prefix(spec: LidModelSpec, label: str) -> str:
    """Remove the model's declared label prefix from one raw label.

    Args:
        spec: Specification of the model that produced the label.
        label: Raw label string as returned by the classifier.

    Returns:
        The label with its prefix removed, e.g. ``kaz_Cyrl`` or ``kk``.

    Raises:
        LidLabelError: If the label does not carry the declared prefix,
            which means the loaded weights are not the declared model.
    """
    prefix = spec["label_prefix"]
    if not label.startswith(prefix):
        raise LidLabelError(spec["model_id"], label, prefix)
    return label[len(prefix) :]


__all__ = [
    "LidModelSpec",
    "decode_lid_model_spec",
    "encode_lid_model_spec",
    "require_bool",
    "require_non_empty_str",
    "require_present",
    "strip_label_prefix",
]
