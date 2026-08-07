"""Typed specification of a language-identification model.

A :class:`LidModelSpec` is the whole truth about one classifier: what it
is called, what file holds its weights, where that file comes from, how
its labels are shaped, and whether those labels distinguish script. The
mapping is treated as immutable; nothing in this package mutates a spec
after construction.

Decoding is total and strict. :func:`decode_lid_model_spec` accepts an
untyped mapping only after every field has cleared a check from
:mod:`turkic_translit.validation`, so a malformed registry entry fails at
the boundary rather than surfacing later as a confusing classification
result.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from turkic_translit.lid.errors import LidLabelError
from turkic_translit.validation import (
    require_bool,
    require_non_empty_str,
    require_present,
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


def decode_lid_model_spec(source: Mapping[str, str | bool]) -> LidModelSpec:
    """Validate a loosely-typed mapping into a :class:`LidModelSpec`.

    Args:
        source: Mapping holding the five specification fields.

    Returns:
        A fully validated specification.

    Raises:
        FieldError: If any field is missing, of the wrong type, or empty.
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
    "strip_label_prefix",
]
