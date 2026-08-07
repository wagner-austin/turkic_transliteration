"""Explicit, typed selection of language-identification models.

The corpora behind the Turkic mutual-intelligibility work were filtered
with NLLB's ``lid218e``, while the shipped CLI historically resolved only
fastText's ``lid.176``. Those two classifiers keep different lines at the
same probability threshold, so the published tool could not rebuild the
published corpora. This package makes the choice explicit, typed, and
recorded, so a filtered corpus names the classifier that produced it.
"""

from __future__ import annotations

from turkic_translit.lid.errors import (
    LidError,
    LidLabelError,
    LidModelFileEmptyError,
    LidModelFileMissingError,
    LidSpecFieldError,
    UnknownLidModelError,
)
from turkic_translit.lid.registry import (
    REGISTRY,
    get_spec,
    known_model_ids,
    resolve_model_path,
)
from turkic_translit.lid.spec import (
    LidModelSpec,
    decode_lid_model_spec,
    encode_lid_model_spec,
    strip_label_prefix,
)

__all__ = [
    "REGISTRY",
    "LidError",
    "LidLabelError",
    "LidModelFileEmptyError",
    "LidModelFileMissingError",
    "LidModelSpec",
    "LidSpecFieldError",
    "UnknownLidModelError",
    "decode_lid_model_spec",
    "encode_lid_model_spec",
    "get_spec",
    "known_model_ids",
    "resolve_model_path",
    "strip_label_prefix",
]
