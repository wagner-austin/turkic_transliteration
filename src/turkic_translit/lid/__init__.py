"""Explicit, typed selection of language-identification models.

The corpora behind the Turkic mutual-intelligibility work were filtered
with NLLB's ``lid218e``, while the shipped CLI historically resolved only
fastText's ``lid.176``. Those two classifiers keep different lines at the
same probability threshold, so the published tool could not rebuild the
published corpora. This package makes the choice explicit, typed, and
recorded, so a filtered corpus names the classifier that produced it.
"""

from __future__ import annotations

from turkic_translit.lid.classifier import (
    FastTextModel,
    LidClassifier,
    LidPrediction,
    encode_lid_prediction,
)
from turkic_translit.lid.errors import (
    EmptyClassificationTextError,
    LidError,
    LidLabelError,
    LidModelFileEmptyError,
    LidModelFileMissingError,
    LidSpecFieldError,
    UnknownLidModelError,
)
from turkic_translit.lid.factory import (
    LidRunRecord,
    build_classifier,
    encode_lid_run_record,
)
from turkic_translit.lid.fetch import ensure_lid_model
from turkic_translit.lid.registry import (
    REGISTRY,
    find_model_path,
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
    "EmptyClassificationTextError",
    "FastTextModel",
    "LidClassifier",
    "LidError",
    "LidLabelError",
    "LidModelFileEmptyError",
    "LidModelFileMissingError",
    "LidModelSpec",
    "LidPrediction",
    "LidRunRecord",
    "LidSpecFieldError",
    "UnknownLidModelError",
    "build_classifier",
    "decode_lid_model_spec",
    "encode_lid_model_spec",
    "encode_lid_prediction",
    "encode_lid_run_record",
    "ensure_lid_model",
    "find_model_path",
    "get_spec",
    "known_model_ids",
    "resolve_model_path",
    "strip_label_prefix",
]
