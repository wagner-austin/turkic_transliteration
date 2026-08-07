"""Construction of a ready classifier, and the record of what was used.

This is the layer a pipeline calls. It takes a model id and returns both
a working classifier and a :class:`LidRunRecord` describing exactly which
weights backed it, so the corpus that comes out can carry the identity of
the filter that produced it.

That record is the whole point. The corpora behind this project were
filtered by a classifier whose identity survived only in an ad-hoc
manifest, which is why they could not later be rebuilt from the released
tool. A run that writes its own filter identity cannot develop that gap.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from turkic_translit.lid import _test_hooks
from turkic_translit.lid.classifier import LidClassifier
from turkic_translit.lid.fetch import ensure_lid_model
from turkic_translit.lid.registry import get_spec


class LidRunRecord(TypedDict):
    """Identity of the classifier that filtered one corpus run.

    Attributes:
        model_id: Registry key of the model used, e.g. ``lid218e``.
        weights_path: Absolute path of the weights actually loaded.
        weights_bytes: Size of those weights, which distinguishes a
            complete model from a truncated one after the fact.
        threshold: Probability threshold applied to keep a line.
        script_aware: Whether the model's labels encode script.
    """

    model_id: str
    weights_path: str
    weights_bytes: int
    threshold: float
    script_aware: bool


def encode_lid_run_record(record: LidRunRecord) -> dict[str, str | int | float | bool]:
    """Render a run record as a plain mapping for manifest writing.

    Args:
        record: The record to encode.

    Returns:
        A mapping carrying exactly the five record fields.
    """
    return {
        "model_id": record["model_id"],
        "weights_path": record["weights_path"],
        "weights_bytes": record["weights_bytes"],
        "threshold": record["threshold"],
        "script_aware": record["script_aware"],
    }


def build_classifier(
    model_id: str,
    search_dirs: Sequence[Path],
    destination_dir: Path,
    threshold: float,
) -> tuple[LidClassifier, LidRunRecord]:
    """Build a classifier and the record describing what backs it.

    Args:
        model_id: Registry key naming the model to use.
        search_dirs: Directories to consult for existing weights.
        destination_dir: Directory to download into when absent.
        threshold: Probability threshold this run will apply, recorded so
            the filter is reproducible from the manifest alone.

    Returns:
        The ready classifier and its run record.

    Raises:
        UnknownLidModelError: If the model id is not registered.
        LidModelFileEmptyError: If the weights are or download as empty.
    """
    spec = get_spec(model_id)
    weights = ensure_lid_model(model_id, search_dirs, destination_dir)
    model = _test_hooks.model_loader.load(weights)
    record = LidRunRecord(
        model_id=spec["model_id"],
        weights_path=str(weights),
        weights_bytes=_test_hooks.probe.size_bytes(weights),
        threshold=threshold,
        script_aware=spec["script_aware"],
    )
    return LidClassifier(spec, model), record


__all__ = ["LidRunRecord", "build_classifier", "encode_lid_run_record"]
