"""Registry of language-identification models and their resolution.

Two classifiers are registered. ``lid.176`` is fastText's 176-language
model, whose labels name a language but not a script. ``lid218e`` is
NLLB's 218-language model, whose labels name both, which is what lets it
separate Latin-script from Cyrillic-script Uzbek.

Selection is explicit and total. There is no preference order and no
fallback: a caller names the model it wants, and if those weights are not
present the call fails with a code naming the model and the path. This is
deliberate. A pipeline that silently substitutes one classifier for
another produces a corpus that cannot be reproduced, because the filter
that built it is no longer recoverable from the recorded configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from turkic_translit.lid import _test_hooks
from turkic_translit.lid.errors import (
    LidModelFileEmptyError,
    LidModelFileMissingError,
    UnknownLidModelError,
)
from turkic_translit.lid.spec import LidModelSpec, decode_lid_model_spec

_RAW_SPECS: Final[tuple[Mapping[str, str | bool], ...]] = (
    {
        "model_id": "lid.176",
        "filename": "lid.176.bin",
        "url": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        "label_prefix": "__label__",
        "script_aware": False,
    },
    {
        "model_id": "lid218e",
        "filename": "lid218e.bin",
        "url": "https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin",
        "label_prefix": "__label__",
        "script_aware": True,
    },
)

REGISTRY: Final[Mapping[str, LidModelSpec]] = {
    spec["model_id"]: spec for spec in (decode_lid_model_spec(raw) for raw in _RAW_SPECS)
}


def known_model_ids() -> tuple[str, ...]:
    """List every registered model identifier in registry order.

    Returns:
        The registered identifiers.
    """
    return tuple(REGISTRY)


def get_spec(model_id: str) -> LidModelSpec:
    """Look up one model specification by identifier.

    Args:
        model_id: Registry key, e.g. ``lid218e``.

    Returns:
        The registered specification.

    Raises:
        UnknownLidModelError: If no model is registered under that id.
    """
    spec = REGISTRY.get(model_id)
    if spec is None:
        raise UnknownLidModelError(model_id, known_model_ids())
    return spec


def resolve_model_path(model_id: str, search_dirs: Sequence[Path]) -> Path:
    """Find the weights file for one model across candidate directories.

    Directories are searched in order and the first hit wins. No other
    model's weights are ever substituted, and a present-but-empty file is
    an error rather than a cache miss, because a truncated download that
    silently re-downloads is how a filter changes underneath a corpus.

    Args:
        model_id: Registry key naming the model to resolve.
        search_dirs: Directories to search, in priority order.

    Returns:
        Absolute path to the weights file.

    Raises:
        UnknownLidModelError: If the model id is not registered.
        LidModelFileMissingError: If no candidate directory holds the file.
        LidModelFileEmptyError: If the file is found but has zero bytes.
    """
    spec = get_spec(model_id)
    filename = spec["filename"]
    searched: list[Path] = []

    for directory in search_dirs:
        candidate = directory / filename
        searched.append(candidate)
        if not _test_hooks.probe.exists(candidate):
            continue
        if _test_hooks.probe.size_bytes(candidate) == 0:
            raise LidModelFileEmptyError(model_id, candidate)
        return candidate

    raise LidModelFileMissingError(model_id, searched[-1] if searched else Path(filename))


__all__ = ["REGISTRY", "get_spec", "known_model_ids", "resolve_model_path"]
