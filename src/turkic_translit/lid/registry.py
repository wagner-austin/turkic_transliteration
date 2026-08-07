"""Registry of language-identification models and their resolution.

Two classifiers are registered. ``lid.176`` is fastText's 176-language
model, whose labels name a language but not a script. ``lid218e`` is
NLLB's 218-language model, whose labels name both.

Script-awareness is not why the choice matters, and it is worth writing
down what was actually measured so the next reader does not assume it.
Over a 25,676-line OSCAR Uzbek slice, filtering at ``p >= 0.95`` keeps
14,207 lines (55.3%) under ``lid218e`` and 3,215 (12.5%) under
``lid.176``: a 4.4x difference in surviving corpus. That gap is entirely
a Latin-script confidence effect, 58.0% against 13.1%. On the Cyrillic
subset (1,172 lines) neither model is usable at all, keeping 0.4% and
0.0% respectively; both label Cyrillic Uzbek predominantly Russian, and
``uzb_Cyrl`` never appears in ``lid218e``'s output. A pipeline that
sources Uzbek from OSCAR through either classifier therefore feeds
almost nothing to a Cyrillic-script rule file.

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


def find_model_path(model_id: str, search_dirs: Sequence[Path]) -> Path | None:
    """Locate a model's weights, reporting absence as ``None``.

    This is the single lookup both :func:`resolve_model_path` and
    :func:`turkic_translit.lid.fetch.ensure_lid_model` are built on, so
    neither has to drive control flow off an exception. Directories are
    searched in order and the first hit wins. No other model's weights
    are ever substituted.

    Args:
        model_id: Registry key naming the model to locate.
        search_dirs: Directories to search, in priority order.

    Returns:
        Absolute path to the weights, or ``None`` if no directory holds
        them.

    Raises:
        UnknownLidModelError: If the model id is not registered.
        LidModelFileEmptyError: If the file is found but has zero bytes.
            A truncated download is an error rather than a cache miss,
            because silently re-fetching is how a filter changes
            underneath a corpus.
    """
    spec = get_spec(model_id)
    for directory in search_dirs:
        candidate = directory / spec["filename"]
        if not _test_hooks.probe.exists(candidate):
            continue
        if _test_hooks.probe.size_bytes(candidate) == 0:
            raise LidModelFileEmptyError(model_id, candidate)
        return candidate
    return None


def resolve_model_path(model_id: str, search_dirs: Sequence[Path]) -> Path:
    """Locate a model's weights, treating absence as an error.

    Never reaches the network. A caller that must not download can rely
    on this; a caller that wants weights fetched calls
    :func:`turkic_translit.lid.fetch.ensure_lid_model` instead.

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
    found = find_model_path(model_id, search_dirs)
    if found is not None:
        return found
    spec = get_spec(model_id)
    searched = [directory / spec["filename"] for directory in search_dirs]
    raise LidModelFileMissingError(
        model_id, searched[-1] if searched else Path(spec["filename"])
    )


__all__ = [
    "REGISTRY",
    "find_model_path",
    "get_spec",
    "known_model_ids",
    "resolve_model_path",
]
