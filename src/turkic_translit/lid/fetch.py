"""Obtaining model weights that are not yet on disk.

This is deliberately a second, separately named operation rather than a
retry path inside :func:`resolve_model_path`. Resolution answers "where
are these weights"; it never reaches the network, so a caller that must
not download can rely on it. :func:`ensure_lid_model` answers "make these
weights available", and the caller chooses which question it is asking.

A download that produces zero bytes raises rather than being retried or
substituted, on the same reasoning as the rest of the package: a corpus
filtered by weights nobody can identify is a corpus nobody can rebuild.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from turkic_translit.lid import _test_hooks
from turkic_translit.lid.errors import LidModelFileEmptyError
from turkic_translit.lid.registry import find_model_path, get_spec


def ensure_lid_model(model_id: str, search_dirs: Sequence[Path], destination_dir: Path) -> Path:
    """Return a path to the model's weights, downloading them if absent.

    The search directories are consulted first. Only when none of them
    holds the file is the spec's URL fetched into ``destination_dir``.

    Args:
        model_id: Registry key naming the model to make available.
        search_dirs: Directories to consult before downloading.
        destination_dir: Directory to download into when not found. It is
            created if it does not exist.

    Returns:
        Absolute path to usable weights.

    Raises:
        UnknownLidModelError: If the model id is not registered.
        LidModelFileEmptyError: If existing weights are zero bytes, or if
            the download produced zero bytes.
    """
    spec = get_spec(model_id)

    found = find_model_path(model_id, search_dirs)
    if found is not None:
        return found

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / spec["filename"]
    written = _test_hooks.downloader.fetch(spec["url"], destination)
    if written == 0:
        raise LidModelFileEmptyError(model_id, destination)
    return destination


__all__ = ["ensure_lid_model"]
