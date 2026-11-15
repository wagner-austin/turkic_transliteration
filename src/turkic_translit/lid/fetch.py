"""Obtaining model weights that are not yet on disk.

This is deliberately a second, separately named operation rather than a
retry path inside :func:`resolve_model_path`. Resolution answers "where
are these weights"; it never reaches the network, so a caller that must
not download can rely on it. :func:`ensure_lid_model` answers "make these
weights available", and the caller chooses which question it is asking.

Downloaded bytes reach the weights' own filename only once they are
known to be the right ones. They land on a ``.part`` sibling, which the
resolver never looks for, and are promoted only when they are not empty
and their SHA-256 is the one the spec pins. A download failing either
test is removed and raises rather than being retried or substituted, on
the same reasoning as the rest of the package: a corpus filtered by
weights nobody can identify is a corpus nobody can rebuild.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from turkic_translit.digests import sha256_of_file
from turkic_translit.lid import _test_hooks
from turkic_translit.lid.errors import LidModelDigestMismatchError, LidModelFileEmptyError
from turkic_translit.lid.registry import find_model_path, get_spec
from turkic_translit.lid.spec import LidModelSpec


def download_verified(spec: LidModelSpec, destination_dir: Path) -> Path:
    """Download a model's weights and keep them only if they are the pinned bytes.

    Args:
        spec: Specification naming the URL to fetch and the digest the
            bytes must have.
        destination_dir: Directory to download into. It is created if it
            does not exist.

    Returns:
        Absolute path to the verified weights.

    Raises:
        LidModelFileEmptyError: If the download produced zero bytes.
        LidModelDigestMismatchError: If the bytes' SHA-256 is not the
            spec's.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / spec["filename"]
    staging = destination.with_suffix(destination.suffix + ".part")
    written = _test_hooks.downloader.fetch(spec["url"], staging)
    if written == 0:
        staging.unlink()
        raise LidModelFileEmptyError(spec["model_id"], destination)
    actual = sha256_of_file(staging)
    if actual != spec["sha256"]:
        staging.unlink()
        raise LidModelDigestMismatchError(spec["model_id"], spec["url"], spec["sha256"], actual)
    staging.replace(destination)
    return destination


def ensure_lid_model(model_id: str, search_dirs: Sequence[Path], destination_dir: Path) -> Path:
    """Return a path to the model's weights, downloading them if absent.

    The search directories are consulted first. Only when none of them
    holds the file is the spec's URL fetched into ``destination_dir``,
    through :func:`download_verified`.

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
        LidModelDigestMismatchError: If the downloaded bytes are not the
            ones the registry pins.
    """
    spec = get_spec(model_id)

    found = find_model_path(model_id, search_dirs)
    if found is not None:
        return found

    return download_verified(spec, destination_dir)


__all__ = ["download_verified", "ensure_lid_model"]
