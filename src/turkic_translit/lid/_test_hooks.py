"""Injection seam for filesystem access during model resolution.

Production binds :data:`probe` to the real filesystem at import time and
never rebinds it. Tests bind it to an in-memory probe. Resolution code
calls ``probe`` unconditionally, so there is no production branch that
exists only to support testing.

The module is private (leading underscore) because the seam is internal
to this package and is not part of the published API.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol


class FileProbe(Protocol):
    """Minimal filesystem interface needed to resolve model weights."""

    def exists(self, path: Path) -> bool:
        """Report whether ``path`` names an existing regular file.

        Args:
            path: Absolute path to test.

        Returns:
            True when the path is a regular file.
        """
        ...

    def size_bytes(self, path: Path) -> int:
        """Report the size of ``path`` in bytes.

        Args:
            path: Absolute path to measure; guaranteed to exist.

        Returns:
            Size of the file in bytes.
        """
        ...


class RealFileProbe:
    """Filesystem probe backed by :mod:`pathlib`."""

    def exists(self, path: Path) -> bool:
        """Return whether ``path`` is an existing regular file.

        Args:
            path: Absolute path to test.

        Returns:
            True when the path is a regular file.
        """
        return path.is_file()

    def size_bytes(self, path: Path) -> int:
        """Return the size of ``path`` in bytes.

        Args:
            path: Absolute path to measure.

        Returns:
            Size of the file in bytes.
        """
        return path.stat().st_size


class MappingFileProbe:
    """Probe answering from an in-memory mapping of path to size.

    Args:
        sizes: Mapping of absolute path to byte size. A path absent from
            the mapping is reported as non-existent.
    """

    def __init__(self, sizes: Mapping[Path, int]) -> None:
        """Store the path-to-size mapping backing this probe."""
        self._sizes = dict(sizes)

    def exists(self, path: Path) -> bool:
        """Return whether ``path`` is present in the backing mapping.

        Args:
            path: Absolute path to test.

        Returns:
            True when the path has a recorded size.
        """
        return path in self._sizes

    def size_bytes(self, path: Path) -> int:
        """Return the recorded size for ``path``.

        Args:
            path: Absolute path to measure; must be present.

        Returns:
            The recorded byte size.
        """
        return self._sizes[path]


probe: FileProbe = RealFileProbe()

__all__ = ["FileProbe", "MappingFileProbe", "RealFileProbe", "probe"]
