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
from urllib.request import urlopen

from turkic_translit.lid.classifier import FastTextModel


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


class Downloader(Protocol):
    """Minimal interface for retrieving model weights over the network."""

    def fetch(self, url: str, destination: Path) -> int:
        """Write the bytes at ``url`` to ``destination``.

        Args:
            url: Canonical download location taken from the model spec.
            destination: Absolute path to write; its parent exists.

        Returns:
            Number of bytes written.
        """
        ...


class UrlDownloader:
    """Downloader streaming over HTTP with :mod:`urllib`.

    Network and filesystem failures propagate unchanged. A partial
    transfer is never promoted to a usable model: bytes land on a
    ``.part`` sibling and are renamed only once the stream completes, so
    an interrupted download leaves no file for the resolver to find.
    """

    def fetch(self, url: str, destination: Path) -> int:
        """Stream ``url`` to ``destination`` via a temporary sibling.

        Args:
            url: Canonical download location.
            destination: Absolute path to write.

        Returns:
            Number of bytes written.
        """
        partial = destination.with_suffix(destination.suffix + ".part")
        written = 0
        with urlopen(url) as response, partial.open("wb") as sink:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                sink.write(chunk)
                written += len(chunk)
        partial.replace(destination)
        return written


class RecordingDownloader:
    """Downloader writing fixed bytes and recording every request.

    Args:
        payload: Bytes to write for any requested URL.
    """

    def __init__(self, payload: bytes) -> None:
        """Store the payload and start an empty request log."""
        self._payload = payload
        self.requests: list[tuple[str, Path]] = []

    def fetch(self, url: str, destination: Path) -> int:
        """Record the request and write the fixed payload.

        Args:
            url: Requested download location.
            destination: Absolute path to write.

        Returns:
            Number of bytes written.
        """
        self.requests.append((url, destination))
        destination.write_bytes(self._payload)
        return len(self._payload)


class ModelLoader(Protocol):
    """Loader turning a weights path into a usable fastText model."""

    def load(self, path: Path) -> FastTextModel:
        """Load the model stored at ``path``.

        Args:
            path: Absolute path to a fastText ``.bin`` file.

        Returns:
            The loaded model.
        """
        ...


class FastTextLoader:
    """Loader backed by the ``fasttext`` package.

    The import is deferred to call time and its result is assigned
    directly to the :class:`FastTextModel` protocol, so the untyped
    third-party surface is narrowed to the one method this package uses
    rather than propagating outward.
    """

    def load(self, path: Path) -> FastTextModel:
        """Load a fastText model from disk.

        Args:
            path: Absolute path to a fastText ``.bin`` file.

        Returns:
            The loaded model, narrowed to :class:`FastTextModel`.
        """
        module = __import__("fasttext")
        loaded: FastTextModel = module.load_model(str(path))
        return loaded


probe: FileProbe = RealFileProbe()
downloader: Downloader = UrlDownloader()
model_loader: ModelLoader = FastTextLoader()

__all__ = [
    "Downloader",
    "FastTextLoader",
    "FileProbe",
    "MappingFileProbe",
    "ModelLoader",
    "RealFileProbe",
    "RecordingDownloader",
    "UrlDownloader",
    "downloader",
    "model_loader",
    "probe",
]
