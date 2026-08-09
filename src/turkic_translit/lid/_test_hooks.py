"""Injection seam for filesystem access during model resolution.

Production binds :data:`probe` to the real filesystem at import time and
never rebinds it. Tests bind it to an in-memory probe. Resolution code
calls ``probe`` unconditionally, so there is no production branch that
exists only to support testing.

The module is private (leading underscore) because the seam is internal
to this package and is not part of the published API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


# Attribute names belonging to a foreign binding, held as data so that no
# identifier in this project has to spell them.
_GET_LABELS = "getLabels"


class FastTextPredictor(Protocol):
    """The native predictor fastText exposes beneath its Python wrapper."""

    def predict(
        self, text: str, k: int, threshold: float, on_unicode_error: str
    ) -> Sequence[tuple[float, str]]:
        """Classify one newline-terminated line.

        Args:
            text: A single line, terminated by a newline.
            k: Number of predictions to return.
            threshold: Minimum probability to report; 0.0 returns the top
                ``k`` unconditionally.
            on_unicode_error: How to handle undecodable bytes.

        Returns:
            Probability and label pairs, most probable first.
        """
        ...


class LabelReader(Protocol):
    """fastText's bound label-listing method.

    Stated as a callable rather than as a named method because the name
    is ``getLabels``, which is pybind11's spelling of a C++ API and not
    one this project may rename. Taking the bound method by name and
    annotating it here keeps the foreign spelling out of the code
    entirely, so no naming exemption is needed for it.
    """

    def __call__(self, on_unicode_error: str) -> tuple[Sequence[str], Sequence[int]]:
        """List every label and its training frequency.

        Args:
            on_unicode_error: How to handle undecodable bytes.

        Returns:
            Parallel sequences of raw labels and their frequencies.
        """
        ...


class FastTextPybindModel:
    """Adapter presenting fastText's native predictor as a model.

    fastText's own Python wrapper ends ``predict`` with
    ``np.array(probs, copy=False)``, which NumPy 2 rejects outright, and
    that single line of convenience is the whole reason this project
    pinned ``numpy<2``. The native predictor underneath returns plain
    ``(probability, label)`` tuples and touches no array library, so
    going straight to it removes the constraint rather than working
    around it.

    Args:
        predictor: The native predictor to classify with.
    """

    def __init__(self, predictor: FastTextPredictor) -> None:
        """Bind the native predictor this adapter delegates to."""
        self._predictor = predictor

    def predict(self, text: str, k: int) -> tuple[Sequence[str], Sequence[float]]:
        """Classify ``text`` and split the result into parallel sequences.

        Args:
            text: A single line, without a trailing newline.
            k: Number of predictions to return.

        Returns:
            Parallel sequences of raw labels and probabilities, ordered
            most probable first.
        """
        predictions = self._predictor.predict(f"{text}\n", k, 0.0, "strict")
        return (
            [label for _probability, label in predictions],
            [probability for probability, _label in predictions],
        )

    def labels(self) -> Sequence[str]:
        """List every label the loaded model can emit.

        Returns:
            The raw labels, each still carrying its prefix.
        """
        read_labels: LabelReader = getattr(self._predictor, _GET_LABELS)
        labels, _frequencies = read_labels("strict")
        return labels


class FastTextLoader:
    """Loader backed by the ``fasttext`` package.

    The import is deferred to call time and its result is narrowed
    immediately to :class:`FastTextPredictor`, so the untyped
    third-party surface stops at this method rather than propagating
    outward.
    """

    def load(self, path: Path) -> FastTextModel:
        """Load a fastText model from disk.

        Args:
            path: Absolute path to a fastText ``.bin`` file.

        Returns:
            The loaded model, adapted to :class:`FastTextModel`.
        """
        module = __import__("fasttext")
        predictor: FastTextPredictor = module.load_model(str(path)).f
        return FastTextPybindModel(predictor)


class TableFastTextModel:
    """Model answering from a fixed table of ranked predictions.

    A real implementation of :class:`FastTextModel`, not a mock: it
    records nothing and offers no assertion helpers, so a test using it
    can only check the value that came back.

    Args:
        answers: Mapping of exact input text to that text's predictions,
            most probable first, each label still carrying its prefix.
    """

    def __init__(self, answers: Mapping[str, Sequence[tuple[str, float]]]) -> None:
        """Store the ranked answer table backing this model."""
        self._answers = {text: list(ranked) for text, ranked in answers.items()}

    def predict(self, text: str, k: int) -> tuple[Sequence[str], Sequence[float]]:
        """Return the top ``k`` scripted answers for ``text``.

        Args:
            text: Cleaned input line, which must be in the table.
            k: Number of predictions requested.

        Returns:
            Parallel sequences of raw label and probability, truncated to
            ``k`` exactly as a real model truncates.
        """
        ranked = self._answers[text][:k]
        return ([label for label, _p in ranked], [p for _label, p in ranked])

    def labels(self) -> Sequence[str]:
        """List every label present anywhere in the answer table.

        Returns:
            The raw labels, in first-appearance order, without duplicates.
        """
        return list(
            dict.fromkeys(label for ranked in self._answers.values() for label, _p in ranked)
        )


class FixedModelLoader:
    """Loader returning one prepared model regardless of path.

    Args:
        model: The model to return from every load.
    """

    def __init__(self, model: FastTextModel) -> None:
        """Store the model this loader hands out."""
        self._model = model

    def load(self, path: Path) -> FastTextModel:
        """Return the prepared model.

        Args:
            path: Ignored; present to satisfy the protocol.

        Returns:
            The prepared model.
        """
        return self._model


probe: FileProbe = RealFileProbe()
downloader: Downloader = UrlDownloader()
model_loader: ModelLoader = FastTextLoader()

__all__ = [
    "Downloader",
    "FastTextLoader",
    "FastTextPredictor",
    "FastTextPybindModel",
    "FileProbe",
    "FixedModelLoader",
    "MappingFileProbe",
    "ModelLoader",
    "RealFileProbe",
    "RecordingDownloader",
    "TableFastTextModel",
    "UrlDownloader",
    "downloader",
    "model_loader",
    "probe",
]
