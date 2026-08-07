"""Tests for obtaining language-identification weights.

The network seam is bound to :class:`RecordingDownloader`, a real
implementation of the :class:`Downloader` protocol that writes bytes and
logs its calls. :class:`UrlDownloader` is exercised for real against a
``file://`` URL, so the streaming and atomic-rename logic is covered by
running it rather than by asserting on a mock.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from turkic_translit.lid import _test_hooks
from turkic_translit.lid.errors import ERR_MODEL_FILE_EMPTY, LidModelFileEmptyError
from turkic_translit.lid.fetch import ensure_lid_model
from turkic_translit.lid.registry import find_model_path, get_spec


@pytest.fixture
def restore_hooks() -> Generator[None, None, None]:
    """Restore both production hooks after a test rebinds them.

    Yields:
        None, once, with the original hooks captured.
    """
    probe, downloader = _test_hooks.probe, _test_hooks.downloader
    yield
    _test_hooks.probe, _test_hooks.downloader = probe, downloader


def test_present_weights_are_used_without_downloading(
    restore_hooks: None, tmp_path: Path
) -> None:
    """A model already on disk is returned and no request is made."""
    weights = tmp_path / "lid218e.bin"
    weights.write_bytes(b"weights")
    recorder = _test_hooks.RecordingDownloader(b"downloaded")
    _test_hooks.downloader = recorder

    resolved = ensure_lid_model("lid218e", [tmp_path], tmp_path / "dl")

    assert resolved == weights
    assert recorder.requests == []


def test_absent_weights_are_downloaded_from_the_spec_url(
    restore_hooks: None, tmp_path: Path
) -> None:
    """A missing model is fetched from the URL recorded in its spec."""
    recorder = _test_hooks.RecordingDownloader(b"fresh-weights")
    _test_hooks.downloader = recorder
    destination_dir = tmp_path / "dl"

    resolved = ensure_lid_model("lid218e", [tmp_path / "empty"], destination_dir)

    assert resolved == destination_dir / "lid218e.bin"
    assert resolved.read_bytes() == b"fresh-weights"
    assert recorder.requests == [(get_spec("lid218e")["url"], resolved)]


def test_zero_byte_download_is_an_error(restore_hooks: None, tmp_path: Path) -> None:
    """A download yielding no bytes raises instead of returning a path."""
    _test_hooks.downloader = _test_hooks.RecordingDownloader(b"")

    with pytest.raises(LidModelFileEmptyError) as excinfo:
        ensure_lid_model("lid.176", [tmp_path / "empty"], tmp_path / "dl")

    assert excinfo.value.code == ERR_MODEL_FILE_EMPTY


def test_existing_zero_byte_weights_are_an_error(
    restore_hooks: None, tmp_path: Path
) -> None:
    """A truncated file on disk is rejected rather than re-downloaded."""
    (tmp_path / "lid.176.bin").write_bytes(b"")
    recorder = _test_hooks.RecordingDownloader(b"replacement")
    _test_hooks.downloader = recorder

    with pytest.raises(LidModelFileEmptyError):
        ensure_lid_model("lid.176", [tmp_path], tmp_path / "dl")

    assert recorder.requests == []


def test_find_reports_absence_as_none(restore_hooks: None, tmp_path: Path) -> None:
    """The shared lookup returns None rather than raising when absent."""
    assert find_model_path("lid218e", [tmp_path]) is None


def test_url_downloader_streams_a_real_file(tmp_path: Path) -> None:
    """The production downloader copies bytes and leaves no .part file."""
    payload = bytes(range(256)) * 8192
    origin = tmp_path / "origin.bin"
    origin.write_bytes(payload)
    destination = tmp_path / "copy.bin"

    written = _test_hooks.UrlDownloader().fetch(origin.as_uri(), destination)

    assert written == len(payload)
    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".bin.part").exists()
