"""Tests for obtaining language-identification weights.

The network seam is bound to :class:`RecordingDownloader`, a real
implementation of the :class:`Downloader` protocol that writes bytes and
logs its calls. :class:`UrlDownloader` is exercised for real against a
``file://`` URL, so the streaming, the digest check and the promotion
are covered by running them rather than by asserting on a mock.

The registered specs pin the SHA-256 of Meta's real files, which no test
payload can match, so the tests that need a download to succeed build a
spec pinning their own payload's digest and call
:func:`download_verified` with it. The tests that go through
:func:`ensure_lid_model` and the real registry show what the registry
does with bytes that are not Meta's: it refuses them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from pathlib import Path

import pytest

from turkic_translit.lid import _test_hooks
from turkic_translit.lid.errors import (
    ERR_MODEL_DIGEST_MISMATCH,
    ERR_MODEL_FILE_EMPTY,
    LidModelDigestMismatchError,
    LidModelFileEmptyError,
)
from turkic_translit.lid.fetch import download_verified, ensure_lid_model
from turkic_translit.lid.registry import find_model_path, get_spec
from turkic_translit.lid.spec import LidModelSpec


@pytest.fixture
def restore_hooks() -> Generator[None, None, None]:
    """Restore both production hooks after a test rebinds them.

    Yields:
        None, once, with the original hooks captured.
    """
    probe, downloader = _test_hooks.probe, _test_hooks.downloader
    yield
    _test_hooks.probe, _test_hooks.downloader = probe, downloader


def _spec_pinning(payload: bytes, url: str) -> LidModelSpec:
    """Build a spec whose pinned digest is ``payload``'s.

    Args:
        payload: The bytes the spec must accept.
        url: Where the spec says to download them from.

    Returns:
        A spec for ``weights.bin`` pinning the SHA-256 of ``payload``.
    """
    return LidModelSpec(
        model_id="pinned",
        filename="weights.bin",
        url=url,
        origin_url="https://example.org/origin/weights.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        label_prefix="__label__",
        script_aware=False,
    )


def test_present_weights_are_used_without_downloading(restore_hooks: None, tmp_path: Path) -> None:
    """A model already on disk is returned and no request is made."""
    weights = tmp_path / "lid218e.bin"
    weights.write_bytes(b"weights")
    recorder = _test_hooks.RecordingDownloader(b"downloaded")
    _test_hooks.downloader = recorder

    resolved = ensure_lid_model("lid218e", [tmp_path], tmp_path / "dl")

    assert resolved == weights
    assert recorder.requests == []


def test_bytes_with_the_pinned_digest_are_kept(restore_hooks: None, tmp_path: Path) -> None:
    """A download whose digest is the spec's lands under the weights' own name."""
    recorder = _test_hooks.RecordingDownloader(b"fresh-weights")
    _test_hooks.downloader = recorder
    destination_dir = tmp_path / "dl"
    spec = _spec_pinning(b"fresh-weights", "https://example.org/weights.bin")

    resolved = download_verified(spec, destination_dir)

    assert resolved == destination_dir / "weights.bin"
    assert resolved.read_bytes() == b"fresh-weights"
    assert recorder.requests == [
        ("https://example.org/weights.bin", destination_dir / "weights.bin.part")
    ]
    assert sorted(destination_dir.iterdir()) == [resolved]


def test_a_registered_model_is_fetched_from_the_mirror_and_refused_unless_it_is_metas(
    restore_hooks: None, tmp_path: Path
) -> None:
    """The registry's URL is the one requested, and other bytes are not kept."""
    recorder = _test_hooks.RecordingDownloader(b"not-metas-weights")
    _test_hooks.downloader = recorder
    destination_dir = tmp_path / "dl"
    spec = get_spec("lid.176")

    with pytest.raises(LidModelDigestMismatchError) as excinfo:
        ensure_lid_model("lid.176", [tmp_path / "empty"], destination_dir)

    assert excinfo.value.code == ERR_MODEL_DIGEST_MISMATCH
    assert excinfo.value.url == spec["url"]
    assert excinfo.value.expected == spec["sha256"]
    assert excinfo.value.actual == hashlib.sha256(b"not-metas-weights").hexdigest()
    assert recorder.requests == [(spec["url"], destination_dir / "lid.176.bin.part")]
    assert sorted(destination_dir.iterdir()) == []


def test_a_refused_download_is_not_found_by_the_next_lookup(
    restore_hooks: None, tmp_path: Path
) -> None:
    """Bytes refused once are not served as the model the next time it is asked for."""
    _test_hooks.downloader = _test_hooks.RecordingDownloader(b"not-metas-weights")
    destination_dir = tmp_path / "dl"

    with pytest.raises(LidModelDigestMismatchError):
        ensure_lid_model("lid218e", [destination_dir], destination_dir)

    assert find_model_path("lid218e", [destination_dir]) is None


def test_zero_byte_download_is_an_error(restore_hooks: None, tmp_path: Path) -> None:
    """A download yielding no bytes raises, and leaves nothing behind."""
    _test_hooks.downloader = _test_hooks.RecordingDownloader(b"")
    destination_dir = tmp_path / "dl"

    with pytest.raises(LidModelFileEmptyError) as excinfo:
        ensure_lid_model("lid.176", [tmp_path / "empty"], destination_dir)

    assert excinfo.value.code == ERR_MODEL_FILE_EMPTY
    assert excinfo.value.path == destination_dir / "lid.176.bin"
    assert sorted(destination_dir.iterdir()) == []


def test_existing_zero_byte_weights_are_an_error(restore_hooks: None, tmp_path: Path) -> None:
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
    """The production downloader copies bytes to exactly the path it is given."""
    payload = bytes(range(256)) * 8192
    origin = tmp_path / "origin.bin"
    origin.write_bytes(payload)
    destination = tmp_path / "copy.bin"

    written = _test_hooks.UrlDownloader().fetch(origin.as_uri(), destination)

    assert written == len(payload)
    assert destination.read_bytes() == payload
    assert sorted(tmp_path.iterdir()) == [destination, origin]


def test_a_real_download_is_verified_and_promoted(restore_hooks: None, tmp_path: Path) -> None:
    """The production downloader, the digest check and the rename, run together."""
    payload = bytes(range(256)) * 8192
    origin = tmp_path / "origin.bin"
    origin.write_bytes(payload)
    _test_hooks.downloader = _test_hooks.UrlDownloader()
    destination_dir = tmp_path / "dl"

    resolved = download_verified(_spec_pinning(payload, origin.as_uri()), destination_dir)

    assert resolved.read_bytes() == payload
    assert sorted(destination_dir.iterdir()) == [destination_dir / "weights.bin"]
