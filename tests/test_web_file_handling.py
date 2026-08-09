"""Tests for the file paths the web interface reads and writes.

Every test here drives real production code. The uploaded-file cases use
a real implementation of :class:`~turkic_translit.web.web_utils.NamedFile`
rather than a stand-in object, and the download-directory cases bind a
real :class:`~turkic_translit._test_hooks.MappingEnvironment` so the
resolved directory is observable without altering the process.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from turkic_translit import _test_hooks
from turkic_translit.web.web_utils import direct_transliterate, download_dir, median_levenshtein


@dataclass(frozen=True)
class UploadedFile:
    """A file chosen in the browser, as the web module reads it.

    Gradio hands the callbacks an object carrying the path it wrote the
    upload to, and nothing else about it is read. This is that object.

    Args:
        name: Path to the uploaded file's contents on disk.
    """

    name: str


@pytest.fixture
def cron_dir(tmp_path: Path) -> Iterator[Path]:
    """Point the download directory at a disposable path.

    Yields:
        The directory ``download_dir`` will resolve to for the duration
        of the test.
    """
    previous = _test_hooks.environment
    target = tmp_path / "cronjob"
    _test_hooks.environment = _test_hooks.MappingEnvironment({"TURKIC_CRON_DIR": str(target)})
    yield target
    _test_hooks.environment = previous


def test_direct_transliterate_turkish_to_ipa() -> None:
    """Turkish text is rendered in IPA and reported on."""
    result, stats = direct_transliterate("merhaba", "tr", False, "ipa")
    assert result == "meɾhaba"
    assert "Bytes" in stats


def test_direct_transliterate_turkish_to_latin() -> None:
    """Turkish text folds to ASCII Latin, unchanged when already ASCII."""
    result, stats = direct_transliterate("merhaba", "tr", False, "latin")
    assert result == "merhaba"
    assert "Bytes" in stats


def test_direct_transliterate_reports_empty_input() -> None:
    """Empty input produces empty output rather than an error."""
    result, _ = direct_transliterate("", "tr", False, "ipa")
    assert result == ""


@pytest.mark.parametrize(
    ("text", "lang", "fmt", "expected"),
    [
        ("привет", "kk", "latin", "privet"),
        ("салам", "ky", "latin", "salam"),
        ("merhaba", "tr", "ipa", "meɾhaba"),
        ("Merhaba dünya", "tr", "ipa", "meɾhaba dynja"),
    ],
)
def test_direct_transliterate_across_languages(
    text: str, lang: str, fmt: str, expected: str
) -> None:
    """Each supported language transliterates to its documented output.

    Args:
        text: Source text in the language's own script.
        lang: Language code passed to the web helper.
        fmt: Target notation, ``latin`` or ``ipa``.
        expected: The transliteration the helper must produce.
    """
    result, _ = direct_transliterate(text, lang, False, fmt)
    assert result == expected


def test_median_levenshtein_reads_both_uploads(tmp_path: Path) -> None:
    """The comparison helper reads the paths the uploads name."""
    latin = tmp_path / "lat.txt"
    ipa = tmp_path / "ipa.txt"
    latin.write_text("merhaba\ndunya\n", encoding="utf-8")
    ipa.write_text("merhaba\ndunya\n", encoding="utf-8")

    reported = median_levenshtein(UploadedFile(str(latin)), UploadedFile(str(ipa)))

    assert reported == "Median distance: 0.0000"


def test_median_levenshtein_scores_differing_files(tmp_path: Path) -> None:
    """Files that differ report a non-zero distance."""
    latin = tmp_path / "lat.txt"
    ipa = tmp_path / "ipa.txt"
    latin.write_text("merhaba\n", encoding="utf-8")
    ipa.write_text("meɾhaba\n", encoding="utf-8")

    reported = median_levenshtein(UploadedFile(str(latin)), UploadedFile(str(ipa)), sample=1)

    assert reported.startswith("Median distance: 0.1")


def test_median_levenshtein_rejects_unnamed_upload(tmp_path: Path) -> None:
    """An upload with no path is rejected rather than read as empty."""
    latin = tmp_path / "lat.txt"
    latin.write_text("merhaba\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must have a .name attribute"):
        median_levenshtein(UploadedFile(str(latin)), UploadedFile(""))


def test_download_dir_honours_the_configured_path(cron_dir: Path) -> None:
    """``TURKIC_CRON_DIR`` decides where downloads are written.

    Args:
        cron_dir: The directory the environment names.
    """
    assert download_dir() == cron_dir
    assert cron_dir.is_dir()


def test_download_dir_is_reusable(cron_dir: Path) -> None:
    """Resolving twice returns the same existing directory.

    Args:
        cron_dir: The directory the environment names.
    """
    first = download_dir()
    (first / "written.txt").write_text("kept", encoding="utf-8")

    second = download_dir()

    assert second == first
    assert (second / "written.txt").read_text(encoding="utf-8") == "kept"


def test_download_dir_defaults_beside_the_working_directory() -> None:
    """With nothing configured, downloads land in ``cronjob``."""
    previous = _test_hooks.environment
    _test_hooks.environment = _test_hooks.MappingEnvironment({})
    try:
        assert download_dir() == Path.cwd() / "cronjob"
    finally:
        _test_hooks.environment = previous
