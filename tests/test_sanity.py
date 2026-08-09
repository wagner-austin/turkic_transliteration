"""Tests for the corpus sanity checks.

Every check reads real files written by the test, so the numbers below
are the numbers the functions compute rather than stated expectations.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from statistics import StatisticsError

import pytest

from turkic_translit.sanity import bytes_per_char, is_nfc, median_lev


def write(path: Path, lines: list[str]) -> Path:
    """Write one line per entry and return the path.

    Args:
        path: File to write.
        lines: Lines to write, without terminators.

    Returns:
        The written path.
    """
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def test_identical_files_have_no_distance(tmp_path: Path) -> None:
    """Aligned identical lines score zero."""
    latin = write(tmp_path / "a.txt", ["salem", "alem"])
    ipa = write(tmp_path / "b.txt", ["salem", "alem"])

    assert median_lev(str(latin), str(ipa)) == 0.0


def test_completely_different_files_score_one(tmp_path: Path) -> None:
    """Lines sharing no characters score the maximum normalised distance."""
    latin = write(tmp_path / "a.txt", ["aaaa", "aaaa"])
    ipa = write(tmp_path / "b.txt", ["bbbb", "bbbb"])

    assert median_lev(str(latin), str(ipa)) == 1.0


def test_the_median_is_taken_across_lines(tmp_path: Path) -> None:
    """Three lines report the middle distance, not the mean."""
    latin = write(tmp_path / "a.txt", ["aaaa", "aaaa", "aaaa"])
    ipa = write(tmp_path / "b.txt", ["aaaa", "aaab", "bbbb"])

    assert median_lev(str(latin), str(ipa)) == 0.25


def test_the_sample_caps_the_lines_compared(tmp_path: Path) -> None:
    """Only the first ``sample`` lines contribute to the median."""
    latin = write(tmp_path / "a.txt", ["aaaa", "aaaa", "aaaa"])
    ipa = write(tmp_path / "b.txt", ["aaaa", "bbbb", "bbbb"])

    assert median_lev(str(latin), str(ipa), sample=1) == 0.0
    assert median_lev(str(latin), str(ipa), sample=3) == 1.0


def test_the_shorter_file_ends_the_comparison(tmp_path: Path) -> None:
    """Pairing stops at the shorter file rather than padding it."""
    latin = write(tmp_path / "a.txt", ["aaaa", "aaaa", "aaaa"])
    ipa = write(tmp_path / "b.txt", ["aaaa"])

    assert median_lev(str(latin), str(ipa)) == 0.0


def test_ascii_text_is_one_byte_per_character(tmp_path: Path) -> None:
    """Newlines count as characters, so plain ASCII is exactly one."""
    path = write(tmp_path / "a.txt", ["abcd", "efgh"])

    assert bytes_per_char(str(path)) == 1.0


def test_crlf_line_endings_raise_the_ratio(tmp_path: Path) -> None:
    """A file stored with CRLF measures above one byte per character.

    The size is taken from the file while the characters are counted
    after Python's universal-newline decoding, which collapses each
    ``\\r\\n`` to one character. A corpus written on Windows without an
    explicit newline therefore reads as denser than the same text stored
    with LF, and this is what that looks like.
    """
    path = tmp_path / "a.txt"
    path.write_text("abcd\nefgh\n", encoding="utf-8", newline="\r\n")

    assert bytes_per_char(str(path)) == 1.2


def test_cyrillic_text_is_two_bytes_per_character(tmp_path: Path) -> None:
    """Two-byte code points raise the ratio above one."""
    path = tmp_path / "a.txt"
    path.write_text("абвг", encoding="utf-8")

    assert bytes_per_char(str(path)) == 2.0


def test_composed_text_is_reported_as_normalised(tmp_path: Path) -> None:
    """Text already in NFC passes."""
    path = tmp_path / "a.txt"
    path.write_text(unicodedata.normalize("NFC", "ä ö ü\n"), encoding="utf-8")

    assert is_nfc(str(path)) is True


def test_decomposed_text_is_reported_as_unnormalised(tmp_path: Path) -> None:
    """Text in NFD is rejected even though it renders identically."""
    path = tmp_path / "a.txt"
    path.write_text(unicodedata.normalize("NFD", "ä ö ü\n"), encoding="utf-8")

    assert is_nfc(str(path)) is False


def test_an_empty_file_has_no_median(tmp_path: Path) -> None:
    """With no pairs to compare there is no median to report."""
    latin = tmp_path / "a.txt"
    ipa = tmp_path / "b.txt"
    latin.write_text("", encoding="utf-8")
    ipa.write_text("", encoding="utf-8")

    with pytest.raises(StatisticsError):
        median_lev(str(latin), str(ipa))
