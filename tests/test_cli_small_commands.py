"""Tests for the two single-purpose commands and the language labeller.

Both commands are driven through their real Click entry points against
real files, and ``build-spm`` trains SentencePiece for real on a small
synthetic corpus, so the trainer arguments it assembles are proven to be
arguments SentencePiece accepts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from turkic_translit.cli.build_spm import main as build_spm
from turkic_translit.cli.run_leven import main as run_leven
from turkic_translit.lang_utils import pretty_lang

CORPUS = [f"qazaq tili {index} salem alem birinshi" for index in range(120)]


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    """Write a corpus large enough for SentencePiece to train on.

    Returns:
        Path of the written corpus.
    """
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(CORPUS) + "\n", encoding="utf-8", newline="\n")
    return path


def test_run_leven_reports_the_distance_between_two_files(tmp_path: Path) -> None:
    """The command prints what the sanity helper computed."""
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("aaaa\naaaa\n", encoding="utf-8")
    second.write_text("aaaa\naaaa\n", encoding="utf-8")

    result = CliRunner().invoke(run_leven, [str(first), str(second)])

    assert result.exit_code == 0
    assert result.output.strip() == "0.0"


def test_run_leven_honours_the_sample_option(tmp_path: Path) -> None:
    """``--sample`` caps the lines compared, changing the answer."""
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("aaaa\naaaa\n", encoding="utf-8")
    second.write_text("aaaa\nbbbb\n", encoding="utf-8")

    capped = CliRunner().invoke(run_leven, [str(first), str(second), "--sample", "1"])
    uncapped = CliRunner().invoke(run_leven, [str(first), str(second)])

    assert capped.output.strip() == "0.0"
    assert uncapped.output.strip() == "0.5"


def test_run_leven_rejects_a_missing_file(tmp_path: Path) -> None:
    """A path that does not exist fails at parse time."""
    present = tmp_path / "a.txt"
    present.write_text("aaaa\n", encoding="utf-8")

    result = CliRunner().invoke(run_leven, [str(present), str(tmp_path / "absent.txt")])

    assert result.exit_code == 2
    assert "absent.txt" in result.output


def test_build_spm_trains_a_model_and_names_it(corpus_file: Path, tmp_path: Path) -> None:
    """The command trains SentencePiece and reports where it saved."""
    prefix = tmp_path / "turkic"

    result = CliRunner().invoke(
        build_spm,
        [
            "--input",
            str(corpus_file),
            "--model-prefix",
            str(prefix),
            "--vocab-size",
            "40",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"{prefix}.model" in result.output
    assert prefix.with_suffix(".model").exists()
    assert prefix.with_suffix(".vocab").exists()


def test_build_spm_puts_the_user_symbols_in_the_vocabulary(
    corpus_file: Path, tmp_path: Path
) -> None:
    """Every requested symbol appears as its own vocabulary entry."""
    prefix = tmp_path / "turkic"

    result = CliRunner().invoke(
        build_spm,
        [
            "--input",
            str(corpus_file),
            "--model-prefix",
            str(prefix),
            "--vocab-size",
            "40",
            "--user-symbols",
            "<lang_kk>, <lang_ky> ,",
        ],
    )

    assert result.exit_code == 0, result.output
    vocabulary = prefix.with_suffix(".vocab").read_text(encoding="utf-8")
    assert "<lang_kk>" in vocabulary
    assert "<lang_ky>" in vocabulary


def test_build_spm_honours_the_model_type(corpus_file: Path, tmp_path: Path) -> None:
    """A character model trains from the same arguments."""
    prefix = tmp_path / "turkic"

    result = CliRunner().invoke(
        build_spm,
        [
            "--input",
            str(corpus_file),
            "--model-prefix",
            str(prefix),
            "--vocab-size",
            "40",
            "--model-type",
            "char",
        ],
    )

    assert result.exit_code == 0, result.output
    assert prefix.with_suffix(".model").exists()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("bh", "Bhojpuri (bh)"),
        ("kk", "Kazakh (kk)"),
        ("tr", "Turkish (tr)"),
        ("uzn", "Northern Uzbek (uzn)"),
    ],
)
def test_a_known_code_is_labelled_with_its_name(code: str, expected: str) -> None:
    """Two-letter and three-letter codes both resolve to a name.

    Args:
        code: Language code to label.
        expected: The label the UI shows for it.
    """
    assert pretty_lang(code) == expected


def test_an_unknown_code_is_returned_unchanged() -> None:
    """A code pycountry does not know is shown as itself."""
    assert pretty_lang("zzz") == "zzz"
