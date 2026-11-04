"""Tests for ``turkic-clean-corpus``: file discovery, rendering and the command.

Every case writes real corpora to a real directory and runs the real
command or helper over them. These cases were in ``test_corpus_clean.py``
until 2026-11-04, when they moved here unchanged so that the cleaning
tests and the command tests each stay under the 600-line limit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.corpus_clean_helpers import LONG, write_corpora
from turkic_translit.cli.clean_corpus import (
    REPORT_NAME,
    cli,
    discover,
    harmonize,
    language_of,
    render,
)
from turkic_translit.corpus.clean import clean_corpora, decode_clean_report
from turkic_translit.corpus.symbols import PACKAGED_SYMBOL_MAP, read_symbol_map


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("oscar_ky_ipa", "ky"),
        ("ky", "ky"),
        ("corpus_fi", "fi"),
        ("oscar_ky_kk_ipa", None),
        ("corpus", None),
    ],
)
def test_the_language_is_read_from_the_file_name(name: str, expected: str | None) -> None:
    """One unambiguous two-letter field, or nothing."""
    assert language_of(Path(f"{name}.txt")) == expected


def test_two_files_claiming_one_language_are_refused(tmp_path: Path) -> None:
    """Picking one would silently drop the other's text."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "oscar_ky_ipa.txt").write_text(LONG, encoding="utf-8")
    (raw / "wiki_ky_ipa.txt").write_text(LONG, encoding="utf-8")

    with pytest.raises(Exception, match="claim to be ky"):
        discover(raw, "*.txt")


def test_a_file_with_no_language_in_its_name_is_refused(tmp_path: Path) -> None:
    """Guessing which language a file holds is not the tool's to do."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "corpus.txt").write_text(LONG, encoding="utf-8")

    with pytest.raises(Exception, match="cannot tell which language"):
        discover(raw, "*.txt")


def test_the_rendering_names_every_language_and_the_budget(tmp_path: Path) -> None:
    """The terminal output says which language constrained the rest."""
    raw = write_corpora(tmp_path, {"ky": [LONG], "kk": [LONG, LONG + " more"]})
    report = clean_corpora(
        {"ky": raw / "oscar_ky_ipa.txt", "kk": raw / "oscar_kk_ipa.txt"},
        tmp_path / "clean",
        (),
    )

    text = render(report)

    assert "ky" in text
    assert "kk" in text
    assert "lowest-resource" in text


def test_the_command_cleans_a_directory_and_writes_a_report(tmp_path: Path) -> None:
    """End to end, through the console entry point, on real files."""
    raw = write_corpora(
        tmp_path,
        {"ky": [LONG.replace("j", "ʧ"), LONG], "kk": [LONG, LONG + " ekint͡ʃi"]},
    )
    out = tmp_path / "clean"

    result = CliRunner().invoke(
        cli,
        [
            "--input-dir",
            str(raw),
            "--output-dir",
            str(out),
            "--symbol-map",
            str(PACKAGED_SYMBOL_MAP),
        ],
    )

    assert result.exit_code == 0, result.output
    report = decode_clean_report(json.loads((out / REPORT_NAME).read_text(encoding="utf-8")))
    assert set(report["languages"]) == {"ky", "kk"}
    assert "ʧ" not in (out / "oscar_ky_ipa.txt").read_text(encoding="utf-8")


def test_the_command_refuses_an_empty_directory(tmp_path: Path) -> None:
    """A run that cleaned nothing must not report success."""
    raw = tmp_path / "raw"
    raw.mkdir()

    result = CliRunner().invoke(
        cli, ["--input-dir", str(raw), "--output-dir", str(tmp_path / "clean")]
    )

    assert result.exit_code != 0
    assert "no file matching" in result.output


# ---------------------------------------------------------------------------
# Harmonisation: the symbol map with nothing else done
# ---------------------------------------------------------------------------


def test_harmonising_rewrites_symbols_and_touches_nothing_else(tmp_path: Path) -> None:
    """Evaluation texts keep every line; only the symbols change.

    The corpus pipeline drops short lines and deduplicates, which would
    destroy a perception file's section headers and numeric markers. This
    mode exists so those files can share the corpus's symbol space while
    keeping their structure line for line.
    """
    raw = tmp_path / "eval"
    raw.mkdir()
    original = "KYRGYZ\nTEXT 1: s\u0251l\u0251m\n\n1\n" + LONG.replace("j", "\u02a7") + "\n1\n"
    (raw / "perception_ky.txt").write_text(original, encoding="utf-8")
    out = tmp_path / "harmonised"

    written = harmonize(discover(raw, "*.txt"), out, read_symbol_map(PACKAGED_SYMBOL_MAP))

    result = (out / "perception_ky.txt").read_text(encoding="utf-8")
    assert written == ["ky"]
    assert "\u02a7" not in result
    assert "t\u0361\u0283" in result
    assert len(result.splitlines()) == len(original.splitlines())
    assert result.splitlines()[0] == "KYRGYZ"
    assert result.splitlines()[3] == "1"


def test_the_command_harmonises_without_cleaning(tmp_path: Path) -> None:
    """The harmonise pair alone is a complete, report-free run."""
    raw = tmp_path / "eval"
    raw.mkdir()
    (raw / "perception_kk.txt").write_text(LONG + "\n", encoding="utf-8")
    out = tmp_path / "harmonised"

    result = CliRunner().invoke(
        cli,
        ["--harmonize-dir", str(raw), "--harmonize-output-dir", str(out)],
    )

    assert result.exit_code == 0, result.output
    assert "harmonised: kk" in result.output
    assert (out / "perception_kk.txt").exists()
    assert not (out / REPORT_NAME).exists()


def test_the_command_runs_both_modes_in_one_invocation(tmp_path: Path) -> None:
    """Corpora are cleaned and evaluation texts harmonised together."""
    raw = write_corpora(tmp_path, {"ky": [LONG], "kk": [LONG + " ekint͡ʃi"]})
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "perception_ky.txt").write_text("short\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "--input-dir",
            str(raw),
            "--output-dir",
            str(tmp_path / "clean"),
            "--harmonize-dir",
            str(eval_dir),
            "--harmonize-output-dir",
            str(tmp_path / "harmonised"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "clean" / REPORT_NAME).exists()
    # The harmonised file keeps its too-short line; only cleaning filters.
    harmonised = tmp_path / "harmonised" / "perception_ky.txt"
    assert harmonised.read_text(encoding="utf-8") == "short\n"


def test_a_half_given_directory_pair_is_refused(tmp_path: Path) -> None:
    """One directory of a pair is a mistake, not a request."""
    raw = tmp_path / "raw"
    raw.mkdir()

    for arguments, complaint in (
        (["--input-dir", str(raw)], "needs both --input-dir and --output-dir"),
        (["--harmonize-dir", str(raw)], "needs both --harmonize-dir"),
    ):
        result = CliRunner().invoke(cli, arguments)
        assert result.exit_code != 0
        assert complaint in result.output


def test_the_command_with_no_directories_is_refused() -> None:
    """A run that would do nothing must say so rather than succeed."""
    result = CliRunner().invoke(cli, [])

    assert result.exit_code != 0
    assert "nothing to do" in result.output


def test_harmonising_an_empty_directory_is_refused(tmp_path: Path) -> None:
    """No matching files is a mistake, not a quiet no-op."""
    raw = tmp_path / "eval"
    raw.mkdir()

    result = CliRunner().invoke(
        cli,
        ["--harmonize-dir", str(raw), "--harmonize-output-dir", str(tmp_path / "out")],
    )

    assert result.exit_code != 0
    assert "no file matching" in result.output
