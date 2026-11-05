"""Tests for the file-length guard rule.

Each case writes real files of a known length and runs the real rule, or
the whole guard, over them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.guard import run_guards
from scripts.guards import Violation
from scripts.guards.file_length_rules import MAX_LINES, FileLengthRule
from tests.git_tree import init_git_tree


def write_lines(path: Path, count: int) -> Path:
    """Write a file of exactly ``count`` lines.

    Args:
        path: File to create; its parent is created too.
        count: Number of newline-terminated lines.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"x_{index} = {index}\n" for index in range(count)), encoding="utf-8")
    return path


def test_a_file_at_the_limit_passes(tmp_path: Path) -> None:
    """Exactly 600 lines is allowed."""
    at_limit = write_lines(tmp_path / "at_limit.py", MAX_LINES)

    assert FileLengthRule().run([at_limit]) == []


def test_a_file_over_the_limit_is_reported_with_its_count(tmp_path: Path) -> None:
    """One line past the limit is a violation naming the file and its length."""
    over = write_lines(tmp_path / "over.py", MAX_LINES + 1)

    assert FileLengthRule().run([over]) == [
        Violation(
            file=over,
            line_no=601,
            kind="file-too-long",
            line="601 lines, over the 600-line limit: split it by concern",
        )
    ]


def test_a_file_that_is_not_python_is_not_measured(tmp_path: Path) -> None:
    """Rule files and data files are outside the rule."""
    long_rules = write_lines(tmp_path / "kk_ipa.rules", MAX_LINES * 2)

    assert FileLengthRule().run([long_rules]) == []


def test_the_rule_reports_under_its_own_name() -> None:
    """The summary line reads ``file-length``."""
    assert FileLengthRule().name == "file-length"


@pytest.mark.parametrize("relative", ["src/pkg/long.py", "tests/test_long.py", "scripts/long.py"])
def test_a_long_file_fails_the_whole_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], relative: str
) -> None:
    """The runner applies the rule to src, tests and scripts alike.

    One case per scanned directory, so a rule or runner that skipped any
    of them fails the case for that directory.

    Args:
        tmp_path: Project root for the scanned tree.
        capsys: Captures the runner's report.
        relative: Where the long file is written, one per scanned directory.
    """
    init_git_tree(tmp_path)
    write_lines(tmp_path / relative, MAX_LINES + 5)

    status = run_guards(tmp_path)

    captured = capsys.readouterr()
    assert status == 2
    assert [line for line in captured.out.splitlines() if "file-length" in line] == [
        "  file-length: 1 violations"
    ]
    assert [line for line in captured.err.splitlines() if "file-too-long" in line] == [
        f"  {relative}:605: file-too-long 605 lines, over the 600-line limit: split it by concern"
    ]
