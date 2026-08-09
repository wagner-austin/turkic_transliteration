"""Tests for the guard package's shared types and file reader.

Real files are written to disk and read back, so the reader is exercised
against the encodings it exists to handle rather than against a stand-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.guards import Rule, RuleReport, Violation
from scripts.guards.util import read_lines


def test_lines_are_returned_without_their_terminators(tmp_path: Path) -> None:
    """A three-line file yields three strings and no newlines."""
    target = tmp_path / "sample.py"
    target.write_text("first\nsecond\nthird\n", encoding="utf-8")

    assert read_lines(target) == ["first", "second", "third"]


def test_a_byte_order_mark_is_consumed(tmp_path: Path) -> None:
    """A BOM written by a Windows editor does not reach the first line.

    utf-8-sig is the whole reason this helper exists: with plain utf-8
    the first line would begin with a zero-width no-break space and
    every rule matching on column one would miss.
    """
    target = tmp_path / "bom.py"
    target.write_bytes(b"\xef\xbb\xbfimport os\n")

    assert read_lines(target) == ["import os"]


def test_an_empty_file_yields_no_lines(tmp_path: Path) -> None:
    """An empty module is zero lines, not one empty line."""
    target = tmp_path / "empty.py"
    target.write_text("", encoding="utf-8")

    assert read_lines(target) == []


def test_a_missing_file_raises_rather_than_returning_nothing(tmp_path: Path) -> None:
    """An unreadable path is an error the caller must see."""
    with pytest.raises(FileNotFoundError):
        read_lines(tmp_path / "absent.py")


def test_invalid_utf8_raises_rather_than_being_replaced(tmp_path: Path) -> None:
    """Undecodable bytes are reported, not silently mangled."""
    target = tmp_path / "latin1.py"
    target.write_bytes(b"x = '\xff\xfe not utf8'\n")

    with pytest.raises(UnicodeDecodeError):
        read_lines(target)


def test_a_violation_carries_its_location_and_kind(tmp_path: Path) -> None:
    """The violation record is a plain tuple of the reported fields."""
    violation = Violation(file=tmp_path / "a.py", line_no=7, kind="any-usage", line="x: Any")

    assert violation.file == tmp_path / "a.py"
    assert violation.line_no == 7
    assert violation.kind == "any-usage"
    assert violation.line == "x: Any"


def test_a_rule_report_pairs_a_name_with_a_count() -> None:
    """The summary record is the rule name and how many it found."""
    report = RuleReport(name="typing", violations=3)

    assert report.name == "typing"
    assert report.violations == 3


def test_the_rule_protocol_is_satisfied_by_a_name_and_a_run() -> None:
    """A class with the two members is structurally a Rule.

    Checked by binding one to the protocol type, which is what the
    aggregator relies on when it treats every rule uniformly.
    """

    class CountingRule:
        """A rule reporting one violation for every file it is given."""

        name = "counting"

        def run(self, files: list[Path]) -> list[Violation]:
            """Report one violation per file.

            Args:
                files: Files to report on.

            Returns:
                One violation for each file.
            """
            return [Violation(file=path, line_no=1, kind="counting", line="") for path in files]

    rule: Rule = CountingRule()

    assert rule.name == "counting"
    assert [v.kind for v in rule.run([Path("a.py"), Path("b.py")])] == ["counting", "counting"]
