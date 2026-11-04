"""Tests for the patching rule: no monkeypatching and no mock library.

Every case writes real Python to a real ``tests`` directory and runs the
real rule over it. These cases were in ``test_guard_test_quality.py``
until 2026-11-04, when they moved here unchanged beside the rule, which
moved to ``scripts/guards/patching_rules.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.guards.patching_rules import PatchingRule


def patch_kinds(paths: list[Path]) -> list[str]:
    """Run the patching rule and return the violation kinds.

    Args:
        paths: Files to scan.

    Returns:
        The kind of each violation, in the order reported.
    """
    return [violation.kind for violation in PatchingRule().run(paths)]


def test_a_monkeypatch_fixture_argument_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Requesting the fixture is the violation, before any call."""
    path = write_test_module("test_a.py", "def test_x(monkeypatch) -> None:\n    assert True\n")

    assert patch_kinds([path]) == ["monkeypatch-usage"]


def test_a_monkeypatch_call_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Using the name anywhere is flagged, however it arrived."""
    path = write_test_module(
        "test_a.py", "def test_x() -> None:\n    monkeypatch.setenv('A', 'b')\n"
    )

    assert patch_kinds([path]) == ["monkeypatch-usage"]


def test_the_monkeypatch_type_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """An annotation naming the class is the same coupling."""
    path = write_test_module(
        "test_a.py",
        "import pytest\ndef test_x(mp: pytest.MonkeyPatch) -> None:\n    assert True\n",
    )

    assert patch_kinds([path]) == ["monkeypatch-usage"]


def test_one_line_reports_one_violation_per_kind(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The argument and its annotation on one line count once."""
    path = write_test_module(
        "test_a.py",
        "import pytest\ndef test_x(monkeypatch: pytest.MonkeyPatch) -> None:\n    assert True\n",
    )

    assert patch_kinds([path]) == ["monkeypatch-usage"]


@pytest.mark.parametrize(
    "source",
    [
        "import unittest.mock\n",
        "import mock\n",
        "from unittest.mock import Mock\n",
        "from mock import patch\n",
        "from unittest import mock\n",
    ],
)
def test_every_way_of_importing_a_mock_library_is_reported(
    write_test_module: Callable[[str, str], Path], source: str
) -> None:
    """The double answers every call, so the import itself is the defect.

    Args:
        write_test_module: Helper writing the module.
        source: One spelling of the import.
    """
    path = write_test_module("test_a.py", source)

    assert patch_kinds([path]) == ["mock-library-import"]


@pytest.mark.parametrize(
    "source",
    [
        "import unittest\n",
        "from unittest import TestCase\n",
        "from turkic_translit import _test_hooks\n",
        "def test_x() -> None:\n    assert True\n",
    ],
)
def test_ordinary_imports_and_tests_are_left_alone(
    write_test_module: Callable[[str, str], Path], source: str
) -> None:
    """Only the patching libraries are restricted.

    Args:
        write_test_module: Helper writing the module.
        source: Source that must not be flagged.
    """
    path = write_test_module("test_a.py", source)

    assert patch_kinds([path]) == []


def test_violations_are_reported_in_source_order(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Several offences in one file come back top to bottom."""
    path = write_test_module(
        "test_a.py",
        "from unittest.mock import Mock\n\n\ndef test_x(monkeypatch) -> None:\n    assert True\n",
    )

    violations = PatchingRule().run([path])

    assert [violation.kind for violation in violations] == [
        "mock-library-import",
        "monkeypatch-usage",
    ]
    assert [violation.line_no for violation in violations] == [1, 4]
    assert violations[0].line == "from unittest.mock import Mock"


def test_a_module_outside_tests_is_not_scanned(tmp_path: Path) -> None:
    """Production code may name anything it likes."""
    path = tmp_path / "src" / "monkeypatch_helper.py"
    path.parent.mkdir()
    path.write_text("monkeypatch = 1\n", encoding="utf-8")

    assert patch_kinds([path]) == []


def test_a_helper_module_inside_tests_is_not_scanned(tmp_path: Path) -> None:
    """The rule applies to test modules, not to their helpers."""
    path = tmp_path / "tests" / "helpers.py"
    path.parent.mkdir()
    path.write_text("from unittest.mock import Mock\n", encoding="utf-8")

    assert patch_kinds([path]) == []


def test_the_rule_is_named_for_its_report() -> None:
    """The summary line names this rule as ``patching``."""
    assert PatchingRule().name == "patching"
