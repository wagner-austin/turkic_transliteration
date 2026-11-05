"""Tests for the guard runner: which files it walks, how it reports, and its entry point.

Each case writes real Python into a real project tree and runs the real
guard over it. The rules themselves are tested beside their modules:
``test_guard_typing_rules.py``, ``test_guard_exception_rules.py``,
``test_guard_file_length.py`` and the others named ``test_guard_*``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.guard import (
    PROJECT_ROOT,
    _iter_py_files,
    _iter_rule_files,
    main,
    parse_root_argument,
    run_guards,
)
from tests.git_tree import init_git_tree


@pytest.fixture
def project(tmp_path: Path) -> Callable[[str, str], Path]:
    """Return a helper writing a module into a project tree.

    Args:
        tmp_path: Root to build the tree in.

    Returns:
        A function taking a repo-relative path and source, returning the
        file's path. The tree is a git work tree with a ``pyproject.toml``,
        so it is one the entry point accepts and the Windows rule can list.
    """
    init_git_tree(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    def write(relative: str, source: str) -> Path:
        """Write source into the project tree being scanned.

        Args:
            relative: Path within the project, with directories
                created as needed.
            source: File contents.

        Returns:
            The written path.
        """
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    return write


def test_only_the_scanned_directories_are_walked(tmp_path: Path) -> None:
    """src, tests and scripts are scanned; anything else is not."""
    for relative in ("src/a.py", "tests/b.py", "scripts/c.py", "docs/d.py", "e.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    found = sorted(p.name for p in _iter_py_files(tmp_path))

    assert found == ["a.py", "b.py", "c.py"]


def test_a_missing_scanned_directory_is_not_an_error(tmp_path: Path) -> None:
    """A project without a scripts directory still scans the others."""
    path = tmp_path / "src" / "only.py"
    path.parent.mkdir(parents=True)
    path.touch()

    assert [p.name for p in _iter_py_files(tmp_path)] == ["only.py"]


def test_rule_files_are_collected_separately_from_python(tmp_path: Path) -> None:
    """Rule files are not Python, so the walk that finds them is its own.

    The provenance rule reads the header of every ``.rules`` file, and
    would silently check nothing if this walk returned an empty list.
    """
    rules_dir = tmp_path / "src" / "turkic_translit" / "rules"
    rules_dir.mkdir(parents=True)
    for name in ("zz_ipa.rules", "zz_lat.rules", "notes.txt"):
        (rules_dir / name).touch()

    found = [path.name for path in _iter_rule_files(tmp_path)]

    assert found == ["zz_ipa.rules", "zz_lat.rules"]


def test_a_project_without_a_rules_directory_is_not_an_error(tmp_path: Path) -> None:
    """The walk yields nothing rather than raising."""
    assert list(_iter_rule_files(tmp_path)) == []


def test_a_clean_tree_passes(project: Callable[[str, str], Path], tmp_path: Path) -> None:
    """A project with nothing to report exits zero.

    The module is written to satisfy every rule, the docstring one
    included, so that this asserts a clean run rather than the absence
    of whichever rules happen to be registered.
    """
    project(
        "src/clean.py",
        "def f(x: int) -> int:\n"
        '    """Add one.\n\n'
        "    Args:\n"
        "        x: The number.\n\n"
        "    Returns:\n"
        "        One more than the number.\n"
        '    """\n'
        "    return x + 1\n",
    )

    assert run_guards(tmp_path) == 0


def test_a_tree_with_a_violation_fails(project: Callable[[str, str], Path], tmp_path: Path) -> None:
    """Any violation exits two."""
    project("src/dirty.py", "from typing import Any\n")

    assert run_guards(tmp_path) == 2


def reported_violations(captured_err: str) -> list[str]:
    """Extract the violation lines from a guard run's error output.

    Args:
        captured_err: Everything the run wrote to standard error.

    Returns:
        Each violation line, stripped of its leading indentation.
    """
    return [
        line.strip()
        for line in captured_err.splitlines()
        if line.startswith("  ") and ".py:" in line
    ]


def test_a_long_violation_line_is_truncated_in_the_report(
    project: Callable[[str, str], Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A long source line is cut to eighty characters plus an ellipsis."""
    project("src/dirty.py", f"x = 1  # type: ignore {'y' * 120}\n")

    run_guards(tmp_path)
    reported = reported_violations(capsys.readouterr().err)

    quoted = reported[0].split("type-ignore ", 1)[1]
    assert quoted.endswith("...")
    assert len(quoted) == 83


def test_a_short_violation_line_is_quoted_whole(
    project: Callable[[str, str], Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under the limit the source line is reported verbatim."""
    project("src/dirty.py", "x = 1  # type: ignore\n")

    run_guards(tmp_path)
    reported = reported_violations(capsys.readouterr().err)

    assert reported == ["src\\dirty.py:1: type-ignore x = 1  # type: ignore".replace("\\", os.sep)]


def test_a_violation_is_reported_relative_to_the_scanned_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Paths are shortened against the root so reports stay readable."""
    scanned = init_git_tree(tmp_path / "scanned")
    (scanned / "src").mkdir(parents=True)
    (scanned / "src" / "dirty.py").write_text("from typing import Any\n", encoding="utf-8")

    assert run_guards(scanned) == 2
    reported = reported_violations(capsys.readouterr().err)
    assert reported == [f"src{os.sep}dirty.py:1: typing-import-any"]


def test_unparseable_source_raises_naming_the_file(
    project: Callable[[str, str], Path], tmp_path: Path
) -> None:
    """A syntax error is surfaced, not counted as a clean file."""
    project("src/broken.py", "def f(:\n")

    with pytest.raises(SyntaxError):
        run_guards(tmp_path)


def test_the_root_argument_is_read_from_the_command_line(tmp_path: Path) -> None:
    """``--root <path>`` selects the tree to scan."""
    assert parse_root_argument(["--root", str(tmp_path)]) == tmp_path.resolve()


def test_a_root_argument_without_a_value_is_ignored() -> None:
    """A trailing ``--root`` names nothing."""
    assert parse_root_argument(["--root"]) is None


def test_other_arguments_are_stepped_over(tmp_path: Path) -> None:
    """The scan finds ``--root`` wherever it appears."""
    assert parse_root_argument(["-v", "--root", str(tmp_path)]) == tmp_path.resolve()


def test_no_arguments_selects_no_root() -> None:
    """Without the flag the caller falls back to its default."""
    assert parse_root_argument([]) is None


def test_the_entry_point_scans_the_requested_root(
    project: Callable[[str, str], Path], tmp_path: Path
) -> None:
    """``--root`` is honoured in preference to the default."""
    project("src/dirty.py", "from typing import cast\n")

    assert main(["--root", str(tmp_path)]) == 2


def test_the_entry_point_falls_back_to_the_project_root(
    project: Callable[[str, str], Path], tmp_path: Path
) -> None:
    """Without ``--root`` the given default tree is scanned."""
    project("src/clean.py", "x: int = 1\n")

    assert main([], default_root=tmp_path) == 0


def test_a_default_root_that_is_not_a_project_is_refused(tmp_path: Path) -> None:
    """Scanning a tree with no project file would report a false pass."""
    assert main([], default_root=tmp_path / "nowhere") == 1


@pytest.mark.parametrize(
    ("source", "status"),
    [("x: int = 1\n", 0), ("from typing import Any\n", 2)],
)
def test_running_the_module_as_a_script_exits_with_the_rule_status(
    project: Callable[[str, str], Path], tmp_path: Path, source: str, status: int
) -> None:
    """``python -m scripts.guard`` turns the result into an exit code.

    Spawned as a real process so the module-level entry point runs with
    a real ``sys.argv``. Coverage follows into the child, so this needs
    no substitution of process state to be measured.

    Args:
        source: Module written into the scanned tree.
        status: Exit code the run should report.
    """
    project("src/module.py", source)

    completed = subprocess.run(
        [sys.executable, "-m", "scripts.guard", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
        check=False,
    )

    assert completed.returncode == status
