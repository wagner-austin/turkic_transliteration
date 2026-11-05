"""Tests for the Windows guard rule.

Each case builds a real git work tree, tracks real files in it and runs the
real rule, or the whole guard, over what git lists. The words the rule
refuses are assembled from pieces here, so this file never contains them
and the rule's run over this repository can read it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.guard import PROJECT_ROOT, run_guards
from scripts.guards import Violation
from scripts.guards.windows_rules import WindowsRule, tracked_files
from tests.git_tree import init_git_tree, track

SHELL = "Power" + "Shell"
EXTENSION = "ps" + "1"
COMMUNITY_MANAGER = "ch" + "oco"
MICROSOFT_MANAGER = "win" + "get"


def windows_violations(root: Path) -> list[Violation]:
    """Run the rule over every file git tracks under a tree.

    Args:
        root: A git work tree.

    Returns:
        What the rule reports.
    """
    return WindowsRule(root).run(tracked_files(root))


def test_a_tree_with_no_windows_tooling_passes(tmp_path: Path) -> None:
    """Linux instructions and the word Windows itself are not refused."""
    root = init_git_tree(tmp_path)
    track(root, "README.md", "Install with apt on Linux; Windows users run pip the same way.\n")

    assert windows_violations(root) == []


def test_a_windows_script_is_refused_by_name(tmp_path: Path) -> None:
    """A tracked script with the Windows shell's extension fails at line one."""
    root = init_git_tree(tmp_path)
    script = track(root, f"scripts/run.{EXTENSION}", "Write-Output hello\n")

    assert windows_violations(root) == [
        Violation(
            file=script, line_no=1, kind="windows-script", line="a Windows shell script: remove it"
        )
    ]


@pytest.mark.parametrize(
    ("relative", "line"),
    [
        ("README.md", f"On Windows run ./scripts/run.{EXTENSION} instead of make."),
        ("scripts/README.md", f"{SHELL.lower()} -ExecutionPolicy Bypass -File release.{EXTENSION}"),
        ("docs/setup_guide.md", f"{COMMUNITY_MANAGER} install make"),
        ("docs/setup_guide.md", f"{MICROSOFT_MANAGER} install GnuWin32.Make"),
        (".editorconfig", f"[*.{{{EXTENSION},sh}}]"),
        ("Makefile", f"# {SHELL} users: see the script."),
    ],
)
def test_a_line_naming_windows_tooling_is_refused(tmp_path: Path, relative: str, line: str) -> None:
    """The shell, its extension, bare or dotted, and both package managers.

    Args:
        tmp_path: Root of the tree.
        relative: Tracked file holding the line.
        line: The offending line, second in its file.
    """
    root = init_git_tree(tmp_path)
    path = track(root, relative, f"first line\n{line}\nlast line\n")

    assert windows_violations(root) == [
        Violation(file=path, line_no=2, kind="windows-text", line=line)
    ]


@pytest.mark.parametrize(
    "line",
    [
        f"{COMMUNITY_MANAGER}late cake",
        f"steps{EXTENSION} and e{EXTENSION}",
        f"{MICROSOFT_MANAGER}s are not a tool",
    ],
)
def test_words_that_only_contain_the_names_pass(tmp_path: Path, line: str) -> None:
    """The names must stand as words, so longer words holding them pass.

    Args:
        tmp_path: Root of the tree.
        line: Text that holds a refused name inside a longer word.
    """
    root = init_git_tree(tmp_path)
    track(root, "notes.md", f"{line}\n")

    assert windows_violations(root) == []


def test_the_root_changelog_is_not_read_but_a_nested_one_is(tmp_path: Path) -> None:
    """The changelog records what was removed; another file of its name does not."""
    root = init_git_tree(tmp_path)
    history = f"- Removed scripts/run.{EXTENSION}.\n"
    track(root, "CHANGELOG.md", history)
    nested = track(root, "docs/CHANGELOG.md", history)

    assert windows_violations(root) == [
        Violation(file=nested, line_no=1, kind="windows-text", line=history.strip())
    ]


def test_an_untracked_file_is_not_read(tmp_path: Path) -> None:
    """Only what git tracks is the repository's."""
    root = init_git_tree(tmp_path)
    (root / f"local.{EXTENSION}").write_text(f"{SHELL}\n", encoding="utf-8")

    assert tracked_files(root) == []
    assert windows_violations(root) == []


def test_a_tree_that_is_not_a_repository_cannot_be_listed(tmp_path: Path) -> None:
    """git's refusal propagates rather than reading as an empty tree."""
    with pytest.raises(subprocess.CalledProcessError):
        tracked_files(tmp_path / "nowhere")


def test_the_rule_reports_under_its_own_name(tmp_path: Path) -> None:
    """The summary line reads ``windows``."""
    assert WindowsRule(tmp_path).name == "windows"


def test_a_windows_line_fails_the_whole_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The runner applies the rule to every tracked file, not only Python.

    Args:
        tmp_path: Project root for the scanned tree.
        capsys: Captures the runner's report.
    """
    root = init_git_tree(tmp_path)
    track(root, "README.md", f"Install {COMMUNITY_MANAGER} first.\n")

    status = run_guards(root)

    captured = capsys.readouterr()
    assert status == 2
    assert [line for line in captured.out.splitlines() if "windows" in line] == [
        "  windows: 1 violations"
    ]
    assert [line for line in captured.err.splitlines() if "windows-text" in line] == [
        f"  README.md:1: windows-text Install {COMMUNITY_MANAGER} first."
    ]


def test_this_repository_carries_no_windows_tooling() -> None:
    """Every file git tracks here passes, this rule's own source included."""
    assert windows_violations(PROJECT_ROOT) == []
