"""The verification targets read and never write; the format target rewrites.

The coding standard: a verification target never rewrites the tree, and
rewriting lives in a target of its own. ``make lint`` ran ``ruff check .
--fix`` and ``ruff format .`` until 2026-11-04, and on the check run itself
it rewrote a test, so the check passed on code the commit did not hold.
Each case asks real make, with ``-n``, what a target would run, and reads
the ruff invocations out of that, so a ``--fix`` restored to lint, or a
format without ``--check``, fails here rather than passing on a tree that
happened to be clean.
"""

from __future__ import annotations

import subprocess

import pytest

from scripts.guard import PROJECT_ROOT

#: The ruff commands a verification target may run: both in check mode.
CHECKING = ["poetry run ruff check .", "poetry run ruff format --check ."]

#: The ruff commands the format target runs: both rewrite.
REWRITING = ["poetry run ruff check . --fix", "poetry run ruff format ."]


#: The command that runs the guard rules, file length among them.
GUARD = "poetry run python -m scripts.guard"


def planned_commands(target: str) -> list[str]:
    """List every command make would run for a target, in order.

    Args:
        target: The Makefile target, run with ``make -n`` so nothing runs.

    Returns:
        Each printed command line, stripped.
    """
    completed = subprocess.run(
        ["make", "-n", target],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
        check=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def ruff_commands(target: str) -> list[str]:
    """List the ruff commands make would run for a target, in order.

    Args:
        target: The Makefile target.

    Returns:
        Every planned command line that invokes ruff.
    """
    return [line for line in planned_commands(target) if " ruff " in f" {line} "]


@pytest.mark.parametrize("target", ["lint", "check"])
def test_a_verification_target_runs_the_guard_once(target: str) -> None:
    """Lint, and check through it, run the guard rules, so a long file fails them.

    Args:
        target: A verification target.
    """
    assert [line for line in planned_commands(target) if line == GUARD] == [GUARD]


@pytest.mark.parametrize("target", ["lint", "check"])
def test_a_verification_target_runs_ruff_only_in_check_mode(target: str) -> None:
    """Lint, and check through it, run ruff check and ruff format --check.

    Args:
        target: A verification target.
    """
    assert ruff_commands(target) == CHECKING


def test_the_format_target_is_the_one_that_rewrites() -> None:
    """Format runs ruff check --fix, then ruff format."""
    assert ruff_commands("format") == REWRITING
