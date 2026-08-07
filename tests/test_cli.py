"""Tests that every declared console script resolves and answers ``--help``.

These used to shell out to the installed ``.cmd`` shims under
``sys.prefix/Scripts``. That form failed for reasons unrelated to the
code: running a batch shim needs a shell, and a pytest worker whose
environment lacks ``%ComSpec%`` cannot start one. It also proved nothing
unless the package happened to be installed with its scripts on disk.

Resolving each entry point from ``pyproject.toml`` and invoking it
directly tests the thing that can actually be wrong — that the declared
``module:function`` target exists and its command runs — and does so
without a shell, an install layout, or a subprocess.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import click
import pytest
import tomllib
from click.testing import CliRunner

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _console_scripts() -> list[tuple[str, str]]:
    """Read the declared console scripts from ``pyproject.toml``.

    Returns:
        Each script name paired with its ``module:attribute`` target.
    """
    document = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return sorted(document["project"]["scripts"].items())


SCRIPTS = _console_scripts()


def test_the_project_declares_console_scripts() -> None:
    """The table exists, so the parametrised tests below are not vacuous."""
    assert [name for name, _target in SCRIPTS] == [
        "turkic-build-spm",
        "turkic-download-corpus",
        "turkic-eval-lm",
        "turkic-filter-russian",
        "turkic-leven",
        "turkic-pyicu-install",
        "turkic-train-lm",
        "turkic-train-spm",
        "turkic-translit",
        "turkic-web",
    ]


@pytest.mark.parametrize(("name", "target"), SCRIPTS)
def test_console_script_target_resolves(name: str, target: str) -> None:
    """Every declared entry point names something that exists and is callable.

    Args:
        name: The console script name.
        target: Its ``module:attribute`` target.
    """
    module_name, _, attribute = target.partition(":")
    resolved = getattr(importlib.import_module(module_name), attribute)
    assert callable(resolved), f"{name} -> {target} is not callable"


@pytest.mark.parametrize(("name", "target"), SCRIPTS)
def test_console_script_reports_its_usage(name: str, target: str) -> None:
    """Every Click entry point answers ``--help`` with its own usage line.

    ``turkic-pyicu-install`` is a plain function rather than a Click
    command and is checked for resolution only, by the test above.

    Args:
        name: The console script name.
        target: Its ``module:attribute`` target.
    """
    module_name, _, attribute = target.partition(":")
    command = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(command, click.BaseCommand):
        pytest.skip(f"{name} is not a Click command")

    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0
    assert result.output.startswith("Usage:")
