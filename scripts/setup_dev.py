#!/usr/bin/env python
"""Set up a development environment for this project.

Installs the package in editable mode with its development extras and
reports which developer tools are on PATH. It used to bootstrap PyICU on
Windows as well, because no wheel resolved there; ICU now arrives with
the dependencies, on every platform.

Every effect this script performs and every fact it branches on — running
a subprocess, reading from the terminal, ending the process, and what the
interpreter is — goes through :mod:`scripts._test_hooks`. That is what
lets the whole flow be exercised without installing anything and without
a test reaching in to rebind an attribute.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from scripts import _test_hooks

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEVELOPER_TOOLS: tuple[str, ...] = ("ruff", "mypy", "pytest")


def confirm_outside_virtual_env() -> None:
    """Ask before installing into a non-isolated interpreter.

    Installing editable packages into the system interpreter is how a
    machine ends up with one project's pins breaking another's, so this
    asks rather than assuming.
    """
    if _test_hooks.interpreter.is_isolated():
        return
    print("WARNING: this interpreter is not a virtual environment.")
    print("Installing here will change the packages available system-wide.")
    if _test_hooks.prompt.ask("Continue anyway? [y/N]: ").lower() != "y":
        print("Setup aborted. Create a virtual environment and try again.")
        _test_hooks.exiter.fail(1)


def install_editable() -> None:
    """Install this project in editable mode with its dev extras."""
    print("\n=== Installing package with development dependencies ===")
    _test_hooks.commands.require(
        [
            _test_hooks.interpreter.executable(),
            "-m",
            "pip",
            "install",
            "-e",
            f"{PROJECT_ROOT}[dev]",
        ]
    )


def report_tools(tools: Sequence[str]) -> list[str]:
    """Print which developer tools are usable, and name the missing ones.

    Args:
        tools: Programs to probe, each run with ``--version``.

    Returns:
        The tools that could not be run, in the order probed.
    """
    print("\n=== Verifying development tools ===")
    missing: list[str] = []
    for tool in tools:
        usable = _test_hooks.commands.succeeds([tool, "--version"])
        print(f"{'installed' if usable else 'NOT FOUND'}: {tool}")
        if not usable:
            missing.append(tool)
    return missing


def report_next_steps() -> None:
    """Print how to drive the project, according to what is installed."""
    if _test_hooks.commands.succeeds(["make", "--version"]):
        print("\nGNU Make is available:")
        print("  make lint - Run the guards, Ruff and Mypy")
        print("  make test - Run the test suite with coverage")
        print("  make web  - Start the web UI")
        return
    print("\nGNU Make is not installed. Install it to use the documented targets:")
    print("  choco install make    (from an elevated PowerShell)")


def main() -> None:
    """Run the whole setup, in order."""
    print("Setting up the development environment for turkic-translit")
    print(f"Project root: {PROJECT_ROOT}")

    confirm_outside_virtual_env()
    install_editable()
    missing = report_tools(DEVELOPER_TOOLS)
    report_next_steps()

    print("\n=== Setup complete ===")
    if missing:
        print(f"Missing developer tools: {', '.join(missing)}")


if __name__ == "__main__":
    main()
