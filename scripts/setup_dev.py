#!/usr/bin/env python
"""
Development Environment Setup Script

This script helps set up a development environment for the Turkic Transliteration package.
It installs all required development dependencies and performs basic configuration.
"""

import logging
import os
import pathlib
import platform
import subprocess
import sys

logger = logging.getLogger(__name__)


def command_runs(command: list[str]) -> bool:
    """Report whether a command exists and exits successfully.

    Args:
        command: The command and arguments to run. Output is discarded;
            only the exit status is consulted.

    Returns:
        True when the command ran and returned zero.
    """
    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.debug("probe %s failed: %s", command[0], exc)
        return False
    return True


def main() -> None:
    """Set up the development environment."""
    # Get the absolute path to the project root
    script_path = pathlib.Path(__file__).resolve()
    project_root = script_path.parent.parent

    print("Setting up development environment for Turkic Transliteration")
    print(f"Project root: {project_root}")

    # Check if we're running in a virtual environment
    if not is_virtual_env():
        print("WARNING: You're not running in a virtual environment.")
        print("It's recommended to use a virtual environment (virtualenv, conda, etc.)")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            print("Setup aborted. Please create a virtual environment and try again.")
            sys.exit(1)

    # Install the package in development mode with all extras
    print("\n=== Installing package with all development dependencies ===")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", f"{project_root}[dev,ui,winlid]"]
    )

    # Check for PyICU on Windows
    if platform.system() == "Windows":
        print("\n=== Checking PyICU installation on Windows ===")
        if command_runs([sys.executable, "-c", "import icu"]):
            print("PyICU is already installed")
        else:
            print("PyICU is not installed. Running the PyICU installer...")
            subprocess.check_call([sys.executable, "-m", "turkic_translit.pyicu_install"])

    # Run basic configuration checks
    print("\n=== Verifying development tools ===")
    tools = ["black", "ruff", "mypy", "pytest"]
    for tool in tools:
        status = "✓ installed" if command_runs([tool, "--version"]) else "✗ not found"
        print(f"{status}: {tool}")

    print("\n=== Setup complete! ===\n")

    # Provide different instructions based on the platform
    if platform.system() == "Windows":
        # Check if GNU Make is installed
        if command_runs(["make", "--version"]):
            print("GNU Make is installed. You can run:")
            print("  make help - Show available commands")
            print("  make lint - Run linting checks")
            print("  make test - Run tests")
            print("  make web  - Start the web UI")
        else:
            print("You can run (using PowerShell):")
            print("  ./scripts/run.ps1 help    - Show available commands")
            print("  ./scripts/run.ps1 lint    - Run linting checks")
            print("  ./scripts/run.ps1 test    - Run tests")
            print("  ./scripts/run.ps1 web     - Start the web UI")
            print("\nTo install GNU Make (recommended for easier development):")
            print("  1. Open an Admin PowerShell window")
            print("  2. Run: choco install make")
            print("  3. After installation, you can use 'make <command>' instead.")
    else:
        print("You can now run:")
        print("  make help - Show available commands")
        print("  make lint - Run linting checks")
        print("  make test - Run tests")
        print("  make web  - Start the web UI")


def is_virtual_env() -> bool:
    """Check if the script is running in a virtual environment."""
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def is_conda_env() -> bool:
    """Check if running inside a conda environment."""
    return bool(os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"))


if __name__ == "__main__":
    main()
