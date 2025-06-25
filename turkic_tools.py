#!/usr/bin/env python
"""
Turkic Transliteration Tools - Main entry point

This script provides a command-line interface to run various demonstration
and utility tools for the Turkic Transliteration Suite.
"""

import pathlib
import subprocess
import sys
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).parent


def run_web_ui() -> int:
    """Launch the web-based user interface.

    This function imports and launches the web_demo.py implementation directly.
    The browser will automatically open with the interface.

    Note: For direct access to console logs (especially warnings about missing
    dependencies like the FastText wheel for Russian filtering), consider
    running web_demo.py directly instead of using this entry point.
    """
    # Configure logging to ensure warnings are visible
    import importlib.machinery
    import importlib.util
    import logging
    import os

    # Respect PYTHONLOGLEVEL environment variable, default to WARNING for compatibility
    _lvl_str = os.environ.get("PYTHONLOGLEVEL", "WARNING").upper()
    _lvl = getattr(logging, _lvl_str, logging.WARNING)
    logging.basicConfig(level=_lvl, force=True)

    web_app_path = PROJECT_ROOT / "src" / "turkic_translit" / "web" / "web_demo.py"
    print(f"Launching web UI from {web_app_path}...")
    print("Note: Some dependency warnings may not be fully visible in the console.")

    try:
        # Import the web_demo module dynamically
        loader = importlib.machinery.SourceFileLoader("web_demo", str(web_app_path))
        spec = importlib.util.spec_from_loader("web_demo", loader)
        if spec is None:
            raise ImportError(f"Could not load module from {web_app_path}")
        web_demo = importlib.util.module_from_spec(spec)
        loader.exec_module(web_demo)

        # Initialize and launch the UI, opening browser automatically
        print("Starting web server and opening browser automatically...")
        ui = web_demo.build_ui()
        ui.queue().launch(inbrowser=True)
        return 0
    except KeyboardInterrupt:
        print("\nWeb UI shutdown requested. Cleaning up...")
        return 0
    except Exception as e:
        print(f"Error launching web UI: {e}")
        return 1


def run_simple_demo() -> int:
    """Run the simple CLI demo."""
    demo_path = PROJECT_ROOT / "examples" / "simple_demo.py"
    print(f"Running simple demo from {demo_path}...")
    return subprocess.call([sys.executable, str(demo_path)])


def run_full_demo() -> int:
    """Run the comprehensive CLI demo with multiple languages."""
    demo_path = PROJECT_ROOT / "examples" / "demo_translit.py"
    print(f"Running full demo from {demo_path}...")
    return subprocess.call([sys.executable, str(demo_path)])


def show_help() -> int:
    """Display help information about available tools."""
    print("Turkic Transliteration Tools")
    print("===========================")
    print("\nAvailable commands:")

    for cmd, info in COMMANDS.items():
        print(f"  {cmd:12} - {info['description']}")

    print("\nUsage: python turkic_tools.py [command]")
    print("Example: python turkic_tools.py web")
    return 0


# Define available commands and their metadata
COMMANDS: dict[str, dict[str, Any]] = {
    "web": {
        "function": run_web_ui,
        "description": "Launch the web-based user interface",
    },
    "demo": {"function": run_simple_demo, "description": "Run the simple CLI demo"},
    "full-demo": {
        "function": run_full_demo,
        "description": "Run the comprehensive demo with multiple languages",
    },
    "help": {"function": show_help, "description": "Show this help information"},
}


if __name__ == "__main__":
    # Get command from arguments or default to help
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    try:
        # Run the requested command or show help if not found
        if cmd in COMMANDS:
            sys.exit(COMMANDS[cmd]["function"]())
        else:
            print(f"Unknown command: {cmd}")
            show_help()
            sys.exit(1)
    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully
        print("\nOperation cancelled by user. Exiting...")
        sys.exit(0)
