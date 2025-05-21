#!/usr/bin/env python
"""
Turkic Transliteration Tools - Main entry point

This script provides a command-line interface to run various demonstration
and utility tools for the Turkic Transliteration Suite.
"""

import pathlib
import subprocess
import sys
from typing import Any, Dict

PROJECT_ROOT = pathlib.Path(__file__).parent


def run_web_ui() -> int:
    """Launch the web-based user interface."""
    web_app_path = PROJECT_ROOT / "examples" / "web" / "app.py"
    print(f"Launching web UI from {web_app_path}...")

    try:
        return subprocess.call([sys.executable, str(web_app_path)])
    except KeyboardInterrupt:
        print("\nWeb UI shutdown requested. Cleaning up...")
        return 0


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
COMMANDS: Dict[str, Dict[str, Any]] = {
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
