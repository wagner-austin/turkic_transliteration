"""One-shot PyICU installer for Windows / Python 3.10+.

The console script ``turkic-pyicu-install`` is a bootstrap tool: it
must be importable and executable in an environment where PyICU is
not yet installed. That is why this module lives at the top of the
package rather than under :mod:`turkic_translit.cli` — importing the
``cli`` subpackage pulls in every registered subcommand and their
transitive dependencies (transformers, torch, evaluate, …), which
defeats the point of a lightweight bootstrap tool.

Invoke via the console script::

    turkic-pyicu-install [--version VERSION]

Or directly::

    python -m turkic_translit.pyicu_install [--version VERSION]
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import platform
import subprocess
import sys
import urllib.request

if platform.system() == "Windows":
    os.environ["PYTHONUTF8"] = "1"

import click

from .logging_config import setup as _log_setup


@click.command()
@click.option(
    "--version",
    "-v",
    default=None,
    help=(
        "PyICU version to install (default: latest available for your Python version)."
    ),
)
def main(version: str | None) -> None:
    """Download and install a PyICU wheel for Windows / Python >=3.10.

    Prefers a wheel already present in the package's ``vendor/pyicu``
    directory, then falls back to a wheel in the current working
    directory, and finally downloads from the cgohlke pyicu-build
    GitHub releases. Every path is deterministic — no silent
    "best-effort" fallback: a missing wheel raises the appropriate
    error from urllib or pip.

    Args:
        version: Explicit PyICU version to install (e.g. ``"2.15"``).
            When ``None``, the latest release published by
            ``cgohlke/pyicu-build`` is used.

    Raises:
        SystemExit: When the current platform or Python version has no
            pre-built wheel available.
        urllib.error.URLError: When the GitHub API or download URL is
            unreachable.
        subprocess.CalledProcessError: When ``pip install`` fails.
    """
    _log_setup()
    log = logging.getLogger("turkic-pyicu-install")

    major, minor = sys.version_info[:2]
    if platform.system() != "Windows":
        sys.exit(
            "turkic-pyicu-install: Not needed — PyICU wheels are on PyPI for "
            "non-Windows platforms."
        )
    py_tag = f"cp{major}{minor}"
    if py_tag not in {"cp310", "cp311", "cp312", "cp313"}:
        sys.exit(f"turkic-pyicu-install: No pre-built PyICU wheel for {py_tag}.")

    if version is None:
        api_url = "https://api.github.com/repos/cgohlke/pyicu-build/releases/latest"
        with urllib.request.urlopen(api_url) as resp:
            release = json.load(resp)
        assets = release.get("assets", [])
        wheel_asset = next(
            (a for a in assets if py_tag in a["name"] and "win_amd64" in a["name"]),
            None,
        )
        if wheel_asset is None:
            sys.exit(
                f"turkic-pyicu-install: No suitable wheel found for {py_tag} "
                "in the cgohlke/pyicu-build latest release."
            )
        wheel_name = wheel_asset["name"]
        url = wheel_asset["browser_download_url"]
    else:
        wheel_name = f"pyicu-{version}-{py_tag}-{py_tag}-win_amd64.whl"
        url = (
            f"https://github.com/cgohlke/pyicu-build/releases/download/"
            f"v{version}/{wheel_name}"
        )

    # Search order: vendored wheel, current working directory, then
    # download from cgohlke.
    package_dir = pathlib.Path(__file__).parent.parent.parent
    vendor_wheel = package_dir / "vendor" / "pyicu" / wheel_name
    local_wheel = pathlib.Path(wheel_name)

    if vendor_wheel.exists():
        log.info("Found wheel in vendor directory: %s", vendor_wheel)
        wheel_path = vendor_wheel
    elif local_wheel.exists():
        log.info("Found wheel in current directory: %s", local_wheel)
        wheel_path = local_wheel
    else:
        log.info("Downloading %s from %s", wheel_name, url)
        urllib.request.urlretrieve(url, wheel_name)
        wheel_path = local_wheel

    subprocess.check_call([sys.executable, "-m", "pip", "install", str(wheel_path)])
    log.info("PyICU %s installed", wheel_name)


if __name__ == "__main__":  # pragma: no cover
    main()
