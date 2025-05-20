"""
One-shot helper for Windows users on Python ≥3.12:
    python -m turkic_translit.cli.pyicu_install
"""

import sys
import urllib.request
import subprocess
import platform
import pathlib
import logging
import click
import json


@click.command()
@click.option(
    "--version",
    "-v",
    default=None,
    help="PyICU version to install (default: latest available for your Python version)",
)
def main(version: str | None) -> None:
    """Download and install a PyICU wheel for Windows/Python >=3.10."""
    logging.basicConfig(format="turkic-pyicu-install: %(message)s", level=logging.INFO)
    log = logging.getLogger("turkic-pyicu-install")

    major, minor = sys.version_info[:2]
    if platform.system() != "Windows":
        sys.exit(
            "turkic-pyicu-install: Not needed – PyICU wheels are on PyPI for non-Windows."
        )
    py_tag = f"cp{major}{minor}"
    if py_tag not in {"cp310", "cp311", "cp312", "cp313"}:
        sys.exit(f"turkic-pyicu-install: No pre-built PyICU wheel yet for {py_tag}")

    if version is None:
        # Query latest version from GitHub API
        api_url = "https://api.github.com/repos/cgohlke/pyicu-build/releases/latest"
        try:
            with urllib.request.urlopen(api_url) as resp:
                release = json.load(resp)
            assets = release.get("assets", [])
            wheel_asset = next(
                (a for a in assets if py_tag in a["name"] and "win_amd64" in a["name"]),
                None,
            )
            if not wheel_asset:
                sys.exit(
                    f"turkic-pyicu-install: No suitable wheel found for {py_tag} in latest release."
                )
            version = release["tag_name"].lstrip("v")
            wheel_name = wheel_asset["name"]
            url = wheel_asset["browser_download_url"]
        except Exception as e:
            sys.exit(f"turkic-pyicu-install: Failed to fetch latest release info: {e}")
    else:
        wheel_name = f"pyicu-{version}-{py_tag}-{py_tag}-win_amd64.whl"
        url = f"https://github.com/cgohlke/pyicu-build/releases/download/v{version}/{wheel_name}"

    log.info("→ %s", url)
    if not pathlib.Path(wheel_name).exists():
        log.info("Downloading %s", wheel_name)
        urllib.request.urlretrieve(url, wheel_name)
    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_name])
    log.info("✓ PyICU %s installed", wheel_name)


if __name__ == "__main__":  # pragma: no cover
    main()
