"""One-shot PyICU installer for Windows / Python 3.10+.

The console script ``turkic-pyicu-install`` is a bootstrap tool: it
must be importable and executable in an environment where PyICU is
not yet installed. That is why this module lives at the top of the
package rather than under :mod:`turkic_translit.cli` — importing the
``cli`` subpackage pulls in every registered subcommand and their
transitive dependencies (transformers, torch, datasets, …), which
defeats the point of a lightweight bootstrap tool.

Which wheel to install is decided in :mod:`turkic_translit.wheels`;
fetching and installing it happen through the hooks in
:mod:`turkic_translit._test_hooks`. This module only sequences them and
turns a wheel-selection failure into a command-line error.

Invoke via the console script::

    turkic-pyicu-install [--version VERSION]

Or directly::

    python -m turkic_translit.pyicu_install [--version VERSION]
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence

import click

from . import _test_hooks
from .logging_config import default_level
from .logging_config import setup as _log_setup
from .wheels import (
    RELEASES_API_URL,
    ReleaseAsset,
    WheelError,
    pinned_asset,
    python_tag,
    require_installable,
    select_asset,
)

VENDOR_DIRECTORY = pathlib.Path(__file__).parent.parent.parent / "vendor" / "pyicu"

logger = logging.getLogger("turkic-pyicu-install")


def choose_asset(version: str | None, tag: str) -> ReleaseAsset:
    """Name the wheel to install for this interpreter.

    Args:
        version: Explicit PyICU version, or ``None`` to take whatever
            the latest release publishes.
        tag: Interpreter tag from :func:`~turkic_translit.wheels.python_tag`.

    Returns:
        The asset to install, named and located.

    Raises:
        NoMatchingAssetError: If the latest release publishes no wheel
            for this interpreter.
        FieldError: If the release API describes an asset without a name
            or a download URL.
    """
    if version is not None:
        return pinned_asset(version, tag)
    return select_asset(_test_hooks.releases.latest_assets(RELEASES_API_URL), tag)


def resolve_wheel(
    asset: ReleaseAsset, search: Sequence[pathlib.Path], download_to: pathlib.Path
) -> pathlib.Path:
    """Find the wheel on disk, downloading it when it is not there.

    The search order is the caller's: a wheel vendored with a checkout is
    the one that checkout was tested against, a wheel in the working
    directory is one the developer put there on purpose, and only when
    neither exists is the network consulted. Both the places to look and
    the place to write are passed in, so this function reads no ambient
    state and a test can point it anywhere.

    Args:
        asset: The wheel to obtain.
        search: Directories to look in, most specific first.
        download_to: Directory a downloaded wheel is written to.

    Returns:
        Path of the wheel file, which exists on return.

    Raises:
        URLError: If the wheel has to be downloaded and its URL is
            unreachable or does not exist.
    """
    for directory in search:
        candidate = directory / asset["name"]
        if candidate.exists():
            logger.info("Found wheel in %s: %s", directory, candidate)
            return candidate

    destination = download_to / asset["name"]
    logger.info("Downloading %s from %s", asset["name"], asset["download_url"])
    _test_hooks.releases.download(asset["download_url"], destination)
    return destination


@click.command()
@click.option(
    "--version",
    "-v",
    default=None,
    help=("PyICU version to install (default: latest available for your Python version)."),
)
def main(version: str | None) -> None:
    """Download and install a PyICU wheel for Windows / Python >=3.10.

    Args:
        version: Explicit PyICU version to install (e.g. ``"2.15"``).
            When ``None``, the latest release published by
            ``cgohlke/pyicu-build`` is used.

    Raises:
        click.ClickException: If this platform or interpreter has no
            pre-built wheel, or the release publishes none for it. The
            message carries the originating error code.
        URLError: If the GitHub API or the download URL is unreachable.
        CalledProcessError: If ``pip install`` fails.
    """
    _log_setup(default_level())

    tag = python_tag(_test_hooks.interpreter.version())
    try:
        require_installable(_test_hooks.interpreter.platform_name(), tag)
        asset = choose_asset(version, tag)
    except WheelError as exc:
        raise click.ClickException(str(exc)) from exc

    here = pathlib.Path.cwd()
    wheel = resolve_wheel(asset, (VENDOR_DIRECTORY, here), here)
    _test_hooks.installer.install(_test_hooks.interpreter.executable(), wheel)
    logger.info("PyICU %s installed", asset["name"])


__all__ = ["VENDOR_DIRECTORY", "choose_asset", "main", "resolve_wheel"]
