"""Tests for the PyICU bootstrap installer.

The installer's three effects — reading the running interpreter, querying
GitHub, and running pip — are bound to real implementations of the same
protocols that answer from stated values and record what they were asked
for. Nothing here reaches the network or installs anything, and no branch
exists in the installer purely to make that possible.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from turkic_translit import _test_hooks
from turkic_translit.pyicu_install import choose_asset, main, resolve_wheel
from turkic_translit.wheels import (
    ERR_NO_MATCHING_ASSET,
    ERR_UNSUPPORTED_PLATFORM,
    ERR_UNSUPPORTED_PYTHON,
    PLATFORM_TAG,
    RELEASES_API_URL,
    WINDOWS,
    ReleaseAsset,
    pinned_asset,
)

CP313_WHEEL: ReleaseAsset = {
    "name": f"pyicu-2.15-cp313-cp313-{PLATFORM_TAG}.whl",
    "download_url": f"https://example.invalid/pyicu-2.15-cp313-cp313-{PLATFORM_TAG}.whl",
}
UNRELATED_ASSET: ReleaseAsset = {
    "name": "Source code (zip)",
    "download_url": "https://example.invalid/source.zip",
}


class Bootstrap:
    """The hook bindings one installer run was driven with.

    Args:
        releases: The release index the run queried.
        installer: The installer the run ended at.
    """

    def __init__(
        self, releases: _test_hooks.TableReleaseIndex, installer: _test_hooks.RecordingInstaller
    ) -> None:
        """Store both recorders so a test can read what the run did."""
        self.releases = releases
        self.installer = installer


@pytest.fixture
def windows_cp313() -> Iterator[Bootstrap]:
    """Bind the hooks to a Windows cp313 interpreter that installs nothing.

    Yields:
        The recorders the installer run wrote to.
    """
    previous = (_test_hooks.interpreter, _test_hooks.releases, _test_hooks.installer)
    releases = _test_hooks.TableReleaseIndex([UNRELATED_ASSET, CP313_WHEEL])
    installer = _test_hooks.RecordingInstaller()
    _test_hooks.interpreter = _test_hooks.DescribedInterpreter(WINDOWS, (3, 13), "python.exe")
    _test_hooks.releases = releases
    _test_hooks.installer = installer
    yield Bootstrap(releases, installer)
    _test_hooks.interpreter, _test_hooks.releases, _test_hooks.installer = previous


@pytest.fixture
def empty_release() -> Iterator[_test_hooks.TableReleaseIndex]:
    """Bind a Windows cp313 interpreter to a release that publishes nothing.

    Yields:
        The release index the installer queried.
    """
    previous = (_test_hooks.interpreter, _test_hooks.releases)
    releases = _test_hooks.TableReleaseIndex([])
    _test_hooks.interpreter = _test_hooks.DescribedInterpreter(WINDOWS, (3, 13), "python.exe")
    _test_hooks.releases = releases
    yield releases
    _test_hooks.interpreter, _test_hooks.releases = previous


def run(arguments: list[str], working_directory: Path) -> Result:
    """Invoke the installer with the working directory it should search.

    Args:
        arguments: Command-line arguments after the program name.
        working_directory: Directory the command runs in, which is both
            searched for a wheel and written to when one is downloaded.

    Returns:
        The completed invocation.
    """
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=working_directory):
        return runner.invoke(main, arguments)


def test_choose_asset_queries_the_release_for_an_unpinned_run(
    windows_cp313: Bootstrap,
) -> None:
    """With no ``--version``, the latest release decides the wheel.

    Args:
        windows_cp313: The bound recorders.
    """
    assert choose_asset(None, "cp313") == CP313_WHEEL
    assert windows_cp313.releases.queried == [RELEASES_API_URL]


def test_choose_asset_skips_the_release_for_a_pinned_run(
    windows_cp313: Bootstrap,
) -> None:
    """A pinned version is resolved by construction, querying nothing.

    Args:
        windows_cp313: The bound recorders.
    """
    assert choose_asset("2.15", "cp313") == pinned_asset("2.15", "cp313")
    assert windows_cp313.releases.queried == []


def test_resolve_wheel_prefers_the_first_directory_that_has_it(tmp_path: Path) -> None:
    """A vendored wheel wins over the same name in the working directory."""
    vendor = tmp_path / "vendor"
    here = tmp_path / "here"
    vendor.mkdir()
    here.mkdir()
    (vendor / CP313_WHEEL["name"]).write_bytes(b"vendored")
    (here / CP313_WHEEL["name"]).write_bytes(b"local")

    resolved = resolve_wheel(CP313_WHEEL, (vendor, here), here)

    assert resolved == vendor / CP313_WHEEL["name"]
    assert resolved.read_bytes() == b"vendored"


def test_resolve_wheel_falls_through_to_a_later_directory(tmp_path: Path) -> None:
    """A wheel the developer placed locally is used when none is vendored."""
    vendor = tmp_path / "vendor"
    here = tmp_path / "here"
    here.mkdir()
    (here / CP313_WHEEL["name"]).write_bytes(b"local")

    resolved = resolve_wheel(CP313_WHEEL, (vendor, here), here)

    assert resolved == here / CP313_WHEEL["name"]
    assert resolved.read_bytes() == b"local"


def test_resolve_wheel_downloads_when_no_directory_has_it(tmp_path: Path) -> None:
    """With the wheel nowhere on disk, it is fetched to the download path."""
    previous = _test_hooks.releases
    releases = _test_hooks.TableReleaseIndex([CP313_WHEEL], contents=b"downloaded")
    _test_hooks.releases = releases
    try:
        resolved = resolve_wheel(CP313_WHEEL, (tmp_path / "vendor",), tmp_path)
    finally:
        _test_hooks.releases = previous

    assert resolved == tmp_path / CP313_WHEEL["name"]
    assert resolved.read_bytes() == b"downloaded"
    assert releases.downloaded == [(CP313_WHEEL["download_url"], resolved)]


def test_install_downloads_and_installs_the_selected_wheel(
    windows_cp313: Bootstrap, tmp_path: Path
) -> None:
    """The whole run picks a wheel, fetches it, and hands it to pip.

    Args:
        windows_cp313: The bound recorders.
        tmp_path: Directory the command runs in.
    """
    result = run([], tmp_path)

    assert result.exit_code == 0
    assert windows_cp313.releases.queried == [RELEASES_API_URL]
    assert len(windows_cp313.installer.installed) == 1
    executable, wheel = windows_cp313.installer.installed[0]
    assert executable == "python.exe"
    assert wheel.name == CP313_WHEEL["name"]


def test_install_honours_an_explicit_version(windows_cp313: Bootstrap, tmp_path: Path) -> None:
    """``--version`` installs that version without consulting the API.

    Args:
        windows_cp313: The bound recorders.
        tmp_path: Directory the command runs in.
    """
    result = run(["--version", "2.14"], tmp_path)

    assert result.exit_code == 0
    assert windows_cp313.releases.queried == []
    _, wheel = windows_cp313.installer.installed[0]
    assert wheel.name == pinned_asset("2.14", "cp313")["name"]


def test_install_on_another_platform_reports_its_code(tmp_path: Path) -> None:
    """A non-Windows run fails naming the platform code, installing nothing."""
    previous = (_test_hooks.interpreter, _test_hooks.installer)
    installer = _test_hooks.RecordingInstaller()
    _test_hooks.interpreter = _test_hooks.DescribedInterpreter("Linux", (3, 13), "python3")
    _test_hooks.installer = installer
    try:
        result = run([], tmp_path)
    finally:
        _test_hooks.interpreter, _test_hooks.installer = previous

    assert result.exit_code == 1
    assert ERR_UNSUPPORTED_PLATFORM in result.output
    assert installer.installed == []


def test_install_on_an_unbuilt_python_reports_its_code(tmp_path: Path) -> None:
    """An interpreter with no wheel fails naming the interpreter code."""
    previous = (_test_hooks.interpreter, _test_hooks.installer)
    installer = _test_hooks.RecordingInstaller()
    _test_hooks.interpreter = _test_hooks.DescribedInterpreter(WINDOWS, (3, 9), "python.exe")
    _test_hooks.installer = installer
    try:
        result = run([], tmp_path)
    finally:
        _test_hooks.interpreter, _test_hooks.installer = previous

    assert result.exit_code == 1
    assert ERR_UNSUPPORTED_PYTHON in result.output
    assert installer.installed == []


def test_install_reports_a_release_with_no_matching_wheel(
    empty_release: _test_hooks.TableReleaseIndex, tmp_path: Path
) -> None:
    """A release publishing no wheel for this interpreter fails with its code.

    Args:
        empty_release: The queried release index, which publishes nothing.
        tmp_path: Directory the command runs in.
    """
    result = run([], tmp_path)

    assert result.exit_code == 1
    assert ERR_NO_MATCHING_ASSET in result.output
    assert empty_release.queried == [RELEASES_API_URL]
