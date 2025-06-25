"""
End-to-end (but offline) tests for the corpus-download CLI.

The real drivers are monkey-patched with dummy generators so we don’t fetch
gigabytes from OSCAR / Wikipedia during CI.  We still go through Click’s
command-line surface to make sure options, limits, NFC normalisation and
FastText filtering logic are wired correctly.
"""

from __future__ import annotations

import unicodedata as ud
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

# import AFTER pytest so the module-level registry is already read
from turkic_translit.cli import download_corpus as dl


# --------------------------------------------------------------------------- fixtures & helpers
@pytest.fixture(autouse=True)
def tmp_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect $HOME so the downloader never pollutes the real FS."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return


def _dummy_driver(name: str) -> dl.StreamFn:
    """
    Return a driver that yields predictable lines incorporating *name* so
    each test can assert on the file contents.
    """

    def gen(
        lang: str, _cfg: dict[str, Any], _filter: str | None
    ) -> Generator[str, None, None]:
        # ten NFC-normalised lines
        base = f"{name}_{lang}".upper()
        for i in range(10):
            yield ud.normalize("NFC", f"{base}-{i}")

    # mypy likes explicit typing
    return gen


# --------------------------------------------------------------------------- tests
def test_list_sources_and_license(runner: CliRunner) -> None:
    res = runner.invoke(dl.cli, ["list-sources"])
    assert res.exit_code == 0
    # registry always has at least oscar-2301
    assert "oscar-2301" in res.output

    res = runner.invoke(dl.cli, ["license", "--source", "oscar-2301"])
    assert res.exit_code == 0
    assert "CC0" in res.output


def test_download_basic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    """Happy-path download with line shuffle & unicode NFC."""
    outfile = tmp_path / "dummy.txt"

    # patch the single driver we care about
    monkeypatch.setitem(dl._DRIVERS, "oscar", _dummy_driver("oscar"))

    res = runner.invoke(
        dl.cli,
        [
            "download",
            "--source",
            "oscar-2301",
            "--lang",
            "kk",
            "--out",
            str(outfile),
        ],
    )
    assert res.exit_code == 0
    assert outfile.exists()

    lines = outfile.read_text(encoding="utf8").splitlines()
    assert lines[0].startswith("OSCAR_KK-")
    assert len(lines) == 10
    # verify NFC normalisation (just in case)
    assert all(ud.is_normalized("NFC", ln) for ln in lines)


def test_max_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    outfile = tmp_path / "few.txt"
    monkeypatch.setitem(dl._DRIVERS, "oscar", _dummy_driver("few"))

    res = runner.invoke(
        dl.cli,
        [
            "download",
            "--source",
            "oscar-2301",
            "--lang",
            "ky",
            "--out",
            str(outfile),
            "--max-lines",
            "3",
        ],
    )
    assert res.exit_code == 0
    assert outfile.read_text().count("\n") == 3


def test_filter_langid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    """
    Simulate FastText filter: we monkey-patch _get_lid() to return a dummy
    object whose .predict() returns 'dummy' for half the lines → only the
    matching ones should survive.
    """

    class FakeFT:
        def __init__(self) -> None:
            self.call = 0

        def predict(self, _s: str, k: int = 1) -> tuple[list[str], list[list[float]]]:
            # alternate between matching ('ru') and non-matching ('en')
            self.call += 1
            lbl = "__label__ru" if self.call % 2 else "__label__en"
            return ([lbl], [[0.9]])

    monkeypatch.setattr(dl, "_get_lid", lambda: FakeFT())

    monkeypatch.setitem(dl._DRIVERS, "oscar", _dummy_driver("flt"))

    outfile = tmp_path / "flt.txt"
    runner.invoke(
        dl.cli,
        [
            "download",
            "--source",
            "oscar-2301",
            "--lang",
            "kk",
            "--out",
            str(outfile),
            "--filter-langid",
            "ru",
        ],
        catch_exceptions=False,
    )

    data = outfile.read_text().splitlines()
    # every second synthetic line passes the filter → 5 remain
    assert len(data) == 5
    assert all("FLT_KK" in ln for ln in data)


# --------------------------------------------------------------------------- utilities
@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click CliRunner with env isolation."""
    return CliRunner(env={"PYTHONIOENCODING": "utf8"})
