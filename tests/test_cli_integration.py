"""End-to-end integration tests for the ``turkic-translit`` CLI.

These tests spawn a real subprocess running the CLI entry point with
concrete input files and assert against the concrete output files.
Unlike the ``--help``-only checks in :mod:`tests.test_cli`, they detect
regressions that only show up when the CLI actually transliterates.

They deliberately exercise:
* every language advertised by the rules directory, so a hardcoded
  ``choices=["kk", "ky"]`` regression would fail immediately;
* both the IPA-only path (Finnish, Turkish, Azerbaijani, Uzbek, Uyghur)
  and the Latin path (Kazakh, Kyrgyz);
* the ``turkic-translit translit --out-latin/--out-ipa`` Click
  interface that replaced the former single-command argparse shell.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from turkic_translit.core import get_supported_languages


def _resolve_console_script() -> str:
    """Locate the installed ``turkic-translit`` console script.

    ``make check`` always runs ``poetry install`` before pytest so the
    console-script wrapper is guaranteed to be on the venv's ``PATH``
    when these tests execute. A missing wrapper therefore indicates a
    broken environment, not an optional feature — the appropriate
    response is to fail loudly.

    Returns:
        The absolute path to the ``turkic-translit`` executable.

    Raises:
        RuntimeError: When the console script is not on ``PATH``.
    """
    resolved = shutil.which("turkic-translit")
    if resolved is None:
        raise RuntimeError(
            "turkic-translit console script is not on PATH. "
            "This module tests the installed CLI; run "
            "'poetry install' (or 'pip install -e .') and retry."
        )
    return resolved


_TURKIC_TRANSLIT_BIN = _resolve_console_script()

# Deterministic per-language input samples. Values chosen so that the
# language's rule set genuinely changes at least one character during
# transliteration; a pass-through regression is thus detected by the
# ``result != input`` assertion below.
_IPA_INPUTS: dict[str, str] = {
    "kk": "мектеп\n",
    "ky": "мектеп\n",
    "tr": "çocuk\n",
    "az": "uşaq\n",
    "uz": "shamol\n",
    "ug": "بالا\n",
    "fi": "kiitos\n",
}

_LATIN_INPUTS: dict[str, str] = {
    "kk": "мектеп\n",
    "ky": "мектеп\n",
}


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the installed ``turkic-translit`` console script.

    The console-script wrapper is the entry point users actually run and
    is registered under ``[project.scripts]`` in ``pyproject.toml``.
    Testing through that wrapper (rather than through
    ``python -m turkic_translit.cli``) exercises the same path a real
    user would hit and sidesteps the namespace collision between the
    ``cli.py`` module and the ``cli/`` sub-package.

    Args:
        *args: Command-line arguments passed to ``turkic-translit``.

    Returns:
        The completed subprocess result with ``stdout`` and ``stderr``
        captured as UTF-8 text.
    """
    command = [_TURKIC_TRANSLIT_BIN, "translit", *args]
    return subprocess.run(
        command, capture_output=True, text=True, check=False, encoding="utf-8"
    )


@pytest.mark.parametrize(("lang", "sample"), _IPA_INPUTS.items())
def test_cli_produces_ipa_for_supported_language(
    lang: str, sample: str, tmp_path: Path
) -> None:
    """The CLI emits real IPA (not the input) for every supported language."""
    supported = get_supported_languages()
    assert lang in supported, f"precondition: {lang} present in rules directory"
    assert "ipa" in supported[lang], f"precondition: {lang} advertised as IPA-capable"

    input_file = tmp_path / f"input_{lang}.txt"
    input_file.write_text(sample, encoding="utf-8")
    output_file = tmp_path / f"output_{lang}.ipa.txt"

    result = _run_cli(
        "--lang",
        lang,
        "--in",
        str(input_file),
        "--out-ipa",
        str(output_file),
    )
    assert result.returncode == 0, (
        f"CLI exited {result.returncode} for {lang}: stderr={result.stderr!r}"
    )
    produced = output_file.read_text(encoding="utf-8")
    assert produced, f"empty output for {lang}"
    assert produced != sample, (
        f"{lang} output matches input — CLI likely fell back to pass-through "
        f"(input={sample!r}, output={produced!r})"
    )


@pytest.mark.parametrize(("lang", "sample"), _LATIN_INPUTS.items())
def test_cli_produces_latin_for_kk_ky(lang: str, sample: str, tmp_path: Path) -> None:
    """The CLI produces Latin output for the two languages that ship Latin rules."""
    input_file = tmp_path / f"input_{lang}.txt"
    input_file.write_text(sample, encoding="utf-8")
    output_file = tmp_path / f"output_{lang}.lat.txt"

    result = _run_cli(
        "--lang",
        lang,
        "--in",
        str(input_file),
        "--out-latin",
        str(output_file),
    )
    assert result.returncode == 0, (
        f"CLI exited {result.returncode}: stderr={result.stderr!r}"
    )
    produced = output_file.read_text(encoding="utf-8")
    assert produced, f"empty output for {lang}"
    assert produced != sample, (
        f"{lang} Latin output matches input — Latin rules did not fire"
    )


def test_cli_lang_choices_include_every_ipa_language() -> None:
    """``turkic-translit translit --help`` advertises every ISA language code.

    Runs ``--help`` and greps for each ISO code; this catches a
    regression where a future refactor reintroduces a hardcoded choices
    tuple.
    """
    result = _run_cli("--help")
    assert result.returncode == 0, result.stderr
    supported = get_supported_languages()
    ipa_langs = sorted(k for k, v in supported.items() if "ipa" in v)
    for code in ipa_langs:
        assert code in result.stdout, (
            f"language code {code!r} missing from --help output; "
            f"choices list may have been hardcoded again"
        )


def test_cli_rejects_out_latin_for_ipa_only_language(tmp_path: Path) -> None:
    """Requesting Latin output for Finnish exits with a Click usage error.

    Finnish ships only ``fi_ipa.rules``; asking for Latin is a
    programmer error. The CLI must not silently produce nothing.
    """
    input_file = tmp_path / "input_fi.txt"
    input_file.write_text("kiitos\n", encoding="utf-8")
    output_file = tmp_path / "output_fi.lat.txt"

    result = _run_cli(
        "--lang",
        "fi",
        "--in",
        str(input_file),
        "--out-latin",
        str(output_file),
    )
    assert result.returncode == 2, (
        f"expected Click UsageError exit 2, got {result.returncode}: "
        f"stderr={result.stderr!r}"
    )
    assert "no Latin rules" in result.stderr
    assert not output_file.exists(), (
        "usage error was raised but the Latin output file was still created"
    )


def test_cli_requires_at_least_one_output(tmp_path: Path) -> None:
    """Invoking with no ``--out-*`` is a Click usage error."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("kiitos\n", encoding="utf-8")
    result = _run_cli("--lang", "fi", "--in", str(input_file))
    assert result.returncode == 2, result.stderr
    assert "at least one of --out-latin or --out-ipa" in result.stderr
