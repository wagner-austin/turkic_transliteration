"""Tests for the Russian-token filtering command.

This command had no tests. It now resolves its classifier through the
same explicit registry as everything else, so the resolution can be
exercised without a 126 MB model on disk: the filesystem probe reports
the weights present and the loader hands back a table-backed model, both
real implementations of the production protocols.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from click.testing import CliRunner

from turkic_translit.cli.filter_russian import main
from turkic_translit.lid import _test_hooks
from turkic_translit.lid.locations import default_search_dirs

ANSWERS = {
    "привет": [("__label__ru", 0.99), ("__label__uk", 0.01)],
    "мир": [("__label__ru", 0.97), ("__label__uk", 0.02)],
    "hello": [("__label__en", 0.98), ("__label__de", 0.01)],
    "әлем": [("__label__kk", 0.95)],
}


@pytest.fixture
def installed_weights() -> Generator[None, None, None]:
    """Present ``lid.176`` as installed and back it with a table model.

    Yields:
        None, once, with the original hooks captured.
    """
    probe, loader = _test_hooks.probe, _test_hooks.model_loader
    _test_hooks.probe = _test_hooks.MappingFileProbe(
        {default_search_dirs()[0] / "lid.176.bin": 131266198}
    )
    _test_hooks.model_loader = _test_hooks.FixedModelLoader(_test_hooks.TableFastTextModel(ANSWERS))
    yield
    _test_hooks.probe = probe
    _test_hooks.model_loader = loader


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click runner with a UTF-8 environment.

    Returns:
        The runner every command test drives.
    """
    # Logging is silenced so the captured stream carries only the command's
    # own output; the --debug reports below are written independently of
    # the logger and so survive.
    return CliRunner(env={"PYTHONIOENCODING": "utf8", "GRADIO": "", "TURKIC_LOG_LEVEL": "ERROR"})


@pytest.fixture
def corpus(tmp_path: Path) -> Callable[[str], list[str]]:
    """Return a helper that writes input to a file and names it to the CLI.

    Reading a real file rather than piping through the runner's stdin
    keeps the test independent of how a given Click version signals end
    of input: 8.2's CliRunner raises EOFError where 8.3 returns "", which
    made a correct run abort. Passing --input also exercises the option
    the command actually documents.

    Args:
        tmp_path: Directory to write the input file into.

    Returns:
        A function taking the input text and returning the ``--input``
        arguments for it.
    """

    def make(text: str) -> list[str]:
        path = tmp_path / "input.txt"
        path.write_text(text, encoding="utf-8")
        return ["--input", str(path)]

    return make


def first_line(output: str) -> str:
    """Return the command's first output line.

    Logging is configured onto the same captured stream, so the filtered
    text is read as the first line rather than as the whole buffer.

    Args:
        output: Everything the invocation wrote.

    Returns:
        The first line, stripped.
    """
    return output.splitlines()[0].strip()


def test_drop_mode_removes_russian_tokens(
    runner: CliRunner, installed_weights: None, corpus: Callable[[str], list[str]]
) -> None:
    """The default mode deletes Russian tokens from the output."""
    result = runner.invoke(main, ["--thr", "0.5", *corpus("привет мир hello\n")])
    assert result.exit_code == 0
    assert first_line(result.output) == "hello"


def test_mask_mode_replaces_russian_tokens(
    runner: CliRunner, installed_weights: None, corpus: Callable[[str], list[str]]
) -> None:
    """Mask mode preserves position by substituting a marker."""
    result = runner.invoke(main, ["--mode", "mask", "--thr", "0.5", *corpus("привет мир hello\n")])
    assert result.exit_code == 0
    assert first_line(result.output) == "<RU> <RU> hello"


def test_a_kazakh_token_survives_the_filter(
    runner: CliRunner, installed_weights: None, corpus: Callable[[str], list[str]]
) -> None:
    """A Kazakh-specific letter keeps the token out of the Russian bucket."""
    result = runner.invoke(main, ["--thr", "0.5", *corpus("әлем hello\n")])
    assert result.exit_code == 0
    assert first_line(result.output) == "әлем hello"


def test_a_stoplisted_token_is_kept(
    runner: CliRunner,
    installed_weights: None,
    corpus: Callable[[str], list[str]],
    tmp_path: Path,
) -> None:
    """Words listed in the core vocabulary are never dropped."""
    stoplist = tmp_path / "core.txt"
    stoplist.write_text("привет\n", encoding="utf-8")

    result = runner.invoke(
        main,
        ["--thr", "0.5", "--stoplist", str(stoplist), *corpus("привет мир hello\n")],
    )

    assert result.exit_code == 0
    assert first_line(result.output) == "привет hello"


def test_debug_reports_each_token_as_json(
    runner: CliRunner, installed_weights: None, corpus: Callable[[str], list[str]]
) -> None:
    """Debug mode emits one JSON object per token, naming the winner."""
    result = runner.invoke(
        main, ["--thr", "0.5", "--debug", *corpus("привет hello\n")], catch_exceptions=False
    )
    assert result.exit_code == 0
    reports = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert [report["tok"] for report in reports] == ["привет", "hello"]
    assert reports[0]["rank1"] == "ru"
    assert reports[1]["ru_conf"] == 0.0


def test_the_orthography_flag_catches_pure_cyrillic_below_the_threshold(
    runner: CliRunner, installed_weights: None, corpus: Callable[[str], list[str]]
) -> None:
    """``--fallback-orth`` applies the script test at any threshold."""
    result = runner.invoke(
        main,
        ["--thr", "0.99", "--mode", "mask", "--fallback-orth", *corpus("мир hello\n")],
    )
    assert result.exit_code == 0
    assert first_line(result.output) == "<RU> hello"
