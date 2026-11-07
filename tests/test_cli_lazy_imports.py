"""Console scripts import only what the command they run needs.

Until 0.5.11 the ``turkic_translit.cli`` package imported every subcommand
when it was imported, so ``turkic-clean-corpus``, which lives in that
package and uses no model, imported ``torch`` and ``transformers`` through
``eval-lm`` before parsing its arguments. Measured on 2026-11-07: on a host
whose ``torch`` could not load its native library, the corpus cleaner could
not print its own ``--help``. Each case here runs a real interpreter, so the
module table it reads is the one a user's process would have.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from turkic_translit.cli import COMMANDS, load_command, main

HEAVY = ("torch", "transformers", "gradio", "sentencepiece")

# Runs a command the way its console script does, then reports which of
# the heavy modules the process imported. The command's own SystemExit
# from click is caught by standalone_mode=False.
PROBE = """
import json, sys
from importlib import import_module
module_name, attribute, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
command = getattr(import_module(module_name), attribute)
command.main(args=argv, standalone_mode=False)
print(json.dumps(sorted(m for m in {heavy!r} if m in sys.modules)))
"""


def _heavy_modules_after(module: str, attribute: str, argv: list[str], cwd: Path) -> list[str]:
    """Run one command in a fresh interpreter and list the heavy modules it loaded.

    Args:
        module: Module defining the command.
        attribute: The command's name in that module.
        argv: Arguments to the command.
        cwd: Directory to run in.

    Returns:
        The heavy modules present in ``sys.modules`` once the command returned.
    """
    completed = subprocess.run(
        [sys.executable, "-c", PROBE.format(heavy=HEAVY), module, attribute, *argv],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=True,
    )
    loaded: list[str] = json.loads(completed.stdout.strip().splitlines()[-1])
    return loaded


def test_transliterating_and_cleaning_load_no_model_library(tmp_path: Path) -> None:
    """translit then turkic-clean-corpus, each in its own process, load none."""
    source = tmp_path / "kk.txt"
    source.write_text("саясат туралы мақала және саясат\n", encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir()

    after_translit = _heavy_modules_after(
        "turkic_translit.cli",
        "main",
        ["translit", "--lang", "kk", "--in", str(source), "--out-ipa", str(raw / "kk_ipa.txt")],
        tmp_path,
    )
    after_clean = _heavy_modules_after(
        "turkic_translit.cli.clean_corpus",
        "cli",
        ["--input-dir", str(raw), "--output-dir", str(tmp_path / "clean")],
        tmp_path,
    )

    assert after_translit == []
    assert after_clean == []
    assert "sɑjɑsɑt" in (tmp_path / "clean" / "kk_ipa.txt").read_text(encoding="utf-8")


def test_the_language_model_commands_still_load_their_libraries(tmp_path: Path) -> None:
    """eval-lm's help imports torch: laziness defers it, it does not drop it."""
    assert _heavy_modules_after("turkic_translit.cli", "main", ["eval-lm", "--help"], tmp_path) == [
        "torch",
        "transformers",
    ]


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_every_listed_command_runs_under_its_own_name(name: str) -> None:
    """Each table entry loads a command whose help click prints under that name."""
    command = load_command(name)
    result = CliRunner().invoke(main, [name, "--help"])

    assert result.exit_code == 0, result.output
    assert result.output.startswith(f"Usage: main {name} [OPTIONS]")
    assert command.help is not None
    assert command.help.split("\n")[0].strip() in result.output


def test_help_lists_every_command_and_an_unknown_one_is_refused() -> None:
    """The group's help names all eight; a typo is click's usage error."""
    listed = CliRunner().invoke(main, ["--help"])
    unknown = CliRunner().invoke(main, ["translitt"])

    assert listed.exit_code == 0, listed.output
    for name in (
        "translit",
        "web",
        "build-spm",
        "download-corpus",
        "filter-russian",
        "train-spm",
        "train-lm",
        "eval-lm",
    ):
        assert name in listed.output
    assert unknown.exit_code == 2
    assert "No such command 'translitt'" in unknown.output
