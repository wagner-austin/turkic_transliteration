"""``turkic_translit.cli`` package — the ``turkic-translit`` console script.

Exposes a :class:`click.Group` named :func:`main` that dispatches to
eight subcommands:

* ``translit`` — file-to-file IPA/Latin transliteration
* ``web`` — the Gradio demo
* ``build-spm`` — assemble a SentencePiece training corpus
* ``download-corpus`` — stream text from OSCAR or Wikipedia
* ``filter-russian`` — drop code-switched Russian tokens
* ``train-spm`` — train a SentencePiece model
* ``train-lm`` — fine-tune a causal LM on a Turkic corpus
* ``eval-lm`` — evaluate a causal LM on a held-out corpus

Every command's implementation lives in a sibling module named after the
command, and the group imports that module only when the command is
invoked. Importing them all here, as this package did until 0.5.11, made
every console script under it, ``turkic-clean-corpus`` included, import
``torch`` and ``transformers`` through ``eval-lm`` before doing anything:
cleaning a corpus could not start without the language-model stack, and
on a host without its native libraries it could not start at all.
"""

from __future__ import annotations

import importlib
from typing import Final

import click

from .. import _test_hooks
from ..error_service import init_error_service, set_correlation_id
from ..logging_config import setup as _log_setup

COMMANDS: Final[dict[str, tuple[str, str]]] = {
    "translit": ("translit", "translit"),
    "web": ("web", "cli"),
    "build-spm": ("build_spm", "main"),
    "download-corpus": ("download_corpus", "cli"),
    "filter-russian": ("filter_russian", "main"),
    "train-spm": ("train_spm", "main"),
    "train-lm": ("train_lm", "cli"),
    "eval-lm": ("eval_lm", "cli"),
}
"""Command name to the sibling module defining it and the command's name there."""


def load_command(name: str) -> click.Command:
    """Import the module defining a subcommand and return the command.

    Args:
        name: A key of :data:`COMMANDS`.

    Returns:
        The click command registered under ``name``.

    Raises:
        KeyError: If ``name`` is not a key of :data:`COMMANDS`.
        ModuleNotFoundError: If the module defining it, or one it imports,
            is not installed.
    """
    module_name, attribute = COMMANDS[name]
    module = importlib.import_module(f"{__name__}.{module_name}")
    command: click.Command = getattr(module, attribute)
    return command


class LazyCommandGroup(click.Group):
    """A group that imports a subcommand's module only when it is invoked."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Name every subcommand without importing any.

        Args:
            ctx: The click context.

        Returns:
            The subcommand names, sorted.
        """
        return sorted(COMMANDS)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Import and return one subcommand, or ``None`` for an unknown name.

        Args:
            ctx: The click context.
            cmd_name: The name typed on the command line.

        Returns:
            The command, or ``None`` so click reports the name as unknown.
        """
        if cmd_name not in COMMANDS:
            return None
        return load_command(cmd_name)


@click.group(cls=LazyCommandGroup)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
    show_default=True,
    help="Set logging level for all commands",
)
def main(log_level: str) -> None:
    """Turkic-Transliterate command-line tools.

    Args:
        log_level: Value of ``--log-level``; passed straight to the
            logging setup routine, which configures the root logger, so
            every subcommand inherits the same level.
    """
    _log_setup(log_level.upper())
    init_error_service()
    set_correlation_id(_test_hooks.environment.get("TURKIC_CORRELATION_ID"))


__all__ = ["COMMANDS", "LazyCommandGroup", "load_command", "main"]
