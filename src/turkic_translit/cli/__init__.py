"""``turkic_translit.cli`` package — the ``turkic-translit`` console script.

Exposes a :class:`click.Group` named :func:`main` that dispatches to
seven subcommands:

* ``translit`` — file-to-file IPA/Latin transliteration
* ``build-spm`` — assemble a SentencePiece training corpus
* ``download-corpus`` — stream text from OSCAR / Wikipedia / Leipzig
* ``filter-russian`` — drop code-switched Russian tokens
* ``train-spm`` — train a SentencePiece model
* ``train-lm`` — fine-tune a causal LM on a Turkic corpus
* ``eval-lm`` — evaluate a causal LM on a held-out corpus

Every command's implementation lives in a sibling module named after
the command; the corresponding function is imported and registered
here. All dependencies are declared in :file:`pyproject.toml` as core
dependencies, so imports never need to be guarded.
"""

from __future__ import annotations

import os

import click

from ..error_service import init_error_service, set_correlation_id
from ..logging_config import setup as _log_setup
from .build_spm import main as _build_spm
from .download_corpus import cli as _download_corpus
from .eval_lm import cli as _eval_lm
from .filter_russian import main as _filter_russian
from .train_lm import cli as _train_lm
from .train_spm import main as _train_spm
from .translit import translit as _translit


@click.group()
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
        log_level: Value of ``--log-level``; written to the
            ``TURKIC_LOG_LEVEL`` environment variable and consumed by
            the logging setup routine so every subcommand inherits the
            same level.
    """
    os.environ["TURKIC_LOG_LEVEL"] = log_level.upper()
    _log_setup()
    init_error_service()
    set_correlation_id(os.getenv("TURKIC_CORRELATION_ID"))


main.add_command(_translit, "translit")
main.add_command(_build_spm, "build-spm")
main.add_command(_download_corpus, "download-corpus")
main.add_command(_filter_russian, "filter-russian")
main.add_command(_train_spm, "train-spm")
main.add_command(_train_lm, "train-lm")
main.add_command(_eval_lm, "eval-lm")


# Re-exported for ``from turkic_translit.cli import train_spm``.
train_spm = _train_spm

__all__ = ["main", "train_spm"]
