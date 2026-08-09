"""CLI: fine-tune a causal LM on Turkic corpora.

The command's own job is to turn language codes into one continuous
sentence stream and hand it to a trainer. Training itself happens behind
:data:`turkic_translit.lm._test_hooks.trainer`, so the stream this command
builds is exactly the stream that reaches Hugging Face.
"""

from __future__ import annotations

import itertools
import logging

import click

from ..lm import _test_hooks
from ..lm.data import DatasetStream
from ..logging_config import default_level
from ..logging_config import setup as _log_setup

logger = logging.getLogger(__name__)

MAX_SENTENCES_PER_LANGUAGE = 500_000
SOURCE_ID = "oscar-2301"


def parse_language_codes(langs: str) -> list[str]:
    """Split the ``--langs`` value into individual ISO codes.

    Args:
        langs: Comma-separated codes as typed on the command line.

    Returns:
        The codes, stripped of surrounding whitespace, in the order given.

    Raises:
        click.BadParameter: If no code survives stripping, since training
            on no language would produce an empty model rather than an
            error.
    """
    codes = [iso.strip() for iso in langs.split(",") if iso.strip()]
    if not codes:
        raise click.BadParameter("--langs must contain at least one ISO code")
    return codes


@click.command("train-lm")
@click.option("--langs", required=True, help="Comma-separated ISO codes (e.g. kk,ky)")
@click.option("--base-model", default="bigscience/bloom-560m", show_default=True)
@click.option("--epochs", default=3, show_default=True, type=int)
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
def cli(langs: str, base_model: str, epochs: int, output_dir: str) -> None:
    """Fine-tune ``base_model`` on multiple Turkic languages.

    The languages are streamed in the order given and chained rather than
    materialised, so memory does not scale with the corpus.

    Args:
        langs: Comma-separated ISO codes to train on.
        base_model: Local directory or Hub repository id to start from.
        epochs: Number of training epochs.
        output_dir: Directory the fine-tuned model is saved into.

    Raises:
        click.BadParameter: If ``--langs`` names no language.
    """
    _log_setup(default_level())
    codes = parse_language_codes(langs)

    streams = []
    for iso in codes:
        logger.info("Streaming sentences for %s", iso)
        streams.append(DatasetStream(SOURCE_ID, iso, max_sentences=MAX_SENTENCES_PER_LANGUAGE))

    _test_hooks.trainer.fine_tune(
        base_model,
        epochs=epochs,
        sentences=itertools.chain.from_iterable(streams),
        output_dir=output_dir,
    )
    click.echo(f"✓ model saved → {output_dir}")


__all__ = ["MAX_SENTENCES_PER_LANGUAGE", "SOURCE_ID", "cli", "parse_language_codes"]
