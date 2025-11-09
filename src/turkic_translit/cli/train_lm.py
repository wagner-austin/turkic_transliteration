"""CLI: fine-tune a causal LM on Turkic corpora."""

# mypy: ignore-errors
from __future__ import annotations

import logging

import click

from ..lm import DatasetStream, LMModel
from ..logging_config import setup as _log_setup

logger = logging.getLogger(__name__)


@click.command("train-lm")
@click.option("--langs", required=True, help="Comma-separated ISO codes (e.g. kk,ky)")
@click.option("--base-model", default="bigscience/bloom-560m", show_default=True)
@click.option("--epochs", default=3, show_default=True, type=int)
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
def cli(langs: str, base_model: str, epochs: int, output_dir: str) -> None:
    """Fine-tune *base_model* on multiple Turkic languages.

    Example:
        turkic-train-lm --langs kk,ky --base-model facebook/mGPT --epochs 1 \
            --output-dir runs/kkky_mgpt
    """
    _log_setup()
    iso_list = [iso.strip() for iso in langs.split(",") if iso.strip()]
    if not iso_list:
        raise click.BadParameter("--langs must contain at least one ISO code")

    sent_iterables = []
    for iso in iso_list:
        logger.info("Streaming sentences for %s", iso)
        sent_iterables.append(DatasetStream("oscar-2301", iso, max_sentences=500_000))

    # Chain without materialising full list to keep memory low
    import itertools as _it

    sentences = _it.chain.from_iterable(sent_iterables)

    LMModel.fresh(
        base_model,
        epochs=epochs,
        sentences=sentences,
        output_dir=output_dir,
    )
    click.echo(f"✓ model saved → {output_dir}")
