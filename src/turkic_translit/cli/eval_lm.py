"""CLI: evaluate LM cross-perplexity on a target language."""

# mypy: ignore-errors
from __future__ import annotations

import logging

import click

from ..lm import DatasetStream, LMModel, cross_perplexity

logger = logging.getLogger(__name__)


@click.command("eval-lm")
@click.option("--model", required=True, help="Path or HF repo of the trained model")
@click.option("--eval-lang", required=True, help="ISO code for evaluation corpus")
@click.option("--sample", default=50_000, show_default=True, type=int)
def cli(model: str, eval_lang: str, sample: int) -> None:
    """Compute sliding-window perplexity of *model* on *eval_lang* corpus."""
    logger.info("Loading model from %s", model)
    lm = LMModel.from_pretrained(model)

    logger.info("Streaming %d sentences of %s", sample, eval_lang)
    data = DatasetStream("oscar-2301", eval_lang, max_sentences=sample)

    ppl = cross_perplexity(lm, data)
    click.echo(f"{ppl:.2f}")
