"""CLI: evaluate LM cross-perplexity on a target language.

The command streams held-out sentences and reports the score. Loading the
model and computing perplexity happen behind
:data:`turkic_translit.lm._test_hooks.evaluator`, so the sample this
command streams is exactly the sample that gets scored.
"""

from __future__ import annotations

import logging

import click

from ..lm import _test_hooks
from ..lm.data import DatasetStream
from ..logging_config import default_level
from ..logging_config import setup as _log_setup

logger = logging.getLogger(__name__)

SOURCE_ID = "oscar-2301"


@click.command("eval-lm")
@click.option("--model", required=True, help="Path or HF repo of the trained model")
@click.option("--eval-lang", required=True, help="ISO code for evaluation corpus")
@click.option("--sample", default=50_000, show_default=True, type=int)
def cli(model: str, eval_lang: str, sample: int) -> None:
    """Compute sliding-window perplexity of ``model`` on a corpus.

    Args:
        model: Local directory or Hub repository id of the model.
        eval_lang: ISO code of the language to evaluate on.
        sample: Number of held-out sentences to score.
    """
    _log_setup(default_level())
    logger.info("Streaming %d sentences of %s", sample, eval_lang)
    sentences = DatasetStream(SOURCE_ID, eval_lang, max_sentences=sample)

    logger.info("Scoring with model from %s", model)
    perplexity = _test_hooks.evaluator.perplexity(model, sentences)
    click.echo(f"{perplexity:.2f}")


__all__ = ["SOURCE_ID", "cli"]
