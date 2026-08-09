#!/usr/bin/env python3
import json
import logging
from typing import TextIO

import click

from .. import _test_hooks
from ..error_service import init_error_service, set_correlation_id
from ..lang_filter import PREDICTIONS_PER_TOKEN, RU_ONLY, is_russian_token
from ..lid.factory import load_installed_classifier
from ..logging_config import default_level
from ..logging_config import setup as _log_setup

logger = logging.getLogger(__name__)

LANGUAGE_MODEL_ID = "lid.176"


@click.command()
@click.option(
    "--input",
    "-i",
    type=click.File("r", encoding="utf-8"),
    default="-",
    show_default=True,
    help="Input file (default: stdin)",
)
@click.option(
    "--output",
    "-o",
    type=click.File("w", encoding="utf-8"),
    default="-",
    show_default=True,
    help="Output file (default: stdout)",
)
@click.option(
    "--mode",
    type=click.Choice(["drop", "mask"]),
    default="drop",
    show_default=True,
    help="How to handle Russian tokens",
)
@click.option(
    "--thr",
    type=float,
    default=0.5,
    show_default=True,
    help="Confidence threshold for Russian detection",
)
@click.option("--min-len", type=int, default=3, show_default=True, help="Minimum token length")
@click.option(
    "--stoplist",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to core-vocab/stoplist file (one word per line)",
)
@click.option(
    "--margin",
    type=float,
    default=0.10,
    show_default=True,
    help="Maximum margin for accepting RU when not the top label",
)
@click.option(
    "--fallback-orth/--no-fallback-orth",
    default=False,
    show_default=True,
    help="Apply pure-Cyrillic orthography fallback regardless of threshold",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Emit structured debug output to stderr",
)
def main(
    input: TextIO,
    output: TextIO,
    mode: str,
    thr: float,
    min_len: int,
    stoplist: str | None,
    margin: float,
    fallback_orth: bool,
    debug: bool,
) -> None:
    """Drop or mask the Russian tokens in a stream of text.

    Args:
        input: Stream to read, one line at a time.
        output: Stream to write the filtered text to.
        mode: ``drop`` removes a Russian token; ``mask`` replaces it
            with ``<RU>`` so the position is preserved.
        thr: Probability at or above which a token counts as Russian.
        min_len: Tokens shorter than this are never classified, because
            short strings carry too little signal to judge.
        stoplist: Path to a file of words to exempt, one per line, or
            ``None``. Used for target-language words that a classifier
            confuses with Russian.
        margin: How far below the top language Russian may rank and
            still be accepted.
        fallback_orth: Whether a token written wholly in Russian-only
            Cyrillic letters counts as Russian regardless of the
            threshold.
        debug: Whether to write each token's top predictions to stderr
            as JSON.

    Raises:
        LidError: If the language-identification model is missing or its
            weights cannot be read.
        OSError: If the stoplist cannot be read.
    """
    # Configure logging and error service for direct module execution
    _log_setup(default_level())
    init_error_service()
    set_correlation_id(_test_hooks.environment.get("TURKIC_CORRELATION_ID"))

    # Resolution is explicit and total: a missing or truncated model
    # raises with a code naming the model and the path, rather than being
    # substituted by whatever else happens to be on disk.
    lid = load_installed_classifier(LANGUAGE_MODEL_ID)
    logger.info("Using language-identification model %s", lid.model_id)

    uz_core = set()
    if stoplist:
        with open(stoplist, encoding="utf-8") as f:
            uz_core = {line.strip().lower() for line in f if line.strip()}

    # Do not override global logging configuration in web context.

    def debug_token(tok: str) -> None:
        """Emit one token's top predictions to stderr as JSON.

        Only called when --debug is set, so the ordinary path classifies
        each token exactly once.

        Args:
            tok: The token to report on.
        """
        predictions = lid.classify_many(tok.lower(), PREDICTIONS_PER_TOKEN)
        russian = [p for p in predictions if p["label"] == "ru"]
        debug_info = {
            "tok": tok,
            "rank1": predictions[0]["label"],
            "conf1": round(predictions[0]["probability"], 2),
            "ru_conf": round(russian[0]["probability"], 2) if russian else 0.0,
        }
        click.echo(json.dumps(debug_info), err=True)

    for line in input:
        out = []
        for tok in line.strip().split():
            t = tok.lower()
            if debug and len(t) >= min_len:
                debug_token(tok)

            # Make the decision using the shared language filter
            decision = is_russian_token(
                tok, thr=thr, min_len=min_len, lid=lid, stoplist=uz_core, margin=margin
            )

            # Apply orthography fallback if requested
            if fallback_orth and not decision and len(t) >= min_len:
                decision = RU_ONLY.fullmatch(t) is not None

            if not decision:
                out.append(tok)
            elif mode == "mask":
                out.append("<RU>")
        click.echo(" ".join(out), file=output)
