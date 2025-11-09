#!/usr/bin/env python3
import json
import logging
import os
import pathlib
import sys
from typing import TextIO

import click
import fasttext

from ..error_service import init_error_service, set_correlation_id
from ..lang_filter import RU_ONLY, is_russian_token
from ..logging_config import setup as _log_setup
from ..model_utils import ensure_fasttext_model

logger = logging.getLogger(__name__)


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
@click.option(
    "--min-len", type=int, default=3, show_default=True, help="Minimum token length"
)
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
    """Filter or mask Russian tokens in text, with configurable detection settings.

    Example usage:
        python -m turkic_translit.cli.filter_russian --mode mask --thr 0.5 --margin 0.1 --fallback-orth < input.txt > output.txt

    Options:
        --thr: Confidence threshold for identifying Russian (0.0 to 1.0)
        --margin: Maximum gap allowed when Russian is not the top language (0.0 to 1.0)
        --fallback-orth: Apply pure-Cyrillic test regardless of threshold
        --debug: Emit structured debug info (JSON) to stderr
    """
    # Configure logging and error service for direct module execution
    _log_setup()
    init_error_service()
    set_correlation_id(os.getenv("TURKIC_CORRELATION_ID"))

    # Use our model_utils to find or download the model
    try:
        model_path = ensure_fasttext_model()
        logger.info(f"Using FastText model at {model_path}")

        # Set FastText parameter to load the full model in memory
        # This is important for the .bin model to work correctly
        fasttext.FastText.eprint = lambda x: None  # Suppress C++ warnings
        lid = fasttext.load_model(str(model_path))
        logger.info(
            f"Loaded model of size {pathlib.Path(model_path).stat().st_size} bytes"
        )

        # Verify model works by testing a known Russian word
        test_result = lid.predict("привет", k=1)
        logger.info(f"Model test with 'привет': {test_result}")

        if test_result[0][0] != "__label__ru":
            logger.warning(
                f"Model may not be working correctly! Test prediction for 'привет': {test_result}"
            )
    except Exception as e:
        logger.exception("Failed to load FastText model")
        raise click.ClickException(f"Failed to load FastText model: {e}") from e

    uz_core = set()
    if stoplist:
        with open(stoplist, encoding="utf-8") as f:
            uz_core = {line.strip().lower() for line in f if line.strip()}

    # Do not override global logging configuration in web context.

    # Debug output function for CLI mode
    def debug_token(tok: str, lbl: list[str], conf: list[float]) -> None:
        if debug and not os.environ.get("GRADIO"):
            # Find the index of Russian label, if present
            ru_idx = lbl.index("__label__ru") if "__label__ru" in lbl else None
            ru_conf = conf[ru_idx] if ru_idx is not None else 0.0

            # Create debug info object
            debug_info = {
                "tok": tok,
                "rank1": lbl[0].replace("__label__", ""),
                "conf1": round(conf[0], 2),
                "ru_conf": round(ru_conf, 2),
            }

            # Write to stderr as JSON
            print(json.dumps(debug_info), file=sys.stderr)

    for line in input:
        out = []
        for tok in line.strip().split():
            # Get the token prediction once for both decision and debug
            t = tok.lower()
            lbl, conf = lid.predict(t, k=3) if len(t) >= min_len else ([], [])
            conf = conf.tolist() if len(conf) > 0 else []

            # Debug output in CLI mode
            debug_token(tok, lbl, conf)

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
        print(" ".join(out), file=output)


if __name__ == "__main__":  # pragma: no cover
    main()
