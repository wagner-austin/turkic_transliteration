#!/usr/bin/env python3
from typing import TextIO

import click
import fasttext


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
    default=0.8,
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
def main(
    input: TextIO,
    output: TextIO,
    mode: str,
    thr: float,
    min_len: int,
    stoplist: str | None,
) -> None:
    """Filter or mask Russian tokens in text, optionally using a core-vocab stoplist."""
    lid = fasttext.load_model("lid.176.ftz")
    uz_core = set()
    if stoplist:
        with open(stoplist, encoding="utf-8") as f:
            uz_core = {line.strip().lower() for line in f if line.strip()}

    def is_ru(tok: str, thr: float) -> bool:
        lbl, conf = lid.predict(tok.lower(), k=1)
        return lbl[0] == "__label__ru" and conf[0] >= thr and tok.lower() not in uz_core

    for line in input:
        out = []
        for tok in line.strip().split():
            if len(tok) < min_len or not is_ru(tok, thr):
                out.append(tok)
            elif mode == "mask":
                out.append("<RU>")
        print(" ".join(out), file=output)


if __name__ == "__main__":  # pragma: no cover
    main()
