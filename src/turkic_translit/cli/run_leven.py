#!/usr/bin/env python3
import click

from turkic_translit.sanity import median_lev


@click.command()
@click.argument("file_a", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument("file_b", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "--sample",
    type=int,
    default=None,
    help="Sample N lines (if supported by median_lev)",
)
def main(file_a: str, file_b: str, sample: int | None) -> None:
    """Compute median Levenshtein distance between two files.

    Args:
        file_a: Path to the first file.
        file_b: Path to the second file, compared line by line.
        sample: Number of lines to sample, or ``None`` for the default.
    """
    if sample is None:
        click.echo(median_lev(file_a, file_b))
        return
    click.echo(median_lev(file_a, file_b, sample=sample))
