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
    """Compute median Levenshtein distance between two files."""
    # If median_lev supports sample, pass it; otherwise, ignore
    try:
        if sample is not None:
            result = median_lev(file_a, file_b, sample=sample)
        else:
            result = median_lev(file_a, file_b)
    except TypeError:
        # Backward compatibility if sample is not supported
        result = median_lev(file_a, file_b)
    click.echo(result)


if __name__ == "__main__":  # pragma: no cover
    main()
