"""Command line for turning raw transliterated corpora into training corpora.

The step this project's own results depend on, published so that they can
be reproduced from the released tool rather than from a script that lived
beside them. See :mod:`turkic_translit.corpus.clean` for what each stage
does and why it is in the order it is.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click

from turkic_translit.corpus.clean import (
    DEFAULT_MIN_IPA_RATIO,
    DEFAULT_MIN_LINE_CHARS,
    CleanReport,
    clean_corpora,
    encode_clean_report,
)
from turkic_translit.corpus.symbols import PACKAGED_SYMBOL_MAP, read_symbol_map
from turkic_translit.logging_config import default_level
from turkic_translit.logging_config import setup as configure_logging

REPORT_NAME = "cleaning_manifest.json"

# A corpus file is named for its language somewhere in the stem, as
# oscar_ky_ipa.txt is. The code is the field that is two letters long.
# A lookahead rather than a consuming match, so that two adjacent
# candidate fields are both seen and the name is reported as ambiguous
# rather than resolved by whichever the scan reached first.
LANGUAGE_FIELD = re.compile(r"(?:^|_)([a-z]{2})(?=_|$)")


def language_of(path: Path) -> str | None:
    """Read a corpus file's language code out of its name.

    Args:
        path: The corpus file.

    Returns:
        The two-letter code, or ``None`` when the name carries exactly
        one field that could be one, so a caller can report the file
        rather than guess.
    """
    found = LANGUAGE_FIELD.findall(path.stem)
    return found[0] if len(found) == 1 else None


def discover(input_dir: Path, pattern: str) -> dict[str, Path]:
    """Find the corpora to clean and the language each belongs to.

    Args:
        input_dir: Directory to look in.
        pattern: Glob selecting corpus files.

    Returns:
        Language code to corpus path.

    Raises:
        click.ClickException: When two files claim the same language, or
            a file's name carries no single language field. Both are
            ambiguities the run must not resolve by picking one.
    """
    found: dict[str, Path] = {}
    for path in sorted(input_dir.glob(pattern)):
        code = language_of(path)
        if code is None:
            message = f"cannot tell which language {path.name} holds from its name"
            raise click.ClickException(message)
        if code in found:
            message = f"both {found[code].name} and {path.name} claim to be {code}"
            raise click.ClickException(message)
        found[code] = path
    return found


def render(report: CleanReport) -> str:
    """Lay out a run report for a terminal.

    Args:
        report: The report to render.

    Returns:
        One line per language, then the budget and which language set it.
    """
    lines = []
    for language, stats in sorted(report["languages"].items()):
        lines.append(
            f"  {language}: {stats['lines_in']:,} lines -> {stats['lines_kept']:,} kept "
            f"(duplicate {stats['dropped_duplicate']:,}, "
            f"short {stats['dropped_short']:,}, "
            f"junk {stats['dropped_low_ipa']:,}), "
            f"wrote {stats['chars_written']:,} chars"
        )
    lines.append(
        f"  budget {report['equalized_char_budget']:,} chars, set by "
        f"{report['budget_language']}, the lowest-resource language of this set"
    )
    return "\n".join(lines)


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory holding the raw transliterated corpora.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Where to write the cleaned corpora and the run report.",
)
@click.option(
    "--pattern",
    default="*.txt",
    show_default=True,
    help="Glob selecting corpus files within the input directory.",
)
@click.option(
    "--symbol-map",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=PACKAGED_SYMBOL_MAP,
    show_default="the packaged map",
    help="CSV of symbol decisions to apply.",
)
@click.option(
    "--min-line-chars",
    type=int,
    default=DEFAULT_MIN_LINE_CHARS,
    show_default=True,
    help="Drop lines shorter than this.",
)
@click.option(
    "--min-ipa-ratio",
    type=float,
    default=DEFAULT_MIN_IPA_RATIO,
    show_default=True,
    help="Drop lines with a smaller fraction of transcription characters.",
)
def cli(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    symbol_map: Path,
    min_line_chars: int,
    min_ipa_ratio: float,
) -> None:
    """Clean, harmonise and equalise a set of transliterated corpora.

    Args:
        input_dir: Directory holding the raw corpora.
        output_dir: Where the cleaned corpora and report are written.
        pattern: Glob selecting corpus files.
        symbol_map: CSV of symbol decisions.
        min_line_chars: Shortest line to keep.
        min_ipa_ratio: Lowest transcription-character ratio to keep.

    Raises:
        click.ClickException: When the input directory holds no corpus
            matching the pattern, or the names are ambiguous.
    """
    configure_logging(default_level())
    inputs = discover(input_dir, pattern)
    if not inputs:
        message = f"no file matching {pattern!r} in {input_dir}"
        raise click.ClickException(message)

    rules = read_symbol_map(symbol_map)
    report = clean_corpora(inputs, output_dir, rules, min_line_chars, min_ipa_ratio)

    (output_dir / REPORT_NAME).write_text(
        json.dumps(encode_clean_report(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    click.echo(render(report))
    click.echo(f"report: {output_dir / REPORT_NAME}")
