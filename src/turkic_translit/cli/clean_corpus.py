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
from turkic_translit.corpus.symbols import (
    PACKAGED_SYMBOL_MAP,
    SymbolRule,
    apply_substitutions,
    read_symbol_map,
    substitutions_for,
)
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


def harmonize(
    inputs: dict[str, Path], output_dir: Path, rules: tuple[SymbolRule, ...]
) -> list[str]:
    """Rewrite files with the symbol map applied and nothing else done.

    For evaluation texts that must stay line-for-line intact: no filtering,
    no deduplication, no equalisation. Each file gets its language's
    substitutions and keeps its name, so section headers and markers
    survive for whatever parses the file downstream.

    Args:
        inputs: Language code to the file for that language.
        output_dir: Where the harmonised files are written.
        rules: Rows of a symbol map.

    Returns:
        The language codes written, in sorted order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for language in sorted(inputs):
        text = inputs[language].read_text(encoding="utf-8")
        harmonised = apply_substitutions(text, substitutions_for(rules, language))
        (output_dir / inputs[language].name).write_text(harmonised, encoding="utf-8")
        written.append(language)
    return written


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
            f"dropped {stats['dropped_foreign_tokens']:,} foreign tokens, "
            f"replaced {stats['chars_replaced']:,} chars, "
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
    default=None,
    help="Directory holding the raw transliterated corpora.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write the cleaned corpora and the run report.",
)
@click.option(
    "--harmonize-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory of evaluation texts to rewrite with the symbol map only.",
)
@click.option(
    "--harmonize-output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write the harmonised evaluation texts.",
)
@click.option(
    "--pattern",
    default="*.txt",
    show_default=True,
    help="Glob selecting files within either input directory.",
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
    input_dir: Path | None,
    output_dir: Path | None,
    harmonize_dir: Path | None,
    harmonize_output_dir: Path | None,
    pattern: str,
    symbol_map: Path,
    min_line_chars: int,
    min_ipa_ratio: float,
) -> None:
    """Clean training corpora, harmonise evaluation texts, or both.

    Corpus cleaning filters, deduplicates and equalises; harmonisation
    applies the symbol map and nothing else, for evaluation texts whose
    line structure must survive. Each mode takes a directory pair, and at
    least one pair must be given.

    Args:
        input_dir: Directory holding the raw corpora.
        output_dir: Where the cleaned corpora and report are written.
        harmonize_dir: Directory of evaluation texts to harmonise.
        harmonize_output_dir: Where the harmonised texts are written.
        pattern: Glob selecting files in either input directory.
        symbol_map: CSV of symbol decisions.
        min_line_chars: Shortest line to keep.
        min_ipa_ratio: Lowest transcription-character ratio to keep.

    Raises:
        click.ClickException: When a directory pair is half-given, no pair
            is given at all, a directory holds nothing matching the
            pattern, or a file name is ambiguous about its language.
    """
    configure_logging(default_level())
    if (input_dir is None) != (output_dir is None):
        raise click.ClickException("corpus cleaning needs both --input-dir and --output-dir")
    if (harmonize_dir is None) != (harmonize_output_dir is None):
        raise click.ClickException(
            "harmonising needs both --harmonize-dir and --harmonize-output-dir"
        )
    if input_dir is None and harmonize_dir is None:
        raise click.ClickException(
            "nothing to do: give --input-dir/--output-dir, "
            "--harmonize-dir/--harmonize-output-dir, or both"
        )

    rules = read_symbol_map(symbol_map)

    if input_dir is not None and output_dir is not None:
        inputs = discover(input_dir, pattern)
        if not inputs:
            message = f"no file matching {pattern!r} in {input_dir}"
            raise click.ClickException(message)
        report = clean_corpora(inputs, output_dir, rules, min_line_chars, min_ipa_ratio)
        (output_dir / REPORT_NAME).write_text(
            json.dumps(encode_clean_report(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        click.echo(render(report))
        click.echo(f"report: {output_dir / REPORT_NAME}")

    if harmonize_dir is not None and harmonize_output_dir is not None:
        texts = discover(harmonize_dir, pattern)
        if not texts:
            message = f"no file matching {pattern!r} in {harmonize_dir}"
            raise click.ClickException(message)
        written = harmonize(texts, harmonize_output_dir, rules)
        click.echo(f"harmonised: {', '.join(written)}")
