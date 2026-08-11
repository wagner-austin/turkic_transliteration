"""Turn raw transliterated corpora into comparable training corpora.

A corpus streamed from the web and pushed through a transliterator is not
yet trainable. It repeats its site boilerplate, carries emoji and stretches
of other scripts that no rule mapped, and is a different size from the
corpus beside it — so a model trained on one has seen more data than a
model trained on another, and the comparison between them measures the
difference in data as much as the difference in language.

This module is the step that closes those gaps, in a fixed order:

1. Apply the symbol map, so every language spells the same sound the
   same way. See :mod:`turkic_translit.corpus.symbols`.
2. Drop lines whose transcription-character ratio is below the threshold,
   which removes the foreign-script and emoji leakage wholesale.
3. Replace whatever stray characters survive with spaces, so no junk
   symbol reaches the vocabulary. A space rather than a deletion, so that
   removing a character never fuses its neighbours into a sequence the
   language does not have.
4. Drop lines shorter than the minimum, which are menu items and
   section headings rather than prose.
5. Drop repeated lines, keeping the first, which removes navigation
   furniture and registry dumps.
6. Truncate every corpus to the size of the smallest, so the languages
   are trained on equal amounts of text.

The last step is why the lowest-resource language in the set decides the
budget for all of them, and why the report says which one that was.

The thresholds and the allowed-character set are the ones this project's
own corpora were built with, and are kept here rather than in a private
script so that the published tool can rebuild what the published results
were trained on.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

from turkic_translit.corpus.errors import NoCorporaError
from turkic_translit.corpus.symbols import (
    SymbolRule,
    apply_substitutions,
    substitutions_for,
)
from turkic_translit.validation import (
    require_non_empty_str,
    require_non_negative_int,
    require_present,
)

DEFAULT_MIN_LINE_CHARS: Final = 30
DEFAULT_MIN_IPA_RATIO: Final = 0.95

# Characters that count as transcription content. The Latin base letters
# cover plain segments; the extension block covers every IPA symbol the
# cited descriptions use for these languages; the diacritics cover length,
# the tie bar, palatalisation and the marks that ride on a segment; and
# the punctuation and digits occur in running prose.
#
# The two precomposed affricate ligatures are here although the symbol map
# rewrites them, because the ratio test runs on text that has already been
# mapped only for the languages the map scopes them to. Treating them as
# junk would drop lines for using a notation this project itself emitted.
_IPA_LETTERS: Final = "abcdefghijklmnopqrstuvwxyz"
_IPA_EXTENSIONS: Final = (
    "ɑæɐɒəɘɵɛɜɪɨɯɔœøʊʏʌ"  # vowels
    "ʁʔʕɣɟɡɢɦɥɰʝŋɲɴɸɹɾʀʂʃʈʋʍχʐʑʒθðβçħɕ"  # consonants
    "ʧʤ"  # affricate ligatures, mapped away but legal before the map
)
_DIACRITICS: Final = "ː͡ʲʼ̥̃̆"
_PUNCTUATION: Final = " \t.,!?;:()«»\"'’`-–—%/0123456789"
ALLOWED_CHARS: Final = frozenset(_IPA_LETTERS + _IPA_EXTENSIONS + _DIACRITICS + _PUNCTUATION)


class CleanStats(TypedDict):
    """What happened to one language's corpus.

    Attributes:
        lines_in: Lines the raw corpus held.
        dropped_duplicate: Lines dropped as repeats of an earlier line.
        dropped_short: Lines dropped for falling under the minimum length.
        dropped_low_ipa: Lines dropped for too few transcription
            characters.
        lines_kept: Lines surviving every filter, before equalisation.
        chars_kept: Characters surviving every filter, counting one
            newline per line.
        chars_written: Characters that reached the file after truncation
            to the shared budget.
    """

    lines_in: int
    dropped_duplicate: int
    dropped_short: int
    dropped_low_ipa: int
    lines_kept: int
    chars_kept: int
    chars_written: int


class CleanReport(TypedDict):
    """What happened to the whole run.

    Attributes:
        min_line_chars: The shortest line kept.
        min_ipa_ratio: The lowest transcription-character ratio kept.
        equalized_char_budget: The size every corpus was cut to.
        budget_language: The language whose surviving text was smallest
            and therefore set that budget. This is the set's
            lowest-resource language, and naming it is the point of
            recording it: the figure is otherwise a number with no owner.
        languages: Per-language statistics, keyed by language code.
    """

    min_line_chars: int
    min_ipa_ratio: float
    equalized_char_budget: int
    budget_language: str
    languages: dict[str, CleanStats]


def decode_clean_stats(
    source: Mapping[str, str | int | float | bool | None | Mapping[str, str | int | float | bool]],
) -> CleanStats:
    """Validate a loosely-typed mapping into :class:`CleanStats`.

    Args:
        source: Mapping holding the seven counts.

    Returns:
        A fully validated statistics record.

    Raises:
        FieldError: If a count is missing, not an integer, or negative.
    """
    return CleanStats(
        lines_in=require_non_negative_int("lines_in", require_present("lines_in", source)),
        dropped_duplicate=require_non_negative_int(
            "dropped_duplicate", require_present("dropped_duplicate", source)
        ),
        dropped_short=require_non_negative_int(
            "dropped_short", require_present("dropped_short", source)
        ),
        dropped_low_ipa=require_non_negative_int(
            "dropped_low_ipa", require_present("dropped_low_ipa", source)
        ),
        lines_kept=require_non_negative_int("lines_kept", require_present("lines_kept", source)),
        chars_kept=require_non_negative_int("chars_kept", require_present("chars_kept", source)),
        chars_written=require_non_negative_int(
            "chars_written", require_present("chars_written", source)
        ),
    )


def encode_clean_stats(stats: CleanStats) -> dict[str, int]:
    """Render a statistics record to a plain mapping.

    Args:
        stats: The record to encode.

    Returns:
        A mapping carrying exactly the seven counts.
    """
    return {
        "lines_in": stats["lines_in"],
        "dropped_duplicate": stats["dropped_duplicate"],
        "dropped_short": stats["dropped_short"],
        "dropped_low_ipa": stats["dropped_low_ipa"],
        "lines_kept": stats["lines_kept"],
        "chars_kept": stats["chars_kept"],
        "chars_written": stats["chars_written"],
    }


def encode_clean_report(
    report: CleanReport,
) -> dict[str, str | int | float | dict[str, dict[str, int]]]:
    """Render a run report to a plain mapping, for writing as JSON.

    Args:
        report: The report to encode.

    Returns:
        A mapping whose ``languages`` value holds one encoded statistics
        record per language.
    """
    return {
        "min_line_chars": report["min_line_chars"],
        "min_ipa_ratio": report["min_ipa_ratio"],
        "equalized_char_budget": report["equalized_char_budget"],
        "budget_language": report["budget_language"],
        "languages": {
            language: encode_clean_stats(stats) for language, stats in report["languages"].items()
        },
    }


def decode_clean_report(
    source: Mapping[str, str | int | float | bool | None | Mapping[str, str | int | float | bool]],
) -> CleanReport:
    """Validate a loosely-typed mapping into :class:`CleanReport`.

    Args:
        source: Mapping holding the run configuration and per-language
            statistics.

    Returns:
        A fully validated report.

    Raises:
        FieldError: If a field is missing, of the wrong type, or the
            ``languages`` value is not a mapping of mappings.
    """
    languages_value = require_present("languages", source)
    if not isinstance(languages_value, Mapping):
        raise TypeError(f"languages must be a mapping, got {type(languages_value).__name__}")
    ratio = require_present("min_ipa_ratio", source)
    if not isinstance(ratio, float):
        raise TypeError(f"min_ipa_ratio must be a float, got {type(ratio).__name__}")
    languages: dict[str, CleanStats] = {}
    for language, stats in languages_value.items():
        if not isinstance(stats, Mapping):
            raise TypeError(f"statistics for {language!r} must be a mapping")
        languages[str(language)] = decode_clean_stats(stats)
    return CleanReport(
        min_line_chars=require_non_negative_int(
            "min_line_chars", require_present("min_line_chars", source)
        ),
        min_ipa_ratio=ratio,
        equalized_char_budget=require_non_negative_int(
            "equalized_char_budget", require_present("equalized_char_budget", source)
        ),
        budget_language=require_non_empty_str(
            "budget_language", require_present("budget_language", source)
        ),
        languages=languages,
    )


def transcription_ratio(line: str) -> float:
    """Fraction of a line that is transcription content.

    Args:
        line: A non-empty line.

    Returns:
        A value between 0 and 1.

    Raises:
        ZeroDivisionError: If the line is empty. Callers filter empty
            lines before reaching here; a ratio for no characters has no
            value that would be right.
    """
    allowed = sum(1 for char in line if char in ALLOWED_CHARS)
    return allowed / len(line)


def sanitize_line(line: str) -> str:
    """Replace whatever is not transcription content with a space.

    Applied only to lines that already passed the ratio test, so at most
    a few characters per line are touched. A space rather than a deletion:
    removing a character would put its neighbours next to each other and
    invent an adjacency the language never had, which is exactly the kind
    of false signal a character-level model would learn.

    Args:
        line: A line that passed the ratio test.

    Returns:
        The line reduced to allowed characters, single-spaced and
        stripped.
    """
    replaced = "".join(char if char in ALLOWED_CHARS else " " for char in line)
    return " ".join(replaced.split())


def clean_lines(
    lines: list[str],
    substitutions: Mapping[str, str],
    min_line_chars: int,
    min_ipa_ratio: float,
) -> tuple[list[str], CleanStats]:
    """Run the per-line pipeline over one language.

    The length test runs twice, before and after sanitising, because
    sanitising can shorten a line past the minimum.

    Args:
        lines: Raw corpus lines.
        substitutions: This language's symbol-map rewrites.
        min_line_chars: Shortest line to keep.
        min_ipa_ratio: Lowest transcription-character ratio to keep.

    Returns:
        The surviving lines and the statistics describing what happened.
        ``chars_written`` is zero here; equalisation fills it in.
    """
    seen: set[str] = set()
    kept: list[str] = []
    stats: CleanStats = {
        "lines_in": len(lines),
        "dropped_duplicate": 0,
        "dropped_short": 0,
        "dropped_low_ipa": 0,
        "lines_kept": 0,
        "chars_kept": 0,
        "chars_written": 0,
    }
    for raw in lines:
        line = apply_substitutions(raw, substitutions).strip()
        if len(line) < min_line_chars:
            stats["dropped_short"] += 1
            continue
        if transcription_ratio(line) < min_ipa_ratio:
            stats["dropped_low_ipa"] += 1
            continue
        line = sanitize_line(line)
        if len(line) < min_line_chars:
            stats["dropped_short"] += 1
            continue
        if line in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(line)
        kept.append(line)
    stats["lines_kept"] = len(kept)
    stats["chars_kept"] = sum(len(line) + 1 for line in kept)
    return kept, stats


def truncate_to_budget(lines: list[str], budget: int) -> list[str]:
    """Keep whole lines from the start until the budget is reached.

    Whole lines rather than a character cut, so that no corpus ends in a
    fragment of a word.

    Args:
        lines: Cleaned lines.
        budget: Most characters to keep, counting one newline per line.

    Returns:
        The longest prefix that fits.
    """
    total = 0
    kept: list[str] = []
    for line in lines:
        cost = len(line) + 1
        if total + cost > budget:
            break
        total += cost
        kept.append(line)
    return kept


def clean_corpora(
    inputs: Mapping[str, Path],
    output_dir: Path,
    rules: tuple[SymbolRule, ...],
    min_line_chars: int = DEFAULT_MIN_LINE_CHARS,
    min_ipa_ratio: float = DEFAULT_MIN_IPA_RATIO,
) -> CleanReport:
    """Clean every corpus, equalise their sizes, and write them out.

    Args:
        inputs: Language code to the raw corpus for that language.
        output_dir: Where to write the cleaned corpora. Each keeps the
            name of the file it came from.
        rules: Rows of a symbol map.
        min_line_chars: Shortest line to keep.
        min_ipa_ratio: Lowest transcription-character ratio to keep.

    Returns:
        The report, naming the budget and the language that set it.

    Raises:
        NoCorporaError: If ``inputs`` is empty. Equalising one corpus
            against no others is not a meaningful operation, and
            returning a report saying so would hide the mistake.
        FileNotFoundError: If a named corpus does not exist.
    """
    if not inputs:
        raise NoCorporaError(str(output_dir))

    languages = tuple(sorted(inputs))
    cleaned: dict[str, list[str]] = {}
    statistics: dict[str, CleanStats] = {}
    for language in languages:
        substitutions = substitutions_for(rules, language)
        raw = inputs[language].read_text(encoding="utf-8").splitlines()
        kept, stats = clean_lines(raw, substitutions, min_line_chars, min_ipa_ratio)
        cleaned[language] = kept
        statistics[language] = stats

    budget_language = min(languages, key=lambda code: statistics[code]["chars_kept"])
    budget = statistics[budget_language]["chars_kept"]

    output_dir.mkdir(parents=True, exist_ok=True)
    for language in languages:
        final = truncate_to_budget(cleaned[language], budget)
        text = "\n".join(final) + "\n" if final else ""
        (output_dir / inputs[language].name).write_text(text, encoding="utf-8")
        statistics[language]["chars_written"] = len(text)

    return CleanReport(
        min_line_chars=min_line_chars,
        min_ipa_ratio=min_ipa_ratio,
        equalized_char_budget=budget,
        budget_language=budget_language,
        languages=statistics,
    )


__all__ = [
    "ALLOWED_CHARS",
    "DEFAULT_MIN_IPA_RATIO",
    "DEFAULT_MIN_LINE_CHARS",
    "CleanReport",
    "CleanStats",
    "clean_corpora",
    "clean_lines",
    "decode_clean_report",
    "decode_clean_stats",
    "encode_clean_report",
    "encode_clean_stats",
    "sanitize_line",
    "transcription_ratio",
    "truncate_to_budget",
]
