"""Turn raw transliterated corpora into comparable training corpora.

A corpus streamed from the web and pushed through a transliterator is not
yet trainable. It repeats its site boilerplate, carries stretches of
foreign material the rules rightly passed through, and is a different
size from the corpus beside it — so a model trained on one has seen more
data than a model trained on another, and the comparison between them
measures the difference in data as much as the difference in language.

This module is the step that closes those gaps, in a fixed order:

1. Apply the symbol map, so every language spells the same sound the
   same way. See :mod:`turkic_translit.corpus.symbols`.
2. Drop lines whose transcription-character ratio is below the threshold,
   which removes the foreign-script and emoji leakage wholesale.
3. Drop every token carrying a letter the language's rules cannot emit.
   Such a letter got there by passing through untransliterated, so the
   token is quoted foreign material; dropping it whole avoids shredding
   it into fragments that read as native words.
4. Replace whatever else the rules cannot emit — punctuation, digits,
   stray symbols — with spaces. Punctuation is corpus style, not
   phonology: its frequency profile differs between the source sites of
   different languages, and a model reads that difference as a
   difference between the languages. A space rather than a deletion, so
   that removing a character never fuses its neighbours into a sequence
   the language does not have.
5. Drop lines shorter than the minimum, which are menu items and
   section headings rather than prose.
6. Drop repeated lines, keeping the first, which removes navigation
   furniture and registry dumps.
7. Truncate every corpus to the size of the smallest, so the languages
   are trained on equal amounts of text.

A cleaned corpus therefore contains exactly the characters its rules can
emit, plus the space and the newline, and its vocabulary is the
transcription inventory rather than an inventory plus residue. What was
dropped or replaced is counted in the report, per language, so an
upstream defect surfaces as a number in the manifest instead of
vanishing into spaces. The report also fingerprints the rule files and
the symbol map, so a corpus can be checked against the rules that exist
now rather than trusted to match them.

The last cleaning step is why the lowest-resource language in the set
decides the budget for all of them, and why the report says which one
that was.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata as ud
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

from turkic_translit.core import _RULE_DIR
from turkic_translit.corpus.errors import NoCorporaError
from turkic_translit.corpus.inventory import emitted_characters
from turkic_translit.corpus.symbols import (
    SymbolRule,
    apply_substitutions,
    encode_symbol_rule,
    substitutions_for,
)
from turkic_translit.validation import (
    require_non_empty_str,
    require_non_negative_int,
    require_present,
)

DEFAULT_MIN_LINE_CHARS: Final = 30
DEFAULT_MIN_IPA_RATIO: Final = 0.95

# Characters that count as transcription content for the line-keeping
# ratio. The Latin base letters cover plain segments; the extension block
# covers every IPA symbol the cited descriptions use for these languages;
# the diacritics cover length, the tie bar, palatalisation and the marks
# that ride on a segment; and the punctuation and digits occur in running
# prose, so a normal sentence is not dropped for carrying them. The ratio
# set is wider than what survives sanitising: a line earns its place by
# being prose, and is then reduced to the characters the rules can emit.
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
        dropped_foreign_tokens: Tokens dropped for carrying a letter the
            language's rules cannot emit.
        chars_replaced: Characters replaced with spaces because the rules
            cannot emit them, punctuation and digits included.
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
    dropped_foreign_tokens: int
    chars_replaced: int
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
        rules_fingerprint: SHA-256 of each language's rule file and of
            the symbol map the run applied, so the corpora can be
            checked against the rules that exist now.
        languages: Per-language statistics, keyed by language code.
    """

    min_line_chars: int
    min_ipa_ratio: float
    equalized_char_budget: int
    budget_language: str
    rules_fingerprint: dict[str, str]
    languages: dict[str, CleanStats]


_STAT_FIELDS: Final = (
    "lines_in",
    "dropped_duplicate",
    "dropped_short",
    "dropped_low_ipa",
    "dropped_foreign_tokens",
    "chars_replaced",
    "lines_kept",
    "chars_kept",
    "chars_written",
)


def decode_clean_stats(
    source: Mapping[str, str | int | float | bool | None | Mapping[str, str | int | float | bool]],
) -> CleanStats:
    """Validate a loosely-typed mapping into :class:`CleanStats`.

    Args:
        source: Mapping holding the nine counts.

    Returns:
        A fully validated statistics record.

    Raises:
        FieldError: If a count is missing, not an integer, or negative.
    """
    counts = {
        field: require_non_negative_int(field, require_present(field, source))
        for field in _STAT_FIELDS
    }
    return CleanStats(
        lines_in=counts["lines_in"],
        dropped_duplicate=counts["dropped_duplicate"],
        dropped_short=counts["dropped_short"],
        dropped_low_ipa=counts["dropped_low_ipa"],
        dropped_foreign_tokens=counts["dropped_foreign_tokens"],
        chars_replaced=counts["chars_replaced"],
        lines_kept=counts["lines_kept"],
        chars_kept=counts["chars_kept"],
        chars_written=counts["chars_written"],
    )


def encode_clean_stats(stats: CleanStats) -> dict[str, int]:
    """Render a statistics record to a plain mapping.

    Args:
        stats: The record to encode.

    Returns:
        A mapping carrying exactly the nine counts.
    """
    return {
        "lines_in": stats["lines_in"],
        "dropped_duplicate": stats["dropped_duplicate"],
        "dropped_short": stats["dropped_short"],
        "dropped_low_ipa": stats["dropped_low_ipa"],
        "dropped_foreign_tokens": stats["dropped_foreign_tokens"],
        "chars_replaced": stats["chars_replaced"],
        "lines_kept": stats["lines_kept"],
        "chars_kept": stats["chars_kept"],
        "chars_written": stats["chars_written"],
    }


def encode_clean_report(
    report: CleanReport,
) -> dict[str, str | int | float | dict[str, str] | dict[str, dict[str, int]]]:
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
        "rules_fingerprint": dict(report["rules_fingerprint"]),
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
    fingerprint_value = require_present("rules_fingerprint", source)
    if not isinstance(fingerprint_value, Mapping):
        raise TypeError(
            f"rules_fingerprint must be a mapping, got {type(fingerprint_value).__name__}"
        )
    fingerprint = {
        str(name): require_non_empty_str("rules_fingerprint value", digest)
        for name, digest in fingerprint_value.items()
    }
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
        rules_fingerprint=fingerprint,
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


def sanitize_line(line: str, emitted: frozenset[str]) -> tuple[str, int, int]:
    """Reduce a line to the characters the language's rules can emit.

    A token carrying a letter outside the emitted set is quoted foreign
    material — the rules cannot produce that letter, so it passed
    through untransliterated — and is dropped whole. Stripping single
    characters out of such a token would leave fragments that read as
    native words, which is exactly the false signal a character-level
    model would learn. In the surviving tokens, every other character
    the rules cannot emit is replaced by a space.

    Args:
        line: A line that passed the ratio test.
        emitted: The language's emitted character set.

    Returns:
        The sanitised line, the count of dropped tokens, and the count
        of replaced characters.
    """
    kept: list[str] = []
    dropped_tokens = 0
    replaced = 0
    for token in line.split():
        if any(ud.category(char).startswith("L") and char not in emitted for char in token):
            dropped_tokens += 1
            continue
        chars: list[str] = []
        for char in token:
            if char in emitted:
                chars.append(char)
            else:
                replaced += 1
                chars.append(" ")
        kept.append("".join(chars))
    return " ".join(" ".join(kept).split()), dropped_tokens, replaced


def clean_lines(
    lines: list[str],
    substitutions: Mapping[str, str],
    min_line_chars: int,
    min_ipa_ratio: float,
    emitted: frozenset[str],
) -> tuple[list[str], CleanStats]:
    """Run the per-line pipeline over one language.

    The length test runs twice, before and after sanitising, because
    sanitising can shorten a line past the minimum.

    Args:
        lines: Raw corpus lines.
        substitutions: This language's symbol-map rewrites.
        min_line_chars: Shortest line to keep.
        min_ipa_ratio: Lowest transcription-character ratio to keep.
        emitted: The characters this language's rules can emit.

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
        "dropped_foreign_tokens": 0,
        "chars_replaced": 0,
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
        line, dropped_tokens, replaced = sanitize_line(line, emitted)
        stats["dropped_foreign_tokens"] += dropped_tokens
        stats["chars_replaced"] += replaced
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


def rules_fingerprint(languages: tuple[str, ...], rules: tuple[SymbolRule, ...]) -> dict[str, str]:
    """SHA-256 of each rule file used, and of the symbol map applied.

    The rule files are hashed from the packaged directory because that
    is where :func:`turkic_translit.core.to_ipa` reads them; the symbol
    map is hashed from the rows the run actually applied, which may
    have come from a caller-supplied file.

    Args:
        languages: The language codes cleaned in this run.
        rules: Rows of the symbol map the run applied.

    Returns:
        File or table name to hex digest.
    """
    fingerprint: dict[str, str] = {}
    for language in languages:
        name = f"{language}_ipa.rules"
        fingerprint[name] = hashlib.sha256((_RULE_DIR / name).read_bytes()).hexdigest()
    encoded = json.dumps([encode_symbol_rule(rule) for rule in rules], ensure_ascii=False)
    fingerprint["symbol_map"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return fingerprint


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
        KeyError: If a language has no rule file to derive its emitted
            characters from.
    """
    if not inputs:
        raise NoCorporaError(str(output_dir))

    languages = tuple(sorted(inputs))
    cleaned: dict[str, list[str]] = {}
    statistics: dict[str, CleanStats] = {}
    for language in languages:
        substitutions = substitutions_for(rules, language)
        emitted = emitted_characters(language)
        raw = inputs[language].read_text(encoding="utf-8").splitlines()
        kept, stats = clean_lines(raw, substitutions, min_line_chars, min_ipa_ratio, emitted)
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
        rules_fingerprint=rules_fingerprint(languages, rules),
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
    "rules_fingerprint",
    "sanitize_line",
    "transcription_ratio",
    "truncate_to_budget",
]
