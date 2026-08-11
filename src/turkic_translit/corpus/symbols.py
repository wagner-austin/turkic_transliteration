"""The symbol map that makes seven transcriptions comparable.

Seven rule files written against seven descriptions do not agree on
notation even when they agree on the sound. One writes the affricate with
a precomposed ligature and another with a tie bar; one writes the rhotic
as a tap because its source does, another as a trill. A character-level
model reads those as differences between the languages, when they are
differences between typographic conventions.

The map is a table of decisions, one row per symbol, each carrying the
verdict that justified it and the source behind that verdict. Rows marked
``merge`` rewrite one symbol to another; rows marked ``keep`` record a
contrast that was examined and deliberately preserved, so that the
absence of a merge is a decision on the record rather than an oversight.

Scope matters: some merges apply to every language, others to one. The
Turkish low vowel is written /a/ by convention in its Illustration and
/ɑ/ everywhere else, so that row is scoped to Turkish alone.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, TypedDict

from turkic_translit.corpus.errors import SymbolMapMalformedError
from turkic_translit.validation import (
    require_non_empty_str,
    require_present,
    require_str,
)

PACKAGED_SYMBOL_MAP: Final = Path(__file__).parent / "symbol_map.csv"

MERGE_ACTION: Final = "merge"
KEEP_ACTION: Final = "keep"
ALL_SCOPE: Final = "all"
SCOPE_SEPARATOR: Final = "+"

REQUIRED_COLUMNS: Final = ("action", "scope", "from", "to", "verdict", "rationale", "citation")


class SymbolRule(TypedDict):
    """One decision about one symbol.

    Attributes:
        action: ``merge`` to rewrite, ``keep`` to record a preserved
            contrast.
        scope: ``all``, or language codes joined by ``+``.
        source: The symbol the row is about.
        target: What a merge rewrites it to. Empty for a ``keep`` row.
        verdict: The judgement, e.g. ``NOTATION``, that classifies how
            much phonological information the merge costs.
        rationale: Why, in prose.
        citation: The description the verdict rests on.
    """

    action: str
    scope: str
    source: str
    target: str
    verdict: str
    rationale: str
    citation: str


def decode_symbol_rule(row: Mapping[str, str], origin: str) -> SymbolRule:
    """Validate one CSV row into a :class:`SymbolRule`.

    Args:
        row: Mapping of column name to value, as read from the CSV.
        origin: Name of the file, used in error messages.

    Returns:
        The validated rule.

    Raises:
        SymbolMapMalformedError: If a column is absent, or a merge row
            names no symbol to rewrite.
        FieldError: If a present column is not a string.
    """
    for column in REQUIRED_COLUMNS:
        if column not in row:
            raise SymbolMapMalformedError(origin, f"row has no {column!r} column")

    action = require_non_empty_str("action", require_present("action", row))
    source = require_str("from", require_present("from", row))
    if action == MERGE_ACTION and source == "":
        raise SymbolMapMalformedError(origin, "merge row names no symbol in 'from'")

    return SymbolRule(
        action=action,
        scope=require_non_empty_str("scope", require_present("scope", row)),
        source=source,
        target=require_str("to", require_present("to", row)),
        verdict=require_str("verdict", require_present("verdict", row)),
        rationale=require_str("rationale", require_present("rationale", row)),
        citation=require_str("citation", require_present("citation", row)),
    )


def encode_symbol_rule(rule: SymbolRule) -> dict[str, str]:
    """Render a rule back to the column names the CSV uses.

    Args:
        rule: The rule to encode.

    Returns:
        A mapping carrying exactly the seven columns.
    """
    return {
        "action": rule["action"],
        "scope": rule["scope"],
        "from": rule["source"],
        "to": rule["target"],
        "verdict": rule["verdict"],
        "rationale": rule["rationale"],
        "citation": rule["citation"],
    }


def parse_symbol_map(text: str, origin: str) -> tuple[SymbolRule, ...]:
    """Read every row of a symbol-map CSV.

    Both actions are returned. A caller applying substitutions filters to
    the merges itself, so that the kept rows stay readable as the record
    they are.

    Args:
        text: Full contents of the CSV.
        origin: Name of the file, used in error messages.

    Returns:
        Every row, in file order.

    Raises:
        SymbolMapMalformedError: If the file has no header, or a row is
            malformed.
        FieldError: If a column holds something other than a string.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise SymbolMapMalformedError(origin, "file is empty, so it has no header row")
    return tuple(decode_symbol_rule(row, origin) for row in reader)


def read_symbol_map(path: Path = PACKAGED_SYMBOL_MAP) -> tuple[SymbolRule, ...]:
    """Read a symbol map from disk, defaulting to the packaged one.

    Args:
        path: CSV to read.

    Returns:
        Every row, in file order.

    Raises:
        SymbolMapMalformedError: If the file is empty or a row is
            malformed.
        FieldError: If a column holds something other than a string.
    """
    return parse_symbol_map(path.read_text(encoding="utf-8"), path.name)


def substitutions_for(rules: Iterable[SymbolRule], language: str) -> dict[str, str]:
    """Collect the rewrites that apply to one language.

    A row scoped to some other language is skipped rather than refused.
    The map is authored for the whole language set, and cleaning a subset
    of that set is an ordinary thing to want; a run of two languages must
    not fail because the map also has an opinion about a third.

    That leaves a typo in a scope silently disabling its merge, which is
    caught where it can be caught properly: :func:`scopes_of` lists every
    scope a map names, and the packaged map's scopes are asserted against
    the languages this project ships rules for.

    Args:
        rules: Rows from a symbol map.
        language: The language code to select rows for.

    Returns:
        Mapping of symbol to replacement, in file order.
    """
    substitutions: dict[str, str] = {}
    for rule in rules:
        if rule["action"] != MERGE_ACTION:
            continue
        if rule["scope"] == ALL_SCOPE or language in rule["scope"].split(SCOPE_SEPARATOR):
            substitutions[rule["source"]] = rule["target"]
    return substitutions


def scopes_of(rules: Iterable[SymbolRule]) -> frozenset[str]:
    """List every language code a map's scopes name.

    Args:
        rules: Rows from a symbol map.

    Returns:
        Each code appearing in a scope, with ``all`` left out because it
        names no language.
    """
    named: set[str] = set()
    for rule in rules:
        if rule["scope"] == ALL_SCOPE:
            continue
        named.update(rule["scope"].split(SCOPE_SEPARATOR))
    return frozenset(named)


def apply_substitutions(text: str, substitutions: Mapping[str, str]) -> str:
    """Rewrite every mapped symbol in a text.

    Applied in the map's own order, which matters when one row's target
    is another row's source.

    Args:
        text: Input text.
        substitutions: Mapping of symbol to replacement.

    Returns:
        The text with every substitution applied.
    """
    for source, target in substitutions.items():
        text = text.replace(source, target)
    return text


__all__ = [
    "ALL_SCOPE",
    "KEEP_ACTION",
    "MERGE_ACTION",
    "PACKAGED_SYMBOL_MAP",
    "REQUIRED_COLUMNS",
    "SCOPE_SEPARATOR",
    "SymbolRule",
    "apply_substitutions",
    "decode_symbol_rule",
    "encode_symbol_rule",
    "parse_symbol_map",
    "read_symbol_map",
    "scopes_of",
    "substitutions_for",
]
