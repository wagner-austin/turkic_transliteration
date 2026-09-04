"""Source-text normalization applied before transliteration.

Three classes of character reached the published corpora because raw web
text carries codepoints the rule files cannot see past. An invisible
format character sits inside a word, where it blocks a contextual rule
at transliteration time and later splits the word. An Arabic
presentation form spells a native Uyghur letter as a display codepoint
the rules do not name, so whole native words passed through
untransliterated. And a page authored in one single-byte codepage but
decoded as another arrives with a native letter swapped for a lookalike,
as Turkish satın does when it arrives spelled satýn.

The first two classes are closed by Unicode itself: every format
character is category Cf, and compatibility normalization folds every
presentation form to the letter it displays. The third is a table of
verified repairs with the same columns and reader as the symbol map, so
each fold carries its scope, its evidence and its citation.
"""

from __future__ import annotations

import unicodedata as ud
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

from turkic_translit.corpus.symbols import read_symbol_map, symbol_table_digest
from turkic_translit.digests import require_sha256_hex
from turkic_translit.validation import require_non_empty_str, require_present

FORMAT_CATEGORY: Final = "Cf"

PACKAGED_FOLDS: Final = Path(__file__).parent / "folds.csv"


def strip_format_characters(text: str) -> str:
    """Remove every Unicode format character.

    Category Cf covers the whole class in one test: soft hyphens,
    zero-width spaces and joiners, byte-order marks and directional
    controls. None of them is text, and every one of them can sit inside
    a word, where it blocks a transliteration rule from matching across
    it.

    Args:
        text: Text possibly carrying format characters.

    Returns:
        The text with every format character removed.
    """
    return "".join(char for char in text if ud.category(char) != FORMAT_CATEGORY)


def normalize_line(fragment: str) -> str:
    """Collapse a raw fragment into a single normalised line.

    Format characters are removed first, so a soft hyphen never reaches
    a rule file or a vocabulary. Compatibility normalization then folds
    presentation forms, ligatures and width variants to the characters
    they display, and composes diacritics, so the same word is one
    string regardless of how the source encoded it. Whitespace is
    collapsed last, because the normalization itself can turn a
    no-break space into a plain one.

    Args:
        fragment: Raw text as a driver yielded it.

    Returns:
        The normalised line, empty when the fragment held no content.
    """
    folded = ud.normalize("NFKC", strip_format_characters(fragment))
    return " ".join(folded.split())


class NormalizationRecord(TypedDict):
    """What decided a corpus's normalisation, as three checkable values.

    Attributes:
        unicode_version: The Unicode database NFKC folded against.
        format_category: The general category stripped as non-text.
        folds_sha256: Digest of the parsed folds table.

    This exists because a corpus's text depends on the normaliser as much
    as on the source, and until 2026-09-03 nothing recorded which
    normaliser ran. The gap is not hypothetical. The 2026-08-12 archives
    were downloaded two days before ``3bfd89e`` added format-character
    stripping, so re-running the same command on the same shards now
    yields text differing at every line that carried a directional mark.
    Uzbek reproduces 7,270 of 7,708 lines exactly and differs on the rest
    by characters that are invisible, which is the hardest kind of
    difference to diagnose and the easiest to mistake for resampling.

    ``unicode_version`` is here rather than assumed because NFKC is
    defined against a Unicode release: the same call on two Python builds
    can fold differently, and a record naming only the folds table would
    look complete while missing that.
    """

    unicode_version: str
    format_category: str
    folds_sha256: str


def normalization_record(folds: Path = PACKAGED_FOLDS) -> NormalizationRecord:
    """Describe the normalisation this build applies.

    Args:
        folds: Folds table to digest, defaulting to the packaged one.

    Returns:
        The record, for a manifest to carry beside the corpus.
    """
    return NormalizationRecord(
        unicode_version=ud.unidata_version,
        format_category=FORMAT_CATEGORY,
        folds_sha256=symbol_table_digest(read_symbol_map(folds)),
    )


def encode_normalization_record(
    record: NormalizationRecord,
) -> dict[str, str | int | float | bool]:
    """Render a normalisation record as a plain mapping ready for JSON.

    The value type is wider than the three strings this returns because a
    dict's value type is invariant: a ``dict[str, str]`` is not accepted
    where the manifest's nested-record type is expected, even though every
    value would satisfy it.

    Args:
        record: The record to encode.

    Returns:
        A mapping carrying every field.
    """
    return {
        "unicode_version": record["unicode_version"],
        "format_category": record["format_category"],
        "folds_sha256": record["folds_sha256"],
    }


def decode_normalization_record(
    source: Mapping[str, str | int | float | bool],
) -> NormalizationRecord:
    """Validate a loosely-typed mapping into a :class:`NormalizationRecord`.

    Args:
        source: Mapping holding every field, as read from JSON.

    Returns:
        A fully validated record.

    Raises:
        FieldError: If any field is missing, of the wrong type, or empty.
    """
    return NormalizationRecord(
        unicode_version=require_non_empty_str(
            "unicode_version", require_present("unicode_version", source)
        ),
        format_category=require_non_empty_str(
            "format_category", require_present("format_category", source)
        ),
        folds_sha256=require_sha256_hex("folds_sha256", require_present("folds_sha256", source)),
    )


__all__ = [
    "FORMAT_CATEGORY",
    "PACKAGED_FOLDS",
    "NormalizationRecord",
    "decode_normalization_record",
    "encode_normalization_record",
    "normalization_record",
    "normalize_line",
    "strip_format_characters",
]
