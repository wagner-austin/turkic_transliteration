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
from pathlib import Path
from typing import Final

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


__all__ = [
    "FORMAT_CATEGORY",
    "PACKAGED_FOLDS",
    "normalize_line",
    "strip_format_characters",
]
