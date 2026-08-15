"""Finnish words checked against Karlsson's grammar.

Karlsson's Finnish: A Comprehensive Grammar (Routledge, 2018) prints
IPA for whole words in its pronunciation chapter, which the monograph
fi_ipa.rules cites mostly does not. The fourteen words here cover the
two velar-nasal contexts (n before k, intervocalic ng), vowel and
consonant length in every position Karlsson illustrates, and the plain
letter values. Words whose printed IPA carries the h allophones [ç x ɦ]
(vihko, tuhka, miehen) are excluded, because the rule file declares
those allophones out of scope; sähkö and huono, which Karlsson prints
with plain [h], are included instead.

Two conventions separate Karlsson's printed forms from this project's:
the primary-stress mark on the first syllable is dropped, because the
rules do not mark stress, and the v glyph his bracketed forms use is
the ʋ his own letter table assigns the letter (p. 16).
"""

from __future__ import annotations

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source
from turkic_translit.testing import as_project_notation

# Karlsson's printed forms are citation forms in brackets; his letter
# table assigns ⟨v⟩ the value ʋ, so the v glyph of his bracketed forms is
# the same segment.
NOTATION: tuple[tuple[str, str, str], ...] = (
    ("[", "", "citation brackets, not segments"),
    ("]", "", "citation brackets, not segments"),
    ("v", "ʋ", "his letter table assigns the letter ʋ, p. 16"),
)

DECLARED_SIMPLIFICATIONS: tuple[tuple[str, str, str], ...] = (
    ("ˈ", "", "the rules do not mark stress; Finnish stress is fixed initial"),
)

PRIMARY_SOURCE = "https://urn.fi/URN:ISBN:9789514289842"
CORROBORATING_ISBN = "978-1-138-82103-3"

# (word, expected IPA, Karlsson's printed form, page)
WORDS: tuple[tuple[str, str, str, str], ...] = (
    ("rengas", "reŋːɑs", "[reŋːɑs]", "'tyre', p. 14"),
    ("rangaista", "rɑŋːɑistɑ", "[rɑŋːɑistɑ]", "'to punish', p. 14"),
    ("Helsinki", "helsiŋki", "[helsiŋki]", "n before k, p. 14"),
    ("Helsingissä", "helsiŋːisːæ", "[ˈhelsiŋːisːæ]", "'in Helsinki', p. 21"),
    ("taloon", "tɑloːn", "[ˈtɑloːn]", "'into the house', p. 20"),
    ("hyppään", "hypːæːn", "[ˈhypːæːn]", "'I jump', p. 20"),
    ("kaappiin", "kɑːpːiːn", "[ˈkɑːpːiːn]", "'into the cupboard', p. 20"),
    ("ravintolaan", "rɑʋintolɑːn", "[ˈrɑvintolɑːn]", "'into the restaurant', p. 20"),
    ("talossaan", "tɑlosːɑːn", "[ˈtɑlosːɑːn]", "'in her/his house', p. 20"),
    ("aatteellinen", "ɑːtːeːlːinen", "[ˈɑːtːeːlːinen]", "'ideological', p. 21"),
    ("aatelinen", "ɑːtelinen", "[ˈɑːtelinen]", "'belonging to the nobility', p. 21"),
    ("ateelinen", "ɑteːlinen", "[ˈɑteːlinen]", "'atelic', p. 21"),
    ("huono", "huono", "[huono]", "'bad', p. 17"),
    ("sähkö", "sæhkø", "[sæhkø]", "'electricity', p. 17"),
)


def test_the_declared_source_is_still_the_monograph() -> None:
    """Corroboration adds a source; it does not replace the cited one."""
    declared = read_rule_source(_RULE_DIR / "fi_ipa.rules")

    assert declared["identifier"] == PRIMARY_SOURCE


def test_the_rule_file_names_the_corroborating_grammar() -> None:
    """The corroboration lives in the rule file, not only in this test."""
    text = (_RULE_DIR / "fi_ipa.rules").read_text(encoding="utf-8")

    assert CORROBORATING_ISBN in text


@pytest.mark.parametrize(
    ("word", "expected", "printed", "where"), WORDS, ids=[w for w, _, _, _ in WORDS]
)
def test_word_matches_the_printed_ipa(word: str, expected: str, printed: str, where: str) -> None:
    """Each printed word transliterates to Karlsson's IPA, our conventions.

    Args:
        word: The word in Finnish orthography.
        expected: Karlsson's IPA with stress dropped and v as ʋ.
        printed: The IPA as the grammar prints it.
        where: Gloss or context, and the page that prints the pair.
    """
    assert printed != ""
    assert where != ""
    assert as_project_notation(printed, NOTATION, DECLARED_SIMPLIFICATIONS) == expected
    assert to_ipa(word, "fi") == expected
