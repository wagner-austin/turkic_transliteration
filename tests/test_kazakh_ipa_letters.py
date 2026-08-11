"""Kazakh transliteration checked against the description the rules cite.

McCollum & Chen print the Cyrillic beside each transcription, so nothing
has to be inferred about the orthography: the expectations below are
their values, at the page they appear on. The rule file declares that it
drops the dental diacritics and the diphthongs, and those two
simplifications are applied to the published form here rather than being
folded into hand-edited constants, so the difference stays visible.

The previous version of this file described itself as a gold standard
"synchronised with" the paper while its rows had been edited to match the
rules, and annotated the edits with notes like "was d̪". That records the
divergence in a comment instead of in an assertion. Applying the declared
simplification explicitly makes the same information testable.
"""

from __future__ import annotations

import unicodedata as ud

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

INHERITS_SOURCE = "https://doi.org/10.1017/S0025100319000185"

BRIDGE_BELOW = "̪"
TIE_BAR = "͡"

# Vowel sequences the source writes as diphthongs and this project writes
# as their second and first element respectively.
COLLAPSED_DIPHTHONGS: tuple[tuple[str, str, str], ...] = (
    ("ie", "e", "the front mid diphthong is written as its offglide alone"),
    ("ij", "i", "the close front diphthong is written without its glide"),
)

TYPESET_SUBSTITUTIONS: tuple[tuple[str, str, str], ...] = (
    ("g", "ɡ", "the journal typesets the voiced velar plosive as U+0067; IPA is U+0261"),
)

# Cyrillic, McCollum & Chen's transcription, gloss, page
KEYWORDS: tuple[tuple[str, str, str, int], ...] = (
    ("піл", "pɪl̪", "elephant", 277),
    ("бас", "bɑs̪", "head", 277),
    ("мал", "mɑl̪", "livestock", 277),
    ("уақ", "wɑq", "time", 277),
    ("тас", "t̪ɑs̪", "stone", 277),
    ("дала", "d̪ɑl̪ɑ", "field", 277),
    ("нар", "n̪ɑr̪", "dromedary", 277),
    ("сат", "s̪ɑt̪", "sell.IMP", 277),
    ("зат", "z̪ɑt̪", "thing", 277),
    ("лас", "l̪ɑs̪", "dirty", 277),
    ("шақ", "ʃɑq", "tense", 277),
    ("жіп", "ʒɪp", "string", 277),
    ("кім", "kɪm", "who", 277),
    ("тау", "t̪ɑw", "mountain", 277),
    ("тән", "t̪æn̪", "body", 277),
    ("біз", "bɪz̪", "we", 277),
    ("қаш", "qɑʃ", "flee.IMP", 277),
    ("жат", "ʒɑt̪", "lie down.IMP", 278),
    ("қан", "qɑn̪", "blood", 278),
    ("хан", "χɑn̪", "khan", 278),
    ("тәж", "t̪æʒ", "crown", 278),
    ("ай", "ɑj", "moon", 278),
    ("гүл", "gʏl̪", "flower", 278),
    ("ғашық", "ʁɑʃəq", "love", 278),
)

# Keywords whose published form carries a diphthong, kept apart so the
# collapse is exercised deliberately rather than incidentally.
DIPHTHONG_KEYWORDS: tuple[tuple[str, str, str, int], ...] = (
    ("кең", "ki͡eŋ", "wide", 278),
    ("шек", "ʃi͡ek", "edge", 278),
    ("тарих", "t̪ɑr̪i͡jχ", "history", 278),
    ("риза", "r̪i͡jz̪ɑ", "satisfied", 277),
)


def as_this_project_writes_it(published: str) -> str:
    """Apply the declared simplifications to a published transcription.

    The dental diacritic and the tie bar are removed first, so the
    diphthong collapse does not depend on where the source places the
    tie over the vowel pair.

    Args:
        published: The transcription exactly as the source prints it.

    Returns:
        The same transcription in the notation these rules produce.
    """
    decomposed = ud.normalize("NFD", published)
    without_marks = "".join(c for c in decomposed if c not in {BRIDGE_BELOW, TIE_BAR})
    text = ud.normalize("NFC", without_marks)
    for diphthong, ours, _reason in COLLAPSED_DIPHTHONGS:
        text = text.replace(diphthong, ours)
    for typeset, ours, _reason in TYPESET_SUBSTITUTIONS:
        text = text.replace(typeset, ours)
    return text


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """These expectations come from the paper the rule file names."""
    declared = read_rule_source(_RULE_DIR / "kk_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "McCollum, A. G. & Chen, S."
    assert declared["year"] == 2021


@pytest.mark.parametrize(("cyrillic", "published", "gloss", "page"), KEYWORDS)
def test_keyword_transliterates_as_the_source_prints_it(
    cyrillic: str, published: str, gloss: str, page: int
) -> None:
    """Each keyword matches the source, allowing the declared simplifications."""
    assert page in {277, 278}
    assert gloss != ""
    assert to_ipa(cyrillic, "kk") == as_this_project_writes_it(published)


@pytest.mark.parametrize(("cyrillic", "published", "gloss", "page"), DIPHTHONG_KEYWORDS)
def test_published_diphthong_is_written_as_a_single_vowel(
    cyrillic: str, published: str, gloss: str, page: int
) -> None:
    """A published diphthong collapses to the single vowel these rules use."""
    assert page in {277, 278}
    assert gloss != ""
    assert TIE_BAR in ud.normalize("NFD", published), "this keyword carries no diphthong"
    assert to_ipa(cyrillic, "kk") == as_this_project_writes_it(published)


def test_the_dental_diacritic_is_dropped_rather_than_transcribed() -> None:
    """No output carries the dental mark the source uses throughout.

    The source marks every coronal as dental. These rules declare that
    they do not, so the mark must appear nowhere in their output; a rule
    reintroducing it would make the corpus inconsistent with itself.
    """
    for cyrillic, _published, _gloss, _page in KEYWORDS:
        assert BRIDGE_BELOW not in ud.normalize("NFD", to_ipa(cyrillic, "kk"))
