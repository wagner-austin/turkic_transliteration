"""Kazakh words checked against the Routledge handbook chapter.

Abish's chapter on Kazakh in the second Routledge The Turkic Languages
prints dozens of words in both Cyrillic orthography and Turcological
transcription. Unlike the Illustration kk_ipa.rules cites, some of her
transcriptions carry positional phonetics these rules declare out of
scope — prothetic glides (yel ‹ел›, woy ‹ой›), ḳ realized as χ (žaχsï
‹жақсы›), fronted a (kịtäp ‹кітап›). This file pins only the pairs whose
transcription is the plain letter-by-letter value, so every assertion
tests the map and not an unmodelled process.

The Turcological glyphs correspond to this project's IPA by fixed
conventions: a=ɑ, ä=æ, ö=ɵ, ï=ə, ị=ɪ, ụ=ʊ, ü/ụ̈=ʏ, ḳ=q, γ=ʁ, ž=ʒ, š=ʃ,
y=j. Abish's under-dot marks near-high lax vowels, which are the lax
values ɪ ə ʊ ʏ this file's cited source assigns to і ы ұ ү; her dotting
is positional, so the expected forms below are stated directly in this
project's glyphs with her printed transcription kept beside each word.
"""

from __future__ import annotations

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source
from turkic_translit.testing import as_project_notation

# Abish's Turcological glyphs in this project's notation. The lax
# near-high vowels carry a combining dot below, so those entries are
# multi-codepoint and must precede the plain letters they contain.
NOTATION: tuple[tuple[str, str, str], ...] = (
    ("ụ̈", "ʏ", "lax high front rounded, her dotted ü, our ʏ for ү"),
    ("ị", "ɪ", "lax high front unrounded, our ɪ for і"),
    ("ụ", "ʊ", "lax high back rounded, our ʊ for ұ"),
    ("ḳ", "q", "her dotted k is the back stop, our q for қ"),
    ("ï", "ə", "her high back unrounded is the central vowel of the cited source, ы"),
    ("ö", "ɵ", "her ö is the cited source's central ɵ for ө"),
    ("γ", "ʁ", "her gamma is the uvular fricative, our ʁ for ғ"),
    ("ž", "ʒ", "her haček z is the fricative, our ʒ for ж"),
    ("g", "ɡ", "Turcological g is the IPA voiced velar plosive U+0261"),
    ("a", "ɑ", "Turcological low back vowel"),
)

INHERITS_SOURCE = "https://doi.org/10.1017/S0025100319000185"
CORROBORATING_SOURCE = "https://doi.org/10.4324/9781003243809-22"

# (Cyrillic word, expected IPA, Abish's printed transcription, gloss and page)
WORDS: tuple[tuple[str, str, str, str], ...] = (
    ("ал", "ɑl", "al", "'take', p. 337"),
    ("қыс", "qəs", "ḳïs", "'winter', p. 337"),
    ("із", "ɪz", "ịz", "'trace', p. 337"),
    ("құс", "qʊs", "ḳụs", "'bird', p. 337"),
    ("күн", "kʏn", "kụ̈n", "'sun', 'day', p. 337"),
    ("көл", "kɵl", "köl", "'lake', p. 337"),
    ("гүл", "ɡʏl", "gụ̈l", "'flower', p. 337"),
    ("ғылым", "ʁələm", "γïlïm", "'science', p. 337"),
    ("кез", "kez", "kez", "'time', p. 337"),
    ("қаз", "qɑz", "ḳaz", "'goose', p. 337"),
    ("жол", "ʒol", "žol", "'road', p. 338"),
    ("жел", "ʒel", "žel", "'wind', p. 338"),
    ("жат", "ʒɑt", "žat", "'outsider', p. 338"),
    ("бала", "bɑlɑ", "bala", "'child', p. 340"),
)


def test_the_declared_source_is_still_the_illustration() -> None:
    """Corroboration adds a source; it does not replace the cited one."""
    declared = read_rule_source(_RULE_DIR / "kk_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE


def test_the_rule_file_names_the_corroborating_description() -> None:
    """The corroboration lives in the rule file, not only in this test."""
    text = (_RULE_DIR / "kk_ipa.rules").read_text(encoding="utf-8")

    assert CORROBORATING_SOURCE in text


@pytest.mark.parametrize(
    ("word", "expected", "printed", "where"), WORDS, ids=[w for w, _, _, _ in WORDS]
)
def test_word_matches_the_printed_transcription(
    word: str, expected: str, printed: str, where: str
) -> None:
    """Each word transliterates to the chapter's transcription, our glyphs.

    Args:
        word: The word in Kazakh Cyrillic orthography.
        expected: Abish's transcription rewritten in this project's IPA.
        printed: The transcription as the chapter prints it.
        where: Gloss and the page that prints the pair.
    """
    assert printed != ""
    assert where != ""
    assert as_project_notation(printed, NOTATION) == expected
    assert to_ipa(word, "kk") == expected


def test_the_standard_fricative_not_the_dialectal_affricate() -> None:
    """<ж> is standard ž, not the Chinese Altay variants' ǰ.

    Abish states the affricates ǰ and č are "Typical of Chinese Altay
    variants" (p. 338). The same letter in Kyrgyz is the affricate, and
    that contrast is the isogloss test_kyrgyz_ipa_letters.py pins from
    the other side.
    """
    assert to_ipa("ж", "kk") == "ʒ"
    assert to_ipa("ж", "ky") == "d͡ʒ"
