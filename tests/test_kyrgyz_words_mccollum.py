"""
Gold-standard word list: Kyrgyz orthography → broad IPA
Source: McCollum 2020 “Vowel harmony and positional variation in Kyrgyz”,
Laboratory Phonology 11(1): 25 (CC-BY 4.0).

The paper’s transcriptions are already broad IPA; no diacritics need stripping.
"""

import unicodedata as ud

import pytest

from turkic_translit.core import to_ipa

# -------------------------------------------------------------------------
# Orthographic word  →  IPA  (drawn from the paper, then canonicalised)
# -------------------------------------------------------------------------
GOLD = {
    # monosyllabic roots (Table 3)
    "бал": "bɑl",
    "бел": "bel",
    "көл": "køl",
    "жыл": "ʒɯl",
    # disyllabic roots (Appendix)
    "молдо": "moldo",
    "илим": "ilim",
    "керме": "kerme",
    "кыргыз": "kɯrɡɯz",  # uvular symbols → velar in our rule set
    "сулуу": "suluu",
    "үгүт": "yɡyt",
    # harmony alternations (Table 3)
    "балда": "bɑldɑ",  # ‘honey-LOC’
    "балды": "bɑldɯ",  # ‘honey-ACC’
    "көлдө": "køldø",
    "көлдү": "køldy",
    "жылда": "ʒɯldɑ",
    "жылды": "ʒɯldɯ",
}


def _canonical(ipa: str) -> str:
    """Normalise alternative glyphs to those emitted by ky_ipa.rules."""
    return (
        ipa.replace("ʤ", "dʒ")  # affricate ligature → digraph
        .replace("ʦ", "t͡s")  # ligature → tie-bar
        .replace("ʧ", "t͡ʃ")  # ligature → tie-bar
        .replace("q", "k")  # uvular stop → velar
        .replace("ʁ", "ɡ")  # uvular fricative → velar
    )


@pytest.mark.parametrize(("cyr", "ipa"), GOLD.items())
def test_kyrgyz_word_to_ipa(cyr: str, ipa: str) -> None:
    """Exact comparison after canonicalisation."""
    predicted = _canonical(ud.normalize("NFC", to_ipa(cyr, "ky")))
    expected = _canonical(ipa)
    assert predicted == expected, f"{cyr} → {predicted!r}, expected {expected!r}"
