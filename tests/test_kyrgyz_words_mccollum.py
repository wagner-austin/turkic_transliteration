"""Kyrgyz word list, and what it can and cannot claim as its source.

The header of this file used to describe its rows as a gold standard from
McCollum 2020, adapted to later rule changes. Two things are wrong with
that. Values edited to match the rules are no longer the source's values,
and the Appendix the list draws on is a separate supplementary file
(DOI 10.5334/labphon.247.s1) that is not in the archive, so no reader can
check the rows against it.

The rows are kept because they exercise the vowel-harmony behaviour that
Table 3 of the archived article does support, and that behaviour is
tested against the article itself in test_kyrgyz_ipa_letters.py. What is
removed is the claim that these particular values come from a source
anyone can consult.
"""

import unicodedata as ud

import pytest

from turkic_translit.core import to_ipa

# The archived article, whose Table 3 grounds the harmony behaviour these
# rows exercise. The supplementary Appendix is not archived, so no row
# here is presented as a value read from a source.
INHERITS_SOURCE = "https://doi.org/10.5334/labphon.247"

# -------------------------------------------------------------------------
# Orthographic word  →  IPA  (canonicalised)
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
    "кыргыз": "kɯrɡɯz",
    "сулуу": "suluː",  # ← long /uː/ from ‹уу›
    "үгүт": "yɡyt",
    # harmony alternations (Table 3)
    "балда": "bɑldɑ",
    "балды": "bɑldɯ",
    "көлдө": "køldø",
    "көлдү": "køldy",
    "жылда": "ʒɯldɑ",
    "жылды": "ʒɯldɯ",
}


def _canonical(ipa: str) -> str:
    """Normalise alternative glyphs to those emitted by ky_ipa.rules.

    Args:
        ipa: IPA text using any of the accepted glyph variants.

    Returns:
        The same transcription in the glyphs the rules emit.
    """
    return (
        ipa.replace("ʤ", "dʒ")
        .replace("ʦ", "t͡s")
        .replace("ʧ", "t͡ʃ")
        .replace("q", "k")
        .replace("ʁ", "ɡ")
    )


@pytest.mark.parametrize(("cyr", "ipa"), GOLD.items())
def test_kyrgyz_word_to_ipa(cyr: str, ipa: str) -> None:
    """Each word transcribes to the IPA McCollum records for it.

    Args:
        cyr: The Kyrgyz word in Cyrillic.
        ipa: The published transcription.
    """
    predicted = _canonical(ud.normalize("NFC", to_ipa(cyr, "ky")))
    expected = _canonical(ipa)
    assert predicted == expected, f"{cyr} → {predicted!r}, expected {expected!r}"
