"""Test Azerbaijani → IPA transliteration using core to_ipa()."""

import pytest

from turkic_translit.core import to_ipa

# ---------------------------------------------------------------------------
# 1.  Single-letter gold standard
# Based on Mokari & Werner (2017) "Azerbaijani." JIPA 47(2): 207–212
GOLD = {
    "a": "ɑ",
    "b": "b",
    "c": "d͡ʒ",
    "ç": "t͡ʃ",
    "d": "d",
    "e": "e",
    "ə": "æ",
    "f": "f",
    "g": "ɟ",
    "ğ": "ɣ",
    "h": "h",
    "x": "x",
    "ı": "ɯ",
    "i": "i",
    "j": "ʒ",
    "k": "k",
    "q": "ɡ",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "ö": "œ",
    "p": "p",
    "r": "ɾ",
    "s": "s",
    "ş": "ʃ",
    "t": "t",
    "u": "u",
    "ü": "y",
    "v": "v",
    "y": "j",
    "z": "z",
}

# ---------------------------------------------------------------------------
# 2.  Word-level sanity checks

WORD_TESTS = {
    "salam": "sɑlɑm",
    "xoş": "xoʃ",
    "gəlmisiniz": "ɟælmisiniz",
    "sağol": "sɑɣol",
}

# ---------------------------------------------------------------------------
# 3.  Parametrised tests


@pytest.mark.parametrize(("input", "expected"), GOLD.items())
def test_letter_to_ipa(input: str, expected: str) -> None:
    """Single Azerbaijani letters."""
    assert to_ipa(input, "az") == expected


def test_azerbaijani_word_examples() -> None:
    """Common Azerbaijani words."""
    for az, ipa in WORD_TESTS.items():
        assert to_ipa(az, "az") == ipa
