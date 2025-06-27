"""
Single Cyrillic characters → IPA
Reference: Help:IPA/Kyrgyz (rev. 6 Jan 2025), updated for the 2025-06 long-vowel
& ɕː revision.
"""

import pytest

from turkic_translit.core import to_ipa

GOLD = {
    "а": "ɑ",
    "б": "b",
    "в": "v",
    "г": "ɡ",
    "д": "d",
    "е": "e",
    "ё": "jo",
    "ж": "ʒ",
    "з": "z",
    "и": "i",
    "й": "j",
    "к": "k",
    "қ": "q",
    "л": "l",
    "м": "m",
    "н": "n",
    "ң": "ŋ",
    "о": "o",
    "ө": "ø",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ү": "y",
    "ф": "f",
    "х": "x",
    "ц": "ʦ",
    "ч": "ʧ",
    "ш": "ʃ",
    "щ": "ɕː",  # ← updated (was ʃt͡ʃ)
    "ы": "ɯ",
    "э": "e",
    "ю": "ju",
    "я": "ja",
}


@pytest.mark.parametrize(("cyr", "ipa"), GOLD.items())
def test_kyrgyz_letter_to_ipa(cyr: str, ipa: str) -> None:
    assert to_ipa(cyr, "ky") == ipa, f"{cyr} → {to_ipa(cyr, 'ky')}, expected {ipa}"
