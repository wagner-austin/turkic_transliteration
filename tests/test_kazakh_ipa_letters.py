"""
Single-letter Cyrillic → IPA checks
Reference: https://en.wikipedia.org/wiki/Help:IPA/Kazakh (10 Nov 2024).

We skip ‹е/Э› because their value depends on position
(^Е → jɪ, CЕ → e). Those are covered by the sentence test.
"""

import pytest

from turkic_translit.core import to_ipa

GOLD = {
    "б": "b",
    "в": "v",
    "г": "ɡ",
    "ғ": "ʁ",
    "д": "d",
    "ё": "jo",
    "ж": "ʑ",
    "з": "z",
    "и": "əj",
    "й": "j",
    "к": "k",
    "қ": "q",
    "л": "l",
    "м": "m",
    "н": "n",
    "ң": "ɴ",
    "о": "o",
    "ө": "ø",
    "п": "p",
    "р": "ɾ",
    "с": "s",
    "т": "t",
    "у": "w",
    "ұ": "ʊ",
    "ү": "ʏ",
    "ф": "f",
    "х": "χ",
    "һ": "h",
    "ц": "t͡s",
    "ч": "t͡ɕ",
    "ш": "ɕ",
    "щ": "ɕ",
    "ы": "ə",
    "і": "ɪ",
    "ю": "ju",
    "я": "ja",
}


@pytest.mark.parametrize(("cyr", "ipa"), GOLD.items())
def test_letter_to_ipa(cyr: str, ipa: str) -> None:
    assert to_ipa(cyr, "kk") == ipa
