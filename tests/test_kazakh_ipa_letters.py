"""
Single-letter Cyrillic → IPA checks
Reference set synchronised with McCollum & Chen 2020 (JIPA 51 (2): 276-298)

‹е/Э› are omitted because their value is position-dependent and
already covered in sentence-level tests.
"""

import pytest

from turkic_translit.core import to_ipa

GOLD = {
    "б": "b",
    "в": "v",
    "г": "ɡ",
    "ғ": "ʁ",
    "д": "d̪",  # dental plosive
    "ё": "jo",
    "ж": "ʑ",
    "з": "z̪",  # dental fricative
    "и": "i͡j",
    "й": "j",
    "к": "k",
    "қ": "q",
    "л": "l̪",  # dental lateral
    "м": "m",
    "н": "n̪",  # dental nasal
    "ң": "ŋ",  # velar nasal
    "о": "o",
    "ө": "ɵ",
    "п": "p",
    "р": "r̪",  # dental trill/tap
    "с": "s̪",  # dental fricative
    "т": "t̪",  # dental plosive
    "у": "u͡w",
    "ұ": "ʊ",
    "ү": "ʏ",
    "ф": "f",
    "х": "χ",
    "һ": "h",
    "ц": "t͡s",
    "ч": "t͡ɕ",
    "ш": "ɕ",
    "щ": "ɕː",  # long alveolo-palatal fricative
    "ы": "ə",
    "і": "ɪ",
    "ю": "ju",
    "я": "ja",
}


@pytest.mark.parametrize(("cyr", "ipa"), GOLD.items())
def test_letter_to_ipa(cyr: str, ipa: str) -> None:
    assert to_ipa(cyr, "kk") == ipa
