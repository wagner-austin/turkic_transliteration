"""Shared letters agree across languages after harmonization, or say why not.

The rule files are faithful to their per-language sources and the corpus
symbol map harmonizes notation across languages, so after harmonization
the same letter should produce the same characters in every language
that has it. Where it does not, the difference must be one declared
below with a reason, each a real property of the languages or their
orthographies rather than notation drift. Any new divergence fails.
"""

from __future__ import annotations

import pytest

from turkic_translit.core import to_ipa
from turkic_translit.corpus.symbols import (
    PACKAGED_SYMBOL_MAP,
    apply_substitutions,
    read_symbol_map,
    substitutions_for,
)

# The declared contrasts inherit their expectations from the per-language
# rule sources named in each reason; the harmonization side is the
# packaged symbol map, whose rows carry their own citations.
INHERITS_SOURCE = "src/turkic_translit/corpus/symbol_map.csv"

CYRILLIC = "абвгдежзийклмнопрстуфхчшщыэюя"
CYRILLIC_LANGS = ("kk", "ky", "uzc")
LATIN = "abdefghijklmnopqrstuvyz"
LATIN_LANGS = ("tr", "az", "uz")

# Letter -> (per-language harmonized outputs, reason). Everything not
# listed must be identical across the group's languages.
DECLARED_CONTRASTS: dict[str, tuple[dict[str, str], str]] = {
    "а": (
        {"kk": "ɑ", "ky": "ɑ", "uzc": "a"},
        "Uzbek's low vowel is a different phonological object after harmony loss "
        "(Ido 2025); the symbol map's keep-row forbids merging it with ɑ",
    ),
    "е": (
        {"kk": "e", "ky": "e", "uzc": "je"},
        "Uzbek Cyrillic е is iotated word-initially by orthographic rule; a bare "
        "letter is word-initial",
    ),
    "ж": (
        {"kk": "ʒ", "ky": "d͡ʒ", "uzc": "d͡ʒ"},
        "the Kipchak isogloss: standard Kazakh has the fricative and calls the "
        "affricate dialectal (Abish 2021: 338); Kyrgyz has the affricate "
        "(McCollum 2020, Table 3)",
    ),
    "о": (
        {"kk": "o", "ky": "o", "uzc": "ɔ"},
        "Uzbek's о is the open back vowel (Ido 2025, p. 154)",
    ),
    "х": (
        {"kk": "χ", "ky": "x", "uzc": "x"},
        "the cited descriptions assign different places: uvular for Kazakh "
        "(McCollum & Chen 2021), velar for Kyrgyz and Uzbek",
    ),
    "ы": (
        {"kk": "ə", "ky": "ɯ", "uzc": "ɨ"},
        "three analyses of the high unrounded vowel: central ə (McCollum & Chen "
        "2021), back ɯ (McCollum 2020), central ɨ (Ido 2025)",
    ),
    "a": (
        {"tr": "ɑ", "az": "ɑ", "uz": "a"},
        "Turkish a merges to ɑ in the symbol map; Uzbek's a stays distinct (Ido 2025)",
    ),
    "g": (
        {"tr": "ɡ", "az": "ɟ", "uz": "ɡ"},
        "Azerbaijani ⟨g⟩ is the palatal plosive (Mokari & Werner 2017, p. 208; "
        "Ragagnin 2021 palatalized front g)",
    ),
    "j": (
        {"tr": "ʒ", "az": "ʒ", "uz": "d͡ʒ"},
        "the alphabets assign the letter different phonemes: loan fricative in "
        "Turkish and Azerbaijani, affricate in Uzbek",
    ),
    "o": (
        {"tr": "o", "az": "o", "uz": "ɔ"},
        "Uzbek's ⟨o⟩ is the open back vowel (Ido 2025, p. 154)",
    ),
    "q": (
        {"tr": "q", "az": "ɡ", "uz": "q"},
        "Azerbaijani ⟨q⟩ is the voiced back stop (Ragagnin 2021: ġatïġ). "
        "Turkish has no ⟨q⟩; it passes through untranslated in foreign "
        "material and collides with Uzbek's uvular /q/ at 704 occurrences in "
        "the Turkish corpus, a known bounded false-friend pathway",
    ),
}


def harmonized(letter: str, lang: str) -> str:
    """One letter's rule output with the corpus symbol map applied.

    Args:
        letter: A single orthographic letter.
        lang: The rule-file language code.

    Returns:
        The harmonized transcription, as the training corpora carry it.
    """
    rules = read_symbol_map(PACKAGED_SYMBOL_MAP)
    subs = substitutions_for(rules, "uz" if lang == "uzc" else lang)
    return apply_substitutions(to_ipa(letter, lang), subs)


@pytest.mark.parametrize(
    ("letters", "langs"),
    [(CYRILLIC, CYRILLIC_LANGS), (LATIN, LATIN_LANGS)],
    ids=["cyrillic", "latin"],
)
def test_shared_letters_agree_unless_declared(letters: str, langs: tuple[str, ...]) -> None:
    """Each shared letter is identical across the group or declared.

    Args:
        letters: The letters shared by every language in the group.
        langs: The language codes of the group.
    """
    for letter in letters:
        outs = {lang: harmonized(letter, lang) for lang in langs}
        if letter in DECLARED_CONTRASTS:
            expected, reason = DECLARED_CONTRASTS[letter]
            assert reason != ""
            assert outs == expected, f"{letter!r}: {outs} != declared {expected}"
        else:
            assert len(set(outs.values())) == 1, f"{letter!r} undeclared: {outs}"


def test_every_declared_contrast_is_a_shared_letter() -> None:
    """No declared contrast for a letter the groups do not test."""
    for letter in DECLARED_CONTRASTS:
        assert letter in CYRILLIC or letter in LATIN
