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
    "я": (
        {"kk": "jɑ", "ky": "jɑ", "uzc": "ja"},
        "the glide plus each language's own low vowel: the contrast is the "
        "same one declared for bare а above",
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


# The letters ‹ч щ ш ж ц з› carry, in the three Turkic Cyrillic files, the
# sounds those files' own comments call Russian loan phonemes. Russian is
# where they are native, so the four files describe one segment each and
# should not spell it four ways. Russian is not folded into
# CYRILLIC_LANGS above, because it differs from the Turkic three across
# most of the alphabet for a real reason -- it marks a palatalisation
# contrast none of them has -- and a whole-alphabet comparison would be a
# list of genuine differences with these few notation questions lost in it.
RUSSIAN_LOAN_LETTERS: dict[str, tuple[dict[str, str], str]] = {
    "з": ({"ru": "z", "kk": "z", "ky": "z", "uzc": "z"}, ""),
    "ц": ({"ru": "t͡s", "kk": "t͡s", "ky": "t͡s", "uzc": "t͡s"}, ""),
    "ш": ({"ru": "ʃ", "kk": "ʃ", "ky": "ʃ", "uzc": "ʃ"}, ""),
    "щ": (
        {"ru": "ʃʲː", "kk": "ʃː", "ky": "ʃː", "uzc": "ʃː"},
        "All four write the postalveolar; Russian adds the ʲ because ‹щ› is one "
        "of its two unpaired SOFT consonants and palatalisation is contrastive "
        "there (Yanushevskaya & Bunčić p. 222), while in the Turkic three the "
        "letter occurs only in Russian loans and their descriptions record no "
        "palatalisation opposition to place it in. Until 2026-09-03 all four "
        "wrote ɕː, which McCollum & Chen's chart (p. 277) excludes: it has no "
        "alveolo-palatal column, and the prose under it enumerates the "
        "non-native sounds Kazakh takes on as [f], [v] and [h].",
    ),
    "ж": (
        {"ru": "ʒ", "kk": "ʒ", "ky": "d͡ʒ", "uzc": "d͡ʒ"},
        "Kyrgyz and Uzbek ‹ж› is the affricate their sources print (McCollum "
        "2020 Table 3; Ido 2025), not the Russian fricative. Giving it the "
        "Russian value is the defect test_kyrgyz_ipa_letters.py was written "
        "to catch, so this contrast is the fix rather than drift.",
    ),
    "ч": (
        {"ru": "t͡ʃʲ", "kk": "t͡ʃ", "ky": "t͡ʃ", "uzc": "t͡ʃ"},
        "Russian ‹ч› is unpaired soft and always palatalised (Yanushevskaya & "
        "Bunčić p. 222); the Turkic affricate is not. A real contrast.",
    ),
}


@pytest.mark.parametrize(
    ("letter", "expected", "reason"),
    [(letter, expected, reason) for letter, (expected, reason) in RUSSIAN_LOAN_LETTERS.items()],
    ids=list(RUSSIAN_LOAN_LETTERS),
)
def test_a_russian_loan_letter_is_spelled_one_way_or_says_why_not(
    letter: str, expected: dict[str, str], reason: str
) -> None:
    """The four Cyrillic files agree on each borrowed letter, or declare why not.

    ‹щ› is the case this was written for. It was ʃʲː in Russian and ɕː in
    the other three, and the ɕː turned out to be excluded by the Kazakh
    chart rather than merely different. All four now write the postalveolar
    and differ only in the ʲ, where the languages themselves differ. Nothing
    was checking any of it, because Russian sits outside the group above.

    Args:
        letter: The Cyrillic letter under test.
        expected: The transcription each of the four files produces.
        reason: Why they differ, or empty when they agree.
    """
    produced = {lang: to_ipa(letter, lang) for lang in ("ru", "kk", "ky", "uzc")}

    assert produced == expected
    if reason == "":
        assert len(set(produced.values())) == 1, f"{letter!r} differs with no reason given"
    else:
        assert len(set(produced.values())) > 1, f"{letter!r} agrees, so the reason is stale"
