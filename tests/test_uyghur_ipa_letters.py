"""Uyghur transliteration checked against the description the rules cite.

This source cannot ground a keyword test, and says why. McCollum used
pictorial prompts rather than written ones "to avoid an orthographic
confound, since Uyghur orthographies do not represent [ɯ]" (p. 8), so his
stimuli are given in IPA with glosses and no spelling. There is no
orthography-to-IPA pair in the article to check a transliterator against.

What the article does state is the inventory, and that is what is checked
here: the vowels these rules emit from Arabic-script input must be the
vowels Table 1 lists. That claim needs no expected value for any
particular word, so it cannot be circular, and it holds whatever the
right transcription of a given word turns out to be.
"""

from __future__ import annotations

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

INHERITS_SOURCE = "https://doi.org/10.5334/labphon.239"

# Table 1, p. 2: seven contrastive vowels, plus /e/ which the text calls
# marginal, "typically occurring in non-nativized loans, and in the
# initial-syllable only".
PUBLISHED_VOWELS = frozenset("ɑouæøiy")
MARGINAL_VOWEL = "e"

# Every vowel symbol any of this project's rule files can emit. A vowel
# from this set that is not in the Uyghur inventory would be a defect.
ANY_VOWEL = frozenset("aeiouyæøɑɒɔəɘɛɜɪɨɯʉʊʌœɵ")

ARABIC_BLOCK = range(0x0600, 0x0700)
ARABIC_SUPPLEMENT = range(0x0750, 0x0780)

# Uyghur written in its own script, so nothing here is foreign text
# passing through unmapped.
ARABIC_SCRIPT_WORDS: tuple[str, ...] = (
    "شىنجاڭ",
    "ئۇيغۇر",
    "ئاپتونوم",
    "رايونى",
    "جۇڭگونىڭ",
    "غەربىدە",
    "جايلاشقان",
    "بالا",
    "ئۆي",
    "كۆل",
    "گۈل",
    "قۇل",
    "يول",
    "باش",
)


def is_arabic_script(text: str) -> bool:
    """Report whether every character is in an Arabic script block.

    Args:
        text: Word to test.

    Returns:
        True when the word carries no character from another script.
    """
    return all(ord(c) in ARABIC_BLOCK or ord(c) in ARABIC_SUPPLEMENT for c in text)


def vowels_of(text: str) -> set[str]:
    """Collect the vowel symbols in transliterated output.

    Args:
        text: IPA output from a transliteration.

    Returns:
        The distinct vowel characters the output contains.
    """
    return {char for char in text if char in ANY_VOWEL}


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """The inventory below comes from the article the rule file names."""
    declared = read_rule_source(_RULE_DIR / "ug_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "McCollum, A. G."
    assert declared["year"] == 2021


@pytest.mark.parametrize("word", ARABIC_SCRIPT_WORDS)
def test_output_uses_only_the_published_vowels(word: str) -> None:
    """Arabic-script input yields no vowel the source does not list."""
    assert is_arabic_script(word), "this word carries characters from another script"

    produced = vowels_of(to_ipa(word, "ug"))

    assert produced <= PUBLISHED_VOWELS | {MARGINAL_VOWEL}
    assert produced, "a real word should produce at least one vowel"


def test_the_gap_in_the_inventory_is_respected() -> None:
    """No output carries the back unrounded high vowel.

    Table 1 leaves that cell empty, and the text states the point
    directly: "there is one notable gap in the Uyghur inventory -- there
    is no [+back] counterpart of /i/" (p. 2). The orthography does not
    represent it either, which is why the study used pictures instead of
    written prompts.
    """
    for word in ARABIC_SCRIPT_WORDS:
        assert "ɯ" not in to_ipa(word, "ug")


def test_characters_from_another_script_are_left_alone() -> None:
    """Text outside the Arabic block passes through unchanged.

    A Uyghur corpus drawn from the web carries Latin fragments, and the
    rules rewrite only what they map. Stating it here keeps the vowel
    check above honest: it restricts itself to Arabic-script words
    precisely because Latin ones would contribute their own vowels.
    """
    assert to_ipa("wiki", "ug") == "wiki"
    assert not is_arabic_script("wiki")
