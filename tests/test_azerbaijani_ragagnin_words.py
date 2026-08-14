"""Azerbaijani rules checked against the variety-matched description.

The Illustration az_ipa.rules cites describes Tabriz and prints no
orthography, so for a long time no source-printed word could check these
rules; test_azerbaijani_ipa_letters.py explains the two reasons and tests
an inventory instead. Ragagnin's chapter on Azeri in the second Routledge
The Turkic Languages describes the standard these rules map — "The
standard language ... is based on the dialect of the capital Baku"
(p. 242) — and prints four words in both Latin orthography and
Turcological transcription (pp. 243-244). This file pins those words,
together with the five special-character equations of its orthography
section (p. 242).

Turcological transcription is not IPA. The fixed glyph conventions
separating the two are declared below and applied to the source's side,
the same shape as the two MFA conventions in
test_uyghur_mfa_agreement.py: same letters, same phonemes, different
glyph choices. Vowel length, which the chapter marks with ː in loanword
pronunciations, is dropped, because the rule file declares length
unmarked for this language.
"""

from __future__ import annotations

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

PRIMARY_SOURCE = "https://doi.org/10.1017/S0025100317000184"
CORROBORATING_SOURCE = "https://doi.org/10.4324/9781003243809-17"

# Turcological glyph → this project's IPA glyph. A single-pass character
# translation, not sequential replacement, so no output feeds a later rule
# (ü→y and y→j would otherwise interact). Length marks are dropped per the
# rule file's declared simplifications.
TURCOLOGICAL_TO_IPA: dict[int, str] = {
    ord("a"): "ɑ",
    ord("ä"): "æ",
    ord("ï"): "ɯ",
    ord("ü"): "y",
    ord("ö"): "œ",
    ord("γ"): "ɣ",
    ord("χ"): "x",
    ord("č"): "t͡ʃ",
    ord("ǰ"): "d͡ʒ",
    ord("š"): "ʃ",
    ord("ž"): "ʒ",
    ord("y"): "j",
    ord("r"): "ɾ",
    ord("ː"): "",
}

# The four words the chapter prints in both spellings. The chapter
# typesets the schwa letter as ǝ (U+01DD); the Azerbaijani alphabet's
# letter, which these rules map and OSCAR text carries, is ə (U+0259).
WORDS: tuple[tuple[str, str, str], ...] = (
    ("kitab", "kitab", "'book', in the word-final devoicing examples, p. 244"),
    ("bulud", "bulud", "'cloud', in the word-final devoicing examples, p. 244"),
    ("ağac", "aγaǰ", "'tree', in the word-final devoicing examples, p. 244"),
    ("rütubət", "rütuːbät", "'humidity', conservative loan pronunciation, p. 243"),
)

# p. 242: "Special characters include ‹ǝ› for ä, ‹ı› for ï, ‹ğ› for γ,
# ‹x› for χ, and ‹h› for the glottal fricative h."
LETTER_EQUATIONS: tuple[tuple[str, str], ...] = (
    ("ə", "ä"),
    ("ı", "ï"),
    ("ğ", "γ"),
    ("x", "χ"),
    ("h", "h"),
)


def in_this_project_glyphs(turcological: str) -> str:
    """Rewrite a Turcological transcription in this project's IPA glyphs.

    Args:
        turcological: The transcription as the chapter prints it.

    Returns:
        The same segments in the glyphs az_ipa.rules emits.
    """
    return turcological.translate(TURCOLOGICAL_TO_IPA)


def test_the_declared_source_is_still_the_illustration() -> None:
    """Corroboration adds a source; it does not replace the cited one."""
    declared = read_rule_source(_RULE_DIR / "az_ipa.rules")

    assert declared["identifier"] == PRIMARY_SOURCE


def test_the_rule_file_names_the_corroborating_description() -> None:
    """The corroboration lives in the rule file, not only in this test."""
    text = (_RULE_DIR / "az_ipa.rules").read_text(encoding="utf-8")

    assert CORROBORATING_SOURCE in text


@pytest.mark.parametrize(("spelling", "turcological", "where"), WORDS, ids=[w for w, _, _ in WORDS])
def test_word_matches_the_printed_transcription(spelling: str, turcological: str, where: str) -> None:
    """Each spelled word transliterates to the chapter's transcription.

    Args:
        spelling: The word in the Latin orthography of the Baku standard.
        turcological: The transcription printed beside it in the chapter.
        where: Which example list of the chapter prints the pair.
    """
    assert where != ""
    assert to_ipa(spelling, "az") == in_this_project_glyphs(turcological)


@pytest.mark.parametrize(("letter", "turcological"), LETTER_EQUATIONS, ids=[a for a, _ in LETTER_EQUATIONS])
def test_special_character_equation_holds(letter: str, turcological: str) -> None:
    """Each special character maps to the value the chapter equates it with.

    Args:
        letter: The special character of the 1991 Latin alphabet.
        turcological: The Turcological value p. 242 gives for it.
    """
    assert to_ipa(letter, "az") == in_this_project_glyphs(turcological)
