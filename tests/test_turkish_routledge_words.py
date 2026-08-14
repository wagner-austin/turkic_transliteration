"""The two soft-g words the Routledge Turkish chapter prints in IPA.

Csató and Johanson print ‹dağ› [dɑː] 'mountain' and ‹düğme› [dyːmε]
'button' (p. 195), one word for each of the two soft-g contexts in
tr_ipa.rules: coda before a word boundary and coda before a consonant.
The cited Illustration states the same phonetics as prose; this chapter
is an independent description that prints the words, so the two rules
are pinned to spellings a source published rather than to examples this
project invented.

Their [ɑ] is this file's a and their [ε] this file's e — the broad
values Zimmer & Orgun, the cited source, assign the letters.
"""

from __future__ import annotations

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

PRIMARY_SOURCE = "https://doi.org/10.1017/S0025100300004588"
CORROBORATING_SOURCE = "https://doi.org/10.4324/9781003243809-13"

# (word, expected IPA, the chapter's printed IPA, which context it pins)
WORDS: tuple[tuple[str, str, str, str], ...] = (
    ("dağ", "daː", "[dɑː]", "soft g in word-final coda lengthens, p. 195"),
    ("düğme", "dyːme", "[dyːmε]", "soft g in coda before a consonant lengthens, p. 195"),
)


def test_the_declared_source_is_still_the_illustration() -> None:
    """Corroboration adds a source; it does not replace the cited one."""
    declared = read_rule_source(_RULE_DIR / "tr_ipa.rules")

    assert declared["identifier"] == PRIMARY_SOURCE


def test_the_rule_file_names_the_corroborating_description() -> None:
    """The corroboration lives in the rule file, not only in this test."""
    text = (_RULE_DIR / "tr_ipa.rules").read_text(encoding="utf-8")

    assert CORROBORATING_SOURCE in text


@pytest.mark.parametrize(
    ("word", "expected", "printed", "where"), WORDS, ids=[w for w, _, _, _ in WORDS]
)
def test_word_matches_the_printed_transcription(
    word: str, expected: str, printed: str, where: str
) -> None:
    """Each printed word transliterates to the chapter's IPA, our glyphs.

    Args:
        word: The word in Turkish orthography as the chapter prints it.
        expected: The chapter's IPA rewritten in this project's glyphs.
        printed: The IPA as the chapter prints it.
        where: Which soft-g context the word pins, and the page.
    """
    assert printed != ""
    assert where != ""
    assert to_ipa(word, "tr") == expected
