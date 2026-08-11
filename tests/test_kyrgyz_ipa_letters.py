"""Kyrgyz transliteration checked against the description the rules cite.

McCollum 2020 is a research article rather than an Illustration, so it
carries no keyword chart. What it does carry is Table 1 (the consonant
inventory), Table 2 (the vowel inventory) and Table 3, which gives eight
roots in IPA alongside their suffixed forms. Those eight are the
expectations here.

The previous version of this file cited a Wikipedia page rather than the
article the rule file names, which is why the mismatch below went
unnoticed for so long.
"""

from __future__ import annotations

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

INHERITS_SOURCE = "https://doi.org/10.5334/labphon.247"

# Simplifications the rules make deliberately, applied to the published
# form to obtain what they produce.
DECLARED_SIMPLIFICATIONS: tuple[tuple[str, str, str], ...] = (
    ("q", "k", "the rules carry no dorsal backness allophony; Table 1 lists k and q separately"),
    ("g", "ɡ", "the journal typesets the voiced velar plosive as U+0067; IPA is U+0261"),
)

# A difference that is probably a defect rather than a choice, recorded
# here so it is visible and testable until a phonologist rules on it.
# McCollum writes the voiced affricate for Cyrillic <ж> in both roots
# Table 3 supplies, and Table 1 lists the fricative and the affricate as
# separate phonemes. These rules emit the fricative, which is the value
# the letter carries in Russian rather than in Kyrgyz.
SUSPECTED_DEFECT: tuple[tuple[str, str, str], ...] = (
    ("d͡ʒ", "ʒ", "Cyrillic <ж> is transcribed as a fricative where the source has an affricate"),
)

# Cyrillic, McCollum's root as Table 3 gives it, gloss, page
TABLE_3_ROOTS: tuple[tuple[str, str, str, int], ...] = (
    ("тил", "til", "language", 4),
    ("бел", "bel", "lower back", 4),
    ("гүл", "gyl", "flower", 4),
    ("көл", "køl", "lake", 4),
    ("бал", "bɑl", "honey", 4),
    ("кул", "qul", "slave", 4),
    ("жыл", "d͡ʒɯl", "year", 4),
    ("жол", "d͡ʒol", "road", 4),
)


def as_this_project_writes_it(published: str) -> str:
    """Apply the declared simplifications and the suspected defect.

    Args:
        published: The transcription exactly as the source prints it.

    Returns:
        The same transcription in the notation these rules produce.
    """
    for source_symbol, ours, _reason in (*SUSPECTED_DEFECT, *DECLARED_SIMPLIFICATIONS):
        published = published.replace(source_symbol, ours)
    return published


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """These expectations come from the article the rule file names."""
    declared = read_rule_source(_RULE_DIR / "ky_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "McCollum, A. G."
    assert declared["year"] == 2020


@pytest.mark.parametrize(("cyrillic", "published", "gloss", "page"), TABLE_3_ROOTS)
def test_table_3_root_transliterates_as_the_source_prints_it(
    cyrillic: str, published: str, gloss: str, page: int
) -> None:
    """Each Table 3 root matches, allowing the recorded differences."""
    assert page == 4
    assert gloss != ""
    assert to_ipa(cyrillic, "ky") == as_this_project_writes_it(published)


def test_the_affricate_difference_is_still_present() -> None:
    """The suspected defect is pinned so a silent change is impossible.

    If Cyrillic <ж> is corrected to the affricate the source uses, this
    test fails and the entry above must be removed rather than the
    correction being absorbed unnoticed.
    """
    assert to_ipa("жол", "ky") == "ʒol"
    assert to_ipa("жыл", "ky") == "ʒɯl"


def test_affricates_are_written_with_a_tie_bar_not_a_ligature() -> None:
    """Kyrgyz writes its affricates the way the other six languages do.

    These rules used the withdrawn precomposed ligatures while every
    other rule file used a tie bar, so one phoneme was one character in
    Kyrgyz and three elsewhere. A character-level model comparing
    languages reads that as a difference between the languages.
    """
    assert to_ipa("чын", "ky") == "t͡ʃɯn"
    for ligature in ("ʧ", "ʤ", "ʥ", "ʨ"):
        assert ligature not in to_ipa("чын жол", "ky")
