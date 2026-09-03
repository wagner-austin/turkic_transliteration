"""Russian palatalisation checked against the second description.

`test_russian_ipa_letters.py` holds `ru_ipa.rules` to the Illustration it
cites. This file is the corroborating pass every other rule file has had:
the same mappings read out of an independent, variety-matched description,
so that a claim resting on one author's chart is not the only thing behind
them.

Timberlake (2004) is a reference grammar rather than an Illustration, so it
carries no IPA keyword chart. What it carries is the classification the
whole rule file turns on. Consonants are `mutable` when they can be either
palatalised or not, and `immutable` when they cannot, and the immutable set
splits into intrinsically palatalised and intrinsically not (p. 18). That
three-way split is exactly what `$Cpair` and the unpaired rules encode, so
it can be checked directly rather than through example words.

The letters below are read from the grammar's own lists, not derived from
the rule file. If someone widens `$Cpair`, or drops the ʲ from ‹ч›, or lets
‹ш› palatalise, these fail — which is what the Kyrgyz file's history says a
second source is for.
"""

from __future__ import annotations

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

# The rule file's own cited source. This module corroborates it; it does not
# replace it, and it must not silently become the only thing checked.
INHERITS_SOURCE = "https://doi.org/10.1017/S0025100314000395"

# Timberlake 2004, p. 18: the consonants that can be either palatalised or
# not. Ordered as the grammar prints them.
MUTABLE: str = "пбвфмтдсзнкгхрл"

# Timberlake 2004, p. 18: immutable and intrinsically NOT palatalised.
IMMUTABLE_HARD: str = "шжц"

# Timberlake 2004, p. 18: immutable and intrinsically palatalised.
IMMUTABLE_SOFT: str = "чщ"

PALATALISATION = "ʲ"
CLOSE_CENTRAL = "ɨ"


def test_the_rule_file_still_cites_the_illustration_this_corroborates() -> None:
    """The corroboration is bound to the file it corroborates.

    Args:
        None.
    """
    assert read_rule_source(_RULE_DIR / "ru_ipa.rules")["identifier"] == INHERITS_SOURCE


def test_the_three_classes_partition_the_consonant_letters() -> None:
    """The grammar's classes are disjoint and total over the letters used.

    Twenty of the thirty-three letters are consonants that this file
    classifies; ‹й› is the approximant and is not in the palatalisation
    opposition at all, which is why it is absent from all three lists.
    """
    classes = [set(MUTABLE), set(IMMUTABLE_HARD), set(IMMUTABLE_SOFT)]

    assert len(MUTABLE) == 15
    assert set.intersection(*classes) == set()
    assert sum(len(group) for group in classes) == len(set().union(*classes))


@pytest.mark.parametrize("letter", MUTABLE, ids=list(MUTABLE))
def test_a_mutable_consonant_takes_palatalisation_before_ya(letter: str) -> None:
    """Each mutable consonant is soft before ‹я› and hard before ‹а›.

    Args:
        letter: One Cyrillic consonant from the grammar's mutable list.
    """
    hard = to_ipa(letter + "а", "ru")
    soft = to_ipa(letter + "я", "ru")

    assert PALATALISATION not in hard
    assert soft == hard.replace("a", PALATALISATION + "a")


@pytest.mark.parametrize("letter", IMMUTABLE_HARD, ids=list(IMMUTABLE_HARD))
def test_an_immutable_hard_consonant_never_takes_palatalisation(letter: str) -> None:
    """‹ш ж ц› stay hard before every vowel letter, including ‹е› and ‹и›.

    Args:
        letter: One Cyrillic consonant from the grammar's immutable-hard list.
    """
    for vowel in "аеёиюя":
        assert PALATALISATION not in to_ipa(letter + vowel, "ru")


@pytest.mark.parametrize(
    ("letter", "expected"),
    [("ш", "ʃɨ"), ("ж", "ʒɨ"), ("ц", "t͡sɨ")],
    ids=list(IMMUTABLE_HARD),
)
def test_i_after_an_unpaired_hard_consonant_is_the_close_central_vowel(
    letter: str, expected: str
) -> None:
    """p. 35: [ɨ] occurs after a hard consonant only if that consonant is unpaired.

    Args:
        letter: One Cyrillic consonant from the grammar's immutable-hard list.
        expected: The transcription these rules produce for letter plus ‹и›.
    """
    assert expected.endswith(CLOSE_CENTRAL)
    assert to_ipa(letter + "и", "ru") == expected


@pytest.mark.parametrize(
    ("letter", "alone", "before_i"),
    [("ч", "t͡ʃʲ", "t͡ʃʲi"), ("щ", "ʃʲː", "ʃʲːi")],
    ids=list(IMMUTABLE_SOFT),
)
def test_an_immutable_soft_consonant_is_soft_however_it_is_written(
    letter: str, alone: str, before_i: str
) -> None:
    """‹ч щ› are intrinsically palatalised, and neither needs a following vowel to be.

    The two are written differently and both are correct. ‹ч› carries an
    explicit ʲ on a postalveolar; ‹щ› is written as an alveolo-palatal,
    where softness is the symbol rather than a diacritic on it. Asserting
    "contains ʲ" would therefore be asserting one spelling rather than the
    property, so the transcriptions are pinned outright.

    The vowel stays the close front one either way: retraction to [ɨ] is a
    property of the unpaired HARD consonants, and these two are unpaired soft.

    Args:
        letter: One Cyrillic consonant from the grammar's immutable-soft list.
        alone: The transcription of the bare consonant.
        before_i: The transcription of the consonant plus ‹и›.
    """
    assert CLOSE_CENTRAL not in before_i
    assert to_ipa(letter, "ru") == alone
    assert to_ipa(letter + "и", "ru") == before_i


def test_no_letter_outside_the_mutable_list_is_palatalised_by_a_front_vowel() -> None:
    """The mutable list is a ceiling, not just a floor.

    A rule file can satisfy every case above and still be wrong by
    palatalising something the grammar calls immutable. This checks the
    complement: for the five immutable consonants, ‹е› adds no ʲ either.
    """
    for letter in IMMUTABLE_HARD:
        assert PALATALISATION not in to_ipa(letter + "е", "ru")

    # The soft pair keeps whatever softness its own spelling carries and
    # gains nothing from the vowel: ‹ч› its ʲ, ‹щ› its alveolo-palatal symbol.
    assert to_ipa("че", "ru") == "t͡ʃʲe"
    assert to_ipa("ще", "ru") == "ʃʲːe"
