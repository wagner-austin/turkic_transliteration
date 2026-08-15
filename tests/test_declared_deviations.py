"""Every declared deviation is exercised, named, and justified.

The source-fidelity tests declare how their output may differ from the
source's printed transcription in two tables: NOTATION for glyph choices
that keep the phoneme identical, and DECLARED_SIMPLIFICATIONS for real
distinctions the rules drop. This file enforces the scheme suite-wide:
only those two names exist, every entry carries a reason (enforced by
as_project_notation at application time), and every entry is exercised
by at least one printed datum, so no table can accumulate dead entries
that read as justified deviations while testing nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import test_azerbaijani_ragagnin_words as az_words
from tests import test_finnish_ipa_letters as fi_letters
from tests import test_finnish_karlsson_words as fi_words
from tests import test_kazakh_routledge_words as kk_words
from tests import test_kyrgyz_ipa_letters as ky_letters
from tests import test_turkish_ipa_letters as tr_letters
from tests import test_uyghur_mfa_agreement as ug_mfa
from tests import test_uzbek_cyr_ipa_letters as uzc_letters
from tests import test_uzbek_lat_ipa_letters as uz_letters
from turkic_translit.testing import Deviation, unexercised_entries

# Module name, its deviation tables, and every published transcription
# its data supplies. The Uyghur data are the dictionary pronunciations of
# the pinned MFA sample.
CASES: tuple[tuple[str, tuple[tuple[Deviation, ...], ...], tuple[str, ...]], ...] = (
    (
        "turkish",
        (tr_letters.NOTATION, tr_letters.DECLARED_SIMPLIFICATIONS),
        tuple(row[1] for row in tr_letters.VOWEL_KEYWORDS)
        + tuple(row[1] for row in tr_letters.KEYWORDS),
    ),
    (
        "kyrgyz",
        (ky_letters.NOTATION, ky_letters.DECLARED_SIMPLIFICATIONS),
        tuple(row[1] for row in ky_letters.TABLE_3_ROOTS),
    ),
    (
        "uzbek-latin",
        (uz_letters.NOTATION,),
        tuple(row[1] for row in uz_letters.KEYWORDS),
    ),
    (
        "uzbek-cyrillic",
        (uzc_letters.NOTATION,),
        tuple(row[2] for row in uzc_letters.KEYWORDS),
    ),
    (
        "uyghur",
        (ug_mfa.NOTATION,),
        tuple(pron for _word, prons in ug_mfa.ROWS for pron in prons),
    ),
    (
        "azerbaijani",
        (az_words.NOTATION, az_words.DECLARED_SIMPLIFICATIONS),
        tuple(row[1] for row in az_words.WORDS)
        + tuple(row[1] for row in az_words.LETTER_EQUATIONS),
    ),
    (
        "kazakh",
        (kk_words.NOTATION,),
        tuple(row[2] for row in kk_words.WORDS),
    ),
    (
        "finnish",
        (fi_words.NOTATION, fi_words.DECLARED_SIMPLIFICATIONS),
        tuple(row[2] for row in fi_words.WORDS),
    ),
    (
        "finnish-letters",
        (fi_letters.NOTATION,),
        tuple(row[1] for row in fi_letters.QUANTITY_SERIES),
    ),
)

BANNED_NAMES = ("DECLARED_DEVIATIONS", "CONVENTIONS", "TURCOLOGICAL_TO_IPA", "_canonical")


@pytest.mark.parametrize(("label", "tables", "data"), CASES, ids=[c[0] for c in CASES])
def test_every_deviation_entry_is_exercised(
    label: str,
    tables: tuple[tuple[Deviation, ...], ...],
    data: tuple[str, ...],
) -> None:
    """No table carries an entry that no printed datum exercises.

    Args:
        label: The language the tables belong to.
        tables: That file's NOTATION and simplification tables.
        data: Every published transcription the file compares against.
    """
    assert label != ""
    assert unexercised_entries(data, *tables) == ()


def test_only_the_two_standard_table_names_exist() -> None:
    """Retired table names and mechanisms stay retired."""
    here = Path(__file__)
    for test_file in sorted(here.parent.glob("test_*.py")):
        if test_file == here:
            continue
        source = test_file.read_text(encoding="utf-8")
        for name in BANNED_NAMES:
            assert name not in source, f"{test_file.name} uses retired name {name}"
