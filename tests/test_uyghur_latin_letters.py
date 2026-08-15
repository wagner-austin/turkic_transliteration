"""Arabic-Script Uyghur to Latin checked against the standard the rules cite.

Unlike the phonological descriptions the IPA rule files rest on, this
source prints the very thing the rule file implements: Table 3 (p. 11) is
the letter-for-letter correspondence between Arabic-Script Uyghur and
Latin-Script Uyghur that the Xinjiang University conferences settled on in
July 2001, and pp. 12-13 give worked examples of the one place the
correspondence is not letter-for-letter, the apostrophe. So the table and
the examples are pinned here directly.

This file exists because ``ar_lat.rules`` shipped for a long time with no
source at all, and carried three defects that a citation would have
caught. Two apostrophes written bare on one line paired with each other in
the ICU rule syntax and swallowed the rule between them, so ع emitted
nothing at all. ي had no rule and passed through as Arabic. And ﺋﯥ emitted
ë, which Duval & Janbaz record (p. 9) as one of the candidates the
committee rejected in favour of é.
"""

from __future__ import annotations

import unicodedata as ud

import pytest

from turkic_translit.core import _RULE_DIR, to_latin
from turkic_translit.rule_provenance import read_rule_source

INHERITS_SOURCE = (
    "https://web.archive.org/web/20061201110649/"
    "http://www.uyghurdictionary.org/excerpts/An%20Introduction%20to%20LSU.pdf"
)

# Table 3, p. 11, and Annex 1, p. 15. The vowels are listed there in their
# initial form, carrier included, because that is how ASU writes them.
TABLE_3: tuple[tuple[str, str], ...] = (
    ("ئا", "a"),
    ("ئە", "e"),
    ("ب", "b"),
    ("پ", "p"),
    ("ت", "t"),
    ("ج", "j"),
    ("چ", "ch"),
    ("خ", "x"),
    ("د", "d"),
    ("ر", "r"),
    ("ز", "z"),
    ("ژ", "zh"),
    ("س", "s"),
    ("ش", "sh"),
    ("غ", "gh"),
    ("ف", "f"),
    ("ق", "q"),
    ("ك", "k"),
    ("گ", "g"),
    ("ڭ", "ng"),
    ("ل", "l"),
    ("م", "m"),
    ("ن", "n"),
    ("ھ", "h"),
    ("ئو", "o"),
    ("ئۇ", "u"),
    ("ئۆ", "ö"),
    ("ئۈ", "ü"),
    ("ۋ", "w"),
    ("ئې", "é"),
    ("ئى", "i"),
    ("ي", "y"),
)

# The apostrophe cases, pp. 12-13. Each pairs an ASU spelling with the LSU
# form the source prints, lower-cased here because these rules transliterate
# letters and do not capitalise proper nouns.
APOSTROPHE_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("قۇرئان", "qur'an"),  # a vowel separated from a consonant by a hiatus
    ("ئىنگلىز", "in'gliz"),  # ن plus گ, which is not the single letter ڭ
    ("ئىسھاق", "is'haq"),  # س plus ھ, which is not the single letter ش
    ("چوڭھاجى", "chong'haji"),  # ڭ plus ھ, which is not ng followed by h
    ("ۋۇقۇئ", "wuqu'"),  # a final hiatus standing for the ﻉ of an Arabic loan
)

# p. 12: "it was deemed unnecessary to do this in cases where there are two
# sequential vowels", with saet among the three words given.
NO_APOSTROPHE_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("سائەت", "saet"),
    ("مەكتەپ", "mektep"),  # p. 8, the word used to argue against ae for ﺋﻪ
    ("ئۆلتۈرۈش", "öltürüsh"),  # p. 13, the word used to argue for diacritics
)


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """The table below comes from the standard the rule file names."""
    declared = read_rule_source(_RULE_DIR / "ar_lat.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "Duval, J. R.; Janbaz, W. A."
    assert declared["year"] == 2006
    assert declared["title"] == "An Introduction to Latin-Script Uyghur"


@pytest.mark.parametrize(("arabic", "latin"), TABLE_3)
def test_every_letter_of_table_3_transliterates_as_the_standard_gives_it(
    arabic: str, latin: str
) -> None:
    """Each ASU letter yields exactly the LSU letter the source prints.

    Args:
        arabic: The Arabic-Script Uyghur letter, in its initial form for
            the vowels, as Table 3 lists it.
        latin: The Latin-Script Uyghur letter the same row of the table
            gives for it.
    """
    assert to_latin(arabic, "ar") == latin


def test_table_3_is_the_whole_alphabet() -> None:
    """The pinned table is the 32 letters the standard settled, not a subset.

    Duval & Janbaz reach 32 by adding the 14 letters decided one at a time
    on pp. 8-11 to the 18 that Table 1 (p. 7) records as already agreed.
    A table quietly trimmed to the letters that happen to pass would make
    every check above vacuous for the ones removed.
    """
    assert len(TABLE_3) == 32
    assert len({arabic for arabic, _ in TABLE_3}) == 32


@pytest.mark.parametrize(("arabic", "latin"), APOSTROPHE_EXAMPLES)
def test_the_apostrophe_appears_where_the_source_writes_it(arabic: str, latin: str) -> None:
    """The hiatus and reading-guard cases produce the source's spelling.

    Args:
        arabic: The Arabic-Script Uyghur spelling.
        latin: The Latin-Script Uyghur spelling the source prints.
    """
    assert to_latin(arabic, "ar") == latin
    assert "'" in latin, "this example is here for its apostrophe"


@pytest.mark.parametrize(("arabic", "latin"), NO_APOSTROPHE_EXAMPLES)
def test_no_apostrophe_appears_where_the_source_omits_it(arabic: str, latin: str) -> None:
    """A hiatus between two vowels is left unwritten, as the source says.

    Args:
        arabic: The Arabic-Script Uyghur spelling.
        latin: The Latin-Script Uyghur spelling the source prints.
    """
    assert to_latin(arabic, "ar") == latin
    assert "'" not in latin, "this example is here for the apostrophe it lacks"


def test_ain_and_hamza_both_emit_an_apostrophe_and_nothing_else() -> None:
    """The quoting defect that lost a whole rule stays fixed.

    ``ء > ' ; ع > ' ;`` reads in ICU as one rule whose output is the
    literal text between the two apostrophes, so ء emitted "; ع > " and
    the rule for ع never existed, leaving it untransliterated. Both
    letters are checked for an exact result, because the failure mode was
    output that merely looked plausible.
    """
    assert to_latin("ء", "ar") == "'"
    assert to_latin("ع", "ar") == "'"


def test_no_arabic_letter_survives_into_the_output() -> None:
    """Every letter the standard lists maps; none passes through raw.

    ي had no rule of its own and so was emitted unchanged, which put
    Arabic script into text that is supposed to be Latin. Stated over the
    whole table so a newly dropped rule cannot pass unnoticed.
    """
    leaked = {
        arabic: to_latin(arabic, "ar")
        for arabic, _ in TABLE_3
        if any(ud.name(char, "").startswith("ARABIC") for char in to_latin(arabic, "ar"))
    }

    assert leaked == {}


def test_the_rules_distinguish_the_digraph_from_the_letter_pair() -> None:
    """ng and n+g are different words, and the apostrophe is what separates them.

    Duval & Janbaz give In'gliz for the pair (p. 12) precisely because ASU
    has a single letter ڭ for [ŋ] that the pair ن+گ must not collide with.
    Both spellings are checked together so the guard cannot be satisfied by
    a rule that inserts an apostrophe everywhere.
    """
    assert to_latin("چوڭ", "ar") == "chong"
    assert to_latin("ئىنگلىز", "ar") == "in'gliz"


def test_the_arabic_pre_pass_reaches_the_same_rules() -> None:
    """``include_arabic`` routes through this file, so it inherits the fix.

    ``to_latin(..., include_arabic=True)`` runs ``ar_lat.rules`` ahead of
    the target language's own rules, which is how an Arabic-script token
    embedded in a Cyrillic-script corpus gets romanised. The Kazakh rules
    map Cyrillic and leave Latin alone, so the pre-pass result survives
    them unchanged and can be compared against the direct call.
    """
    assert to_latin("قۇرئان", "kk", include_arabic=True) == to_latin("قۇرئان", "ar")
    assert to_latin("قۇرئان", "kk", include_arabic=True) == "qur'an"
    assert to_latin("قۇرئان", "kk") == "قۇرئان", "without the pre-pass the token is untouched"
