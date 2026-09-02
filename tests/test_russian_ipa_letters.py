"""Russian transliteration checked against the Illustration the rules cite.

Yanushevskaya & Bunčić (2015) is an Illustration of the IPA, so it carries
a keyword chart: one word per consonant phoneme (pp. 222-223) and one per
vowel in two consonantal contexts (p. 224), each printed in broad
transcription beside its scholarly transliteration and gloss. Those words
are the expectations here.

The chart prints the transliteration rather than Cyrillic, so the Cyrillic
column below is the orthographic form of the word the chart names, taken
from the transliteration table the Illustration gives on p. 221. Nothing
here is produced by running the transliterator over anything.

Two kinds of difference from the published form are expected, and every
word that shows one names which:

* notation, where this project writes a phoneme differently but means the
  same segment, and
* a simplification the rule file declares in its header, where the
  published form records something orthography does not determine.

A word with neither must match the Illustration exactly. That is the
property this file exists to hold, and it is what would catch a rule that
quietly stopped marking palatalisation.
"""

from __future__ import annotations

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

INHERITS_SOURCE = "https://doi.org/10.1017/S0025100314000395"

# The Illustration prints affricates as two symbols with no tie bar. These
# rules use the tie bar, as the seven Turkic files do, so that one phoneme
# is the same number of characters in every language a character-level
# model compares. Notation, not phonology.
NOTATION: tuple[tuple[str, str, str], ...] = (
    ("ts", "t͡s", "the tie bar binds the affricate, as in every sibling file"),
    ("tʃ", "t͡ʃ", "the tie bar binds the affricate, as in every sibling file"),
)

# The classes the rule file's header declares out of scope, each with the
# page of the Illustration that states the process. A word below may differ
# from its published form only by naming one of these.
DECLARED_SIMPLIFICATIONS: dict[str, str] = {
    "final devoicing": "p. 223: only voiceless obstruents occur word-finally",
    "voicing assimilation": "p. 223: regressive assimilation across a cluster",
    "palatalisation assimilation": "p. 223: assimilation can affect palatalisation",
    "vowel reduction": "p. 225: /e o/ do not occur in unstressed syllables",
    "loanword hard consonant": "p. 223: loanwords admit a hard consonant before /e/",
}

# Cyrillic, the transcription as the Illustration prints it, the form these
# rules produce, and the declared simplification when the two differ.
# ``None`` means the rules are expected to reproduce the published form.
KEYWORDS: tuple[tuple[str, str, str, str | None], ...] = (
    # Consonants, pp. 222-223
    ("пальцы", "palʲtsɨ", "palʲtsɨ", None),
    ("пяльцы", "pʲalʲtsɨ", "pʲalʲtsɨ", None),
    ("рад", "rat", "rad", "final devoicing"),
    ("ряд", "rʲat", "rʲad", "final devoicing"),
    ("бас", "bas", "bas", None),
    ("сад", "sat", "sad", "final devoicing"),
    ("бязь", "bʲasʲ", "bʲazʲ", "final devoicing"),
    ("сядь", "sʲatʲ", "sʲadʲ", "final devoicing"),
    ("мало", "mala", "malo", "vowel reduction"),
    ("запах", "zapax", "zapax", None),
    ("мяла", "mʲala", "mʲala", None),
    ("зябкий", "zʲapkʲij", "zʲabkʲij", "voicing assimilation"),
    ("фото", "fota", "foto", "vowel reduction"),
    ("царь", "tsarʲ", "tsarʲ", None),
    ("Фёдор", "fʲodar", "fʲodor", "vowel reduction"),
    ("чары", "tʃʲarɨ", "tʃʲarɨ", None),
    ("валик", "valʲik", "valʲik", None),
    ("щука", "ʃʲːuka", "ʃʲːuka", None),
    ("вялит", "vʲalʲit", "vʲalʲit", None),
    ("шар", "ʃar", "ʃar", None),
    ("тапка", "tapka", "tapka", None),
    ("жар", "ʒar", "ʒar", None),
    ("тяпка", "tʲapka", "tʲapka", None),
    ("яма", "jama", "jama", None),
    ("дом", "dom", "dom", None),
    ("кот", "kot", "kot", None),
    ("дёрн", "dʲorn", "dʲorn", None),
    ("ткёт", "tkʲot", "tkʲot", None),
    ("нос", "nos", "nos", None),
    ("год", "ɡot", "ɡod", "final devoicing"),
    ("нёс", "nʲos", "nʲos", None),
    ("Гёте", "ɡʲote", "ɡʲotʲe", "loanword hard consonant"),
    ("лот", "lot", "lot", None),
    ("хунта", "xunta", "xunta", None),
    ("лёд", "lʲot", "lʲod", "final devoicing"),
    ("Хюбнер", "xʲubner", "xʲubnʲer", "loanword hard consonant"),
    # Vowels in CVC and CʲVCʲ context, p. 224
    ("пили", "pʲilʲi", "pʲilʲi", None),
    ("пыл", "pɨl", "pɨl", None),
    ("шест", "ʃest", "ʃest", None),
    ("честь", "tʃʲesʲtʲ", "tʃʲestʲ", "palatalisation assimilation"),
    ("стопы", "stopɨ", "stopɨ", None),
    ("Стёпин", "stʲopʲin", "stʲopʲin", None),
    ("лук", "luk", "luk", None),
    ("люки", "lʲukʲi", "lʲukʲi", None),
)


def as_this_project_writes_it(published: str) -> str:
    """Rewrite a published transcription in the notation these rules use.

    Args:
        published: The transcription exactly as the Illustration prints it.

    Returns:
        The same transcription with this project's affricate notation.
    """
    rewritten = published
    for printed, ours, _reason in NOTATION:
        rewritten = rewritten.replace(printed, ours)
    return rewritten


def test_the_rule_file_cites_the_illustration_this_file_checks_against() -> None:
    """The binding between file and source is read, not restated.

    A test naming a source the rule file does not cite would check one
    description against another, which is the failure the Kyrgyz file
    records having had.
    """
    assert read_rule_source(_RULE_DIR / "ru_ipa.rules")["identifier"] == INHERITS_SOURCE


@pytest.mark.parametrize(
    ("cyrillic", "published", "ours", "simplification"),
    KEYWORDS,
    ids=[case[0] for case in KEYWORDS],
)
def test_keyword_transliterates_as_the_illustration_prints_it(
    cyrillic: str, published: str, ours: str, simplification: str | None
) -> None:
    """Each keyword transliterates to the chart's transcription.

    Args:
        cyrillic: The keyword in Russian orthography.
        published: The transcription the Illustration prints.
        ours: The transcription these rules are expected to produce.
        simplification: The declared class explaining a difference, or
            ``None`` when the rules reproduce the published form.
    """
    if simplification is None:
        assert ours == published, f"{cyrillic} claims no simplification but differs"
    else:
        assert simplification in DECLARED_SIMPLIFICATIONS
        assert ours != published, f"{cyrillic} names {simplification} but matches"

    assert to_ipa(cyrillic, "ru") == as_this_project_writes_it(ours)


def test_every_consonant_phoneme_of_the_chart_is_exercised() -> None:
    """The keyword list is not silently short of the chart it comes from.

    A parametrised test reports success over whatever it was given, so the
    count the Illustration's consonant chart supplies is pinned here. Thirty
    six consonants and eight vowel keywords are what pp. 222-224 print.
    """
    assert len(KEYWORDS) == 44
    assert sum(1 for case in KEYWORDS if case[3] is None) == 30


def test_palatalisation_is_marked_rather_than_spelled_as_a_glide() -> None:
    """A consonant before ‹я ё ю е и› is soft, and no /j/ is inserted.

    This is the defect the file was rewritten to fix. The earlier version
    mapped every iotated vowel to a j-sequence unconditionally, so
    ‹северный› came out with /sj/ where the Illustration's own passage
    transcription (p. 226) prints /sʲ/. Palatalisation is contrastive in
    Russian (p. 222), so that was a wrong phoneme rather than a coarse one.
    """
    assert to_ipa("северный", "ru") == "sʲevʲernɨj"
    assert to_ipa("ветер", "ru") == "vʲetʲer"


def test_a_glide_survives_where_the_orthography_really_has_one() -> None:
    """Word-initially, after a vowel and after either sign, /j/ is written.

    The palatalisation rule above must not swallow the cases that are
    genuinely iotated, which is the opposite error.
    """
    assert to_ipa("яма", "ru") == "jama"
    assert to_ipa("моя", "ru") == "moja"
    assert to_ipa("объект", "ru") == "objekt"
    assert to_ipa("статья", "ru") == "statʲja"


def test_the_unpaired_consonants_take_no_palatalisation() -> None:
    """‹ж ш ц› stay hard and ‹и› after them is ɨ; ‹ч щ› carry their own ʲ.

    ``солнце`` is the case that motivated spelling ц out ahead of the
    general rule: /t͡s/ ends in the same s the pairable-consonant set
    contains, so the general rule palatalised it.
    """
    assert to_ipa("солнце", "ru") == "solnt͡se"
    assert to_ipa("цена", "ru") == "t͡sena"
    assert to_ipa("жил", "ru") == "ʒɨl"
    assert to_ipa("цирк", "ru") == "t͡sɨrk"
    assert to_ipa("шест", "ru") == "ʃest"
