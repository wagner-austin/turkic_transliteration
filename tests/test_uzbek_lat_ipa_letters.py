"""Uzbek transliteration checked against the description the rules cite.

Ido prints the Latin orthography beside each transcription, so nothing is
inferred about the spelling. All twenty-three keywords match once six
declared deviations are applied: five are notation, and one is a
difference in how a rhotic is written.

Both this file and the Cyrillic one draw on the same Illustration, since
the two rule sets transcribe the same language in its two alphabets.
"""

from __future__ import annotations

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

INHERITS_SOURCE = "https://doi.org/10.1017/S0025100324000148"

# Applied to the published transcription to obtain what these rules
# produce. The first three follow from the consonant chart on p. 153,
# where the source places these fricatives and affricates at the
# alveolo-palatal position and this project writes the postalveolar
# series instead.
NOTATION: tuple[tuple[str, str, str], ...] = (
    ("d͡ʑ", "d͡ʒ", "the source writes the alveolo-palatal affricate, p. 153"),
    ("t͡ɕ", "t͡ʃ", "the source writes the alveolo-palatal affricate, p. 153"),
    ("ɕ", "ʃ", "the source writes the alveolo-palatal fricative, p. 153"),
    ("ɾ", "r", "the source writes the rhotic as a tap, p. 153"),
    ("χ", "x", "the source writes the voiceless fricative as uvular, p. 153"),
    ("g", "ɡ", "the journal typesets the voiced velar plosive as U+0067; IPA is U+0261"),
)

# Latin orthography, Ido's transcription, gloss, page
KEYWORDS: tuple[tuple[str, str, str, int], ...] = (
    ("pesh", "peɕ", "front", 154),
    ("besh", "beɕ", "five", 154),
    ("tor", "tɔɾ", "narrow", 154),
    ("dor", "dɔɾ", "rope", 154),
    ("sol", "sɔl", "raft", 154),
    ("zol", "zɔl", "adept", 154),
    ("shol", "ɕɔl", "woolen fabric", 154),
    ("mol", "mɔl", "livestock", 154),
    ("nol", "nɔl", "zero", 154),
    ("fahm", "fahm", "quick-wittedness", 154),
    ("vahm", "vahm", "fright", 154),
    ("chin", "t͡ɕin", "genuine", 154),
    ("jin", "d͡ʑin", "genie", 154),
    ("jing", "d͡ʑiŋ", "complaints", 154),
    ("ko'r", "koɾ", "blind", 154),
    ("go'r", "goɾ", "tomb", 154),
    ("qo'r", "qoɾ", "coal", 154),
    ("lol", "lɔl", "speechless", 154),
    ("yol", "jɔl", "mane", 154),
    ("rol", "ɾɔl", "role", 154),
    ("xam", "χam", "adroop", 154),
    ("g'am", "ʁam", "grief", 154),
    ("ham", "ham", "also", 154),
)

# The two vowels the orthography distinguishes with the modifier letter,
# which the keyword list contrasts directly: plain <o> is the open vowel,
# <oʻ> the close one.
O_CONTRAST: tuple[tuple[str, str, str], ...] = (
    ("tor", "tɔr", "plain o is the open vowel"),
    ("ko'r", "kor", "o with the modifier is the close vowel"),
    ("yol", "jɔl", "the yo digraph carries the open vowel, like plain o"),
)

# Sequences where a shorter digraph rule used to steal a letter from the
# apostrophe letter behind it: <yoʻ> is y + oʻ and <ngʻ> is n + gʻ,
# because the apostrophe exists only as part of oʻ and gʻ. The expected
# values compose from the letter values Ido prints (y=j, oʻ=o, gʻ=ʁ,
# ng=ŋ), and the Cyrillic spelling of each word writes the same parse
# with plain letters, so both alphabets are checked and must agree.
SEAMS: tuple[tuple[str, str, str, str], ...] = (
    ("yo'l", "йўл", "jol", "road"),
    ("yo'q", "йўқ", "joq", "absent"),
    ("qo'ng'iz", "қўнғиз", "qonʁiz", "beetle"),
    ("to'ng'ich", "тўнғич", "tonʁit͡ʃ", "firstborn"),
    ("jing", "жинг", "d͡ʒiŋ", "complaints, plain ng stays the velar nasal"),
    ("ta'sir", "таъсир", "taʔsir", "effect, a bare apostrophe is the tutuq belgisi"),
    ("ba'zi", "баъзи", "baʔzi", "some"),
)

# The corpora spell the modifier with several codepoints; the rules
# declare them equivalent in $Apo, so the parse must not depend on which
# one a text uses.
APOSTROPHES: tuple[str, ...] = ("'", "ʻ", "’", "ʼ")


def as_this_project_writes_it(published: str) -> str:
    """Apply the declared deviations to a published transcription.

    Args:
        published: The transcription exactly as the source prints it.

    Returns:
        The same transcription in the notation these rules produce.
    """
    for source_symbol, ours, _reason in NOTATION:
        published = published.replace(source_symbol, ours)
    return published


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """These expectations come from the paper the rule file names."""
    declared = read_rule_source(_RULE_DIR / "uz_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "Ido, S."
    assert declared["year"] == 2025


@pytest.mark.parametrize(("orthography", "published", "gloss", "page"), KEYWORDS)
def test_keyword_transliterates_as_the_source_prints_it(
    orthography: str, published: str, gloss: str, page: int
) -> None:
    """Each keyword matches the source, allowing the declared deviations."""
    assert page == 154
    assert gloss != ""
    assert to_ipa(orthography, "uz") == as_this_project_writes_it(published)


@pytest.mark.parametrize(
    ("latin", "cyrillic", "expected", "gloss"), SEAMS, ids=[s[0] for s in SEAMS]
)
def test_digraph_boundary_parses_as_the_orthography_defines(
    latin: str, cyrillic: str, expected: str, gloss: str
) -> None:
    """The apostrophe letter binds tighter than the digraph before it.

    Args:
        latin: The word in Latin orthography, apostrophe written ASCII.
        cyrillic: The same word in Cyrillic orthography.
        expected: The transcription composed from the pinned letter values.
        gloss: What the word means, for the reader.
    """
    assert gloss != ""
    assert to_ipa(latin, "uz") == expected
    assert to_ipa(cyrillic, "uzc") == expected


@pytest.mark.parametrize("mark", APOSTROPHES, ids=[f"U+{ord(m):04X}" for m in APOSTROPHES])
def test_the_parse_does_not_depend_on_the_apostrophe_codepoint(mark: str) -> None:
    """Every declared apostrophe variant yields the same parse.

    Args:
        mark: One apostrophe-like codepoint from the rules' $Apo class.
    """
    assert to_ipa(f"yo{mark}l", "uz") == "jol"
    assert to_ipa(f"qo{mark}ng{mark}iz", "uz") == "qonʁiz"


@pytest.mark.parametrize(("orthography", "expected", "reason"), O_CONTRAST)
def test_the_two_o_vowels_stay_distinct(orthography: str, expected: str, reason: str) -> None:
    """Plain o and o-with-modifier map to different vowels.

    The digraph rule for <yo> used to emit the close vowel, which put the
    wrong vowel on every word spelled with it while plain <o> elsewhere
    was correct. Ido gives yol 'mane' as the open vowel (p. 154).
    """
    assert reason != ""
    assert to_ipa(orthography, "uz") == expected
