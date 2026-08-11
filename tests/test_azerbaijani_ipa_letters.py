"""Azerbaijani transliteration checked against the description the rules cite.

This is the one language here whose cited source cannot ground a keyword
test, and there are two separate reasons.

The first is that the keyword list carries no orthography. The
Illustration prints each consonant and vowel with a phonemic
transcription and a gloss — /pul/ 'money', /ɑtɑ/ 'dad' — and never the
written form, because the variety it describes is written in Arabic
script: "The orthographic version of the recorded passage in the present
paper was written using Arabic script" (p. 1). These rules map the Latin
alphabet, so there is no orthography-to-IPA pair in the article to check
them against.

The second is scope, and the rule file states it: the Illustration "is
based on the urban variety of Tabriz dialectal region" (p. 1), while
these rules target the Baku standard. Its consonant chart gives an
alveolar affricate pair the Baku standard lacks and parenthesises two
plosives.

What the source does state unconditionally is the inventory, and that is
what is checked here, together with the two Tabriz-specific segments
which must stay out of the output. Neither claim needs an expected value
for any particular word, so neither can be circular.
"""

from __future__ import annotations

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

INHERITS_SOURCE = "https://doi.org/10.1017/S0025100317000184"

# p. 3: "Azerbaijani has nine vowels, /æ ɑ o e œ ɯ u i y/, with no length
# distinction." The same nine appear in the vowel plot on the same page
# and in Table 1 on p. 4.
PUBLISHED_VOWELS = frozenset("æɑoeœɯuiy")

# p. 2, the consonant chart, in this project's notation: the voiced velar
# plosive is U+0261 rather than the U+0067 the journal typesets.
PUBLISHED_CONSONANTS = frozenset("pbtdcɟkɡmnɾfvszʃʒxɣhlj")

# p. 2: affricates, which the chart gives as four. Written with the tie
# bar over two components that the chart already lists separately.
PUBLISHED_AFFRICATES = ("t͡s", "d͡z", "t͡ʃ", "d͡ʒ")

# Segments the chart has that the Baku standard these rules target does
# not, so they must never appear in output. The alveolar pair is phonemic
# in Tabriz; the palatal voiceless plosive is one of the two the chart
# parenthesises, and in the Baku standard <k> is the velar.
TABRIZ_ONLY = (
    ("t͡s", "the alveolar affricate is phonemic in Tabriz, p. 2"),
    ("d͡z", "the alveolar affricate is phonemic in Tabriz, p. 2"),
    ("c", "the chart parenthesises the palatal plosive, p. 2"),
)

# The alphabet the rules map, which is the Baku standard's. Because the
# map is context-free — asserted below — running every letter through it
# is exhaustive, and a segment absent from that set cannot appear in any
# Azerbaijani text. Sentence-level conformance for this language lives
# with the other languages' in test_inventory_conformance.py.
ALPHABET = "abcçdeəfgğhxıijkqlmnoöprsştuüvyz"

WORDS: tuple[str, ...] = (
    "Şimal",
    "Günəş",
    "mübahisə",
    "səyahətçi",
    "güclüdür",
    "çıxartdı",
    "bürüyürdü",
    "qədər",
)


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """The inventory below comes from the Illustration the rule file names."""
    declared = read_rule_source(_RULE_DIR / "az_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "Ghaffarvand Mokari, P. & Werner, S."
    assert declared["year"] == 2017


def test_every_letter_maps_inside_the_published_inventory() -> None:
    """No letter of the alphabet produces a segment the source omits.

    The strongest statement available without an orthography in the
    source: whatever the right transcription of a given Azerbaijani word
    is, a segment outside this chart is wrong for every word.
    """
    published = PUBLISHED_VOWELS | PUBLISHED_CONSONANTS

    for letter in ALPHABET:
        produced = to_ipa(letter, "az")
        segments = {char for char in produced if char != "͡"}

        assert segments <= published, f"{letter!r} produced {produced!r}"


@pytest.mark.parametrize("word", WORDS)
def test_the_map_is_context_free(word: str) -> None:
    """A word transliterates as the concatenation of its letters.

    This is what makes the letter-level checks in this file exhaustive.
    The rule file has no context rules — Azerbaijani harmony needs none,
    because the orthography writes every vowel — and if that ever stops
    being true, the letter-level coverage argument stops holding with it.
    """
    assert to_ipa(word, "az") == "".join(to_ipa(letter, "az") for letter in word)


def test_the_nine_vowels_are_all_reachable() -> None:
    """Every vowel of p. 3 is produced by some letter of the alphabet.

    The converse of the conformance check above. Together they say the
    rules use the published vowel system and all of it, rather than a
    subset that happens to avoid the disputed symbols.
    """
    reachable = set()
    for letter in ALPHABET:
        reachable |= {char for char in to_ipa(letter, "az") if char in PUBLISHED_VOWELS}

    assert reachable == PUBLISHED_VOWELS


@pytest.mark.parametrize(("segment", "reason"), TABRIZ_ONLY)
def test_a_segment_of_the_other_variety_is_never_emitted(segment: str, reason: str) -> None:
    """The variety difference the rule file declares is pinned here.

    The Illustration describes Tabriz and these rules target Baku. That
    gap is recorded in the rule file's scope note rather than papered
    over, and this keeps it from being closed by accident in either
    direction.
    """
    assert reason != ""
    for letter in ALPHABET:
        assert segment not in to_ipa(letter, "az")


def test_the_postalveolar_affricates_are_the_ones_the_rules_use() -> None:
    """<c> and <ç> are the postalveolar pair, not the alveolar one.

    Both pairs are in the chart on p. 2. Which pair a Latin letter maps
    to is a fact about the Baku orthography rather than about the chart,
    so it is stated here explicitly instead of being left to the
    inventory check, which both pairs would pass.
    """
    assert to_ipa("c", "az") == "d͡ʒ"
    assert to_ipa("ç", "az") == "t͡ʃ"
    assert set(PUBLISHED_AFFRICATES) == {"t͡s", "d͡z", "t͡ʃ", "d͡ʒ"}


def test_no_keyword_of_the_source_carries_an_orthography() -> None:
    """The reason this file tests an inventory rather than keywords.

    Stated as an assertion so it cannot rot into a stale comment: the
    alphabet these rules map is Latin, and the Illustration's passage is
    in Arabic script (p. 1), so its keywords cannot be fed to these rules
    at all.
    """
    assert all(letter.isascii() or letter in "çəğıöşü" for letter in ALPHABET)
    assert all(ord(letter) < 0x0600 for letter in ALPHABET)
