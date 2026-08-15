"""Tests for the alphabet and emitted-character inventory.

The head parser is exercised on synthetic rule text covering every shape
a line can take, and the emitted sets are pinned on the facts that make
the cleaner's token filter work: the letters foreign material rides in
on are not emittable, and the letters the languages actually write are.
"""

from __future__ import annotations

from turkic_translit.corpus.inventory import (
    APOSTROPHES,
    emitted_characters,
    heads_from_source,
    multi_char_rule_heads,
    seam_inputs,
    source_alphabets,
)

SYNTHETIC = """# comment line, ignored entirely
$Apo = [ʼ ' ʻ] ;
ng $Apo > nʁ ;   # expands to one head per apostrophe
$Apo > ʔ ;       # bare macro head, skipped
$Vow $Apo > x ;  # macro-bearing base, skipped
sh > ʃ ;
a > ɑ ;          # single character, not a head
e { j > i ;      # context rule, skipped
:: NFC ;
"""


def test_heads_expand_apostrophes_and_skip_what_they_cannot_exercise() -> None:
    """One synthetic file covers every line shape the parser handles."""
    heads = heads_from_source(SYNTHETIC)

    assert heads == [*(f"ng{mark}" for mark in APOSTROPHES), "sh"]


def test_a_headless_left_side_is_not_a_head() -> None:
    """A rule whose left side is empty contributes nothing."""
    assert heads_from_source("> x ;") == []


def test_the_uzbek_rule_file_declares_the_apostrophe_digraphs() -> None:
    """The heads that caused the seam defect are read from the real file."""
    heads = multi_char_rule_heads("uz")

    assert "ng" + APOSTROPHES[0] in heads
    assert "yo" + APOSTROPHES[0] in heads
    assert "sh" in heads


def test_every_language_has_an_alphabet_and_letters_to_sweep() -> None:
    """The inventory serves the full language set with real content."""
    alphabets = source_alphabets()

    assert set(alphabets) == {"az", "fi", "kk", "ky", "tr", "ug", "uz", "uzc"}
    assert "ب" in alphabets["ug"]
    assert len(seam_inputs("uz")) > len(alphabets["uz"]) ** 2


def test_emitted_characters_cover_digraph_products() -> None:
    """The seam sweep's outputs are in the emitted set, not just letters'."""
    emitted = emitted_characters("uz")

    assert "ʁ" in emitted  # only <gʻ> produces it
    assert "ŋ" in emitted  # only <ng> produces it
    assert "ʃ" in emitted  # only <sh> produces it


def test_the_letters_foreign_material_rides_in_on_are_not_emittable() -> None:
    """The measured passthrough letters stay outside each emitted set.

    These are the letters the corpus audit found carrying English and
    quoted material: they are what the cleaner's token filter keys on,
    so their absence from the emitted sets is the property that makes
    the filter mean something.
    """
    assert "c" not in emitted_characters("ky")
    assert "g" not in emitted_characters("ky")
    assert "w" not in emitted_characters("tr")
    assert "q" not in emitted_characters("tr")
    assert "c" not in emitted_characters("uz")
    assert "A" not in emitted_characters("kk")


def test_loan_letters_keep_their_values_in_the_emitted_sets() -> None:
    """A letter of a neighbouring orthography is emittable, deliberately.

    The Cyrillic rule files map every letter the mixed-script corpora
    carry, so ky emits h for һ and ҳ even though Kyrgyz itself has no
    /h/. The token filter therefore cannot catch an English word made
    only of such letters; that residue is what the attestation floor at
    scoring time exists for.
    """
    assert "h" in emitted_characters("ky")
    assert "h" in emitted_characters("kk")
