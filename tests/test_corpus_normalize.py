"""Tests for source-text normalization and the packaged fold table.

The three character classes this stage exists for are each pinned with
the real defect that motivated them: a soft hyphen inside a Kazakh word,
an Arabic presentation form of a native Uyghur letter, and the cp1254
mojibake the Turkish raw corpus carries.
"""

from __future__ import annotations

from turkic_translit.corpus.normalize import (
    PACKAGED_FOLDS,
    normalize_line,
    strip_format_characters,
)
from turkic_translit.corpus.symbols import (
    MERGE_ACTION,
    apply_substitutions,
    read_symbol_map,
    scopes_of,
    substitutions_for,
)

SOFT_HYPHEN = "­"
ZERO_WIDTH_SPACE = "​"
BOM = "﻿"
NO_BREAK_SPACE = "\N{NO-BREAK SPACE}"

# Presentation forms from the Uyghur raw corpus, and the letters they
# display: U+FBE9 initial alef maksura, U+FEEA final heh, U+FE8E final
# alef.
PRESENTATION_FORMS = "ﯩﻪﺎ"
DISPLAYED_LETTERS = "ىها"


def test_normalize_collapses_whitespace() -> None:
    """Tabs, newlines and runs of spaces become single spaces."""
    assert normalize_line(" salom\tdunyo\n qalaysiz  ") == "salom dunyo qalaysiz"


def test_normalize_composes_decomposed_characters() -> None:
    """A decomposed diacritic becomes one code point, so words compare equal."""
    assert normalize_line("üch") == "üch"


def test_normalize_reports_a_blank_fragment_as_empty() -> None:
    """A fragment of pure whitespace normalises to the empty string."""
    assert normalize_line("  \t\n ") == ""


def test_format_characters_vanish_without_splitting_the_word() -> None:
    """A soft hyphen inside a word deletes to nothing, not to a space.

    The Kazakh glide rule cannot match across an invisible character,
    and the cleaner used to turn it into a space that split the word.
    Deletion rejoins the word instead.
    """
    assert strip_format_characters(f"та{SOFT_HYPHEN}у") == "тау"
    assert normalize_line(f"та{SOFT_HYPHEN}у {ZERO_WIDTH_SPACE}{BOM}bir") == "тау bir"


def test_presentation_forms_fold_to_the_letters_they_display() -> None:
    """An Arabic presentation form becomes the letter the rules map.

    The Uyghur raw corpus carried thousands of these display codepoints,
    which passed through transliteration unmapped.
    """
    assert normalize_line(PRESENTATION_FORMS) == DISPLAYED_LETTERS


def test_a_no_break_space_collapses_like_any_other_space() -> None:
    """Compatibility normalization turns NBSP into a collapsible space."""
    assert normalize_line(f"bir{NO_BREAK_SPACE}eki") == "bir eki"


def test_every_packaged_fold_is_a_justified_merge() -> None:
    """Each fold row rewrites something, says why, and cites its ground."""
    rules = read_symbol_map(PACKAGED_FOLDS)

    assert len(rules) == 8, "folds gained or lost a row; update this pin deliberately"
    for rule in rules:
        assert rule["action"] == MERGE_ACTION
        assert rule["verdict"] == "MISENCODING"
        assert rule["rationale"] != ""
        assert rule["citation"] != ""


def test_fold_scopes_name_only_languages_this_project_ships() -> None:
    """A typo in a scope would silently disable the fold; pin the set."""
    assert scopes_of(read_symbol_map(PACKAGED_FOLDS)) == {"ky", "tr"}


def test_the_kyrgyz_fold_repairs_the_tailed_nasal() -> None:
    """The en-with-tail lookalike becomes the alphabet's own letter."""
    folds = substitutions_for(read_symbol_map(PACKAGED_FOLDS), "ky")

    assert apply_substitutions("кеӊеш жаӊы Ӊ", folds) == "кеңеш жаңы Ң"


def test_the_turkish_folds_repair_the_codepage_mojibake() -> None:
    """Text decoded with the wrong codepage gets its letters back."""
    folds = substitutions_for(read_symbol_map(PACKAGED_FOLDS), "tr")

    assert apply_substitutions("satýn þekilde doðru Ýstanbul", folds) == (
        "satın şekilde doğru İstanbul"
    )


def test_folds_for_an_unaffected_language_are_empty() -> None:
    """A language with no measured misencoding gets no rewrites."""
    assert substitutions_for(read_symbol_map(PACKAGED_FOLDS), "kk") == {}
