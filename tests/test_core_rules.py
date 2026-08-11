"""Tests for rule discovery, PyICU loading and the transliteration API.

The absent-PyICU path is exercised through a real implementation of the
provider protocol that raises exactly what the import raises on a machine
without the package, so the message a user would actually see is the
message under test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from turkic_translit import _test_hooks
from turkic_translit.core import (
    _RULE_DIR,
    FORWARD,
    _icu_trans,
    _require_icu,
    get_supported_languages,
    missing_icu_message,
    scan_rule_directory,
    to_ipa,
    to_latin,
)


@pytest.fixture
def without_pyicu() -> Iterator[None]:
    """Present PyICU as not installed.

    The compiled-transliterator cache is cleared on the way in and out,
    because a transliterator built while PyICU was present would
    otherwise be served without the provider being consulted.

    Yields:
        None, once, with the original provider captured.
    """
    previous = _test_hooks.icu
    _test_hooks.icu = _test_hooks.AbsentIcu()
    _icu_trans.cache_clear()
    yield
    _test_hooks.icu = previous
    _icu_trans.cache_clear()


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("win32", "turkic-pyicu-install"),
        ("linux", "libicu-dev"),
        ("darwin", "brew install icu4c"),
    ],
)
def test_each_known_platform_gets_its_own_command(platform: str, expected: str) -> None:
    """The message names the command that works on that platform.

    Args:
        platform: Value of ``sys.platform``.
        expected: A fragment of the command for that platform.
    """
    message = missing_icu_message("3.11", platform)

    assert expected in message
    assert "PyICU missing on Python 3.11" in message
    assert platform in message


def test_an_unrecorded_platform_gets_generic_advice() -> None:
    """A platform with no recorded command still gets a usable message."""
    message = missing_icu_message("3.11", "aix")

    assert "Please install the ICU C++ libraries for your platform." in message
    assert "PyICU missing on Python 3.11 (aix)" in message


def test_an_absent_pyicu_is_reported_with_install_advice(without_pyicu: None) -> None:
    """The import failure becomes a message naming the install command.

    Args:
        without_pyicu: The bound provider, which reports PyICU absent.
    """
    with pytest.raises(RuntimeError, match="PyICU missing on Python") as raised:
        _require_icu()

    assert str(raised.value.__cause__) == "No module named 'icu'"


def test_transliterating_without_pyicu_reports_the_same_message(without_pyicu: None) -> None:
    """The failure surfaces from the public API, not only the loader.

    Args:
        without_pyicu: The bound provider, which reports PyICU absent.
    """
    with pytest.raises(RuntimeError, match="PyICU missing on Python"):
        to_ipa("мектеп", "kk")


def test_the_installed_provider_returns_a_working_compiler() -> None:
    """The provider production binds hands back a working PyICU."""
    compile_rules = _test_hooks.InstalledIcu().rule_compiler()

    compiled = compile_rules("test", "a > b;", FORWARD)

    assert compiled.transliterate("aaa") == "bbb"


def test_every_rule_file_is_advertised_by_the_language_list() -> None:
    """The advertised formats match the rule files on disk."""
    supported = get_supported_languages()

    for rule_file in _RULE_DIR.glob("*.rules"):
        if "_" not in rule_file.stem:
            continue
        lang, fmt = rule_file.stem.split("_", 1)
        expected = "latin" if fmt in {"lat", "lat2023"} else fmt
        assert expected in supported[lang]


def test_a_rule_file_naming_no_format_is_passed_over(tmp_path: Path) -> None:
    """A file whose name carries no format describes no rule set."""
    (tmp_path / "orphan.rules").touch()
    (tmp_path / "kk_ipa.rules").touch()

    assert scan_rule_directory(tmp_path) == {"kk": ["ipa"]}


def test_both_latin_spellings_collapse_to_one_format(tmp_path: Path) -> None:
    """``lat`` and ``lat2023`` are one format, listed once."""
    (tmp_path / "kk_lat.rules").touch()
    (tmp_path / "kk_lat2023.rules").touch()

    assert scan_rule_directory(tmp_path) == {"kk": ["latin"]}


def test_formats_accumulate_per_language(tmp_path: Path) -> None:
    """Each language collects every format its files advertise."""
    (tmp_path / "kk_ipa.rules").touch()
    (tmp_path / "kk_lat.rules").touch()
    (tmp_path / "az_ipa.rules").touch()

    assert scan_rule_directory(tmp_path) == {"az": ["ipa"], "kk": ["ipa", "latin"]}


def test_a_directory_with_no_rules_advertises_nothing(tmp_path: Path) -> None:
    """An empty directory yields an empty mapping, not an error."""
    assert scan_rule_directory(tmp_path) == {}


def test_a_format_with_further_underscores_is_kept_whole(tmp_path: Path) -> None:
    """Only the first underscore separates the language from the format."""
    (tmp_path / "uz_cyr_ipa.rules").touch()

    assert scan_rule_directory(tmp_path) == {"uz": ["cyr_ipa"]}


def test_latin_variants_are_reported_under_one_name() -> None:
    """``lat`` and ``lat2023`` are both advertised as ``latin``."""
    supported = get_supported_languages()

    assert "latin" in supported["kk"]
    assert "lat2023" not in supported["kk"]
    assert "lat" not in supported["kk"]


def test_a_format_is_listed_once_per_language() -> None:
    """Two rule files mapping to ``latin`` produce one entry."""
    for formats in get_supported_languages().values():
        assert len(formats) == len(set(formats))


def test_an_unsupported_latin_language_lists_the_ones_that_work() -> None:
    """The rejection names every language Latin output is available for."""
    with pytest.raises(ValueError, match="Latin transliteration not supported for 'zz'"):
        to_latin("text", "zz")


def test_a_language_with_only_ipa_rules_is_rejected_for_latin() -> None:
    """Azerbaijani has IPA rules and no Latin ones, and says so."""
    with pytest.raises(ValueError, match="Latin transliteration not supported for 'az'"):
        to_latin("uşaq", "az")


def test_a_language_with_only_latin_rules_is_rejected_for_ipa() -> None:
    """Arabic has Latin rules and no IPA ones, and says so."""
    with pytest.raises(ValueError, match="IPA transliteration not supported for 'ar'"):
        to_ipa("سلام", "ar")


def test_an_unsupported_ipa_language_lists_the_ones_that_work() -> None:
    """The rejection names every language IPA output is available for."""
    with pytest.raises(ValueError, match="IPA transliteration not supported for 'zz'"):
        to_ipa("text", "zz")


def test_arabic_is_folded_before_the_target_rules_when_asked() -> None:
    """The Arabic pre-pass changes the result for Arabic-script input."""
    without = to_latin("سلام", "kk", include_arabic=False)
    with_arabic = to_latin("سلام", "kk", include_arabic=True)

    assert with_arabic != without
    assert with_arabic.isascii()


def test_output_is_normalised_to_composed_form() -> None:
    """Results come back in NFC, whatever the rules emit."""
    import unicodedata

    result = to_ipa("мектеп", "kk")

    assert result == unicodedata.normalize("NFC", result)


COMBINING_BREVE = "\u0306"
COMBINING_DIAERESIS = "\u0308"
SOFT_G_COMPOSED = "da\u011f"
SOFT_G_DECOMPOSED = "dag" + COMBINING_BREVE


def test_decomposed_input_transliterates_as_its_composed_form() -> None:
    """A decomposed letter is mapped, not left as a base plus its mark.

    The rule files spell precomposed letters, so decomposed input matched
    no rule: Turkish soft g arrived as ``g`` plus a combining breve, the
    base letter was rewritten, and the breve survived into the output as
    a segment no Turkic inventory contains.
    """
    composed = to_ipa(SOFT_G_COMPOSED, "tr")
    decomposed = to_ipa(SOFT_G_DECOMPOSED, "tr")

    assert composed == "da\u02d0"
    assert decomposed == composed
    assert COMBINING_BREVE not in decomposed


def test_decomposed_input_reaches_the_latin_rules_composed() -> None:
    """The Latin path normalises its input for the same reason.

    The decomposed form is derived with NFD rather than written out, so
    the test cannot accidentally compare two different letters.
    """
    import unicodedata

    word = "\u0430\u0439\u0434\u044b\u04a3"
    decomposed = unicodedata.normalize("NFD", word)

    assert decomposed != word
    assert to_latin(decomposed, "kk") == to_latin(word, "kk")
    assert COMBINING_BREVE not in to_latin(decomposed, "kk")
