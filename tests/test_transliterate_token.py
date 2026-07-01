"""Smoke tests for :func:`turkic_translit.transliterate.transliterate_token`.

Covers the previously-untested regression where the function hardcoded
``kk``/``ky`` and silently pass-through-returned tokens for the other five
supported languages, silently breaking
:class:`turkic_translit.pipeline.TurkicTransliterationPipeline` for anything
except Kazakh and Kyrgyz.
"""

import pytest

from turkic_translit.core import get_supported_languages
from turkic_translit.transliterate import transliterate_token

# Representative one-token samples per language. Values chosen so that a
# correct IPA rule application produces a string that differs from the input
# (i.e. at least one character is mapped by the rules), making regressions
# to pass-through behaviour trivially detectable.
IPA_SAMPLES: dict[str, str] = {
    "kk": "мектеп",  # Kazakh Cyrillic → IPA
    "ky": "мектеп",  # Kyrgyz Cyrillic → IPA
    "tr": "çocuk",  # Turkish Latin (diacritics) → IPA
    "az": "uşaq",  # Azerbaijani Latin (diacritics) → IPA
    "uz": "shamol",  # Uzbek Latin (digraph) → IPA
    "ug": "بالا",  # Uyghur Arabic → IPA
    "fi": "kiitos",  # Finnish Latin → IPA
}

LATIN_SAMPLES: dict[str, str] = {
    "kk": "мектеп",
    "ky": "мектеп",
}


@pytest.mark.parametrize(("lang", "token"), IPA_SAMPLES.items())
def test_transliterate_token_ipa_all_supported(lang: str, token: str) -> None:
    """Every language with an ``_ipa.rules`` file yields non-identity IPA."""
    supported = get_supported_languages()
    assert lang in supported, f"precondition: {lang} known to rules directory"
    assert "ipa" in supported[lang], (
        f"precondition: {lang} advertised as IPA-capable by rules directory"
    )
    result = transliterate_token(token, lang, "ipa")
    assert result, f"{lang} IPA returned empty for {token!r}"
    assert result != token, (
        f"{lang} IPA returned input unchanged — regression to pre-fix "
        f"pass-through behaviour ({token!r} -> {result!r})"
    )


@pytest.mark.parametrize(("lang", "token"), LATIN_SAMPLES.items())
def test_transliterate_token_latin_available_languages(lang: str, token: str) -> None:
    """Latin output for the languages that ship a ``_lat.rules`` file."""
    result = transliterate_token(token, lang, "latin")
    assert result, f"{lang} Latin returned empty for {token!r}"
    assert result != token, f"{lang} Latin returned input unchanged for {token!r}"


def test_transliterate_token_russian_pass_through() -> None:
    """Russian tokens are returned unchanged (documented behaviour)."""
    assert transliterate_token("привет", "ru", "ipa") == "привет"
    assert transliterate_token("привет", "ru", "latin") == "привет"


def test_transliterate_token_unknown_language_pass_through() -> None:
    """An unrecognised language code returns the token unchanged.

    This preserves the mixed-stream behaviour that the pipeline relies on:
    tokens whose language is outside the configured target set flow through
    without being corrupted.
    """
    assert transliterate_token("hello", "en", "ipa") == "hello"
    assert transliterate_token("hello", "xx", "latin") == "hello"


def test_transliterate_token_invalid_mode_raises() -> None:
    """Unknown ``mode`` values are a programmer error, not silent pass-through."""
    with pytest.raises(ValueError, match="Unknown transliteration mode"):
        transliterate_token("kitap", "tr", "hieroglyph")


def test_transliterate_token_latin_unsupported_falls_through() -> None:
    """Requesting Latin for a language without ``_lat.rules`` passes through.

    This is a deliberate design choice: the function is called on every
    token in a mixed-language stream, so an unsupported ``(lang, mode)``
    pair must not raise — the token simply isn't transliterated.
    """
    supported = get_supported_languages()
    # Finnish has ipa rules but no latin rules; requesting latin should
    # fall through rather than raise.
    if "fi" in supported and "latin" not in supported["fi"]:
        assert transliterate_token("kiitos", "fi", "latin") == "kiitos"
