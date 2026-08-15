"""Tests for the Direct Transliteration tab.

The handlers are driven directly, so every decision the tab makes is
exercised on real transliteration output rather than inferred from the
rendered interface. The download directory is bound to a disposable path
so the tab's file writes are observable.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import pytest

from tests import test_azerbaijani_ragagnin_words as az_words
from tests import test_finnish_karlsson_words as fi_words
from tests import test_kazakh_routledge_words as kk_words
from tests import test_kyrgyz_words_mccollum as ky_words
from tests import test_turkish_routledge_words as tr_words
from tests import test_uyghur_mfa_agreement as ug_words
from tests import test_uzbek_cyr_ipa_letters as uzc_letters
from tests import test_uzbek_lat_ipa_letters as uz_letters
from turkic_translit import _test_hooks
from turkic_translit.web.tabs import direct

LONG_TURKISH = "Merhaba dünya, bu metin indirme esigini asacak kadar uzun bir metindir."

# The words each gold-standard module holds to its published source.
# The tab's examples are drawn from these, and the test below says so,
# so an example cannot quietly become a word nothing checks.
PINNED_WORDS: dict[str, frozenset[str]] = {
    "az": frozenset(word for word, *_rest in az_words.WORDS),
    "fi": frozenset(word for word, *_rest in fi_words.WORDS),
    "kk": frozenset(word for word, *_rest in kk_words.WORDS),
    "ky": frozenset(ky_words.GOLD),
    "tr": frozenset(word for word, *_rest in tr_words.WORDS),
    "ug": frozenset(word for word, _prons in ug_words.ROWS),
    "uz": frozenset(word for word, *_rest in uz_letters.KEYWORDS),
    "uzc": frozenset(word for word, *_rest in uzc_letters.KEYWORDS),
}


@pytest.fixture
def cron_dir(tmp_path: Path) -> Iterator[Path]:
    """Point the tab's download directory at a disposable path.

    Yields:
        The directory the tab writes downloadable copies into.
    """
    previous = _test_hooks.environment
    target = tmp_path / "cronjob"
    _test_hooks.environment = _test_hooks.MappingEnvironment({"TURKIC_CRON_DIR": str(target)})
    yield target
    _test_hooks.environment = previous


def test_the_tab_offers_only_languages_with_ipa_rules() -> None:
    """Every offered code has an ``<lang>_ipa.rules`` file behind it."""
    from turkic_translit.core import get_supported_languages

    offered = direct.ipa_languages()

    assert offered == sorted(offered)
    assert offered
    assert all("ipa" in get_supported_languages()[code] for code in offered)


def test_every_offered_language_has_an_example() -> None:
    """The example table covers the offered languages and nothing else.

    A rules file arriving without an example would raise a KeyError
    while the interface is built, which is a crash at startup rather
    than a missing row. This fails first, and in ``make check``.
    """
    assert set(direct.EXAMPLE_WORDS) == set(direct.ipa_languages())


@pytest.mark.parametrize(("code", "word"), sorted(direct.EXAMPLE_WORDS.items()))
def test_each_example_is_a_word_this_suite_pins(code: str, word: str) -> None:
    """Every example is a word checked against a published description.

    The demo's claim is that its output is auditable against the
    literature, so the words it invites a visitor to try are the words
    whose transcriptions this suite already holds to a source.
    """
    assert word in PINNED_WORDS[code]


def test_typed_text_is_transliterated_to_ipa(cron_dir: Path) -> None:
    """Typing transliterates and writes nothing to disk.

    Args:
        cron_dir: The bound download directory, which stays empty.
    """
    result, stats = direct.transliterate_request("merhaba", "tr")

    assert result == "meɾhaba"
    assert "Bytes" in stats
    assert not cron_dir.exists()


def test_typing_a_long_text_still_writes_nothing(cron_dir: Path) -> None:
    """Length no longer decides what lands on disk.

    Args:
        cron_dir: The bound download directory, which stays empty.
    """
    result, _stats = direct.transliterate_request(LONG_TURKISH, "tr")

    assert result != LONG_TURKISH
    assert not cron_dir.exists()


def test_the_download_handler_writes_the_result(cron_dir: Path) -> None:
    """Asking for a file produces one, whatever the length.

    Args:
        cron_dir: The bound download directory, which receives the file.
    """
    result, _stats, download = direct.transliterated_download("merhaba", "tr")

    written = sorted(cron_dir.iterdir())[0]
    assert written.read_text(encoding="utf-8") == result
    assert download.visible is True
    # Gradio serves its own copy, so the bytes a visitor receives are
    # the ones worth asserting on, not only the ones this tab wrote.
    assert download.value["orig_name"] == written.name
    assert Path(download.value["path"]).read_text(encoding="utf-8") == result


def test_nothing_to_transliterate_offers_no_file(cron_dir: Path) -> None:
    """An empty result leaves the download slot empty and hidden.

    Args:
        cron_dir: The bound download directory, which stays empty.
    """
    result, stats, download = direct.transliterated_download("   ", "tr")

    assert result == ""
    assert "Please enter some text" in stats
    assert download.value is None
    assert download.visible is False


def test_an_uploaded_file_replaces_the_text_box(cron_dir: Path, tmp_path: Path) -> None:
    """The upload wins over whatever was typed.

    Args:
        cron_dir: The bound download directory.
        tmp_path: Directory holding the upload.
    """
    upload = tmp_path / "input.txt"
    upload.write_text("merhaba", encoding="utf-8")

    result, stats = direct.transliterate_request("ignored", "tr", str(upload))

    assert result == "meɾhaba"
    assert "Source: Uploaded file" in stats


def test_an_unreadable_upload_is_reported_not_raised(cron_dir: Path, tmp_path: Path) -> None:
    """A missing upload path becomes a message, not a traceback.

    Args:
        cron_dir: The bound download directory.
        tmp_path: Directory the absent file would have been in.
    """
    result, stats = direct.transliterate_request("merhaba", "tr", str(tmp_path / "absent.txt"))

    assert result == ""
    assert "Error reading file:" in stats


def test_an_empty_upload_falls_back_to_nothing(cron_dir: Path, tmp_path: Path) -> None:
    """An upload with no content produces the empty-input notice.

    Args:
        cron_dir: The bound download directory.
        tmp_path: Directory holding the upload.
    """
    upload = tmp_path / "empty.txt"
    upload.write_text("   \n", encoding="utf-8")

    result, stats = direct.transliterate_request("", "tr", str(upload))

    assert result == ""
    assert "Please enter some text" in stats


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_blank_input_asks_for_text(cron_dir: Path, text: str) -> None:
    """Whitespace-only input is treated as no input at all.

    Args:
        cron_dir: The bound download directory.
        text: A value carrying no characters to transliterate.
    """
    result, stats = direct.transliterate_request(text, "tr")

    assert result == ""
    assert "Please enter some text" in stats


def test_an_unsupported_language_is_reported_not_raised(cron_dir: Path) -> None:
    """A language with no rules becomes a message, not a traceback.

    Args:
        cron_dir: The bound download directory.
    """
    result, stats = direct.transliterate_request("merhaba", "xx")

    assert result == ""
    assert stats.startswith("**Error**")


def test_reading_an_absent_upload_returns_the_failure_text(tmp_path: Path) -> None:
    """The upload reader names the failure rather than propagating it."""
    message = direct._handle_file_upload(str(tmp_path / "absent.txt"))

    assert message.startswith("Error reading file:")


def test_reading_no_upload_returns_nothing() -> None:
    """No upload is not a failure and produces no message."""
    assert direct._handle_file_upload(None) == ""
    assert direct._handle_file_upload("") == ""


def test_the_tab_renders_into_a_blocks_app() -> None:
    """Registering the tab wires every widget without error."""
    with gr.Blocks() as blocks:
        direct.register()

    radios = [block for block in blocks.blocks.values() if isinstance(block, gr.Radio)]
    assert [radio.value for radio in radios] == [direct.ipa_languages()[0]]
