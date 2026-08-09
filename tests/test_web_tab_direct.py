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

from turkic_translit import _test_hooks
from turkic_translit.web.tabs import direct

LONG_TURKISH = "Merhaba dünya, bu metin indirme esigini asacak kadar uzun bir metindir."


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


def test_the_language_caption_names_the_first_three() -> None:
    """A short list is spelled out in full with no overflow count."""
    assert direct.language_info(["kk", "ky", "tr"]).count("=") == 3
    assert "more" not in direct.language_info(["kk", "ky", "tr"])


def test_the_language_caption_counts_the_rest() -> None:
    """A longer list names three and reports how many remain."""
    caption = direct.language_info(["kk", "ky", "tr", "uz", "ug"])

    assert caption.count("=") == 3
    assert caption.endswith("+2 more")


def test_typed_text_is_transliterated_to_ipa(cron_dir: Path) -> None:
    """Short input is transliterated and offered no download.

    Args:
        cron_dir: The bound download directory, which stays empty.
    """
    result, stats, download = direct.transliterate_request("merhaba", "tr")

    assert result == "meɾhaba"
    assert "Bytes" in stats
    assert download is None


def test_long_output_is_written_for_download(cron_dir: Path) -> None:
    """Output past the threshold is saved and the path returned.

    Args:
        cron_dir: The bound download directory, which receives the file.
    """
    result, stats, download = direct.transliterate_request(LONG_TURKISH, "tr")

    assert len(result) > direct.MIN_CHARS_FOR_DOWNLOAD
    written = sorted(cron_dir.iterdir())[0]
    assert download == str(written)
    assert written.read_text(encoding="utf-8") == result
    assert written.name in stats


def test_an_uploaded_file_replaces_the_text_box(cron_dir: Path, tmp_path: Path) -> None:
    """The upload wins over whatever was typed.

    Args:
        cron_dir: The bound download directory.
        tmp_path: Directory holding the upload.
    """
    upload = tmp_path / "input.txt"
    upload.write_text("merhaba", encoding="utf-8")

    result, stats, _download = direct.transliterate_request("ignored", "tr", str(upload))

    assert result == "meɾhaba"
    assert "Source: Uploaded file" in stats


def test_an_unreadable_upload_is_reported_not_raised(cron_dir: Path, tmp_path: Path) -> None:
    """A missing upload path becomes a message, not a traceback.

    Args:
        cron_dir: The bound download directory.
        tmp_path: Directory the absent file would have been in.
    """
    result, stats, download = direct.transliterate_request(
        "merhaba", "tr", str(tmp_path / "absent.txt")
    )

    assert result == ""
    assert "Error reading file:" in stats
    assert download is None


def test_an_empty_upload_falls_back_to_nothing(cron_dir: Path, tmp_path: Path) -> None:
    """An upload with no content produces the empty-input notice.

    Args:
        cron_dir: The bound download directory.
        tmp_path: Directory holding the upload.
    """
    upload = tmp_path / "empty.txt"
    upload.write_text("   \n", encoding="utf-8")

    result, stats, download = direct.transliterate_request("", "tr", str(upload))

    assert result == ""
    assert "Please enter some text" in stats
    assert download is None


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_blank_input_asks_for_text(cron_dir: Path, text: str) -> None:
    """Whitespace-only input is treated as no input at all.

    Args:
        cron_dir: The bound download directory.
        text: A value carrying no characters to transliterate.
    """
    result, stats, download = direct.transliterate_request(text, "tr")

    assert result == ""
    assert "Please enter some text" in stats
    assert download is None


def test_an_unsupported_language_is_reported_not_raised(cron_dir: Path) -> None:
    """A language with no rules becomes a message, not a traceback.

    Args:
        cron_dir: The bound download directory.
    """
    result, stats, download = direct.transliterate_request("merhaba", "xx")

    assert result == ""
    assert stats.startswith("**Error**")
    assert download is None


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
