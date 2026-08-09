"""Tests for the Download Corpus tab.

The tab is driven through its module-level handlers against a corpus
source that answers from a table and a classifier backed by a table
model, so a download really is streamed, filtered, written and previewed
without touching the network or a 126 MB model file.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import pytest

from turkic_translit import _test_hooks
from turkic_translit.corpus import _test_hooks as corpus_hooks
from turkic_translit.lid import _test_hooks as lid_hooks
from turkic_translit.lid.locations import default_search_dirs
from turkic_translit.web import web_utils
from turkic_translit.web.tabs import corpus

TURKISH_LINES = ["merhaba dunya", "", "iyi gunler"]
ANSWERS = {
    "merhaba dunya": [("__label__tr", 0.99)],
    "iyi gunler": [("__label__tr", 0.98)],
    "salom dunyo": [("__label__uz", 0.97)],
    "kein ipa hier": [("__label__xx", 0.96)],
}


@pytest.fixture
def tab(tmp_path: Path) -> Iterator[Path]:
    """Bind the corpus source, the classifier and the download directory.

    The caches the tab keeps over the language lists and the classifier
    are cleared on the way in and out, because a list built from another
    test's bindings would otherwise be served here.

    Yields:
        The directory downloads are written into.
    """
    previous = (
        corpus_hooks.dataset_texts,
        corpus_hooks.languages,
        lid_hooks.probe,
        lid_hooks.model_loader,
        _test_hooks.environment,
    )
    target = tmp_path / "cronjob"
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer(
        {"tr": TURKISH_LINES, "uz": ["salom dunyo"], "xx": ["kein ipa hier"]}
    )
    corpus_hooks.languages = corpus_hooks.MappingLanguageCatalogue(
        {"oscar-corpus/OSCAR-2301": ["tr", "uz", "xx"]}, ["tr", "uz", "xx"]
    )
    lid_hooks.probe = lid_hooks.MappingFileProbe(
        {default_search_dirs()[0] / "lid.176.bin": 131266198}
    )
    lid_hooks.model_loader = lid_hooks.FixedModelLoader(lid_hooks.TableFastTextModel(ANSWERS))
    _test_hooks.environment = _test_hooks.MappingEnvironment({"TURKIC_CRON_DIR": str(target)})
    web_utils.installed_classifier.cache_clear()
    corpus._fasttext_langs.cache_clear()
    corpus._lang_choices.cache_clear()
    yield target
    (
        corpus_hooks.dataset_texts,
        corpus_hooks.languages,
        lid_hooks.probe,
        lid_hooks.model_loader,
        _test_hooks.environment,
    ) = previous
    web_utils.installed_classifier.cache_clear()
    corpus._fasttext_langs.cache_clear()
    corpus._lang_choices.cache_clear()


def test_a_language_with_ipa_rules_enables_the_checkbox(tab: Path) -> None:
    """Turkish has IPA rules, so the transliteration option is offered.

    Args:
        tab: The bound download directory.
    """
    checkbox = corpus.transliterate_checkbox("tr")

    assert checkbox.interactive is True
    assert checkbox.info == corpus.TRANSLITERATE_AVAILABLE


def test_a_language_without_ipa_rules_disables_the_checkbox(tab: Path) -> None:
    """A code with no rules clears the option and says why.

    Args:
        tab: The bound download directory.
    """
    checkbox = corpus.transliterate_checkbox("xx")

    assert checkbox.interactive is False
    assert checkbox.value is False
    assert checkbox.info == corpus.TRANSLITERATE_UNAVAILABLE.format(lang="xx")


def test_no_language_at_all_disables_the_checkbox(tab: Path) -> None:
    """A source offering nothing leaves the option unavailable.

    Args:
        tab: The bound download directory.
    """
    checkbox = corpus.transliterate_checkbox(None)

    assert checkbox.interactive is False
    assert checkbox.info == corpus.TRANSLITERATE_UNAVAILABLE.format(lang=None)


def test_changing_source_rebuilds_both_controls(tab: Path) -> None:
    """A new source repopulates the dropdown and resets the checkbox.

    Args:
        tab: The bound download directory.
    """
    dropdown, checkbox = corpus.language_controls("oscar-2301")

    offered = [value for _label, value in dropdown.choices]
    assert dropdown.value in offered
    assert checkbox.label == corpus.TRANSLITERATE_LABEL


def test_an_unreachable_catalogue_offers_nothing_and_says_so(tab: Path) -> None:
    """A source that cannot be read yields no languages and a reason.

    Offering the locally installed IPA languages instead, as this once
    did, lists codes the source may not carry at all.

    Args:
        tab: The bound download directory.
    """
    previous = corpus_hooks.languages
    corpus_hooks.languages = corpus_hooks.MappingLanguageCatalogue({}, [])
    corpus._lang_choices.cache_clear()
    try:
        offered, label = corpus.offered_languages("oscar-2301")
    finally:
        corpus_hooks.languages = previous
        corpus._lang_choices.cache_clear()

    assert offered == []
    assert label == f"{corpus.LANGUAGE_LABEL} — oscar-2301 could not be reached"


def test_an_unreachable_catalogue_leaves_the_dropdown_empty(tab: Path) -> None:
    """The controls report the failure rather than offering substitutes.

    Args:
        tab: The bound download directory.
    """
    previous = corpus_hooks.languages
    corpus_hooks.languages = corpus_hooks.MappingLanguageCatalogue({}, [])
    corpus._lang_choices.cache_clear()
    try:
        dropdown, checkbox = corpus.language_controls("oscar-2301")
    finally:
        corpus_hooks.languages = previous
        corpus._lang_choices.cache_clear()

    assert list(dropdown.choices) == []
    assert dropdown.value is None
    assert checkbox.interactive is False


def test_a_catalogue_offering_nothing_offers_nothing(tab: Path) -> None:
    """A source that lists no languages is reported as carrying none.

    Args:
        tab: The bound download directory.
    """
    previous = corpus_hooks.languages
    corpus_hooks.languages = corpus_hooks.MappingLanguageCatalogue(
        {"oscar-corpus/OSCAR-2301": []}, []
    )
    corpus._lang_choices.cache_clear()
    try:
        offered, label = corpus.offered_languages("oscar-2301")
    finally:
        corpus_hooks.languages = previous
        corpus._lang_choices.cache_clear()

    assert offered == []
    assert label == corpus.LANGUAGE_LABEL


def test_downloading_writes_the_corpus_and_previews_it(tab: Path) -> None:
    """A plain download writes the lines and previews the first.

    Args:
        tab: The bound download directory.
    """
    info, path, translit, preview, label = corpus.download_request(
        "oscar-2301", "tr", None, False, 0.0, False, progress=gr.Progress()
    )

    written = sorted(tab.iterdir())[0]
    assert path == str(written)
    assert written.read_text(encoding="utf-8") == "merhaba dunya\niyi gunler\n"
    assert translit is None
    assert preview == "merhaba dunya ..."
    assert label == corpus._ORIGINAL_PREVIEW
    assert "Lines written:** 2" in info


def test_downloading_honours_the_sentence_cap(tab: Path) -> None:
    """``max_lines`` stops the stream once that many lines are kept.

    Args:
        tab: The bound download directory.
    """
    _info, path, _translit, preview, _label = corpus.download_request(
        "oscar-2301", "tr", 1, False, 0.0, False, progress=gr.Progress()
    )

    written = sorted(tab.iterdir())[0]
    assert path == str(written)
    assert written.read_text(encoding="utf-8") == "merhaba dunya\n"
    assert preview == "merhaba dunya"


def test_filtering_reports_what_the_classifier_removed(tab: Path) -> None:
    """Asking for Uzbek out of a Turkish corpus keeps nothing.

    Args:
        tab: The bound download directory.
    """
    info, path, _translit, preview, label = corpus.download_request(
        "oscar-2301", "uz", None, True, 0.95, False, progress=gr.Progress()
    )

    written = sorted(tab.iterdir())[0]
    assert path == str(written)
    assert written.read_text(encoding="utf-8") == "salom dunyo\n"
    assert "Lines removed by LangID filter:** 0" in info
    assert preview == "salom dunyo"
    assert label == corpus._ORIGINAL_PREVIEW


def test_transliterating_writes_a_second_file(tab: Path) -> None:
    """The IPA copy is written beside the corpus and previewed instead.

    Args:
        tab: The bound download directory.
    """
    _info, path, translit, preview, label = corpus.download_request(
        "oscar-2301", "tr", None, False, 0.0, True, progress=gr.Progress()
    )

    corpus_file, ipa_file = sorted(tab.iterdir())
    assert path == str(corpus_file)
    assert translit == str(ipa_file)
    assert ipa_file.name == f"{corpus_file.stem}_ipa.txt"
    assert ipa_file.read_text(encoding="utf-8").splitlines() == [
        "meɾhaba dunja",
        "iji ɡunleɾ",
    ]
    assert preview == "meɾhaba dunja ..."
    assert label == corpus._IPA_PREVIEW


def test_transliterating_an_unsupported_language_warns_instead(tab: Path) -> None:
    """A language with no IPA rules is downloaded and flagged.

    Args:
        tab: The bound download directory.
    """
    info, path, translit, _preview, label = corpus.download_request(
        "oscar-2301", "xx", None, False, 0.0, True, progress=gr.Progress()
    )

    written = sorted(tab.iterdir())[0]
    assert path == str(written)
    assert written.read_text(encoding="utf-8") == "kein ipa hier\n"
    assert translit is None
    assert "No IPA rules for language 'xx'" in info
    assert label == corpus._ORIGINAL_PREVIEW


def test_an_unknown_source_is_reported_as_an_error(tab: Path) -> None:
    """An unregistered source produces the error payload, not a file.

    Args:
        tab: The bound download directory.
    """
    info, path, translit, preview, label = corpus.download_request(
        "oscar-9999", "tr", None, False, 0.0, False, progress=gr.Progress()
    )

    assert path == ""
    assert translit is None
    assert preview == ""
    assert label == "**Preview**"
    assert "invalid_source" in info


def test_a_stream_failure_is_reported_as_an_error(tab: Path) -> None:
    """A source that cannot be read renders the failure, not a traceback.

    Args:
        tab: The bound download directory.
    """
    original = corpus_hooks.dataset_texts
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer({})
    try:
        info, path, translit, preview, label = corpus.download_request(
            "oscar-2301", "tr", None, False, 0.0, False, progress=gr.Progress()
        )
    finally:
        corpus_hooks.dataset_texts = original

    assert path is None
    assert translit is None
    assert preview == ""
    assert label == "**Preview**"
    assert "download_failed" in info


def test_a_download_with_no_reporter_makes_its_own(tab: Path) -> None:
    """Omitting the progress reporter still completes the download.

    Args:
        tab: The bound download directory.
    """
    _info, path, _translit, preview, _label = corpus.download_request(
        "oscar-2301", "tr", None, False, 0.0, False
    )

    written = sorted(tab.iterdir())[0]
    assert path == str(written)
    assert written.read_text(encoding="utf-8") == "merhaba dunya\niyi gunler\n"
    assert preview == "merhaba dunya ..."


def test_the_tab_renders_into_a_blocks_app(tab: Path) -> None:
    """Registering the tab wires every widget without error.

    Args:
        tab: The bound download directory.
    """
    with gr.Blocks() as blocks:
        corpus.register()

    dropdowns = [block for block in blocks.blocks.values() if isinstance(block, gr.Dropdown)]
    assert [dropdown.value for dropdown in dropdowns] == ["oscar-2301", "tr"]
