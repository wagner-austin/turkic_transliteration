"""The Download Corpus tab.

The download handler catches only what the operation can produce — a
corpus or classifier error from the library, or an OS error from writing
the file — so a defect in the UI itself still surfaces as a traceback
rather than being rendered to the user as a polite failure message.

Per-line IPA transliteration is likewise all-or-nothing. The previous
version fell back to the untransliterated line whenever a single line
failed, which produced a file silently mixing IPA and source text.

The handlers are module-level functions rather than closures inside
:func:`register`. A closure over the widgets can only be reached by
building the whole interface, which left every decision the tab makes —
which languages a source offers, when the IPA checkbox is available,
what a download produces — exercised by nothing.
"""

from __future__ import annotations

import logging
from functools import cache, lru_cache
from pathlib import Path

import gradio as gr

from turkic_translit.corpus.errors import CorpusError
from turkic_translit.error_service import error_markdown, error_response
from turkic_translit.lid.errors import LidError
from turkic_translit.web.web_utils import (
    direct_transliterate,
    download_corpus_to_file,
    labelise,
)

logger = logging.getLogger(__name__)

_ORIGINAL_PREVIEW = "**Preview** (Original corpus - first line)"
_IPA_PREVIEW = "**Preview** (IPA-transliterated corpus - first line)"

__all__ = [
    "LANGUAGE_LABEL",
    "TRANSLITERATE_AVAILABLE",
    "TRANSLITERATE_LABEL",
    "TRANSLITERATE_UNAVAILABLE",
    "download_request",
    "language_controls",
    "offered_languages",
    "register",
    "transliterate_checkbox",
]


@lru_cache(maxsize=1)
def _fasttext_langs() -> set[str]:
    """List the languages the corpus filter can be asked for.

    Returns:
        Every label the language-identification model can emit.
    """
    from turkic_translit.web.web_utils import LANGUAGE_MODEL_ID, installed_classifier

    return set(installed_classifier(LANGUAGE_MODEL_ID).known_labels())


@lru_cache(maxsize=1)
def _ipa_supported_langs() -> set[str]:
    """List the language codes that have an ``<lang>_ipa.rules`` file.

    Returns:
        Codes this project can transliterate to IPA.
    """
    from turkic_translit.core import get_supported_languages

    return {code for code, fmts in get_supported_languages().items() if "ipa" in fmts}


@cache
def _lang_choices(src: str) -> list[str]:
    """List the languages one source offers, for the dropdown.

    Args:
        src: Registry key of the selected source.

    Returns:
        The codes both the source offers and the classifier knows, so
        that every entry is one the tab can actually download and filter.

    Raises:
        CorpusError: If the source's host cannot be read. The previous
            version substituted the locally installed IPA languages
            here, which is a different set entirely: it offered codes
            the source may not carry, and a download of one of those
            fails with nothing in the list having hinted that it would.
    """
    from turkic_translit.corpus.catalogue import available_languages
    from turkic_translit.corpus.sources import get_source_spec

    offered = available_languages(get_source_spec(src))
    known = _fasttext_langs()
    return [code for code in offered if code in known]


def offered_languages(src: str) -> tuple[list[str], str]:
    """List a source's languages, or report that it cannot be reached.

    Args:
        src: Registry key of the selected source.

    Returns:
        The codes to offer and the label to put on the dropdown. An
        unreachable source yields no codes and a label saying so, which
        is the honest answer: the tab cannot know what it carries.
    """
    try:
        return _lang_choices(src), LANGUAGE_LABEL
    except CorpusError as exc:
        logger.warning("could not list languages for %s: %s", src, exc)
        return [], f"{LANGUAGE_LABEL} — {src} could not be reached"


def _preview_of(path: Path) -> str:
    """Return the file's first line, marked when more lines follow.

    Args:
        path: File to preview.

    Returns:
        The first line, with an ellipsis when the file has more.
    """
    with path.open(encoding="utf-8") as handle:
        first = handle.readline().rstrip()
        return f"{first} ..." if handle.readline() else first


def _write_ipa_copy(path: Path, lang: str) -> Path:
    """Write an IPA transliteration of ``path`` beside it.

    Args:
        path: The downloaded corpus.
        lang: Language code whose IPA rules to apply.

    Returns:
        Path of the transliterated copy.

    Raises:
        ValueError: If the language has no IPA rules. The caller checks
            first; a failure here means the rules went missing between
            the check and the write, which is worth surfacing.
    """
    target = path.with_name(f"{path.stem}_ipa{path.suffix}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    transliterated = [direct_transliterate(line, lang, False, "ipa")[0] for line in lines if line]
    target.write_text("\n".join(transliterated) + "\n", encoding="utf-8")
    return target


LANGUAGE_LABEL = "Language"
TRANSLITERATE_LABEL = "Also create IPA-transliterated version"
TRANSLITERATE_AVAILABLE = "Get both original + IPA-transliterated corpus files"
TRANSLITERATE_UNAVAILABLE = "No IPA rules for '{lang}' — transliteration unavailable"


def transliterate_checkbox(selected_lang: str | None) -> gr.Checkbox:
    """Build the IPA checkbox in the state the selected language allows.

    The checkbox is interactive only when the language has IPA rules;
    otherwise it is cleared and disabled, and its caption names the
    language that lacks them.

    Args:
        selected_lang: Language code chosen in the dropdown, or ``None``
            when the source offers none.

    Returns:
        The checkbox, ready to be returned as a Gradio update.
    """
    if selected_lang is not None and selected_lang in _ipa_supported_langs():
        return gr.Checkbox(
            label=TRANSLITERATE_LABEL,
            interactive=True,
            info=TRANSLITERATE_AVAILABLE,
        )
    return gr.Checkbox(
        label=TRANSLITERATE_LABEL,
        value=False,
        interactive=False,
        info=TRANSLITERATE_UNAVAILABLE.format(lang=selected_lang),
    )


def language_controls(selected_src: str) -> tuple[gr.Dropdown, gr.Checkbox]:
    """Rebuild the language dropdown and IPA checkbox for a new source.

    Args:
        selected_src: Registry key of the newly selected source.

    Returns:
        The dropdown carrying that source's languages, and the checkbox
        in the state the new default language allows. A source that
        cannot be reached yields an empty dropdown whose label says so.
    """
    langs, label = offered_languages(selected_src)
    default = langs[0] if langs else None
    dropdown = gr.Dropdown(choices=labelise(langs), value=default, label=label)
    return dropdown, transliterate_checkbox(default)


def download_request(
    source: str,
    lang: str,
    max_lines: int | None,
    filter_flag: bool,
    conf_thr: float,
    transliterate_flag: bool,
    progress: gr.Progress | None = None,
) -> tuple[str, str | None, str | None, str, str]:
    """Stream a corpus and report what came back.

    Args:
        source: Registry key of the corpus source.
        lang: Language code to stream.
        max_lines: Cap on sentences kept, or ``None`` for all.
        filter_flag: Whether to keep only lines the classifier assigns to
            ``lang``.
        conf_thr: Minimum probability a line must reach to be kept.
        transliterate_flag: Whether to write an IPA copy beside the
            corpus.
        progress: Gradio's progress reporter, or ``None`` to make one.

    Returns:
        A Markdown status line, the corpus path, the IPA copy's path when
        one was written, the preview text, and the preview's heading.
    """
    reporter = gr.Progress(track_tqdm=True) if progress is None else progress
    try:
        path, info = download_corpus_to_file(
            source,
            lang,
            int(max_lines) if max_lines else None,
            filter_flag,
            conf_thr,
            progress=reporter,
        )
    except (CorpusError, LidError, OSError) as exc:
        logger.exception("corpus download failed")
        payload = error_response(str(exc), status=500, code="download_failed")
        return error_markdown(payload), None, None, "", "**Preview**"

    if not path:
        return info, path, None, "", "**Preview**"

    if not transliterate_flag:
        return info, path, None, _preview_of(Path(path)), _ORIGINAL_PREVIEW

    if lang not in _ipa_supported_langs():
        info += f"\n\n**Warning:** No IPA rules for language '{lang}'."
        return info, path, None, _preview_of(Path(path)), _ORIGINAL_PREVIEW

    translit_path = _write_ipa_copy(Path(path), lang)
    return info, path, str(translit_path), _preview_of(translit_path), _IPA_PREVIEW


def register() -> None:
    """Render the Download Corpus tab content."""
    with gr.Column():
        gr.Markdown(
            """
            **Corpus Downloader:** Stream sentences from public corpora (OSCAR or Wikipedia) directly in the browser.
            Select a source and language, optionally cap the number of sentences, and decide whether to filter by FastText language ID.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                from turkic_translit.corpus.sources import known_source_ids

                source_dd = gr.Dropdown(
                    choices=sorted(known_source_ids()),
                    label="Corpus Source",
                    value="oscar-2301",
                )

                initial_langs, language_label = offered_languages("oscar-2301")
                initial_lang = initial_langs[0] if initial_langs else None
                lang_dd = gr.Dropdown(
                    choices=labelise(initial_langs),
                    value=initial_lang,
                    label=language_label,
                )

                max_lines_num = gr.Number(
                    label="Max Sentences (empty = all)", value=10, precision=0
                )
                filter_cb = gr.Checkbox(
                    label="Filter by FastText LangID",
                    value=True,
                    info=(
                        "Keep only sentences whose FastText language-ID matches the code above (uses lid.176 model)."
                    ),
                )
                conf_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.95,
                    label="Min Lang-ID Confidence Threshold",
                )

                gr.Markdown("---")
                gr.Markdown("### IPA Transliteration")
                transliterate_cb = transliterate_checkbox(initial_lang)

                lang_dd.change(
                    transliterate_checkbox,
                    inputs=[lang_dd],
                    outputs=[transliterate_cb],
                )
                source_dd.change(
                    language_controls,
                    inputs=[source_dd],
                    outputs=[lang_dd, transliterate_cb],
                )

                download_btn = gr.Button("Download", variant="primary")
            with gr.Column(scale=1):
                info_md = gr.Markdown()
                file_out = gr.File(label="Original Corpus")
                file_out_translit = gr.File(
                    label="Transliterated Corpus",
                    visible=False,
                )

                preview_label = gr.Markdown("**Preview**")
                preview_text = gr.Textbox(
                    label="",
                    lines=1,
                    interactive=False,
                    buttons=["copy"],
                    elem_id="corpus-preview",
                    show_label=False,
                )

        # Toggle the transliterated-file output visibility with the checkbox.
        transliterate_cb.change(
            lambda x: gr.update(visible=x),
            inputs=[transliterate_cb],
            outputs=[file_out_translit],
        )

        download_btn.click(
            download_request,
            [
                source_dd,
                lang_dd,
                max_lines_num,
                filter_cb,
                conf_slider,
                transliterate_cb,
            ],
            outputs=[info_md, file_out, file_out_translit, preview_text, preview_label],
        )

        # The example's language is taken from the dropdown rather than
        # written in. A pinned code silently stops being offered when
        # the source's language list or the classifier's labels change,
        # and Gradio then loads the tab with a value the dropdown does
        # not contain.
        gr.Examples(
            [["oscar-2301", initial_lang, 100, True, 0.95, False]],
            inputs=[
                source_dd,
                lang_dd,
                max_lines_num,
                filter_cb,
                conf_slider,
                transliterate_cb,
            ],
            outputs=[info_md, file_out, file_out_translit, preview_text, preview_label],
            fn=download_request,
            label="Try this example",
        )
