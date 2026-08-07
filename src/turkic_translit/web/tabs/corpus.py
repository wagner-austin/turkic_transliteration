"""The Download Corpus tab.

The download handler catches only what the operation can produce — a
corpus or classifier error from the library, or an OS error from writing
the file — so a defect in the UI itself still surfaces as a traceback
rather than being rendered to the user as a polite failure message.

Per-line IPA transliteration is likewise all-or-nothing. The previous
version fell back to the untransliterated line whenever a single line
failed, which produced a file silently mixing IPA and source text.
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


@lru_cache(maxsize=1)
def _fasttext_langs() -> set[str]:
    """List the languages the corpus filter can be asked for.

    Returns:
        Every label the language-identification model can emit.
    """
    from turkic_translit.web.web_utils import LANGUAGE_MODEL_ID, _langid_singleton

    return set(_langid_singleton(LANGUAGE_MODEL_ID).known_labels())


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
        Language codes the classifier also knows. When the source's host
        cannot be reached, the locally available IPA languages are
        offered instead, so the tab still works offline.
    """
    from turkic_translit.corpus.catalogue import available_languages
    from turkic_translit.corpus.sources import get_source_spec

    try:
        offered = list(available_languages(get_source_spec(src)))
    except CorpusError as exc:
        logger.warning("could not list languages for %s: %s", src, exc)
        offered = []

    if not offered:
        logger.warning("falling back to the local language list for %s", src)
        offered = sorted(_ipa_supported_langs())

    known = _fasttext_langs()
    return [code for code in offered if code in known]


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


def register() -> None:
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

                initial_langs = _lang_choices("oscar-2301")
                lang_dd = gr.Dropdown(
                    choices=labelise(initial_langs),
                    value=initial_langs[0] if initial_langs else None,
                    label="Language",
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
                _initial_lang = initial_langs[0] if initial_langs else None
                _initial_has_ipa = (
                    _initial_lang is not None and _initial_lang in _ipa_supported_langs()
                )
                transliterate_cb = gr.Checkbox(
                    label="Also create IPA-transliterated version",
                    value=False,
                    interactive=_initial_has_ipa,
                    info=(
                        "Get both original + IPA-transliterated corpus files"
                        if _initial_has_ipa
                        else f"No IPA rules for '{_initial_lang}' — transliteration unavailable"
                    ),
                    elem_id="transliterate-checkbox",
                )

                def _update_transliterate_cb(selected_lang: str | None) -> gr.Checkbox:
                    if selected_lang and selected_lang in _ipa_supported_langs():
                        return gr.Checkbox(
                            label="Also create IPA-transliterated version",
                            interactive=True,
                            info="Get both original + IPA-transliterated corpus files",
                        )
                    return gr.Checkbox(
                        label="Also create IPA-transliterated version",
                        value=False,
                        interactive=False,
                        info=f"No IPA rules for '{selected_lang}' — transliteration unavailable",
                    )

                lang_dd.change(
                    _update_transliterate_cb,
                    inputs=[lang_dd],
                    outputs=[transliterate_cb],
                )

                def _update_langs(
                    selected_src: str,
                ) -> tuple[gr.Dropdown, gr.Checkbox]:
                    langs = _lang_choices(selected_src)
                    new_default = langs[0] if langs else None
                    lang_update = gr.Dropdown(
                        choices=labelise(langs),
                        value=new_default,
                        label="Language",
                    )
                    cb_update = _update_transliterate_cb(new_default)
                    return lang_update, cb_update

                source_dd.change(
                    _update_langs,
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
                    show_copy_button=True,
                    elem_id="corpus-preview",
                    show_label=False,
                )

        # Toggle the transliterated-file output visibility with the checkbox.
        transliterate_cb.change(
            lambda x: gr.update(visible=x),
            inputs=[transliterate_cb],
            outputs=[file_out_translit],
        )

        def _do_download(
            source: str,
            lang: str,
            max_lines: int | None,
            filter_flag: bool,
            conf_thr: float,
            transliterate_flag: bool,
            progress: gr.Progress | None = None,
        ) -> tuple[str, str | None, str | None, str, str]:
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

        download_btn.click(
            _do_download,
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

        gr.Examples(
            [["oscar-2301", "kk", 100, True, 0.95, False]],
            inputs=[
                source_dd,
                lang_dd,
                max_lines_num,
                filter_cb,
                conf_slider,
                transliterate_cb,
            ],
            outputs=[info_md, file_out, file_out_translit, preview_text, preview_label],
            fn=_do_download,
            label="Try this example",
        )
