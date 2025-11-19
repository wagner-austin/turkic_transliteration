from __future__ import annotations

from functools import cache, lru_cache
from pathlib import Path
from typing import Any, cast

import gradio as gr

from turkic_translit.web.web_utils import (
    download_corpus_to_file,
    labelise,
)


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
                from turkic_translit.cli.download_corpus import (
                    _REG,  # dynamic registry
                )

                source_dd = gr.Dropdown(
                    choices=sorted(k for k in _REG if _REG[k]["driver"] != "leipzig"),
                    label="Corpus Source",
                    value="oscar-2301",
                )

                @lru_cache(maxsize=1)
                def _fasttext_langs() -> set[str]:
                    from turkic_translit.langid import FastTextLangID

                    mdl = FastTextLangID()
                    return {
                        lab.replace("__label__", "") for lab in mdl.model.get_labels()
                    }

                @cache
                def _lang_choices(src: str) -> list[str]:
                    import logging

                    from turkic_translit.cli import download_corpus as _dl

                    logger = logging.getLogger(__name__)
                    cfg = _dl._REG[src]
                    lst: list[str] = []
                    try:
                        if cfg["driver"] == "oscar":
                            import os

                            from datasets import get_dataset_config_names

                            token = os.getenv("HF_TOKEN")
                            lst = sorted(
                                get_dataset_config_names(
                                    cfg["hf_name"], token=token, trust_remote_code=True
                                )
                            )
                        elif cfg["driver"] == "wikipedia":
                            try:
                                from turkic_translit.cli.download_corpus import (
                                    _wikipedia_lang_codes_from_sitematrix as _site,
                                )

                                lst = _site()
                            except Exception as e:
                                logger.error(
                                    f"Failed to fetch Wikipedia languages: {e}"
                                )
                                lst = []
                    except Exception as e:
                        logger.error(f"Failed to fetch languages for {src}: {e}")
                        lst = []

                    if not lst:
                        logger.warning(f"Using fallback language list for {src}")
                        lst = [
                            c
                            for c in ["en", "tr", "az", "uz", "kk", "ru"]
                            if c in _fasttext_langs()
                        ]

                    ft = _fasttext_langs()
                    return [code for code in lst if code in ft]

                initial_langs = _lang_choices("oscar-2301")
                lang_dd = gr.Dropdown(
                    choices=labelise(initial_langs),
                    value=initial_langs[0] if initial_langs else None,
                    label="Language",
                )

                def _update_langs(selected_src: str) -> Any:
                    langs = _lang_choices(selected_src)
                    return gr.update(
                        choices=labelise(langs), value=langs[0] if langs else None
                    )

                source_dd.change(_update_langs, inputs=[source_dd], outputs=[lang_dd])
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
                gr.Markdown("### Transliteration Options")
                transliterate_cb = gr.Checkbox(
                    label="Also create transliterated version",
                    value=False,
                    info="Get both original + transliterated corpus files",
                    elem_id="transliterate-checkbox",
                )
                translit_format = gr.Radio(
                    ["Latin", "IPA"],
                    label="Transliteration Format",
                    value="Latin",
                    visible=False,
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

        # Combined callback for transliterate checkbox to update both format radio and file output visibility
        transliterate_cb.change(
            lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[transliterate_cb],
            outputs=[translit_format, file_out_translit],
        )

        def _do_download(
            source: str,
            lang: str,
            max_lines: int | None,
            filter_flag: bool,
            conf_thr: float,
            transliterate_flag: bool,
            translit_fmt: str,
            progress: gr.Progress | None = None,
        ) -> tuple[str, str | None, str | None, str, str]:
            if progress is None:
                progress = gr.Progress(track_tqdm=True)
            try:
                path, info = download_corpus_to_file(
                    source,
                    lang,
                    int(max_lines) if max_lines else None,
                    filter_flag,
                    conf_thr,
                    progress=progress,
                )
                preview = ""
                preview_label_txt = "**Preview** (Original corpus - first line)"
                if path:
                    try:
                        with open(path, encoding="utf-8") as f:
                            first_line = f.readline().rstrip()
                            preview = first_line
                            if f.readline():
                                preview += " ..."
                    except Exception:
                        preview = "Could not generate preview"

                translit_path = None
                if transliterate_flag and path:
                    from turkic_translit.core import get_supported_languages

                    supported = get_supported_languages()
                    info_msg = info
                    if lang not in supported:
                        info_msg += (
                            f"\n\n**Warning:** No rules found for language '{lang}'."
                        )
                    elif translit_fmt.lower() not in supported[lang]:
                        info_msg += f"\n\n**Warning:** {translit_fmt} transliteration not supported for '{lang}'."
                    else:
                        try:
                            with open(path, encoding="utf-8") as f:
                                lines = f.readlines()
                            orig_path = Path(path)
                            translit_filename = (
                                orig_path.stem
                                + f"_{translit_fmt.lower()}"
                                + orig_path.suffix
                            )
                            translit_path = str(orig_path.parent / translit_filename)
                            from turkic_translit.web.web_utils import (
                                direct_transliterate as _dt,
                            )

                            translit_lines = []
                            for line in lines:
                                line = line.strip()
                                if line:
                                    try:
                                        result, _ = _dt(
                                            line, lang, False, translit_fmt.lower()
                                        )
                                        translit_lines.append(result + "\n")
                                    except Exception:
                                        translit_lines.append(line + "\n")
                            with open(translit_path, "w", encoding="utf-8") as f:
                                f.writelines(translit_lines)
                            if translit_lines:
                                preview = translit_lines[0].rstrip()
                                if len(translit_lines) > 1:
                                    preview += " ..."
                                preview_label_txt = f"**Preview** ({translit_fmt} transliterated corpus - first line)"
                            info = info_msg
                        except Exception as e:  # pragma: no cover
                            info = (
                                info_msg + f"\n\n**Transliteration failed:** {str(e)}"
                            )
                            translit_path = None

                return info, path, translit_path, preview, preview_label_txt
            except Exception as exc:  # pragma: no cover
                from turkic_translit.error_service import error_markdown, error_response

                payload = error_response(str(exc), status=500, code="download_failed")
                return error_markdown(payload), None, None, "", "**Preview**"

        download_btn.click(
            _do_download,
            [
                source_dd,
                lang_dd,
                max_lines_num,
                filter_cb,
                conf_slider,
                transliterate_cb,
                translit_format,
            ],
            outputs=[info_md, file_out, file_out_translit, preview_text, preview_label],
        )

        gr.Examples(
            cast(
                list[list[object]],
                [["oscar-2301", "kk", 100, True, 0.95, False, "Latin"]],
            ),
            inputs=[
                source_dd,
                lang_dd,
                max_lines_num,
                filter_cb,
                conf_slider,
                transliterate_cb,
                translit_format,
            ],
            outputs=[info_md, file_out, file_out_translit, preview_text, preview_label],
            fn=_do_download,
            label="Try this example",
        )
