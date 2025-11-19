from __future__ import annotations

from typing import Any, cast

import gradio as gr

from turkic_translit.lang_utils import pretty_lang
from turkic_translit.web.web_utils import _CRON_DIR, direct_transliterate


def _handle_file_upload(file_path: str | None) -> str:
    if not file_path:
        return ""
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:  # pragma: no cover
        return f"Error reading file: {str(e)}"


def register() -> None:
    """Render the Direct Transliteration tab content."""
    with gr.Column():
        gr.Markdown(
            """
            **Direct Transliteration:** Convert text directly to Latin or IPA using language-specific rules.
            Select the language, choose your output format, and click Transliterate.
            """
        )

        # Text input and file upload
        with gr.Row():
            with gr.Column(scale=2):
                translit_textbox = gr.Textbox(
                    label="Input Text",
                    lines=4,
                    elem_id="translit-input-text",
                    placeholder="Enter Turkic language text in Cyrillic script...",
                )
            with gr.Column(scale=1):
                gr.Markdown("**Or upload a text file:**")
                translit_upload_file = gr.File(
                    label="Upload .txt file",
                    file_types=[".txt"],
                    type="filepath",
                    elem_id="translit-file-upload",
                )
                gr.Markdown(
                    "*File content replaces text input*",
                    elem_classes=["file-upload-note"],
                )

        with gr.Row():
            with gr.Column(scale=3):
                # Dynamically get supported languages
                from turkic_translit.core import get_supported_languages

                supported_langs = get_supported_languages()
                lang_choices = sorted(supported_langs.keys())

                lang_info = ", ".join(
                    [f"{code} = {pretty_lang(code)}" for code in lang_choices[:3]]
                )
                if len(lang_choices) > 3:
                    lang_info += f", +{len(lang_choices) - 3} more"

                lang = gr.Radio(
                    lang_choices,
                    label="Language",
                    value=lang_choices[0] if lang_choices else "kk",
                    info=lang_info,
                )
                output_format = gr.Radio(
                    ["Latin", "IPA"],
                    label="Output Format",
                    value="Latin",
                    info="IPA: International Phonetic Alphabet",
                )

                include_arabic = gr.Checkbox(
                    False,
                    label="Also transliterate Arabic script",
                    info="Only applies when using Latin output format",
                )

            with gr.Column(scale=7):
                output = gr.Textbox(label="Output", lines=4, interactive=False)
                stats = gr.Markdown()
                download_file = gr.File(
                    label="Download Result", elem_id="download-output"
                )

        def do_direct(
            text: str,
            lang: str,
            fmt: str,
            include_arabic: bool,
            file_path: str | None = None,
        ) -> tuple[str, str, str | None]:
            if file_path:
                text = _handle_file_upload(file_path)
                if text.startswith("Error reading file:"):
                    return "", f"**{text}**", None
            if not text.strip():
                return (
                    "",
                    "*Please enter some text to transliterate or upload a file*",
                    None,
                )
            try:
                result, stats_md = direct_transliterate(
                    text, lang, include_arabic, fmt.lower()
                )
                if file_path:
                    stats_md += "\n*Source: Uploaded file*"

                download_path = None
                if result and len(result) > 50:
                    import time as _t

                    ts = _t.strftime("%Y%m%d_%H%M%S")
                    filename = f"transliteration_{lang}_{fmt.lower()}_{ts}.txt"
                    filepath = _CRON_DIR / filename
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(result)
                    download_path = str(filepath)
                    stats_md += f"\n*File ready for download: {filename}*"
                return result, stats_md, download_path
            except Exception as e:  # pragma: no cover
                return "", f"**Error**: {str(e)}", None

        with gr.Row(elem_classes=["examples-row"]):
            gr.Examples(
                examples=cast(
                    list[list[Any]],
                    [
                        ["Пример текста", "kk", "Latin", False, None],
                        ["Merhaba dünya", "tr", "IPA", False, None],
                    ],
                ),
                inputs=[
                    translit_textbox,
                    lang,
                    output_format,
                    include_arabic,
                    translit_upload_file,
                ],
                outputs=[output, stats, download_file],
                fn=do_direct,
                label="Try these examples",
            )
            btn = gr.Button("Transliterate", variant="primary")

        btn.click(
            do_direct,
            [
                translit_textbox,
                lang,
                output_format,
                include_arabic,
                translit_upload_file,
            ],
            [output, stats, download_file],
        )
        translit_textbox.change(
            do_direct,
            [
                translit_textbox,
                lang,
                output_format,
                include_arabic,
                translit_upload_file,
            ],
            [output, stats, download_file],
        )
        lang.change(
            do_direct,
            [
                translit_textbox,
                lang,
                output_format,
                include_arabic,
                translit_upload_file,
            ],
            [output, stats, download_file],
        )
        output_format.change(
            do_direct,
            [
                translit_textbox,
                lang,
                output_format,
                include_arabic,
                translit_upload_file,
            ],
            [output, stats, download_file],
        )
        translit_upload_file.change(
            do_direct,
            [
                translit_textbox,
                lang,
                output_format,
                include_arabic,
                translit_upload_file,
            ],
            [output, stats, download_file],
        )
