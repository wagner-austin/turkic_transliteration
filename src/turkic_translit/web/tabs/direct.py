"""The Direct Transliteration tab.

Both handlers below report failure to the user as text rather than
raising, because a Gradio callback that raises shows the user nothing
useful. They catch only what the operation can actually produce — an
``OSError`` from reading an uploaded file, and a ``ValueError`` from an
unsupported language or output format — so a genuine defect still
surfaces as a traceback instead of being rendered as a polite message.

The handlers are module-level functions rather than closures inside
:func:`register`. A closure over the widgets can only be reached by
building the whole interface, which left every decision the tab makes —
which input wins, when a download is offered, how a failure reads —
exercised by nothing. :func:`register` now only wires widgets to them.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import gradio as gr

from turkic_translit.lang_utils import pretty_lang
from turkic_translit.web.web_utils import direct_transliterate, download_dir

logger = logging.getLogger(__name__)

MIN_CHARS_FOR_DOWNLOAD = 50

__all__ = [
    "MIN_CHARS_FOR_DOWNLOAD",
    "ipa_languages",
    "language_info",
    "register",
    "transliterate_request",
]


def _handle_file_upload(file_path: str | None) -> str:
    """Read an uploaded file, reporting read failures as text.

    Args:
        file_path: Path Gradio wrote the upload to, or ``None``.

    Returns:
        The file's contents, empty when nothing was uploaded, or a
        message naming the read failure.
    """
    if not file_path:
        return ""
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read uploaded file %s: %s", file_path, exc)
        return f"Error reading file: {exc!s}"


def ipa_languages() -> list[str]:
    """List the language codes the tab can transliterate to IPA.

    Returns:
        Codes with an ``<lang>_ipa.rules`` file, sorted.
    """
    from turkic_translit.core import get_supported_languages

    return sorted(code for code, fmts in get_supported_languages().items() if "ipa" in fmts)


def language_info(codes: list[str]) -> str:
    """Summarise the language list for the radio group's caption.

    Args:
        codes: Language codes the radio group offers, in display order.

    Returns:
        The first three codes with their names, and a count of the rest.
    """
    shown = ", ".join(f"{code} = {pretty_lang(code)}" for code in codes[:3])
    if len(codes) > 3:
        return f"{shown}, +{len(codes) - 3} more"
    return shown


def transliterate_request(
    text: str, lang: str, file_path: str | None = None
) -> tuple[str, str, str | None]:
    """Transliterate the text box or the uploaded file to IPA.

    Args:
        text: Text typed into the box.
        lang: Language code selected in the radio group.
        file_path: Uploaded file, which takes precedence over the box, or
            ``None`` when nothing was uploaded.

    Returns:
        The IPA output, a Markdown status line, and the path of a
        downloadable copy when the result is long enough to want one.
    """
    if file_path:
        text = _handle_file_upload(file_path)
        if text.startswith("Error reading file:"):
            return "", f"**{text}**", None
    if not text.strip():
        return "", "*Please enter some text to transliterate or upload a file*", None

    try:
        result, stats_md = direct_transliterate(text, lang, False, "ipa")
    except ValueError as exc:
        logger.warning("transliteration of %s failed: %s", lang, exc)
        return "", f"**Error**: {exc!s}", None

    if file_path:
        stats_md += "\n*Source: Uploaded file*"

    if len(result) <= MIN_CHARS_FOR_DOWNLOAD:
        return result, stats_md, None

    stamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"transliteration_{lang}_ipa_{stamp}.txt"
    target = download_dir() / filename
    target.write_text(result, encoding="utf-8")
    stats_md += f"\n*File ready for download: {filename}*"
    return result, stats_md, str(target)


def register() -> None:
    """Render the Direct Transliteration tab content."""
    with gr.Column():
        gr.Markdown(
            """
            **Direct Transliteration:** Convert text to IPA using language-specific rules.
            Select the language and click Transliterate.
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
                # Only expose languages that have an `{lang}_ipa.rules` file.
                lang_choices = ipa_languages()
                lang = gr.Radio(
                    lang_choices,
                    label="Language",
                    value=lang_choices[0],
                    info=language_info(lang_choices),
                )

            with gr.Column(scale=7):
                output = gr.Textbox(label="Output (IPA)", lines=4, interactive=False)
                stats = gr.Markdown()
                download_file = gr.File(label="Download Result", elem_id="download-output")

        with gr.Row(elem_classes=["examples-row"]):
            gr.Examples(
                examples=[
                    ["Пример текста", "kk", None],
                    ["Merhaba dünya", "tr", None],
                ],
                inputs=[
                    translit_textbox,
                    lang,
                    translit_upload_file,
                ],
                outputs=[output, stats, download_file],
                fn=transliterate_request,
                label="Try these examples",
            )
            btn = gr.Button("Transliterate", variant="primary")

        btn.click(
            transliterate_request,
            [translit_textbox, lang, translit_upload_file],
            [output, stats, download_file],
        )
        translit_textbox.change(
            transliterate_request,
            [translit_textbox, lang, translit_upload_file],
            [output, stats, download_file],
        )
        lang.change(
            transliterate_request,
            [translit_textbox, lang, translit_upload_file],
            [output, stats, download_file],
        )
        translit_upload_file.change(
            transliterate_request,
            [translit_textbox, lang, translit_upload_file],
            [output, stats, download_file],
        )
