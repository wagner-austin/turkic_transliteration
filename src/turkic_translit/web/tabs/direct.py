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
from collections.abc import Mapping
from pathlib import Path

import gradio as gr

from turkic_translit.web.web_utils import direct_transliterate, download_dir, labelise

logger = logging.getLogger(__name__)

# One example per language the tab offers, and every word is one this
# project already checks against a published description of that
# language — the same words the gold-standard test modules pin the rules
# against. A visitor's first click therefore lands on output the
# literature agrees with, which is the whole claim this demo makes.
#
# tests/test_web_tab_direct.py fails when a word here is not one of those
# words, and when this table's languages are not exactly the offered
# ones, so a new rule file cannot arrive without an example. The
# previous pair pinned two codes by hand, showed Russian text under the
# Kazakh one, and left the other six languages with no example at all.
EXAMPLE_WORDS: Mapping[str, str] = {
    "az": "kitab",  # 'book', Ragagnin p. 244
    "fi": "rengas",  # 'tyre', Karlsson p. 14
    "kk": "құс",  # 'bird', Abish p. 337
    "ky": "көл",  # McCollum, monosyllabic roots, Table 3
    "tr": "dağ",  # soft g lengthens the vowel, Routledge p. 195
    "ug": "ئوخشاش",  # Montreal Forced Aligner dictionary sample
    "uz": "besh",  # 'five', Ido p. 154
    "uzc": "беш",  # 'five', Ido p. 154
}

__all__ = [
    "EXAMPLE_WORDS",
    "ipa_languages",
    "register",
    "transliterate_request",
    "transliterated_download",
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


def transliterate_request(text: str, lang: str, file_path: str | None = None) -> tuple[str, str]:
    """Transliterate the text box or the uploaded file to IPA.

    Args:
        text: Text typed into the box.
        lang: Language code selected in the radio group.
        file_path: Uploaded file, which takes precedence over the box, or
            ``None`` when nothing was uploaded.

    Returns:
        The IPA output and a Markdown status line. Nothing is written to
        disk: this runs on every keystroke, and the previous version
        wrote a timestamped file on each one whose output passed fifty
        characters, so typing a paragraph left a hundred files behind.
    """
    if file_path:
        text = _handle_file_upload(file_path)
        if text.startswith("Error reading file:"):
            return "", f"**{text}**"
    if not text.strip():
        return "", "*Please enter some text to transliterate or upload a file*"

    try:
        result, stats_md = direct_transliterate(text, lang, False, "ipa")
    except ValueError as exc:
        logger.warning("transliteration of %s failed: %s", lang, exc)
        return "", f"**Error**: {exc!s}"

    if file_path:
        stats_md += "\n*Source: Uploaded file*"
    return result, stats_md


def transliterated_download(
    text: str, lang: str, file_path: str | None = None
) -> tuple[str, str, gr.File]:
    """Transliterate, and write the result out for downloading.

    Asking for a file is what the button means, so length no longer
    decides it. The previous rule offered a download above fifty
    characters and withheld one below, which made the shorter answer the
    harder one to keep.

    Args:
        text: Text typed into the box.
        lang: Language code selected in the radio group.
        file_path: Uploaded file, which takes precedence over the box, or
            ``None`` when nothing was uploaded.

    Returns:
        The IPA output, a Markdown status line, and the download slot —
        holding the written file, or empty and hidden when there was no
        output to write.
    """
    result, stats_md = transliterate_request(text, lang, file_path)
    if not result:
        return result, stats_md, gr.File(value=None, visible=False)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"transliteration_{lang}_ipa_{stamp}.txt"
    target = download_dir() / filename
    target.write_text(result, encoding="utf-8")
    return result, stats_md, gr.File(value=str(target), visible=True)


def register() -> None:
    """Render the Direct Transliteration tab content."""
    with gr.Column():
        gr.Markdown(
            """
            **Direct Transliteration:** Convert text to IPA using language-specific rules.
            Select the language and click Transliterate.
            """
        )

        # Input on the left and output on the right, as the corpus tab
        # arranges them. The previous layout put the input box across the
        # top at double width, which left it beside an empty column, and
        # pushed the output down into a narrow strip off to one side.
        with gr.Row():
            with gr.Column(scale=1):
                translit_textbox = gr.Textbox(
                    label="Input Text",
                    lines=6,
                    elem_id="translit-input-text",
                    placeholder="Enter text in the language selected below...",
                )
                translit_upload_file = gr.File(
                    label="Or upload a .txt file, which replaces the text above",
                    file_types=[".txt"],
                    type="filepath",
                    elem_id="translit-file-upload",
                )
                # Only expose languages that have an `{lang}_ipa.rules` file,
                # named as the corpus tab names them rather than by bare code.
                lang_choices = ipa_languages()
                lang = gr.Radio(
                    labelise(lang_choices),
                    label="Language",
                    value=lang_choices[0],
                )
                btn = gr.Button("Transliterate", variant="primary")

            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Output (IPA)",
                    lines=6,
                    interactive=False,
                    buttons=["copy"],
                )
                stats = gr.Markdown()
                # Empty until the button writes something into it, like
                # the corpus tab's transliterated-file slot.
                download_file = gr.File(
                    label="Download Result",
                    elem_id="download-output",
                    visible=False,
                )

        # Below both columns, so the examples span the tab rather than
        # sharing a row with the button that acts on them.
        gr.Examples(
            examples=[[EXAMPLE_WORDS[code], code, None] for code in lang_choices],
            inputs=[translit_textbox, lang, translit_upload_file],
            outputs=[output, stats, download_file],
            fn=transliterated_download,
            label="Try these examples",
        )

        # Typing, switching language and uploading all show the result;
        # only the button writes a file, because only the button was
        # asked for one.
        inputs = [translit_textbox, lang, translit_upload_file]
        btn.click(transliterated_download, inputs, [output, stats, download_file])
        translit_textbox.change(transliterate_request, inputs, [output, stats])
        lang.change(transliterate_request, inputs, [output, stats])
        translit_upload_file.change(transliterate_request, inputs, [output, stats])
