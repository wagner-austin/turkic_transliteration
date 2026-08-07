from __future__ import annotations

import logging

import gradio as gr

from turkic_translit.lid.locations import default_search_dirs
from turkic_translit.lid.registry import find_model_path
from turkic_translit.web.web_utils import LANGUAGE_MODEL_ID, get_ui_log_handler

from ..error_service import init_error_service
from ..logging_config import setup as _log_setup

"""Gradio-based web UI for the Turkic Transliteration Suite.

This module builds the Blocks app shell and delegates individual tabs to
modular implementations under turkic_translit.web.tabs.*
"""


# Logging is configured centrally when the UI is built/launched.
_logger = logging.getLogger("turkic_translit.web_demo")


def _model_check() -> tuple[str, str]:
    """Verify auxiliary model files exist; attempt download when missing.

    Returns (warning_markdown, fasttext_info_markdown).
    """
    missing: list[str] = []

    weights = find_model_path(LANGUAGE_MODEL_ID, default_search_dirs())
    if weights is None:
        message = (
            f"- Language-identification model {LANGUAGE_MODEL_ID!r} is not "
            f"installed; it will be downloaded on first use"
        )
        _logger.warning(message)
        missing.append(message)
        fasttext_info = f"Language model {LANGUAGE_MODEL_ID}: not installed"
    else:
        size_mb = round(weights.stat().st_size / (1024 * 1024), 2)
        fasttext_info = f"Language model {LANGUAGE_MODEL_ID}: {weights.name} ({size_mb} MB)"
        _logger.info("Language-identification model found at %s", weights)

    # The web UI no longer depends on the SentencePiece tokenizer by default,
    # so we don't warn about a missing `turkic_model.model` here. Tabs that
    # need it should surface their own guidance when used.

    warning_md = ""
    if missing:
        warning_md = (
            "**Model file(s) missing:**\n"
            + "\n".join(missing)
            + "\nPlease ensure all required models are present for full functionality."
        )

    return warning_md, fasttext_info


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    # Ensure logging is configured for the web demo (honours TURKIC_LOG_LEVEL)
    _log_setup()
    init_error_service()
    # Called for the startup log line naming which language-identification
    # model is installed. The result is deliberately not surfaced in the UI:
    # tabs raise their own notice when a feature actually needs the model.
    _model_check()

    css = """
    .container { margin: 0 auto; }
    .tab-content { padding: 10px 15px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 5px 5px; }
    .examples-row { margin-top: 10px; }
    .file-info { margin-top: -5px; font-size: 0.85em; color: #555; }
    footer { margin-top: 20px; text-align: center; font-size: 0.8em; color: #666; }
    """

    blocks: gr.Blocks = gr.Blocks(
        title="Turkic Transliteration Suite", css=css, theme=gr.themes.Soft()
    )
    with blocks:
        gr.Markdown(
            """
            # Turkic Transliteration Suite
            ## Web Interface for exploring Turkic language transliteration tools
            """
        )

        # Avoid surfacing global warnings at load; tabs will surface notices
        # contextually if a dependency is needed when a feature is used.

        gr.Markdown(
            """
            Explore IPA transliteration for Turkic languages.
            Navigate through the tabs below to access different features.
            """
        )

        # Attach shared UI log handler (for tabs that stream logs)
        _ = get_ui_log_handler()

        # Lightweight wrappers that delegate to modular tab code
        def _direct_tab() -> None:
            from turkic_translit.web.tabs import direct as _tab

            _tab.register()

        def _corpus_tab() -> None:
            from turkic_translit.web.tabs import corpus as _tab

            _tab.register()

        with gr.Tabs():
            with gr.Tab("📥 Download Corpus", id="corpus"):
                _corpus_tab()
            with gr.Tab("📝 Transliterate to IPA", id="direct"):
                _direct_tab()

        gr.Markdown(
            """
            <footer>
            <p>Turkic Transliteration Suite - A tool for transliterating Turkic languages between different writing systems</p>
            <p>Use the tabs above to explore different features</p>
            </footer>
            """
        )
    return blocks


def main() -> None:
    """Launch the web UI."""
    ui = build_ui()
    ui.queue().launch()


if __name__ == "__main__":
    main()
