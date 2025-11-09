from __future__ import annotations

import logging
import pathlib
from typing import cast

import gradio as gr

from turkic_translit.web.web_utils import get_ui_log_handler

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
    fasttext_info = ""

    try:
        from ..model_utils import ensure_fasttext_model

        model_path = ensure_fasttext_model()
        model_name = model_path.name
        size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)
        model_type = "Full" if model_name.endswith(".bin") else "Compressed"
        fasttext_info = (
            f"FastText Language Model: {model_name} ({model_type}, {size_mb} MB)"
        )
        logging.info("FastText language identification model found at %s", model_path)
    except Exception as exc:  # noqa: BLE001
        home = pathlib.Path.home()
        pkg_dir = pathlib.Path(__file__).parent.parent
        probe_paths = [
            home / "lid.176.bin",
            home / "lid.176.ftz",
            pkg_dir / "lid.176.bin",
            pkg_dir / "lid.176.ftz",
        ]
        existing = [p for p in probe_paths if p.exists()]
        if existing:
            p = existing[0]
            model_name = p.name
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            model_type = "Full" if model_name.endswith(".bin") else "Compressed"
            fasttext_info = (
                f"FastText Language Model: {model_name} ({model_type}, {size_mb} MB)"
            )
        else:
            msg = f"- FastText language model not found and auto-download failed: {exc}"
            logging.warning(msg)
            missing.append(msg)
            fasttext_info = "FastText Language Model: Not found"

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
    warning_message, _ = _model_check()

    css = """
    .container { margin: 0 auto; }
    .tab-content { padding: 10px 15px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 5px 5px; }
    .examples-row { margin-top: 10px; }
    .file-info { margin-top: -5px; font-size: 0.85em; color: #555; }
    footer { margin-top: 20px; text-align: center; font-size: 0.8em; color: #666; }
    """

    with gr.Blocks(
        title="Turkic Transliteration Suite", css=css, theme=gr.themes.Soft()
    ) as app:
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
            Explore transliteration capabilities for Turkic languages between Cyrillic, Latin, and IPA representations.
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

        def _sentencepiece_tab() -> None:
            from turkic_translit.web.tabs import sentencepiece as _tab

            _tab.register()

        def _mutual_tab() -> None:
            from turkic_translit.web.tabs import mutual as _tab

            _tab.register()

        with gr.Tabs():
            with gr.Tab("📥 Download Corpus", id="corpus"):
                _corpus_tab()
            with gr.Tab("🧩 Train Tokenizer", id="sentencepiece"):
                _sentencepiece_tab()
            with gr.Tab("📝 Transliterate to IPA/Latin", id="direct"):
                _direct_tab()
            with gr.Tab("🤝 Mutual Intelligibility", id="mutual"):
                _mutual_tab()

        gr.Markdown(
            """
            <footer>
            <p>Turkic Transliteration Suite - A tool for transliterating Turkic languages between different writing systems</p>
            <p>Use the tabs above to explore different features</p>
            </footer>
            """
        )
    return cast(gr.Blocks, app)


def main() -> None:
    """Launch the web UI."""
    ui = build_ui()
    ui.queue().launch()


if __name__ == "__main__":
    main()
