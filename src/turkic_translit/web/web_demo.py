# ruff: noqa: E402
from __future__ import annotations

"""
A web interface to explore and test Turkic transliteration features.

This module provides a Gradio-based web UI for the Turkic Transliteration Suite,
offering various transliteration and language analysis features.

IMPORTANT NOTES:
1. This file contains the core implementation of the web UI and can be run directly
   with `python web_demo.py` for development and debugging purposes.

2. For production use, this file is imported by app.py which is called by the `make web`
   command. Some console logs (especially warnings about missing dependencies or model
   files) may be more visible when running this file directly rather than through app.py.

3. Missing dependency warnings (e.g., FastText language model for Russian filtering)
   will be displayed both in the console and in the web UI itself when that feature
   is accessed.
"""
import logging
import pathlib
from typing import Any, cast

import gradio as gr

from turkic_translit.web.web_utils import (
    direct_transliterate,
    mask_russian,
    median_levenshtein,
    pipeline_transliterate,
    token_table_markdown,
    train_sentencepiece_model,
)

logging.basicConfig(level=logging.INFO)


def _model_check() -> str:
    """
    Check for required model files. If any are missing, try to download them automatically.
    Returns a markdown warning string for any models that couldn't be downloaded; else returns an empty string.
    """
    missing = []
    # Try to download and check the fastText model
    try:
        from ..model_utils import ensure_fasttext_model

        model_path = ensure_fasttext_model()
        logging.info(f"FastText language identification model found at {model_path}")
    except Exception as e:
        # Check lid.176.ftz in standard locations
        home_lid = pathlib.Path.home() / "lid.176.ftz"
        root_lid = pathlib.Path(__file__).parent / "lid.176.ftz"
        pkg_lid = pathlib.Path(__file__).parent.parent / "lid.176.ftz"

        if not any(p.exists() for p in [home_lid, root_lid, pkg_lid]):
            msg = f"- `lid.176.ftz` not found and auto-download failed: {str(e)}"
            logging.warning(msg)
            missing.append(msg)
    # Check turkic_model.model in tokenizer.py's directory
    try:
        import turkic_translit.tokenizer as tokenizer

        tok_dir = pathlib.Path(tokenizer.__file__).parent
        model_path = tok_dir / "turkic_model.model"
        if not model_path.exists():
            msg = f"- `turkic_model.model` not found in {model_path}"
            logging.warning(msg)
            missing.append(msg)
    except Exception as e:
        msg = f"- Could not check for `turkic_model.model`: {e}"
        logging.warning(msg)
        missing.append(msg)
    if missing:
        return (
            "**⚠️ Model file(s) missing:**\n"
            + "\n".join(missing)
            + "\nPlease ensure all required models are present for full functionality."
        )
    return ""


def build_ui() -> gr.Blocks:
    """Build and return the Turkic Transliteration Suite UI as a Gradio Blocks app."""
    # Define example inputs for each tab
    examples: dict[str, list[Any]] = {
        "direct": [
            ["сәлем әлем", "kk", "Latin", False],
            ["мен қазақша сөйлеймін", "kk", "IPA", False],
            ["кыргыз тилинде сүйлөйм", "ky", "Latin", False],
        ],
        "pipeline": [["менің атым Айдар", "Latin"], ["сенің атың кім", "IPA"]],
        "tokens": ["сәлем привет қалайсың здравствуй"],
        "filter_ru": [["қазақша текст с русскими словами", 0.5, 3]],
        "spm_examples": ["менің атым Айдар, сенің атың кім?"],
    }

    def do_train_spm(
        input_text: str,
        training_file: Any,
        vocab_size: int,
        model_type: str,
        char_coverage: float,
        user_symbols: str,
    ) -> tuple[str, str]:
        """Train a SentencePiece model with the provided parameters."""
        try:
            model_file, info = train_sentencepiece_model(
                input_text=input_text,
                training_file=training_file,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=char_coverage,
                user_symbols=user_symbols,
            )
            return info, model_file
        except Exception as e:
            return f"**Error training model:** {str(e)}", ""

    def do_compare(lat_file: Any, ipa_file: Any, sample_n: Any) -> str:
        if lat_file is None or ipa_file is None:
            return "**Please upload both Latin and IPA files to compare.**"
        try:
            lat_obj = type("FileObj", (), {"name": lat_file.name})
            ipa_obj = type("FileObj", (), {"name": ipa_file.name})
            sample_val = int(sample_n) if str(sample_n).strip() else None
            result = median_levenshtein(lat_obj, ipa_obj, sample_val)
            return f"**Comparison Result:**\n{result}"
        except Exception as e:
            return f"**Error comparing files**: {str(e)}"

    css = """
    .container { margin: 0 auto; }
    .tab-content { padding: 10px 15px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 5px 5px; }
    .examples-row { margin-top: 10px; }
    .feature-description { margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px; }
    .file-info { margin-top: -5px; font-size: 0.85em; color: #555; }
    footer { margin-top: 20px; text-align: center; font-size: 0.8em; color: #666; }
    """

    with gr.Blocks(
        title="Turkic Transliteration Suite", css=css, theme=gr.themes.Soft()
    ) as app:
        gr.Markdown(
            """
            # 🌐 Turkic Transliteration Suite
            
            Explore transliteration capabilities for Turkic languages between Cyrillic, Latin, and IPA representations.
            Navigate through the tabs below to access different features.
            """
        )

        model_warning = _model_check()
        if model_warning:
            gr.Markdown(model_warning, elem_classes=["warning-box"])

        shared_textbox = gr.Textbox(
            label="Input Text",
            lines=4,
            elem_id="input-text",
            placeholder="Enter Turkic language text in Cyrillic script...",
        )

        # Helper functions defined first so they're available for the UI components
        def do_direct(
            text: str, lang: str, fmt: str, include_arabic: bool
        ) -> tuple[str, str]:
            if not text.strip():
                return "", "*Please enter some text to transliterate*"
            try:
                result, stats = direct_transliterate(
                    text, lang, include_arabic, fmt.lower()
                )
                return result, stats
            except Exception as e:
                return "", f"**Error**: {str(e)}"

        def do_pipeline(text: str, mode: str) -> tuple[str, str]:
            if not text.strip():
                return "", "*Please enter some text to transliterate*"
            try:
                result, stats = pipeline_transliterate(text, mode.lower())
                return result, stats
            except Exception as e:
                return "", f"**Error**: {str(e)}"

        def do_tokens(text: str) -> str:
            if not text.strip():
                return "*Please enter some text to analyze*"
            try:
                result = token_table_markdown(text)
                return result if result else "*No tokens identified*"
            except ImportError as e:
                return f"**pandas missing – install `turkic-transliterate[ui]`**\n\n{e}"
            except Exception as e:
                return f"**Error analyzing tokens**: {str(e)}"

        def do_mask(text: str, threshold: float, min_len: int) -> str:
            if not text.strip():
                return "*Please enter some text to mask*"
            try:
                return mask_russian(text, threshold, min_len)
            except Exception as e:
                return f"**Error masking Russian text**: {str(e)}"

        def do_compare(lat_file: Any, ipa_file: Any, sample_n: Any) -> str:
            if lat_file is None or ipa_file is None:
                return "**Please upload both Latin and IPA files to compare.**"
            try:
                lat_obj = type("FileObj", (), {"name": lat_file.name})
                ipa_obj = type("FileObj", (), {"name": ipa_file.name})
                sample_val = int(sample_n) if str(sample_n).strip() else None
                result = median_levenshtein(lat_obj, ipa_obj, sample_val)
                return f"**Comparison Result:**\n{result}"
            except Exception as e:
                return f"**Error comparing files**: {str(e)}"

        def _direct_tab() -> None:
            with gr.Column():
                gr.Markdown(
                    """
                    <div class="feature-description">
                    <strong>Direct Transliteration:</strong> Convert text directly to Latin or IPA using language-specific rules.
                    Select the language, choose your output format, and click Transliterate.
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        lang = gr.Radio(
                            ["kk", "ky"],
                            label="Language",
                            value="kk",
                            info="kk = Kazakh, ky = Kyrgyz",
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

                with gr.Row(elem_classes=["examples-row"]):
                    gr.Examples(
                        examples=cast(list[list[Any]], examples["direct"]),
                        inputs=[shared_textbox, lang, output_format, include_arabic],
                        outputs=[output, stats],
                        fn=do_direct,
                        label="Try these examples",
                    )
                    btn = gr.Button("Transliterate", variant="primary")

            # Trigger on button click
            btn.click(
                do_direct,
                [shared_textbox, lang, output_format, include_arabic],
                [output, stats],
            )
            # Also trigger on text input change for real-time transliteration
            shared_textbox.change(
                do_direct,
                [shared_textbox, lang, output_format, include_arabic],
                [output, stats],
            )
            # Update when language or format changes
            lang.change(
                do_direct,
                [shared_textbox, lang, output_format, include_arabic],
                [output, stats],
            )
            output_format.change(
                do_direct,
                [shared_textbox, lang, output_format, include_arabic],
                [output, stats],
            )
            include_arabic.change(
                do_direct,
                [shared_textbox, lang, output_format, include_arabic],
                [output, stats],
            )

        def _pipeline_tab() -> None:
            with gr.Column():
                gr.Markdown(
                    """
                    <div class="feature-description">
                    <strong>Pipeline Transliteration:</strong> Process text through the complete transliteration pipeline, 
                    which includes language identification, tokenization, and transliteration.
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        mode = gr.Radio(
                            ["Latin", "IPA"],
                            label="Pipeline Mode",
                            value="Latin",
                            info="Choose output format for the pipeline",
                        )

                    with gr.Column(scale=8):
                        output = gr.Textbox(label="Output", lines=4, interactive=False)
                        stats = gr.Markdown()

                with gr.Row(elem_classes=["examples-row"]):
                    gr.Examples(
                        examples=cast(list[list[Any]], examples["pipeline"]),
                        inputs=[shared_textbox, mode],
                        outputs=[output, stats],
                        fn=do_pipeline,
                        label="Try these examples",
                    )
                    btn = gr.Button("Pipeline Transliterate", variant="primary")

                # Button click for pipeline transliteration
                btn.click(do_pipeline, [shared_textbox, mode], [output, stats])

                # Real-time transliteration as you type
                shared_textbox.change(
                    do_pipeline, [shared_textbox, mode], [output, stats]
                )

                # Update when mode changes
                mode.change(do_pipeline, [shared_textbox, mode], [output, stats])

        def _tokens_tab() -> None:
            with gr.Column():
                gr.Markdown(
                    """
                    <div class="feature-description">
                    <strong>Token Analysis:</strong> Analyze the text by splitting it into tokens and identifying the most likely language for each token.
                    This helps understand how the system processes mixed-language text.
                    </div>
                    """
                )

                token_md = gr.Markdown()
                analyze_btn = gr.Button("Analyze Tokens", variant="primary")

                with gr.Row(elem_classes=["examples-row"]):
                    gr.Examples(
                        examples=cast(list[str], examples["tokens"]),
                        inputs=[shared_textbox],
                        outputs=[token_md],
                        fn=do_tokens,
                        label="Try these examples",
                    )

            # Button click for token analysis
            analyze_btn.click(do_tokens, [shared_textbox], token_md)

            # Real-time token analysis as you type
            shared_textbox.change(do_tokens, [shared_textbox], token_md)

        def _filter_ru_tab() -> None:
            with gr.Column():
                gr.Markdown(
                    """
                    <div class="feature-description">
                    <strong>Russian Text Filter:</strong> Identify and mask Russian text within mixed-language content.
                    Adjust the threshold and minimum token length to control sensitivity.
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        threshold = gr.Slider(
                            0,
                            1,
                            value=0.7,  # Changed from 0.5 to 0.7 as requested
                            label="RU Mask Threshold",
                            step=0.01,
                            info="Higher values are more strict in identifying Russian",
                            elem_id="ru-threshold-slider",
                            show_label=True,
                        )
                        min_len = gr.Slider(
                            1,
                            10,
                            value=3,
                            label="Min Token Length",
                            step=1,
                            info="Skip words shorter than this",  # Updated helper text
                            elem_id="min-len-slider",
                            show_label=True,
                        )

                    with gr.Column(scale=7):
                        output = gr.Textbox(
                            label="Masked Output",
                            lines=4,
                            interactive=False,
                            show_copy_button=True,
                        )

                with gr.Row(elem_classes=["examples-row"]):
                    gr.Examples(
                        examples=[
                            ["қазақша текст с русскими словами", 0.7, 3],
                            ["Все это написано на русском языке", 0.7, 3],
                            ["Бұл мәтін толығымен қазақ тілінде жазылған", 0.7, 3],
                            ["қазақша и русский в одном тексте", 0.7, 2],
                            ["Көптеген орыс сөздер арасында казахский текст", 0.8, 4],
                        ],
                        inputs=[shared_textbox, threshold, min_len],
                        outputs=[output],
                        fn=do_mask,
                        label="Try these examples",
                    )
                    btn = gr.Button("Mask Russian", variant="primary")

            # Button click for masking Russian
            btn.click(do_mask, [shared_textbox, threshold, min_len], output)

            # Real-time masking as you type
            shared_textbox.change(do_mask, [shared_textbox, threshold, min_len], output)

            # Update when parameters change
            threshold.change(do_mask, [shared_textbox, threshold, min_len], output)
            min_len.change(do_mask, [shared_textbox, threshold, min_len], output)

        def _compare_tab() -> None:
            with gr.Column():
                gr.Markdown(
                    """
                    <div class="feature-description">
                    <strong>Compare Transliterations:</strong> Calculate the median Levenshtein distance between Latin and IPA transliterations.
                    Upload two text files for comparison to evaluate transliteration quality.
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column():
                        file_lat = gr.File(label="Latin File", file_types=["text"])
                        gr.Markdown(
                            "*Upload a file with Latin transliteration*",
                            elem_classes=["file-info"],
                        )
                    with gr.Column():
                        file_ipa = gr.File(label="IPA File", file_types=["text"])
                        gr.Markdown(
                            "*Upload a file with IPA transliteration*",
                            elem_classes=["file-info"],
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        sample = gr.Number(label="Sample Size (optional)", precision=0)
                        gr.Markdown(
                            "*Number of lines to sample (leave empty to use all)*",
                            elem_classes=["file-info"],
                        )
                    with gr.Column(scale=1):
                        btn = gr.Button("Compare Files", variant="primary")

                out = gr.Markdown()

            btn.click(do_compare, [file_lat, file_ipa, sample], out)

        def _sentencepiece_tab() -> None:
            with gr.Column():
                gr.Markdown(
                    """
                    <div class="feature-description">
                    <strong>SentencePiece Training:</strong> Train your own SentencePiece model for tokenization.
                    You can use this model for custom tokenization tasks with the Turkic Transliteration toolkit.
                    </div>
                    """
                )

                with gr.Row(), gr.Column():
                    input_text = gr.Textbox(
                        label="Input Text (Optional)",
                        placeholder="Enter text here for training...",
                        lines=5,
                    )
                    gr.Markdown(
                        "*You can provide text directly in the box above, or upload a file below, or both*",
                        elem_classes=["file-info"],
                    )
                    training_file = gr.File(
                        label="Training File (Optional)",
                        file_types=["text"],
                        file_count="single",
                    )

                with gr.Row():
                    with gr.Column():
                        vocab_size = gr.Slider(
                            minimum=100,
                            maximum=50000,
                            value=12000,
                            step=100,
                            label="Vocabulary Size",
                        )
                        gr.Markdown(
                            "*Number of tokens in the vocabulary*",
                            elem_classes=["file-info"],
                        )
                    with gr.Column():
                        model_type = gr.Dropdown(
                            choices=["unigram", "bpe", "char", "word"],
                            value="unigram",
                            label="Model Type",
                        )
                        gr.Markdown(
                            "*SentencePiece algorithm*", elem_classes=["file-info"]
                        )

                with gr.Row():
                    with gr.Column():
                        char_coverage = gr.Slider(
                            minimum=0.9,
                            maximum=1.0,
                            value=1.0,
                            step=0.01,
                            label="Character Coverage",
                        )
                        gr.Markdown(
                            "*Amount of characters covered by the model*",
                            elem_classes=["file-info"],
                        )
                    with gr.Column():
                        user_symbols = gr.Textbox(
                            label="User Symbols (Comma-separated)",
                            placeholder="<lang_kk>,<lang_ky>",
                            value="<lang_kk>,<lang_ky>",
                        )
                        gr.Markdown(
                            "*Special tokens to include in the model*",
                            elem_classes=["file-info"],
                        )

                with gr.Row(), gr.Column():
                    train_btn = gr.Button("Train Model", variant="primary")

                output_info = gr.Markdown("*Click 'Train Model' to begin training*")
                model_download = gr.File(label="Download Trained Model")

                # Examples section
                gr.Examples(
                    examples["spm_examples"],
                    inputs=[input_text],
                    label="Try this example text",
                )

            # Connect the training function to the button
            train_btn.click(
                do_train_spm,
                inputs=[
                    input_text,
                    training_file,
                    vocab_size,
                    model_type,
                    char_coverage,
                    user_symbols,
                ],
                outputs=[output_info, model_download],
            )

        with gr.Tabs():
            # Order tabs by the logical workflow: training → tokenization → processing → analysis
            with gr.Tab("🧩 SentencePiece Training", id="sentencepiece"):
                _sentencepiece_tab()
            with gr.Tab("🔍 Token Analysis", id="tokens"):
                _tokens_tab()
            with gr.Tab("🎭 Filter Russian", id="filter_ru"):
                _filter_ru_tab()
            with gr.Tab("🔄 Pipeline Process", id="pipeline"):
                _pipeline_tab()
            with gr.Tab("📝 Direct Transliteration", id="direct"):
                _direct_tab()
            with gr.Tab("📊 Compare Files", id="compare"):
                _compare_tab()

        gr.Markdown(
            """
            <footer>
            <p>Turkic Transliteration Suite - A tool for transliterating Turkic languages between different writing systems</p>
            <p>Use the tabs above to explore different features</p>
            </footer>
            """
        )
    return app  # type: ignore[no-any-return]


def main() -> None:
    """Launch the Turkic Transliteration web interface with default settings.

    This function serves as an entry point for the web UI and can be called
    directly or through the CLI entry point 'turkic-web'.
    """
    ui = build_ui()
    ui.queue().launch()


if __name__ == "__main__":
    main()
