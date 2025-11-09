from __future__ import annotations

from typing import Any

import gradio as gr

from turkic_translit.web.web_utils import train_sentencepiece_model


def register() -> None:
    with gr.Column():
        gr.Markdown(
            """
            **SentencePiece Training:** Train your own SentencePiece model for tokenization.
            You can use this model for custom tokenization tasks with the Turkic Transliteration toolkit.
            """
        )

        with gr.Row(), gr.Column():
            input_text = gr.Textbox(
                label="Input Text (Optional)",
                placeholder="Enter text here for training...",
                lines=5,
            )
            gr.Markdown(
                "*Provide text in the box above, or upload a file, or both*",
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
                    "*Number of tokens in the vocabulary*", elem_classes=["file-info"]
                )
            with gr.Column():
                model_type = gr.Dropdown(
                    choices=["unigram", "bpe", "char", "word"],
                    value="unigram",
                    label="Model Type",
                )
                gr.Markdown("*SentencePiece algorithm*", elem_classes=["file-info"])

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
                    label="User-defined Symbols (comma-separated)",
                    value="<lang_kk>,<lang_ky>",
                )
                gr.Markdown(
                    "*Optional special tokens to include in the vocabulary*",
                    elem_classes=["file-info"],
                )

        with gr.Row():
            train_btn = gr.Button("Train Model", variant="primary")
            output_info = gr.Markdown()
            model_download = gr.File(label="Download Model")

        def do_train_spm(
            input_text: str,
            training_file: Any,
            vocab_size: int,
            model_type: str,
            char_coverage: float,
            user_symbols: str,
        ) -> tuple[str, str]:
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
            except Exception as e:  # pragma: no cover
                return f"**Error training model:** {str(e)}", ""

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
