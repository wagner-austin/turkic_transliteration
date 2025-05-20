# ruff: noqa: E402
from __future__ import annotations

"""
A simple web interface to demonstrate the Turkish transliteration.
"""
import logging
import pathlib
from typing import Any, Tuple

import gradio as gr

from turkic_translit.web_utils import (
    direct_transliterate,
    pipeline_transliterate,
    token_table_markdown,
    mask_russian,
    median_levenshtein,
)

logging.basicConfig(level=logging.INFO)


def _model_check() -> str:
    """
    Check for required model files. If any are missing, return a markdown warning string; else return an empty string.
    """
    missing = []
    # Check lid.176.ftz in home and project root
    home_lid = pathlib.Path.home() / "lid.176.ftz"
    root_lid = pathlib.Path(__file__).parent / "lid.176.ftz"
    if not home_lid.exists() and not root_lid.exists():
        msg = f"- `lid.176.ftz` not found in {home_lid} or {root_lid}"
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
    with gr.Blocks(title="Turkic Transliteration Suite") as app:
        gr.Markdown(_model_check())
        shared_textbox = gr.Textbox(label="Input Text", lines=4, elem_id="input-text")
        shared_textbox.render()  # type: ignore[no-untyped-call]

        def _direct_tab() -> None:
            lang = gr.Radio(["kk", "ky"], label="Language", value="kk")
            output_format = gr.Radio(
                ["Latin", "IPA"], label="Output Format", value="Latin"
            )
            include_arabic = gr.Checkbox(
                False, label="Also transliterate Arabic script (Latin mode only)"
            )
            btn = gr.Button("Transliterate")
            output = gr.Textbox(label="Output", lines=4, interactive=False)
            stats = gr.Markdown()

            def do_direct(
                text: str, lang: str, fmt: str, include_arabic: bool
            ) -> Tuple[str, str]:
                result, stats = direct_transliterate(
                    text, lang, include_arabic, fmt.lower()
                )
                return result, stats

            btn.click(
                do_direct,
                [shared_textbox, lang, output_format, include_arabic],
                [output, stats],
            )

        def _pipeline_tab() -> None:
            mode = gr.Radio(["Latin", "IPA"], label="Pipeline Mode", value="Latin")
            btn = gr.Button("Pipeline Transliterate")
            output = gr.Textbox(label="Output", lines=4, interactive=False)
            stats = gr.Markdown()

            def do_pipeline(text: str, mode: str) -> Tuple[str, str]:
                result, stats = pipeline_transliterate(text, mode.lower())
                return result, stats

            btn.click(do_pipeline, [shared_textbox, mode], [output, stats])

        def _tokens_tab() -> None:
            token_md = gr.Markdown()

            def do_tokens(text: str) -> str:
                try:
                    return token_table_markdown(text)
                except ImportError as e:
                    return f"**pandas missing – install `turkic-transliterate[ui]`**\n\n{e}"

            shared_textbox.blur(do_tokens, [shared_textbox], token_md)

        def _filter_ru_tab() -> None:
            threshold = gr.Slider(0, 1, value=0.5, label="RU Mask Threshold", step=0.01)
            min_len = gr.Slider(1, 10, value=3, label="Min Token Length", step=1)
            btn = gr.Button("Mask Russian")
            output = gr.Textbox(label="Masked Output", lines=4, interactive=False)

            def do_mask(text: str, threshold: float, min_len: int) -> str:
                return mask_russian(text, threshold, min_len)

            btn.click(do_mask, [shared_textbox, threshold, min_len], output)

        def _compare_tab() -> None:
            file_lat = gr.File(label="Latin File", file_types=["text"])
            file_ipa = gr.File(label="IPA File", file_types=["text"])
            sample = gr.Number(label="Sample Size (optional)", precision=0)
            out = gr.Markdown()
            btn = gr.Button("Compare")

            def do_compare(lat_file: Any, ipa_file: Any, sample_n: Any) -> str:
                if lat_file is None or ipa_file is None:
                    return "Please upload both files."
                lat_obj = type("FileObj", (), {"name": lat_file.name})
                ipa_obj = type("FileObj", (), {"name": ipa_file.name})
                sample_val = int(sample_n) if str(sample_n).strip() else None
                return median_levenshtein(lat_obj, ipa_obj, sample_val)

            btn.click(do_compare, [file_lat, file_ipa, sample], out)

        with gr.Tabs():
            with gr.Tab("Direct"):
                _direct_tab()
            with gr.Tab("Pipeline"):
                _pipeline_tab()
            with gr.Tab("Tokens"):
                _tokens_tab()
            with gr.Tab("Filter RU"):
                _filter_ru_tab()
            with gr.Tab("Compare"):
                _compare_tab()
    return app  # type: ignore[no-any-return]


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=False)
