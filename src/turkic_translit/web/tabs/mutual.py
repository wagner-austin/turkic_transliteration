from __future__ import annotations

from collections.abc import Generator
from typing import Any

import gradio as gr

from turkic_translit.web.web_utils import get_ui_log_handler


def _do_mutual(
    model_a: str,
    model_b: str,
    eval_lang: str,
    sample: int,
    metric: str,
    *,
    progress: gr.Progress | None = None,
) -> Generator[tuple[Any, str], None, None]:
    import time

    # Heavy imports inside the function to avoid slowing initial UI load
    from turkic_translit.lm import (
        DatasetStream,
        LMModel,
        centred_cosine_matrix,
        cross_perplexity,
    )

    try:
        t0 = time.perf_counter()
        if progress is None:
            progress = gr.Progress(track_tqdm=True)

        yield gr.update(value=""), "**loading models...**\n"

        progress(0.00, desc="loading A")
        lm_a = LMModel.from_pretrained(model_a)
        progress(0.10, desc="loading B")
        lm_b = LMModel.from_pretrained(model_b)

        progress(0.15, desc=f"streaming {sample} sentences")
        ds = DatasetStream("oscar-2301", eval_lang, max_sentences=sample)
        sentences = list(ds)

        logs = get_ui_log_handler().dump()
        separator = (
            '<hr style="margin: 0.7em 0; border-top: 1px dashed #ccc;">' if logs else ""
        )
        yield (
            gr.update(),
            (logs + separator if logs else "")
            + '<p style="margin: 0.5em 0; padding: 0.3em 0; font-weight: bold;">encoding / evaluating...</p>',
        )

        if metric == "Cosine":
            sim = centred_cosine_matrix(lm_a, lm_b, sentences)
            interpretation = (
                "The models are extremely similar."
                if sim >= 0.9
                else "The models are very similar."
                if sim >= 0.7
                else "The models have moderate similarity."
                if sim >= 0.5
                else "The models have some differences."
                if sim >= 0.3
                else "The models have significant differences."
            )
            res = f"""### Model Similarity Results

**Cosine Similarity Score:** {sim:.3f}

**What this means:** {interpretation}
"""
        else:

            def _filter_by_maxlen(tok: Any, snts: list[str]) -> list[str]:
                mx: int = getattr(tok, "model_max_length", 1024) or 1024
                out: list[str] = []
                for s in snts:
                    if len(tok(s, truncation=False)["input_ids"]) <= mx:
                        out.append(s)
                return out

            # Filter sentences separately for each model using its tokenizer
            sent_a = _filter_by_maxlen(lm_a.tokenizer, sentences)
            sent_b = _filter_by_maxlen(lm_b.tokenizer, sentences)

            progress(0.80, desc="computing PPL A→lang")
            ppl_a = cross_perplexity(lm_a, sent_a)
            progress(0.90, desc="computing PPL B→lang")
            ppl_b = cross_perplexity(lm_b, sent_b)
            res = f"""### Perplexity Results

**Model A PPL:** {ppl_a:.2f}  |  **Model B PPL:** {ppl_b:.2f}

Lower is better. Significant differences (>10%) indicate one model understands the language better.
"""

        extra = get_ui_log_handler().dump()
        final = (
            f'<p style="margin: 0.5em 0;">done in {(time.perf_counter() - t0):.1f}s</p>'
        )
        separator = (
            '<hr style="margin: 0.7em 0; border-top: 1px dashed #ccc;">'
            if extra
            else ""
        )
        yield res, (extra + separator + final) if extra else final
    except Exception as exc:  # pragma: no cover
        yield (f"Error: {exc}", "*see backend logs for traceback*")


def register() -> None:
    with gr.Column():
        gr.Markdown(
            """
            ### Mutual Intelligibility Analysis

            Compare how similar two language models are in their understanding of a specific Turkic language.
            """
        )

        iso_choices = ["az", "ba", "kk", "ky", "tr", "tk", "uz"]

        with gr.Row():
            with gr.Column(scale=3):
                model_a = gr.Textbox(
                    label="Model A (path or repo)",
                    placeholder="sshleifer/tiny-gpt2",
                    value="sshleifer/tiny-gpt2",
                )
                model_b = gr.Textbox(
                    label="Model B (path or repo)",
                    placeholder="sshleifer/tiny-gpt2",
                    value="sshleifer/tiny-gpt2",
                )
                eval_lang = gr.Dropdown(
                    choices=iso_choices, value="kk", label="Evaluation Language"
                )
                sample = gr.Slider(
                    minimum=100, maximum=1000, value=100, step=100, label="Sample Size"
                )
                metric = gr.Radio(
                    ["Cosine", "Perplexity"], label="Similarity Metric", value="Cosine"
                )

            with gr.Column(scale=7):
                gr.Markdown("""<h4>Results</h4>""")
                output = gr.Markdown()
                gr.Markdown("""<h5>Processing Log</h5>""")
                live_log = gr.Markdown(
                    value="<p style=\"margin: 1em 0; font-style: italic; color: #666;\">Click 'Compare Models' to start analysis...</p>",
                    elem_id="mutual-log",
                    show_label=False,
                    height=300,
                )

        with gr.Row(elem_classes=["examples-row"]):
            gr.Button("Compare Models", variant="primary").click(
                _do_mutual,
                [model_a, model_b, eval_lang, sample, metric],
                [output, live_log],
            )
