# ruff: noqa: E402
from __future__ import annotations

from collections.abc import Generator
from functools import cache, lru_cache
from typing import Any, cast

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
import os
import pathlib
import re  # NEW – used by _to_md_table
import time

from turkic_translit.lang_utils import pretty_lang  # unified helper


def _labelise(codes: list[str]) -> list[tuple[str, str]]:
    """Return (label, value) pairs for Gradio dropdown from ISO codes."""
    return [(pretty_lang(c), c) for c in codes]


import gradio as gr

from turkic_translit.web.web_utils import (
    direct_transliterate,
    download_corpus_to_file,  # NEW
    mask_russian,
    median_levenshtein,
    pipeline_transliterate,
    token_table_markdown,
    train_sentencepiece_model,
)

# Configure logging once. Honour $PYTHONLOGLEVEL if provided (DEBUG, INFO …).
_logger = logging.getLogger("turkic_translit.web_demo")
if not _logger.handlers:
    _lvl_str = os.environ.get("PYTHONLOGLEVEL", "INFO").upper()
    _lvl = getattr(logging, _lvl_str, logging.INFO)
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=_lvl)
    _logger.setLevel(_lvl)


# ── Gradio-aware log handler ───────────────────────────────────────────────
class GradioLogHandler(logging.Handler):
    """Buffers log records so UI callbacks can flush them into the browser."""

    def __init__(self) -> None:
        super().__init__()
        self.buffer: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(self.format(record))

    def dump(self) -> str:
        # Format each log line as a separate paragraph with line breaks for better readability
        formatted_lines = [
            f'<p style="margin: 0.5em 0; padding: 0.3em 0;">{line}</p>'
            for line in self.buffer
        ]
        out, self.buffer = "\n".join(formatted_lines), []
        return out


# ── model presence check helper ────────────────────────────────────────────


def _model_check() -> tuple[str, str]:
    """Verify auxiliary model files exist; attempt download when missing.

    Returns (warning_markdown, fasttext_info_markdown).  *warning_markdown* is an
    empty string when everything is fine; otherwise it lists missing items so the
    UI can surface a friendly notice on load.
    """
    missing: list[str] = []
    fasttext_info = ""

    # Try to download and check the FastText model via utility (preferred path).
    try:
        from ..model_utils import ensure_fasttext_model

        model_path = ensure_fasttext_model()
        model_name = model_path.name
        size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)
        model_type = "Full" if model_name.endswith(".bin") else "Compressed"
        fasttext_info = (
            f"**FastText Language Model:** {model_name} ({model_type}, {size_mb} MB)"
        )
        logging.info("FastText language identification model found at %s", model_path)
    except Exception as exc:  # noqa: BLE001
        # Fallback: check common locations if automatic download failed.
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
            fasttext_info = f"**FastText Language Model:** {model_name} ({model_type}, {size_mb} MB)"
        else:
            msg = f"- FastText language model not found and auto-download failed: {exc}"
            logging.warning(msg)
            missing.append(msg)
            fasttext_info = "**FastText Language Model:** Not found"

    # Check SentencePiece model bundled with tokenizer for direct transliteration.
    try:
        import turkic_translit.tokenizer as tokenizer

        tok_dir = pathlib.Path(tokenizer.__file__).parent
        spm_model = tok_dir / "turkic_model.model"
        if not spm_model.exists():
            msg = f"- `turkic_model.model` not found in {spm_model}"
            logging.warning(msg)
            missing.append(msg)
    except Exception as exc:  # noqa: BLE001
        msg = f"- Could not verify `turkic_model.model`: {exc}"
        logging.warning(msg)
        missing.append(msg)

    warning_md = ""
    if missing:
        warning_md = (
            "**⚠️ Model file(s) missing:**\n"
            + "\n".join(missing)
            + "\nPlease ensure all required models are present for full functionality."
        )

    return warning_md, fasttext_info


# ── shared regex helpers for RU-filter debug table ──────────────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")  # strips any ANSI colour codes
_DBG_ROW = re.compile(
    r"Token:\s*(?P<tok>.*?),\s*Label:\s*__label__(?P<lbl>\w+),"
    r"\s*Confidence:\s*(?P<conf>[0-9.]+)"
)


def _to_md_table(debug_lines: list[str]) -> str:
    """Convert raw debug lines from `mask_russian` into a Markdown table.

    The function removes ANSI escapes, extracts the first-rank language and
    confidence, and returns GitHub-flavoured Markdown.  If no parsable rows
    are found it emits a graceful fallback string, so callers never need to
    special-case empty output.
    """
    rows = [
        f"| {m['tok']} | {m['lbl']} | {float(m['conf']):.3f} |"
        for ln in debug_lines
        if (clean_ln := _ANSI_RE.sub("", ln))  # ANSI scrub (in one place)
        if (m := _DBG_ROW.search(clean_ln))  # pattern match
    ]
    if not rows:
        return "*— no tokens found —*"
    header = "| Token | Lang | Conf |\n|-------|------|------|\n"
    return header + "\n".join(rows)


# ────────────────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    """Build and return the Turkic Transliteration Suite UI as a Gradio Blocks app."""
    # Run model checks and get information
    warning_message, fasttext_model_info = _model_check()

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
        "corpus": [["oscar-2301", "kk", 100, True]],
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
        # Add header with model information
        gr.Markdown(
            """
            # 🌍 Turkic Transliteration Suite
            ## Web Interface for exploring Turkic language transliteration tools
            """
        )

        # FastText model info now only shown in Filter Russian tab

        # Show warning if any models are missing
        if warning_message:
            gr.Markdown(warning_message)

        gr.Markdown(
            """
            Explore transliteration capabilities for Turkic languages between Cyrillic, Latin, and IPA representations.
            Navigate through the tabs below to access different features.
            """
        )

        # Warning message already displayed above if it exists

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

        def do_corpus_download(
            source: str,
            lang: str,
            max_lines: int | None,
            filter_flag: bool,
            conf_thr: float,
            progress: gr.Progress | None = None,
        ) -> tuple[str, str | None]:  # returns markdown + optional file path
            """Download corpus via helper and return (info_markdown, file_path)."""
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
                return info, path
            except Exception as exc:  # noqa: BLE001
                return f"**Error:** {exc}", None

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

        def do_mask(text: str, threshold: float, min_len: int) -> tuple[str, str]:
            """
            Mask Russian tokens and return a tuple of
            (masked_text, markdown_table_of_confidences).

            The helper always calls `mask_russian` with `debug=True`, so the table
            can be rebuilt on every invocation without an extra code-path.
            """
            # Empty/whitespace input → blank outputs, no error dialogs in the UI
            if not text.strip():
                return "", ""

            try:
                # Request debug markup unconditionally
                raw = mask_russian(text, threshold, min_len, debug=True)

                # If the debug blob is missing, supply only the masked string
                if "<!--debug " not in raw:
                    return raw.strip(), ""

                masked, dbg_blob = raw.split("<!--debug ", 1)
                dbg_blob = dbg_blob.rsplit(" -->", 1)[0]  # strip trailing marker

                # Parse the JSON blob
                import json

                try:
                    debug_data = json.loads(dbg_blob)
                    # Convert JSON format to the expected text format for _to_md_table
                    debug_lines = [
                        f"Token: {item['tok']}, Label: __label__{item['winner']}, Confidence: {item['conf']}"
                        for item in debug_data
                    ]
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    debug_lines = dbg_blob.splitlines()

                table_md = _to_md_table(debug_lines)

                return masked.strip(), table_md
            except Exception as e:
                return f"**Error masking Russian text**: {str(e)}", ""

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

        def _do_mutual(  # keep public name unchanged
            model_a: str,
            model_b: str,
            eval_lang: str,
            sample: int,
            metric: str,
            *,  # keyword-only from here ↓
            progress: gr.Progress | None = None,
        ) -> Generator[tuple[Any, str], None, None]:
            """Compute mutual-intelligibility metrics *and* stream updates.

            Yields
            ------
            (first-output, log-markdown) tuples, where *first-output* is either a
            `gr.update()` sentinel or the final Markdown string with the metric.
            """
            # Lazily import heavy LM utilities inside the function to avoid slowing the
            # initial UI load when the tab is not used.
            import gradio as gr  # local import so CLI usage doesn't depend on gradio

            from turkic_translit.lm import (
                DatasetStream,
                LMModel,
                centred_cosine_matrix,
                cross_perplexity,
            )

            try:
                t0 = time.perf_counter()

                # Initialise progress helper lazily to avoid mutable default (ruff B008)
                if progress is None:
                    progress = gr.Progress(track_tqdm=True)

                # Immediately tell UI we started
                yield gr.update(value=""), "⏳ **loading models …**\n"

                progress(0.00, desc="loading A")
                lm_a = LMModel.from_pretrained(model_a)

                progress(0.10, desc="loading B")
                lm_b = LMModel.from_pretrained(model_b)

                progress(0.15, desc=f"streaming {sample} sentences")
                ds = DatasetStream("oscar-2301", eval_lang, max_sentences=sample)
                sentences = list(ds)  # tqdm captured automatically

                # Flush any backend INFO lines collected so far
                logs = _ui_log_handler.dump()
                # Add a visual separator between log sections
                separator = (
                    '<hr style="margin: 0.7em 0; border-top: 1px dashed #ccc;">'
                    if logs
                    else ""
                )
                yield (
                    gr.update(),
                    (logs + separator if logs else "")
                    + '<p style="margin: 0.5em 0; padding: 0.3em 0; font-weight: bold;">⏳ encoding / evaluating...</p>',
                )

                if metric == "Cosine":
                    sim = centred_cosine_matrix(lm_a, lm_b, sentences)
                    interpretation = ""
                    if sim >= 0.9:
                        interpretation = "The models are extremely similar in how they understand this language."
                    elif sim >= 0.7:
                        interpretation = (
                            "The models are very similar in their understanding."
                        )
                    elif sim >= 0.5:
                        interpretation = "The models have moderate similarity."
                    elif sim >= 0.3:
                        interpretation = (
                            "The models have some differences in their understanding."
                        )
                    else:
                        interpretation = "The models have significant differences in how they understand this language."

                    res = f"""### Model Similarity Results

**Cosine Similarity Score:** {sim:.3f}

**What this means:** {interpretation}

**About this measurement:** Cosine similarity measures how similar the two models are in their representation of language. Higher values (closer to 1.0) indicate greater similarity between models."""
                else:

                    def _fit(tok: Any, snts: list[str]) -> list[str]:
                        """Return only sentences that fit *tok* context window."""
                        mx: int = getattr(tok, "model_max_length", 1024) or 1024
                        return [
                            s
                            for s in snts
                            if len(tok(s, truncation=False)["input_ids"]) <= mx
                        ]

                    progress(0.80, desc="computing PPL A→lang")
                    ppl_a = cross_perplexity(lm_a, _fit(lm_a.tokenizer, sentences))
                    progress(0.90, desc="computing PPL B→lang")
                    ppl_b = cross_perplexity(lm_b, _fit(lm_b.tokenizer, sentences))
                    comparison = ""
                    if ppl_a < ppl_b:
                        diff_percent = ((ppl_b - ppl_a) / ppl_b) * 100
                        comparison = f"Model A has {diff_percent:.1f}% lower perplexity, suggesting it understands {eval_lang} better than Model B."
                    elif ppl_b < ppl_a:
                        diff_percent = ((ppl_a - ppl_b) / ppl_a) * 100
                        comparison = f"Model B has {diff_percent:.1f}% lower perplexity, suggesting it understands {eval_lang} better than Model A."
                    else:
                        comparison = "Both models have identical perplexity scores, suggesting they understand the language equally well."

                    res = f"""### Language Understanding Results

**Model A (*{model_a}*) perplexity:** {ppl_a:.1f}

**Model B (*{model_b}*) perplexity:** {ppl_b:.1f}

**What this means:** {comparison}

**About this measurement:** Perplexity measures how well a model understands a language. Lower values indicate better understanding - the model is less "confused" by the text."""

                progress(1.00, desc="done")
                dt = time.perf_counter() - t0
                final = f'<p style="margin: 0.5em 0; padding: 0.5em; background-color: #f0f8ff; border-radius: 4px;">✅ <strong>Done in {dt:,.1f}s</strong></p>'
                extra = _ui_log_handler.dump()
                # Add a visual separator between logs and completion message
                separator = (
                    '<hr style="margin: 0.7em 0; border-top: 1px dashed #ccc;">'
                    if extra
                    else ""
                )
                yield res, (extra + separator + final if extra else final)

            except Exception as exc:  # pragma: no cover
                logging.exception("Mutual intelligibility computation failed: %s", exc)
                yield (f"⚠️ Error: {exc}", "*see backend logs for traceback*")

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

        def _corpus_tab() -> None:
            with gr.Column():
                gr.Markdown(
                    """
                    <div class="feature-description">
                    <strong>Corpus Downloader:</strong> Stream sentences from public corpora (OSCAR or Wikipedia) directly in the browser. Select a source and language, optionally cap the number of sentences, and decide whether to filter by FastText language ID.
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        from turkic_translit.cli.download_corpus import (
                            _REG,  # dynamic registry
                        )

                        source_dd = gr.Dropdown(
                            choices=sorted(
                                k for k in _REG if _REG[k]["driver"] != "leipzig"
                            ),
                            label="Corpus Source",
                            value="oscar-2301",
                        )

                        # FastText supported ISO codes – cached after first call
                        @lru_cache(maxsize=1)
                        def _fasttext_langs() -> set[str]:
                            """Return set of ISO codes recognised by lid.176 FastText model."""
                            from turkic_translit.langid import FastTextLangID

                            mdl = FastTextLangID()
                            return {
                                lab.replace("__label__", "")
                                for lab in mdl.model.get_labels()
                            }

                        # Language choices depend on corpus source. Helper fetches valid ISO codes.
                        @cache
                        def _lang_choices(src: str) -> list[str]:
                            """Return available ISO codes for *src* corpus."""
                            from turkic_translit.cli import download_corpus as _dl

                            cfg = _dl._REG[src]
                            lst: list[str] = []
                            try:
                                if cfg["driver"] == "oscar":
                                    import os

                                    from datasets import (
                                        get_dataset_config_names,  # heavy import
                                    )

                                    token = os.getenv("HF_TOKEN")
                                    logging.info(
                                        f"Attempting to fetch OSCAR configs with token: {'present' if token else 'missing'}"
                                    )
                                    lst = sorted(
                                        get_dataset_config_names(
                                            cfg["hf_name"], token=token, trust_remote_code=True
                                        )
                                    )
                                    logging.info(
                                        f"Successfully fetched {len(lst)} OSCAR configs."
                                    )
                                elif cfg["driver"] == "wikipedia":
                                    try:
                                        from turkic_translit.cli.download_corpus import (
                                            _wikipedia_lang_codes_from_sitematrix as _site,
                                        )

                                        lst = _site()
                                    except Exception:
                                        logging.exception(
                                            "Failed to fetch Wikipedia lang codes"
                                        )
                                        lst = []
                            except Exception:  # noqa: BLE001 – network / import errors
                                logging.exception(
                                    f"Failed to fetch language choices for source: {src}"
                                )
                                lst = []

                            if not lst:
                                logging.warning(
                                    f"Language list for {src} is empty, falling back to hardcoded list."
                                )
                                lst = [
                                    c
                                    for c in ["en", "tr", "az", "uz", "kk", "ru"]
                                    if c in _fasttext_langs()
                                ]

                            # Keep only languages FastText can ID
                            ft = _fasttext_langs()
                            return [code for code in lst if code in ft]

                        initial_langs = _lang_choices(
                            "oscar-2301"
                        )  # default source before change
                        lang_dd = gr.Dropdown(
                            choices=_labelise(initial_langs),
                            value=initial_langs[0] if initial_langs else None,
                            label="Language",
                        )

                        # Dynamically refresh language choices when the corpus source changes
                        def _update_langs(selected_src: str) -> Any:
                            langs = _lang_choices(selected_src)
                            return gr.update(
                                choices=_labelise(langs),
                                value=langs[0] if langs else None,
                            )

                        source_dd.change(
                            _update_langs,
                            inputs=[source_dd],
                            outputs=[lang_dd],
                        )
                        max_lines_num = gr.Number(
                            label="Max Sentences (empty = all)", value=10, precision=0
                        )
                        filter_cb = gr.Checkbox(
                            label="Filter by FastText LangID",
                            value=True,
                            info=(
                                "Keep only sentences whose FastText language-ID "
                                "matches the code above (uses lid.176 model)."
                            ),
                        )
                        conf_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.95,
                            label="Min Lang-ID confidence",
                            info="Discard sentences below this probability threshold when filtering.",
                        )
                        download_btn = gr.Button("Download", variant="primary")
                    with gr.Column(scale=1):
                        info_md = gr.Markdown()
                        file_out = gr.File(label="Corpus File")

                # Hook up button
                download_btn.click(
                    do_corpus_download,
                    [source_dd, lang_dd, max_lines_num, filter_cb, conf_slider],
                    outputs=[info_md, file_out],
                )

                # Examples
                gr.Examples(
                    examples["corpus"],
                    inputs=[source_dd, lang_dd, max_lines_num, filter_cb, conf_slider],
                    outputs=[info_md, file_out],
                    fn=do_corpus_download,
                    label="Try this example",
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

                # Show simplified FastText model info only on this tab, integrated into the description
                gr.Markdown(
                    """
                    <div style="margin-bottom: 10px; font-size: 0.9em; color: #228B22;">
                    ✅ using the {model_type} model: {model_name} for language ID
                    </div>
                    """.format(
                        model_type=(
                            "Full" if "Full" in fasttext_model_info else "Compressed"
                        ),
                        model_name=(
                            "lid.176.bin"
                            if "bin" in fasttext_model_info
                            else "lid.176.ftz"
                        ),
                    )
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        threshold = gr.Slider(
                            0,
                            1,
                            value=0.5,  # Match the CLI default in filter_russian.py
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

                        # ✱✱ NEW component – confidence table ✱✱
                        debug_tbl = gr.Markdown(
                            label="Token confidences (top-1)",
                            value="",
                            elem_id="ru-debug-table",
                        )

                with gr.Row(elem_classes=["examples-row"]):
                    gr.Examples(
                        examples=[
                            ["қазақша текст с русскими словами", 0.5, 3],
                            ["Все это написано на русском языке", 0.5, 3],
                            ["Бұл мәтін толығымен қазақ тілінде жазылған", 0.5, 3],
                            ["қазақша и русский в одном тексте", 0.5, 2],
                            ["Көптеген орыс сөздер арасында казахский текст", 0.6, 4],
                        ],
                        inputs=[shared_textbox, threshold, min_len],
                        outputs=[output, debug_tbl],
                        fn=do_mask,
                        label="Try these examples",
                    )
                    btn = gr.Button("Mask Russian", variant="primary")

            # Button click for masking Russian
            btn.click(
                do_mask, [shared_textbox, threshold, min_len], [output, debug_tbl]
            )

            # Real-time masking as you type
            shared_textbox.change(
                do_mask, [shared_textbox, threshold, min_len], [output, debug_tbl]
            )

            # Update when parameters change
            threshold.change(
                do_mask, [shared_textbox, threshold, min_len], [output, debug_tbl]
            )
            min_len.change(
                do_mask, [shared_textbox, threshold, min_len], [output, debug_tbl]
            )

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

        def _mutual_tab() -> None:
            with gr.Column():
                gr.Markdown(
                    """
                    <div class="feature-description">
                    <h3>🤝 Mutual Intelligibility Analysis</h3>
                    <p>Compare how similar two language models are in their understanding of a specific Turkic language.</p>
                    </div>
                    <div class="feature-explanation">
                    <p>This tool measures how similarly two different language models understand and process the same language. 
                    You can compare any two models to see if they would be mutually intelligible in real-world applications.</p>
                    
                    <h4>How It Works:</h4>
                    <ul>
                        <li><strong>Select two models</strong> to compare (pre-trained from Hugging Face or local models)</li>
                        <li><strong>Choose a Turkic language</strong> for evaluation</li>
                        <li><strong>Select a similarity metric</strong> (see details below)</li>
                        <li><strong>Adjust the sample size</strong> for the comparison (larger samples provide more accurate results but take longer)</li>
                    </ul>
                    </div>
                    """
                )

                # Predefined Oscar-2301 ISO codes likely to be interesting in Turkic context
                iso_choices = [
                    "az",
                    "ba",
                    "kk",
                    "ky",
                    "tr",
                    "tk",
                    "uz",
                ]

                with gr.Row():
                    with gr.Column(scale=3):
                        model_a = gr.Textbox(
                            label="Model A (path or repo)",
                            placeholder="sshleifer/tiny-gpt2 or /path/to/local/model",
                            value="sshleifer/tiny-gpt2",
                        )
                        model_b = gr.Textbox(
                            label="Model B (path or repo)",
                            placeholder="sshleifer/tiny-gpt2 or /path/to/local/model",
                            value="sshleifer/tiny-gpt2",
                        )
                        eval_lang = gr.Dropdown(
                            choices=iso_choices,
                            value="kk",
                            label="Evaluation Language",
                        )
                        sample = gr.Slider(
                            minimum=100,
                            maximum=1000,
                            value=100,
                            step=100,
                            label="Sample Size",
                        )
                        metric = gr.Radio(
                            ["Cosine", "Perplexity"],
                            label="Similarity Metric",
                            value="Cosine",
                            info="Cosine measures overall similarity between models. Perplexity shows how well each model understands the language.",
                        )

                    with gr.Column(scale=7):
                        gr.Markdown("""<h4>📈 Results</h4>""")
                        output = gr.Markdown()
                        gr.Markdown("""<h5>Processing Log</h5>""")
                        live_log = gr.Markdown(
                            value="<p style=\"margin: 1em 0; font-style: italic; color: #666;\">Click 'Compare Models' to start analysis...</p>",
                            elem_id="mutual-log",
                            show_label=False,
                            height=300,  # Increased height for better readability
                        )

                with gr.Row(elem_classes=["examples-row"]):
                    gr.Button("Compare Models", variant="primary").click(
                        _do_mutual,
                        [model_a, model_b, eval_lang, sample, metric],
                        [output, live_log],
                    )

                # Add explanatory text below the main interface
                gr.Markdown(
                    """
                    <div class="metrics-explanation">
                    <h4>About the Similarity Metrics:</h4>
                    
                    <details>
                        <summary><strong>Cosine Similarity</strong> - How similarly do the models represent language?</summary>
                        <p>Cosine similarity measures how closely aligned the internal representations of language are between two models. 
                        It tells you if the models "think" about language in the same way.</p>
                        <ul>
                            <li><strong>Values range from 0 to 1</strong>, where higher values mean greater similarity</li>
                            <li><strong>Above 0.9:</strong> Models are practically identical in how they understand the language</li>
                            <li><strong>0.7-0.9:</strong> Very high similarity; models likely have similar training data or architecture</li>
                            <li><strong>0.5-0.7:</strong> Moderate similarity; models share some common understanding</li>
                            <li><strong>Below 0.5:</strong> Low similarity; models have significant differences</li>
                        </ul>
                    </details>
                    
                    <details>
                        <summary><strong>Perplexity</strong> - How well does each model understand the language?</summary>
                        <p>Perplexity measures how "confused" a model is by new text in a specific language. Lower perplexity means
                        the model is less surprised by the text and therefore understands the language better.</p>
                        <ul>
                            <li>The tool measures perplexity for each model separately against the same language samples</li>
                            <li>The model with <strong>lower perplexity</strong> has a better understanding of the language</li>
                            <li>Significant differences in perplexity (>10%) indicate one model has a substantially better grasp of the language</li>
                            <li>This is useful for determining which model would be more effective for applications in that language</li>
                        </ul>
                    </details>
                    </div>
                    """
                )

        with gr.Tabs():
            # Order tabs by the logical workflow: training → sanity-check → processing → cleanup → ad-hoc → evaluation
            with gr.Tab("📥 Download Corpus", id="corpus"):
                _corpus_tab()
            with gr.Tab("🧩 Train Tokenizer", id="sentencepiece"):
                _sentencepiece_tab()
            with gr.Tab("🔍 ID Token Language", id="tokens"):
                _tokens_tab()
            # The Pipeline tab is temporarily disabled per user request
            # with gr.Tab("🔄 Lang ID, Tokenize, Transliterate", id="pipeline"):
            #     _pipeline_tab()
            with gr.Tab("🎭 Filter Russian", id="filter_ru"):
                _filter_ru_tab()
            with gr.Tab("📝 Transliterate to IPA/Latin", id="direct"):
                _direct_tab()
            with gr.Tab("📊 Levenshtein Distance", id="compare"):
                _compare_tab()
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


class _PrettyLogFilter(logging.Filter):
    """Skip verbose HTTP and backend housekeeping messages for UI logs."""

    _SKIP_PHRASES = (
        "HTTP Request:",
        "turkic_model.model not found",
    )

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 – simple verb
        msg = record.getMessage()
        return not any(p in msg for p in self._SKIP_PHRASES)


_ui_log_handler = GradioLogHandler()
_ui_log_handler.setFormatter(logging.Formatter("%(message)s"))
_ui_log_handler.addFilter(_PrettyLogFilter())

# Attach only to our own package logger to avoid external noise
logging.getLogger("turkic_translit").addHandler(_ui_log_handler)


def main() -> None:
    """Launch the Turkic Transliteration web interface with default settings.

    This function serves as an entry point for the web UI and can be called
    directly or through the CLI entry point 'turkic-web'.
    """
    ui = build_ui()
    ui.queue().launch()


if __name__ == "__main__":
    main()
