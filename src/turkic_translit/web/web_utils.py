from __future__ import annotations

import functools
import json
import logging
import os
import tempfile
import threading
import time
import typing as t
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from turkic_translit.lang_utils import pretty_lang

from ..error_service import (
    error_markdown,
    error_response,
    set_correlation_id,
    set_request_context,
)

# Gradio is optional at runtime but needed for type hints
try:
    import gradio as gr  # type-only; not required outside web UI
except ModuleNotFoundError:  # pragma: no cover
    import typing as _t

    gr = _t.cast(_t.Any, None)

from ..lang_filter import is_russian_token
from ..langid import FastTextLangID

log = logging.getLogger(__name__)

# Directory for temporary corpus downloads – excluded from VCS via .gitignore

_CRON_DIR = Path(os.getenv("TURKIC_CRON_DIR", Path.cwd() / "cronjob"))
_CRON_DIR.mkdir(parents=True, exist_ok=True)


# Start a background janitor thread to purge files older than 10 min.
def _start_janitor(max_age_sec: int = 600) -> None:
    def _janitor() -> None:
        while True:
            try:
                cutoff = time.time() - max_age_sec
                for fp in _CRON_DIR.glob("*"):
                    try:
                        if fp.stat().st_mtime < cutoff:
                            fp.unlink(missing_ok=True)
                    except FileNotFoundError:
                        pass
                time.sleep(max_age_sec)  # run roughly every *max_age_sec*
            except Exception as e:  # pragma: no cover – background safety
                log.warning(f"Janitor error: {e}")
                time.sleep(60)

    th = threading.Thread(target=_janitor, daemon=True, name="cron-janitor")
    th.start()


_start_janitor()


def labelise(codes: list[str]) -> list[tuple[str, str]]:
    """Return (label, value) pairs for Gradio dropdown from ISO codes."""
    return [(pretty_lang(c), c) for c in codes]


class GradioLogHandler(logging.Handler):
    """Buffers log records so UI callbacks can flush them into the browser."""

    def __init__(self) -> None:
        super().__init__()
        self.buffer: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(self.format(record))

    def dump(self) -> str:
        out, self.buffer = "\n".join(self.buffer), []
        return out


class UiPrettyLogFilter(logging.Filter):
    """Skip verbose HTTP and housekeeping messages for UI logs."""

    _SKIP_PHRASES = (
        "HTTP Request:",
        "turkic_model.model not found",
    )

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        msg = record.getMessage()
        return not any(p in msg for p in self._SKIP_PHRASES)


_UI_LOG_HANDLER: GradioLogHandler | None = None


def get_ui_log_handler() -> GradioLogHandler:
    """Return shared UI log handler attached to the package logger."""
    global _UI_LOG_HANDLER
    if _UI_LOG_HANDLER is None:
        h = GradioLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        h.addFilter(UiPrettyLogFilter())
        logging.getLogger("turkic_translit").addHandler(h)
        _UI_LOG_HANDLER = h
    return _UI_LOG_HANDLER


if TYPE_CHECKING:  # for static checkers only
    from ..pipeline import TurkicTransliterationPipeline  # Import from main package


def _make_pipeline() -> TurkicTransliterationPipeline:
    from ..pipeline import TurkicTransliterationPipeline  # Import from main package

    log.info("Instantiating TurkicTransliterationPipeline singleton")
    return TurkicTransliterationPipeline()


_lazy_pipeline = functools.lru_cache(maxsize=1)(_make_pipeline)

# Create a singleton for the language ID model
_langid_singleton = functools.lru_cache(maxsize=1)(FastTextLangID)


def direct_transliterate(
    text: str, lang: str, include_arabic: bool, out_fmt: str
) -> tuple[str, str]:
    """
    Transliterate text directly to Latin or IPA.
    Usage: direct_transliterate('сәлем', 'kk', False, 'latin')
    Returns: (result, stats_markdown)
    Raises: ValueError if out_fmt is invalid.
    """
    from ..core import to_ipa, to_latin  # Import from main package

    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="direct_transliterate", lang=lang, out_fmt=out_fmt)

    fmt = out_fmt.lower()
    if fmt not in {"latin", "ipa"}:
        raise ValueError(f"out_fmt must be 'latin' or 'ipa', got {out_fmt!r}")
    if fmt == "latin":
        result = to_latin(text, lang, include_arabic)
    else:
        result = to_ipa(text, lang)
    stats_markdown = (
        f"**Bytes** — Cyrillic : {len(text.encode('utf8'))}, "
        f"{fmt.upper()} : {len(result.encode('utf8'))}"
    )
    return result, stats_markdown


def pipeline_transliterate(text: str, mode: str) -> tuple[str, str]:
    """
    Transliterate text using the pipeline (mode: 'latin' or 'ipa').
    Usage: pipeline_transliterate('сәлем', 'ipa')
    Returns: (result, stats_markdown)
    Raises: ValueError if mode is invalid (passed to pipeline).
    """
    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="pipeline_transliterate", mode=mode)

    mode = mode.lower()
    if mode not in {"latin", "ipa"}:
        raise ValueError(f"mode must be 'latin' or 'ipa', got {mode!r}")
    pipeline = _lazy_pipeline()
    pipeline.mode = mode
    result = pipeline.process(text)
    stats_markdown = f"**{len(result)} chars**"
    return result, stats_markdown


# Soft import pandas at module level

# Soft import pandas - using TYPE_CHECKING pattern to provide type hints
# without requiring pandas at import time
if TYPE_CHECKING:
    import pandas as pd
else:
    # Runtime pandas import with fallback
    try:
        import pandas as pd
    except ModuleNotFoundError:  # pragma: no cover
        pd: ModuleType | None = None


def token_table_markdown(text: str) -> str:
    """
    Tokenize text and return a markdown table of tokens and language predictions.
    Usage: token_table_markdown('сәлем әлем!')
    Returns: markdown string
    Raises: ImportError if pandas is not installed.
    """
    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="token_table", sample=len(text))

    if pd is None:
        raise ImportError(
            "install turkic-transliterate[ui] to use token_table_markdown"
        )

    try:
        pipeline = _lazy_pipeline()
        tokens = pipeline.tokenizer.tokenize(text)
        langs = pipeline.langid.predict_tokens(tokens)
        df = pd.DataFrame({"Token": tokens, "Lang": langs})
        markdown_table: str = df.to_markdown(index=False)
        return markdown_table
    except OSError as e:
        if "turkic_model.model" in str(e):
            return (
                "**⚠️ Tokenizer model file missing**\n\n"
                "The `turkic_model.model` file is required for tokenization.\n"
                "Please train a model with `turkic-build-spm` command\n"
                "and place it in the package directory."
            )
        # Other OSError not related to missing model
        return f"**Error loading tokenizer:** {e}"


def mask_russian(
    text: str, thr: float, min_len: int, *, margin: float = 0.10, debug: bool = False
) -> str:
    """
    Replace Russian tokens with <RU> using the shared heuristic.

    Args:
        text: Text to process
        thr: Confidence threshold for Russian detection
        min_len: Minimum token length to consider
        margin: Maximum margin for accepting RU when not the top label
        debug: Whether to include debug information in output

    Returns:
        Masked text with <RU> replacing Russian tokens
    """
    import re

    _ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="mask_russian", thr=thr, min_len=min_len)

    try:
        # Get model from singleton
        lid = _langid_singleton().model
        stoplist = None  # future hook – can come from UI later
        masked, dbg = [], []

        for tok in text.strip().split():
            ru = is_russian_token(
                tok, thr=thr, min_len=min_len, lid=lid, stoplist=stoplist, margin=margin
            )
            masked.append("<RU>" if ru else tok)

            if debug:
                # json-serialisable per-token info
                lbls, confs = lid.predict(tok.lower(), k=1)
                dbg.append(
                    {
                        "tok": tok,
                        "ru": ru,
                        "winner": lbls[0][9:],
                        "conf": float(confs[0]),
                    }
                )

        out = " ".join(masked)
        if debug:
            out += "\n\n<!--debug " + json.dumps(dbg, ensure_ascii=False) + " -->"
        return _ansi_re.sub("", out)

    except Exception as e:
        log.warning(f"Failed to process text with FastText model: {e}")
        return (
            "**⚠️ Error in Russian language detection**\n\n"
            "The Russian filter feature requires the FastText language identification model.\n"
            "Please ensure the model is properly installed and accessible.\n\n"
            f"Error: {str(e)}"
        )


def median_levenshtein(
    file_lat: t.Any, file_ipa: t.Any, sample: int | None = None
) -> str:
    """
    Compute median Levenshtein distance between two files (accepts any objects with .name attribute).
    Usage: median_levenshtein(NamedTuple('F', [('name', str)])('lat.txt'), NamedTuple('F', [('name', str)])('ipa.txt'))
    Returns: formatted string prefixed with 'Median distance: ...'.
    Example: 'Median distance: 0.1234'
    Raises: ValueError if file objects are missing .name.
    """
    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="median_lev", sample=sample or 0)

    from .. import sanity  # Import from main package, not web subpackage

    lat_path = getattr(file_lat, "name", None)
    ipa_path = getattr(file_ipa, "name", None)
    if not lat_path or not ipa_path:
        raise ValueError("Both file_lat and file_ipa must have a .name attribute")
    if sample is not None:
        value = sanity.median_lev(lat_path, ipa_path, sample=sample)
    else:
        value = sanity.median_lev(lat_path, ipa_path)
    return f"Median distance: {value:.4f}"


# ──────────────────────────────────────────────────────────────────────────
# NEW: lightweight corpus-to-file streaming helper


def download_corpus_to_file(
    source: str,
    lang: str,
    max_lines: int | None = None,
    filter_langid: bool = False,
    prob_threshold: float = 0.0,
    *,
    progress: gr.Progress | None = None,  # injected by Gradio
) -> tuple[str, str]:
    """
    Stream sentences from *source*/*lang* into a temporary UTF-8 file.

    Returns a pair *(file_path, markdown_info)* so the caller can both expose
    the file for download **and** show a summary message.
    """
    import logging
    from pathlib import Path

    from turkic_translit.cli import download_corpus as dl

    logger = logging.getLogger(__name__)
    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="download_corpus", source=source, lang=lang)
    logger.info(
        f"Web UI corpus download: source={source}, lang={lang}, max_lines={max_lines}, filter_langid={filter_langid}"
    )

    if source not in dl._REG:
        payload = error_response(
            f"Unknown corpus source {source!r}.",
            status=400,
            code="invalid_source",
            details={"available": list(dl._REG.keys())},
        )
        return "", error_markdown(payload)

    driver = dl._DRIVERS[dl._REG[source]["driver"]]
    # We fetch *unfiltered* iterator and apply our own threshold-aware filter
    base_iter = driver(lang, dl._REG[source], None)

    # Determine progress callback (Gradio Progress implements __call__)
    if progress is None:

        def _noop_progress(*_a: object, **_kw: object) -> None:
            """Fallback progress function that does nothing."""

        progress_fn: t.Callable[..., None] = _noop_progress
    else:
        progress_fn = t.cast(t.Callable[..., None], progress)

    # Initial tick so the UI shows the bar immediately
    progress_fn(0, desc="starting stream")

    model: FastTextLangID | None = None
    if filter_langid:
        logger.info("Getting FastText language ID model from singleton...")
        try:
            model = _langid_singleton()
            logger.info(f"FastText model loaded successfully: {model}")
        except Exception:
            logger.exception("Failed to load FastText model")
            raise

    # Counters
    i = 0  # lines written
    removed = 0
    total_processed = 0

    with open(
        _CRON_DIR / f"{source}_{lang}_{int(time.time())}.txt", "w", encoding="utf8"
    ) as tmp:
        tmp_path = tmp.name  # capture early so it is available after context closes
        logger.info(f"Starting to process sentences (max_lines={max_lines})...")
        # Ensure *i* is defined even when the iterator is empty
        for sentence in base_iter:
            # Check if we've already reached the limit before processing
            if max_lines is not None and i >= max_lines:
                logger.info(f"Reached max_lines limit: {i} >= {max_lines}")
                break

            total_processed += 1
            # Safety-net: stop early if we've processed far more lines than requested
            if (
                max_lines is not None
                and filter_langid
                and total_processed >= max_lines * 50
            ):
                logger.warning(
                    "Processing limit reached without enough lines kept; breaking early to avoid long hang."
                )
                break
            clean_sentence = sentence.replace("\n", " ").replace("\r", " ").strip()
            if not clean_sentence:
                continue  # skip blank lines
            # Apply LangID filter if requested - USE PREDICT() LIKE THE CLI DOES
            if model is not None:
                pred_lang, pred_prob = model.predict_with_prob(clean_sentence)
                if total_processed <= 5:
                    logger.info(
                        f"Sentence {total_processed}: '{clean_sentence[:50]}...' -> predicted: {pred_lang}, wanted: {lang}"
                    )
                # Skip sentence if wrong language or below probability threshold
                if pred_lang != lang or pred_prob < prob_threshold:
                    removed += 1
                    continue
            tmp.write(clean_sentence + "\n")
            i += 1

            # Update progress and log
            if i % 10 == 0 or (max_lines and i == max_lines):
                if max_lines:
                    progress_msg = f"{i}/{max_lines} lines kept"
                    progress_fn(min(1.0, i / max_lines), desc=progress_msg)
                    logger.info(f"Progress: {progress_msg}")
                else:
                    progress_msg = f"{i:,} lines kept"
                    progress_fn(None, desc=progress_msg)
                    if i % 100 == 0:  # Less frequent logging when no limit
                        logger.info(f"Progress: {progress_msg}")

    # Capture file path after context manager closes it
    tmp_path = tmp.name

    progress_fn(1.0, desc="completed")

    # Compute how many lines were skipped when language filtering is active
    # removed counter already computed

    logger.info(
        f"Download complete: {i} lines written from {total_processed} sentences processed"
    )

    info_md = (
        "### ✅ Download complete\n\n"
        f"- **Source:** `{source}`\n"
        f"- **Language:** {pretty_lang(lang)}\n"
        f"- **Lines written:** {i:,}\n"
        f"- **Total sentences processed:** {total_processed:,}\n"
        + (
            f"- **Lines removed by LangID filter:** {removed:,} (p ≥ {prob_threshold})\n"
            if filter_langid
            else ""
        )
        + f"- **File:** `{Path(tmp_path).name}`\n"
    )
    return tmp_path, info_md


def train_sentencepiece_model(
    input_text: str,
    training_file: t.Any = None,
    vocab_size: int = 12000,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
    user_symbols: str = "<lang_kk>,<lang_ky>",
) -> tuple[str, str]:
    """
    Train a SentencePiece model using provided text and parameters.

    Args:
        input_text: Text content to use for training
        training_file: Optional file object to use for training (must have .name attribute)
        vocab_size: Size of the vocabulary to create
        model_type: SentencePiece model type (unigram, bpe, char, word)
        character_coverage: Character coverage ratio
        user_symbols: Comma-separated list of user-defined symbols

    Returns:
        Tuple of (output model file path, info markdown string)

    Raises:
        ValueError: If neither input_text nor training_file is provided
        ImportError: If sentencepiece is not installed
    """
    try:
        import sentencepiece as spm
    except ImportError as err:
        raise ImportError(
            "SentencePiece is required for model training. Please install with: pip install sentencepiece"
        ) from err

    if not input_text.strip() and not training_file:
        raise ValueError("Either input text or training file must be provided")

    # Create a temporary directory for training data and model files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        training_data_path = temp_dir_path / "training_data.txt"
        model_prefix = temp_dir_path / "spm_model"

        # Write input text to training file if provided
        input_files = []

        if input_text.strip():
            with open(training_data_path, "w", encoding="utf-8") as f:
                f.write(input_text.strip() + "\n")
            input_files.append(str(training_data_path))

        # If a file was uploaded, use its path directly for SentencePiece training
        # This avoids loading large files into memory
        if training_file:
            file_path = getattr(training_file, "name", None)
            if file_path:
                input_files.append(file_path)

        # Parse user symbols
        user_symbols_list = [s.strip() for s in user_symbols.split(",") if s.strip()]

        # Train the model with all input files
        # This approach is more memory-efficient for large files
        spm.SentencePieceTrainer.train(
            input=",".join(
                input_files
            ),  # SentencePiece accepts comma-separated file paths
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            normalization_rule_name="nfkc",
            user_defined_symbols=user_symbols_list,
            # Additional parameters that help with large corpus files
            input_sentence_size=10000000,  # Process up to 10M sentences (plenty for most use cases)
            shuffle_input_sentence=True,  # Shuffle for better training outcome
            num_threads=os.cpu_count()
            or 4,  # Use multiple threads for faster processing
        )

        # Path to the output model file
        model_file_path = str(model_prefix) + ".model"
        vocab_file_path = str(model_prefix) + ".vocab"

        # Count vocab items for stats
        vocab_count = 0
        with open(vocab_file_path, encoding="utf-8") as vocab_file:
            for _ in vocab_file:
                vocab_count += 1

        # Get model file size
        model_size_bytes = os.path.getsize(model_file_path)
        model_size_kb = model_size_bytes / 1024

        # Copy model to a more permanent location for download
        output_model_path = (
            Path(tempfile.gettempdir())
            / f"turkic_sp_model_{vocab_size}_{model_type}.model"
        )
        with open(model_file_path, "rb") as src, open(output_model_path, "wb") as dst:
            dst.write(src.read())

        # Create info message
        info_md = f"""### Model Training Complete

**Model Statistics:**
- Vocabulary Size: {vocab_count} tokens
- Model Type: {model_type}
- Character Coverage: {character_coverage}
- Model File Size: {model_size_kb:.2f} KB

You can download the model file below. To use this model with the Turkic Transliteration toolkit, 
place it in the appropriate location for your application.

For the default tokenizer, rename it to `turkic_model.model` and place it in the 
same directory as the `tokenizer.py` file.
"""

        return str(output_model_path), info_md


__all__ = [
    "direct_transliterate",
    "pipeline_transliterate",
    "token_table_markdown",
    "mask_russian",
    "median_levenshtein",
    "download_corpus_to_file",
    "train_sentencepiece_model",
    "labelise",
    "get_ui_log_handler",
]
