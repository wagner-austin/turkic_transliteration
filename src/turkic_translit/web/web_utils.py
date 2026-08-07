from __future__ import annotations

import functools
import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import gradio as gr
import pandas as pd

from turkic_translit.lang_utils import pretty_lang
from turkic_translit.tokenizer import default_model_path

from ..error_service import (
    error_markdown,
    error_response,
    set_correlation_id,
    set_request_context,
)
from ..lang_filter import is_russian_token
from ..lid.errors import LidError
from ..lid.factory import load_installed_classifier

log = logging.getLogger(__name__)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# How often, in lines kept, the progress bar is updated.
_PROGRESS_EVERY = 10


class NamedFile(Protocol):
    """A Gradio file upload, which this module only reads a path from."""

    name: str


class ProgressReporter(Protocol):
    """The Gradio progress callback, as this module calls it."""

    def __call__(self, progress: float | None, desc: str = "") -> None:
        """Report progress to the user interface.

        Args:
            progress: Completed fraction, or ``None`` when the total is
                unknown and only the description should update.
            desc: Text shown beside the bar.
        """
        ...


class SilentProgress:
    """The reporter used when Gradio supplies none.

    A real implementation of :class:`ProgressReporter` rather than a
    null check at every call site: outside the web UI there is nowhere
    to report progress to, and that is not an error condition.
    """

    def __call__(self, progress: float | None, desc: str = "") -> None:
        """Discard the report.

        Args:
            progress: Ignored.
            desc: Ignored.
        """


# Directory for temporary corpus downloads – excluded from VCS via .gitignore

_CRON_DIR = Path(os.getenv("TURKIC_CRON_DIR", Path.cwd() / "cronjob"))
_CRON_DIR.mkdir(parents=True, exist_ok=True)


def purge_expired_downloads(max_age_sec: int) -> int:
    """Delete temporary downloads older than ``max_age_sec``.

    A file vanishing between the listing and the delete is the normal
    outcome of two sweeps overlapping, so ``missing_ok`` covers it and
    nothing is caught. Any other filesystem error is a real problem and
    propagates to the janitor, which reports it.

    Args:
        max_age_sec: Age beyond which a file is removed.

    Returns:
        The number of files deleted.
    """
    cutoff = time.time() - max_age_sec
    removed = 0
    for entry in _CRON_DIR.glob("*"):
        if entry.is_file() and entry.stat().st_mtime < cutoff:
            entry.unlink(missing_ok=True)
            removed += 1
    return removed


def _start_janitor(max_age_sec: int = 600) -> None:
    """Start the background thread that purges temporary downloads.

    Args:
        max_age_sec: Age beyond which a download is removed, and the
            interval between sweeps.
    """

    def _janitor() -> None:
        """Sweep expired downloads forever, reporting and continuing."""
        while True:
            try:
                purge_expired_downloads(max_age_sec)
            except OSError:
                # The thread is a daemon serving a long-lived UI: it logs
                # and keeps sweeping rather than dying on one bad file.
                log.exception("janitor sweep failed")
            time.sleep(max_age_sec)

    threading.Thread(target=_janitor, daemon=True, name="cron-janitor").start()


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

    def filter(self, record: logging.LogRecord) -> bool:
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

# Create a singleton for the language ID model. The model is named
# explicitly; there is no preference order to drift.
LANGUAGE_MODEL_ID = "lid.176"
_langid_singleton = functools.lru_cache(maxsize=1)(load_installed_classifier)


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
    result = to_latin(text, lang, include_arabic) if fmt == "latin" else to_ipa(text, lang)
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


def token_table_markdown(text: str) -> str:
    """Tokenise text and tabulate each token's predicted language.

    Args:
        text: Text to tokenise.

    Returns:
        A Markdown table, or a message explaining that the SentencePiece
        model has not been trained yet. That model is not shipped, so its
        absence is an ordinary first-run state rather than a defect, and
        the message says how to produce it.
    """
    set_correlation_id()
    set_request_context(action="token_table", sample=len(text))

    if not default_model_path().is_file():
        return (
            "**⚠️ Tokenizer model file missing**\n\n"
            f"`{default_model_path().name}` is required for tokenization.\n"
            "Train one with the `turkic-build-spm` command and place it in "
            "the package directory."
        )

    pipeline = _lazy_pipeline()
    tokens = pipeline.tokenizer.tokenize(text)
    languages = pipeline.predict_tokens(tokens)
    table: str = pd.DataFrame({"Token": tokens, "Lang": languages}).to_markdown(index=False)
    return table


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
    set_correlation_id()
    set_request_context(action="mask_russian", thr=thr, min_len=min_len)

    try:
        lid = _langid_singleton(LANGUAGE_MODEL_ID)
    except LidError as exc:
        log.warning("language-identification model unavailable: %s", exc)
        return (
            "**⚠️ Language-identification model unavailable**\n\n"
            f"The Russian filter needs the `{LANGUAGE_MODEL_ID}` model.\n\n"
            f"Error: {exc!s}"
        )

    masked: list[str] = []
    reports: list[dict[str, str | bool | float]] = []
    for token in text.strip().split():
        is_russian = is_russian_token(
            token, thr=thr, min_len=min_len, lid=lid, stoplist=None, margin=margin
        )
        masked.append("<RU>" if is_russian else token)
        if debug:
            top = lid.classify(token.lower())
            reports.append(
                {
                    "tok": token,
                    "ru": is_russian,
                    "winner": top["label"],
                    "conf": top["probability"],
                }
            )

    out = " ".join(masked)
    if debug:
        out += "\n\n<!--debug " + json.dumps(reports, ensure_ascii=False) + " -->"
    return _ANSI.sub("", out)


def median_levenshtein(file_lat: NamedFile, file_ipa: NamedFile, sample: int | None = None) -> str:
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


def _report_progress(progress_fn: ProgressReporter, written: int, max_lines: int | None) -> None:
    """Update the progress bar every tenth line kept.

    Args:
        progress_fn: Where to report.
        written: Lines written so far.
        max_lines: The cap, which makes the fraction meaningful, or
            ``None`` when the run is unbounded.
    """
    if written % _PROGRESS_EVERY != 0:
        return
    if max_lines is None:
        progress_fn(None, desc=f"{written:,} lines kept")
        return
    progress_fn(min(1.0, written / max_lines), desc=f"{written}/{max_lines} lines kept")


def _download_summary(
    *,
    source: str,
    lang: str,
    written: int,
    seen: int,
    removed: int | None,
    prob_threshold: float,
    path: Path,
) -> str:
    """Render the Markdown summary shown beside the download link.

    Args:
        source: Registry key of the corpus streamed.
        lang: Language code requested.
        written: Lines written to the file.
        seen: Sentences the source yielded.
        removed: Lines the language filter rejected, or ``None`` when no
            filter ran, which is what omits the line entirely.
        prob_threshold: Threshold the filter applied.
        path: The file written.

    Returns:
        The summary, as Markdown.
    """
    lines = [
        "### ✅ Download complete\n",
        f"- **Source:** `{source}`",
        f"- **Language:** {pretty_lang(lang)}",
        f"- **Lines written:** {written:,}",
        f"- **Total sentences processed:** {seen:,}",
    ]
    if removed is not None:
        lines.append(f"- **Lines removed by LangID filter:** {removed:,} (p ≥ {prob_threshold})")
    lines.append(f"- **File:** `{path.name}`")
    return "\n".join(lines) + "\n"


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

    from turkic_translit.corpus.drivers import stream_source
    from turkic_translit.corpus.sources import SOURCE_REGISTRY, known_source_ids

    logger = logging.getLogger(__name__)
    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="download_corpus", source=source, lang=lang)
    logger.info(
        f"Web UI corpus download: source={source}, lang={lang}, max_lines={max_lines}, filter_langid={filter_langid}"
    )

    if source not in SOURCE_REGISTRY:
        payload = error_response(
            f"Unknown corpus source {source!r}.",
            status=400,
            code="invalid_source",
            details={"available": list(known_source_ids())},
        )
        return "", error_markdown(payload)

    # Drivers no longer filter; they yield raw fragments and this function
    # applies its own threshold-aware filter below.
    base_iter = stream_source(SOURCE_REGISTRY[source], lang, os.getenv("HF_TOKEN"))

    progress_fn: ProgressReporter = SilentProgress() if progress is None else progress

    # Initial tick so the UI shows the bar immediately
    progress_fn(0, desc="starting stream")

    model = _langid_singleton(LANGUAGE_MODEL_ID) if filter_langid else None

    # Counters
    i = 0  # lines written
    removed = 0
    total_processed = 0

    with open(_CRON_DIR / f"{source}_{lang}_{int(time.time())}.txt", "w", encoding="utf8") as tmp:
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
            if max_lines is not None and filter_langid and total_processed >= max_lines * 50:
                logger.warning(
                    "Processing limit reached without enough lines kept; breaking early to avoid long hang."
                )
                break
            clean_sentence = sentence.replace("\n", " ").replace("\r", " ").strip()
            if not clean_sentence:
                continue  # skip blank lines
            # Apply LangID filter if requested - USE PREDICT() LIKE THE CLI DOES
            if model is not None:
                prediction = model.classify(clean_sentence)
                pred_lang = prediction["label"]
                pred_prob = prediction["probability"]
                # Skip sentence if wrong language or below probability threshold
                if pred_lang != lang or pred_prob < prob_threshold:
                    removed += 1
                    continue
            tmp.write(clean_sentence + "\n")
            i += 1

            _report_progress(progress_fn, i, max_lines)

    # Capture file path after context manager closes it
    tmp_path = tmp.name
    progress_fn(1.0, desc="completed")
    logger.info("download complete: %d lines written from %d seen", i, total_processed)

    return tmp_path, _download_summary(
        source=source,
        lang=lang,
        written=i,
        seen=total_processed,
        removed=removed if filter_langid else None,
        prob_threshold=prob_threshold,
        path=Path(tmp_path),
    )


def train_sentencepiece_model(
    input_text: str,
    training_file: NamedFile | None = None,
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
            input=",".join(input_files),  # SentencePiece accepts comma-separated file paths
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            normalization_rule_name="nfkc",
            user_defined_symbols=user_symbols_list,
            # Additional parameters that help with large corpus files
            input_sentence_size=10000000,  # Process up to 10M sentences (plenty for most use cases)
            shuffle_input_sentence=True,  # Shuffle for better training outcome
            num_threads=os.cpu_count() or 4,  # Use multiple threads for faster processing
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
            Path(tempfile.gettempdir()) / f"turkic_sp_model_{vocab_size}_{model_type}.model"
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
    "download_corpus_to_file",
    "get_ui_log_handler",
    "labelise",
    "mask_russian",
    "median_levenshtein",
    "pipeline_transliterate",
    "token_table_markdown",
    "train_sentencepiece_model",
]
