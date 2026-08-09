from __future__ import annotations

import functools
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import gradio as gr
import pandas as pd

from turkic_translit.lang_utils import pretty_lang
from turkic_translit.tokenizer import (
    MODEL_PATH_VARIABLE,
    default_model_path,
    sentencepiece_trainer,
)

from .. import _test_hooks
from ..error_service import (
    error_markdown,
    error_response,
    set_correlation_id,
    set_request_context,
)
from ..lang_filter import is_russian_token
from ..lid.errors import LidError
from ..lid.factory import load_installed_classifier
from ..pipeline import TurkicTransliterationPipeline

log = logging.getLogger(__name__)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# How often, in lines kept, the progress bar is updated.
_PROGRESS_EVERY = 10

# How long a temporary download survives, and how long the janitor waits
# between sweeps.
JANITOR_INTERVAL_SECONDS = 600


class NamedFile(Protocol):
    """A Gradio file upload, which this module only reads a path from.

    The path is declared read-only because that is all this module does
    with it. A mutable attribute would exclude any immutable stand-in
    while granting a capability nothing here uses.
    """

    @property
    def name(self) -> str:
        """Return the path the upload's contents were written to.

        Returns:
            An absolute path on the local filesystem.
        """
        ...


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


def download_dir() -> Path:
    """Return the directory temporary corpus downloads are written to.

    Resolved on each call rather than at import, so nothing is created as
    a side effect of importing this module and a test can point the
    directory somewhere disposable without touching the process.

    Args:
        None.

    Returns:
        The directory named by ``TURKIC_CRON_DIR``, or ``cronjob`` under
        the working directory. It is created if it does not yet exist.
    """
    configured = _test_hooks.environment.get("TURKIC_CRON_DIR")
    directory = Path(configured) if configured else Path.cwd() / "cronjob"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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
    for entry in download_dir().glob("*"):
        if entry.is_file() and entry.stat().st_mtime < cutoff:
            entry.unlink(missing_ok=True)
            removed += 1
    return removed


def sweep_once(max_age_sec: int) -> int:
    """Purge expired downloads, reporting a filesystem error rather than
    letting it end the sweeping thread.

    This is the only place an exception is absorbed in this module, and
    the reason is that the caller is a daemon serving a long-lived UI: a
    single file held open by the browser must not stop every later
    sweep. The failure is logged with its traceback, not discarded.

    Args:
        max_age_sec: Age beyond which a download is removed.

    Returns:
        The number of files deleted, or zero when the sweep failed.
    """
    try:
        return purge_expired_downloads(max_age_sec)
    except OSError:
        log.exception("janitor sweep failed")
        return 0


def forever() -> bool:
    """Report that the janitor should keep sweeping.

    Returns:
        True, always. This is the loop condition a real run uses; a test
        supplies one that stops.
    """
    return True


def run_janitor(max_age_sec: int, running: Callable[[], bool]) -> int:
    """Sweep expired downloads until the loop condition says to stop.

    Args:
        max_age_sec: Age beyond which a download is removed, and the
            interval between sweeps.
        running: Consulted before each sweep. A real run passes
            :func:`forever`.

    Returns:
        The number of sweeps performed.
    """
    sweeps = 0
    while running():
        sweep_once(max_age_sec)
        sweeps += 1
        _test_hooks.clock.sleep(max_age_sec)
    return sweeps


def start_janitor(
    max_age_sec: int = JANITOR_INTERVAL_SECONDS,
    running: Callable[[], bool] = forever,
) -> threading.Thread:
    """Start the background thread that purges temporary downloads.

    Started by the application rather than by importing this module.
    Sweeping on import meant every process that touched a web helper —
    including every test run — spawned a thread and created a download
    directory wherever it happened to be running.

    Args:
        max_age_sec: Age beyond which a download is removed, and the
            interval between sweeps.
        running: Loop condition, defaulting to sweeping indefinitely.

    Returns:
        The started daemon thread.
    """
    thread = threading.Thread(
        target=run_janitor,
        args=(max_age_sec, running),
        daemon=True,
        name="cron-janitor",
    )
    thread.start()
    return thread


def labelise(codes: list[str]) -> list[tuple[str, str]]:
    """Pair each language code with the name a dropdown should show.

    Args:
        codes: ISO language codes, in the order to offer them.

    Returns:
        Label and value pairs, in the same order.
    """
    return [(pretty_lang(c), c) for c in codes]


class GradioLogHandler(logging.Handler):
    """Buffers log records so UI callbacks can flush them into the browser."""

    def __init__(self) -> None:
        """Start with an empty buffer."""
        super().__init__()
        self.buffer: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Add one formatted record to the buffer.

        Args:
            record: The record being logged.
        """
        self.buffer.append(self.format(record))

    def dump(self) -> str:
        """Take everything buffered so far, leaving the buffer empty.

        Returns:
            The buffered records as one newline-separated block.
        """
        out, self.buffer = "\n".join(self.buffer), []
        return out


class UiPrettyLogFilter(logging.Filter):
    """Skip verbose HTTP and housekeeping messages for UI logs."""

    _SKIP_PHRASES = (
        "HTTP Request:",
        "turkic_model.model not found",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether a record belongs in the browser's log.

        Args:
            record: The record being logged.

        Returns:
            False for the housekeeping messages that would bury the ones
            a user is actually watching for.
        """
        msg = record.getMessage()
        return not any(p in msg for p in self._SKIP_PHRASES)


_UI_LOG_HANDLER: GradioLogHandler | None = None


def get_ui_log_handler() -> GradioLogHandler:
    """Return the shared handler that feeds the browser's log panel.

    Returns:
        The handler, attached to the project logger on first call and
        reused afterwards so records are buffered in one place.
    """
    global _UI_LOG_HANDLER
    if _UI_LOG_HANDLER is None:
        h = GradioLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        h.addFilter(UiPrettyLogFilter())
        logging.getLogger("turkic_translit").addHandler(h)
        _UI_LOG_HANDLER = h
    return _UI_LOG_HANDLER


def _make_pipeline() -> TurkicTransliterationPipeline:
    """Build the pipeline the transliteration helpers share.

    Returns:
        A pipeline over the default language-identification model.
    """
    log.info("Instantiating TurkicTransliterationPipeline singleton")
    return TurkicTransliterationPipeline()


_lazy_pipeline = functools.lru_cache(maxsize=1)(_make_pipeline)

# The language-identification model is named explicitly; there is no
# preference order to drift. Loading it costs a 126 MB read, so the
# result is held for the life of the process. The cache is public
# because it is the reason a rebound loader hook would otherwise go
# unnoticed: whoever rebinds the hook clears this.
LANGUAGE_MODEL_ID = "lid.176"
installed_classifier = functools.lru_cache(maxsize=1)(load_installed_classifier)


def direct_transliterate(
    text: str, lang: str, include_arabic: bool, out_fmt: str
) -> tuple[str, str]:
    """Transliterate text to Latin or IPA with the language's rules.

    Args:
        text: Input in the language's native orthography.
        lang: ISO 639-1 language code.
        include_arabic: Whether to fold Arabic-script tokens to Latin
            before applying the target rules.
        out_fmt: ``latin`` or ``ipa``, in any casing.

    Returns:
        The transliteration and a Markdown line reporting the input and
        output sizes in bytes.

    Raises:
        ValueError: If the format is neither ``latin`` nor ``ipa``, or
            the language has no rules for the one requested.
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
    """Transliterate text through the tokenising pipeline.

    Unlike :func:`direct_transliterate`, this splits the text into
    tokens and identifies each one's language first, so mixed-language
    input is transliterated by the rules of the language each token is
    actually in.

    Args:
        text: Input text, which may mix languages.
        mode: ``latin`` or ``ipa``, in any casing.

    Returns:
        The transliteration and a Markdown line reporting its length.

    Raises:
        ValueError: If the mode is neither ``latin`` nor ``ipa``.
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
            f"`{default_model_path()}` is required for tokenization.\n"
            "Train one with the `turkic-build-spm` command, then point "
            f"`{MODEL_PATH_VARIABLE}` at it."
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
        lid = installed_classifier(LANGUAGE_MODEL_ID)
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
    """Report the median edit distance between two aligned files.

    Args:
        file_lat: Upload holding the Latin transliteration, one line per
            sentence.
        file_ipa: Upload holding the IPA transliteration, aligned line
            for line with ``file_lat``.
        sample: Lines to compare, or ``None`` to use the default cap.

    Returns:
        The distance rendered as ``Median distance: 0.1234``.

    Raises:
        ValueError: If either upload names no path, which means the
            browser sent a file reference the UI never wrote.
        OSError: If either path cannot be read.
    """
    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="median_lev", sample=sample or 0)

    from .. import sanity  # Import from main package, not web subpackage

    if file_lat.name == "" or file_ipa.name == "":
        raise ValueError("Both file_lat and file_ipa must have a .name attribute")
    if sample is not None:
        value = sanity.median_lev(file_lat.name, file_ipa.name, sample=sample)
    else:
        value = sanity.median_lev(file_lat.name, file_ipa.name)
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
    """Stream a corpus into a file the browser can download.

    Args:
        source: Registry key of the corpus source.
        lang: Language code within that source.
        max_lines: Stop after this many lines are kept, or ``None`` to
            read the source to exhaustion.
        filter_langid: Whether to keep only lines the classifier assigns
            to ``lang``.
        prob_threshold: Minimum probability a line must reach to be kept
            when filtering.
        progress: Gradio's progress reporter, or ``None`` to report
            nowhere, which is what a caller outside the web UI wants.

    Returns:
        The path written and a Markdown summary of the run. An
        unregistered source returns an empty path and an error payload
        rendered as Markdown, because the caller displays this rather
        than handling an exception.

    Raises:
        CorpusError: If the source is registered but cannot be read.
        LidError: If filtering was asked for and the classifier's
            weights are missing or unreadable.
        OSError: If the output file cannot be written.
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
    base_iter = stream_source(
        SOURCE_REGISTRY[source], lang, _test_hooks.environment.get("HF_TOKEN")
    )

    progress_fn: ProgressReporter = SilentProgress() if progress is None else progress

    # Initial tick so the UI shows the bar immediately
    progress_fn(0, desc="starting stream")

    model = installed_classifier(LANGUAGE_MODEL_ID) if filter_langid else None

    # Counters
    i = 0  # lines written
    removed = 0
    total_processed = 0

    target = download_dir() / f"{source}_{lang}_{int(time.time())}.txt"
    with open(target, "w", encoding="utf8") as tmp:
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
    """
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

        # If a file was uploaded, use its path directly for SentencePiece
        # training, which avoids loading large files into memory.
        if training_file is not None and training_file.name != "":
            input_files.append(training_file.name)

        # Parse user symbols
        user_symbols_list = [s.strip() for s in user_symbols.split(",") if s.strip()]

        # Train the model with all input files
        # This approach is more memory-efficient for large files
        sentencepiece_trainer().train(
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
