"""Streaming a corpus to a file for the corpus tab, with progress.

This was part of ``web/web_utils.py`` until 2026-11-04, when it moved here
unchanged so that each concern of the web helpers stays under the
600-line limit. The function's logger therefore now reports under this
module's name, still beneath ``turkic_translit``, whose handler feeds the
browser's log panel.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import gradio as gr

from turkic_translit.lang_utils import pretty_lang

from .. import _test_hooks
from ..error_service import (
    error_markdown,
    error_response,
    set_correlation_id,
    set_request_context,
)
from .downloads import download_dir
from .web_utils import LANGUAGE_MODEL_ID, installed_classifier

# How often, in lines kept, the progress bar is updated.
_PROGRESS_EVERY = 10


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


__all__ = ["ProgressReporter", "SilentProgress", "download_corpus_to_file"]
