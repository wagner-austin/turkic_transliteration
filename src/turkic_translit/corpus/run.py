"""Running one corpus download from end to end.

This is the only place lines are normalised, counted, filtered and
written. Keeping it in one function is what lets the manifest be true:
the classifier recorded in the manifest is the same object that judged
every line in the file beside it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from turkic_translit.corpus.drivers import stream_source
from turkic_translit.corpus.errors import FilterKeptNothingError
from turkic_translit.corpus.filtering import LidFilterRequest, build_line_filter
from turkic_translit.corpus.manifest import (
    CorpusRunManifest,
    manifest_path_for,
    write_corpus_run_manifest,
)
from turkic_translit.corpus.normalize import (
    language_folds,
    normalization_record,
    normalize_source_line,
)
from turkic_translit.corpus.sources import get_source_spec
from turkic_translit.digests import sha256_of_file

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL: Final = 1000


def download_corpus(
    source_id: str,
    language: str,
    output_path: Path,
    max_lines: int | None,
    access_token: str | None,
    lid_filter: LidFilterRequest | None,
) -> tuple[CorpusRunManifest, Path]:
    """Stream a corpus to disk and write the manifest describing the run.

    Args:
        source_id: Registry key of the corpus to stream.
        language: Language code within that source.
        output_path: File to write the corpus to. Parent directories are
            created.
        max_lines: Stop after this many lines have been written, or
            ``None`` to read the source to exhaustion.
        access_token: Credential for gated datasets, or ``None``.
        lid_filter: Language filter to apply, or ``None`` to keep every
            line.

    Returns:
        The manifest describing the run, and the path it was written to.

    Raises:
        UnknownCorpusSourceError: If the source id is not registered.
        UnknownLidModelError: If the filter names an unregistered model.
        LidLanguageNotEmittedError: If the filter's model reports no label
            for the filter's language, checked before streaming.
        CorpusStreamError: If the source's host cannot be read.
        FilterKeptNothingError: If a filtered run kept none of the lines
            it saw. The empty corpus is removed and no manifest written.
    """
    spec = get_source_spec(source_id)
    line_filter, lid_record = build_line_filter(lid_filter)
    filter_language = None if lid_filter is None else lid_filter["language"]
    # The verified misencoding repairs for this language, applied to
    # every line as it arrives so the corpus is born normalised.
    folds = language_folds(language)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines_seen = 0
    lines_written = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for fragment in stream_source(spec, language, access_token):
            text = normalize_source_line(fragment, folds)
            if text == "":
                continue
            lines_seen += 1
            if not line_filter.keeps(text):
                continue
            sink.write(text + "\n")
            lines_written += 1
            if lines_written % PROGRESS_INTERVAL == 0:
                logger.info(
                    "wrote %d lines of %d seen from %s/%s",
                    lines_written,
                    lines_seen,
                    source_id,
                    language,
                )
            if max_lines is not None and lines_written >= max_lines:
                break

    if lid_filter is not None and lines_written == 0:
        output_path.unlink()
        raise FilterKeptNothingError(
            source_id, lid_filter["language"], lid_filter["model_id"], lines_seen, str(output_path)
        )

    manifest = CorpusRunManifest(
        source_id=spec["source_id"],
        driver=spec["driver"],
        license=spec["license"],
        language=language,
        output_path=str(output_path),
        # Digested after the writer closed, from the bytes on disk rather
        # than from what this function believes it emitted. Hashing the
        # lines as they stream would agree with the run and could still
        # disagree with the file.
        output_sha256=sha256_of_file(output_path),
        normalization=normalization_record(),
        lines_seen=lines_seen,
        lines_written=lines_written,
        filter_language=filter_language,
        language_identification=lid_record,
    )
    manifest_file = manifest_path_for(output_path)
    write_corpus_run_manifest(manifest, manifest_file)
    logger.info(
        "wrote %d lines of %d seen to %s; manifest at %s",
        lines_written,
        lines_seen,
        output_path,
        manifest_file,
    )
    return manifest, manifest_file


__all__ = ["PROGRESS_INTERVAL", "download_corpus"]
