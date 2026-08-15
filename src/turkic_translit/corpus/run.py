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
from turkic_translit.corpus.filtering import LidFilterRequest, build_line_filter
from turkic_translit.corpus.manifest import (
    CorpusRunManifest,
    manifest_path_for,
    write_corpus_run_manifest,
)
from turkic_translit.corpus.normalize import PACKAGED_FOLDS, normalize_line
from turkic_translit.corpus.sources import get_source_spec
from turkic_translit.corpus.symbols import (
    apply_substitutions,
    read_symbol_map,
    substitutions_for,
)

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
        CorpusStreamError: If the source's host cannot be read.
    """
    spec = get_source_spec(source_id)
    line_filter, lid_record = build_line_filter(lid_filter)
    filter_language = None if lid_filter is None else lid_filter["language"]
    # The verified misencoding repairs for this language, applied to
    # every line as it arrives so the corpus is born normalised.
    folds = substitutions_for(read_symbol_map(PACKAGED_FOLDS), language)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines_seen = 0
    lines_written = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for fragment in stream_source(spec, language, access_token):
            text = apply_substitutions(normalize_line(fragment), folds)
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

    manifest = CorpusRunManifest(
        source_id=spec["source_id"],
        driver=spec["driver"],
        license=spec["license"],
        language=language,
        output_path=str(output_path),
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
