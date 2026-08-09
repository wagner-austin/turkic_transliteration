"""Streaming public corpora to disk, with a record of how it was done.

The package separates four concerns that the previous single-module
implementation ran together: which sources exist and what shape their
configuration has (:mod:`sources`), how each source's bytes are read
(:mod:`drivers`), which lines survive (:mod:`filtering`), and what the
run writes down about itself (:mod:`manifest`). :mod:`run` is the one
place that composes them.

The separation is what fixes the reproducibility gap. Filtering used to
happen inside each driver and again in the CLI loop, judged by models
constructed independently in the two places, and nothing was written down
about either. A corpus produced here names the source, the language, the
counts, and the exact classifier weights that decided its contents.
"""

from __future__ import annotations

from turkic_translit.corpus.drivers import (
    stream_source,
    stream_wikipedia,
    wikipedia_dump_url,
)
from turkic_translit.corpus.errors import (
    CorpusError,
    CorpusStreamError,
    UnknownCorpusSourceError,
)
from turkic_translit.corpus.filtering import (
    KeepEveryLine,
    LanguageLineFilter,
    LidFilterRequest,
    LineFilter,
    build_line_filter,
    decode_lid_filter_request,
    encode_lid_filter_request,
)
from turkic_translit.corpus.manifest import (
    CorpusRunManifest,
    decode_corpus_run_manifest,
    encode_corpus_run_manifest,
    manifest_path_for,
    read_corpus_run_manifest,
    write_corpus_run_manifest,
)
from turkic_translit.corpus.run import download_corpus, normalize_line
from turkic_translit.corpus.sources import (
    SOURCE_REGISTRY,
    OscarSourceSpec,
    WikipediaSourceSpec,
    decode_source_registry,
    decode_source_spec,
    encode_source_spec,
    get_source_spec,
    known_source_ids,
    load_source_registry,
)

__all__ = [
    "SOURCE_REGISTRY",
    "CorpusError",
    "CorpusRunManifest",
    "CorpusStreamError",
    "KeepEveryLine",
    "LanguageLineFilter",
    "LidFilterRequest",
    "LineFilter",
    "OscarSourceSpec",
    "UnknownCorpusSourceError",
    "WikipediaSourceSpec",
    "build_line_filter",
    "decode_corpus_run_manifest",
    "decode_lid_filter_request",
    "decode_source_registry",
    "decode_source_spec",
    "download_corpus",
    "encode_corpus_run_manifest",
    "encode_lid_filter_request",
    "encode_source_spec",
    "get_source_spec",
    "known_source_ids",
    "load_source_registry",
    "manifest_path_for",
    "normalize_line",
    "read_corpus_run_manifest",
    "stream_source",
    "stream_wikipedia",
    "wikipedia_dump_url",
    "write_corpus_run_manifest",
]
