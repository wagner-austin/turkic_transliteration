"""The record a corpus carries about how it was produced.

A corpus file is a list of lines with no memory of where it came from.
That is exactly the gap that made this project's own training corpora
irreproducible: they were filtered by a language-identification model
whose identity survived only in a side note, so the released tool could
not rebuild them. Every run therefore writes a manifest beside its output
naming the source, the language, the counts, and — when a filter ran —
the precise weights that decided which lines survived.

``filter_language`` and ``language_identification`` are always present and
are null together when no filter ran. An absent key would make "no filter"
indistinguishable from "manifest written by an older version", which is
the ambiguity this file exists to remove.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

from turkic_translit.lid.factory import (
    LidRunRecord,
    decode_lid_run_record,
    encode_lid_run_record,
)
from turkic_translit.validation import (
    ERR_FIELD_TYPE,
    FieldError,
    require_non_empty_str,
    require_non_negative_int,
    require_optional_mapping,
    require_optional_non_empty_str,
    require_present,
)

MANIFEST_SUFFIX: Final = ".manifest.json"


class CorpusRunManifest(TypedDict):
    """Everything needed to reproduce one corpus download.

    Attributes:
        source_id: Registry key of the corpus streamed.
        driver: Which reader produced the lines, e.g. ``oscar``.
        license: SPDX identifier the corpus is published under.
        language: Language code requested from that source.
        output_path: Path the corpus was written to.
        lines_seen: Non-empty lines the source yielded, before filtering.
        lines_written: Lines that survived and reached the file.
        filter_language: Language the filter required, or ``None`` when
            no filter ran.
        language_identification: Identity of the classifier that applied
            that filter, or ``None`` when no filter ran.
    """

    source_id: str
    driver: str
    license: str
    language: str
    output_path: str
    lines_seen: int
    lines_written: int
    filter_language: str | None
    language_identification: LidRunRecord | None


def manifest_path_for(output_path: Path) -> Path:
    """Name the manifest that belongs beside a corpus file.

    The suffix is appended rather than replacing the existing one, so
    ``kk.txt`` and ``kk.jsonl`` in the same directory keep distinct
    manifests.

    Args:
        output_path: Path the corpus itself was written to.

    Returns:
        Path of the manifest for that corpus.
    """
    return output_path.with_name(output_path.name + MANIFEST_SUFFIX)


def encode_corpus_run_manifest(
    manifest: CorpusRunManifest,
) -> dict[str, str | int | None | dict[str, str | int | float | bool]]:
    """Render a manifest as a plain mapping ready for JSON.

    Args:
        manifest: The manifest to encode.

    Returns:
        A mapping carrying every manifest field, with the nested
        language-identification record encoded in turn.
    """
    record = manifest["language_identification"]
    return {
        "source_id": manifest["source_id"],
        "driver": manifest["driver"],
        "license": manifest["license"],
        "language": manifest["language"],
        "output_path": manifest["output_path"],
        "lines_seen": manifest["lines_seen"],
        "lines_written": manifest["lines_written"],
        "filter_language": manifest["filter_language"],
        "language_identification": (None if record is None else encode_lid_run_record(record)),
    }


def decode_corpus_run_manifest(
    source: Mapping[str, str | int | float | bool | None | Mapping[str, str | int | float | bool]],
) -> CorpusRunManifest:
    """Validate a loosely-typed mapping into a :class:`CorpusRunManifest`.

    Args:
        source: Mapping holding every manifest field, as read from JSON.

    Returns:
        A fully validated manifest.

    Raises:
        FieldError: If any field is missing, of the wrong type, empty, or
            outside its permitted range.
    """
    record = require_optional_mapping(
        "language_identification",
        require_present("language_identification", source),
    )
    return CorpusRunManifest(
        source_id=require_non_empty_str("source_id", require_present("source_id", source)),
        driver=require_non_empty_str("driver", require_present("driver", source)),
        license=require_non_empty_str("license", require_present("license", source)),
        language=require_non_empty_str("language", require_present("language", source)),
        output_path=require_non_empty_str("output_path", require_present("output_path", source)),
        lines_seen=require_non_negative_int("lines_seen", require_present("lines_seen", source)),
        lines_written=require_non_negative_int(
            "lines_written", require_present("lines_written", source)
        ),
        filter_language=require_optional_non_empty_str(
            "filter_language", require_present("filter_language", source)
        ),
        language_identification=(None if record is None else decode_lid_run_record(record)),
    )


def write_corpus_run_manifest(manifest: CorpusRunManifest, path: Path) -> None:
    """Write a manifest to disk as indented, key-sorted JSON.

    Args:
        manifest: The manifest to write.
        path: Destination, ordinarily from :func:`manifest_path_for`.
    """
    path.write_text(
        json.dumps(
            encode_corpus_run_manifest(manifest),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_corpus_run_manifest(path: Path) -> CorpusRunManifest:
    """Read a manifest back from disk and validate it.

    Args:
        path: Location of the manifest.

    Returns:
        The validated manifest.

    Raises:
        FieldError: If the document is not a mapping, or any field fails
            validation.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise FieldError(
            ERR_FIELD_TYPE,
            "<manifest>",
            f"expected a mapping of fields, got {type(document).__name__}",
        )
    return decode_corpus_run_manifest(document)


__all__ = [
    "MANIFEST_SUFFIX",
    "CorpusRunManifest",
    "decode_corpus_run_manifest",
    "encode_corpus_run_manifest",
    "manifest_path_for",
    "read_corpus_run_manifest",
    "write_corpus_run_manifest",
]
