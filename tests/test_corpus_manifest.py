"""Tests for the manifest a corpus run writes beside its output.

The manifest exists so a corpus can name the classifier that produced it,
so the round-trip through disk is tested rather than only the in-memory
encoding: what matters is that a later session reading the file recovers
the same record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turkic_translit.corpus.manifest import (
    CorpusRunManifest,
    decode_corpus_run_manifest,
    encode_corpus_run_manifest,
    manifest_path_for,
    read_corpus_run_manifest,
    write_corpus_run_manifest,
)
from turkic_translit.lid.factory import LidRunRecord
from turkic_translit.validation import (
    ERR_FIELD_MISSING,
    ERR_FIELD_TYPE,
    FieldError,
)

FILTERED = CorpusRunManifest(
    source_id="oscar-2301",
    driver="oscar",
    license="CC0-1.0",
    language="uz",
    output_path="corpora/uz.txt",
    lines_seen=25676,
    lines_written=14207,
    filter_language="uzn",
    language_identification=LidRunRecord(
        model_id="lid218e",
        weights_path="/models/lid218e.bin",
        weights_bytes=1176355829,
        threshold=0.95,
        script_aware=True,
    ),
)

UNFILTERED = CorpusRunManifest(
    source_id="wikipedia",
    driver="wikipedia",
    license="CC-BY-SA-3.0",
    language="kk",
    output_path="corpora/kk.txt",
    lines_seen=40,
    lines_written=40,
    filter_language=None,
    language_identification=None,
)


def test_manifest_path_appends_rather_than_replacing_the_suffix() -> None:
    """Two corpora with different extensions keep distinct manifests."""
    assert manifest_path_for(Path("corpora/uz.txt")) == Path("corpora/uz.txt.manifest.json")
    assert manifest_path_for(Path("corpora/uz.jsonl")) == Path("corpora/uz.jsonl.manifest.json")


def test_filtered_manifest_round_trips_through_encode_decode() -> None:
    """A manifest naming a classifier decodes back to an equal manifest."""
    assert decode_corpus_run_manifest(encode_corpus_run_manifest(FILTERED)) == FILTERED


def test_unfiltered_manifest_round_trips_through_encode_decode() -> None:
    """A manifest with no filter keeps both filter fields null."""
    assert decode_corpus_run_manifest(encode_corpus_run_manifest(UNFILTERED)) == UNFILTERED


def test_encoding_nests_the_classifier_record() -> None:
    """The classifier's identity is written out field by field."""
    encoded = encode_corpus_run_manifest(FILTERED)
    assert encoded["language_identification"] == {
        "model_id": "lid218e",
        "weights_path": "/models/lid218e.bin",
        "weights_bytes": 1176355829,
        "threshold": 0.95,
        "script_aware": True,
    }


def test_manifest_survives_a_round_trip_through_disk(tmp_path: Path) -> None:
    """Writing and reading a manifest recovers exactly what was written."""
    path = manifest_path_for(tmp_path / "uz.txt")
    write_corpus_run_manifest(FILTERED, path)
    assert read_corpus_run_manifest(path) == FILTERED


def test_written_manifest_is_readable_json_with_sorted_keys(tmp_path: Path) -> None:
    """The file is stable text a human or a diff can read."""
    path = manifest_path_for(tmp_path / "kk.txt")
    write_corpus_run_manifest(UNFILTERED, path)
    text = path.read_text(encoding="utf-8")
    assert list(json.loads(text)) == sorted(json.loads(text))
    assert text.endswith("}\n")


def test_reading_rejects_a_document_that_is_not_a_mapping(tmp_path: Path) -> None:
    """A manifest written as a list fails before any field is read."""
    path = tmp_path / "list.manifest.json"
    path.write_text(json.dumps(["oscar-2301"]), encoding="utf-8")
    with pytest.raises(FieldError) as excinfo:
        read_corpus_run_manifest(path)
    assert excinfo.value.field == "<manifest>"
    assert excinfo.value.detail == "expected a mapping of fields, got list"


def test_decoding_requires_the_filter_fields_to_be_present() -> None:
    """An absent filter field is a missing key, never an implied null.

    That distinction is the point: "no filter ran" and "this manifest
    predates filter recording" must not look the same.
    """
    encoded = encode_corpus_run_manifest(UNFILTERED)
    del encoded["language_identification"]
    with pytest.raises(FieldError) as excinfo:
        decode_corpus_run_manifest(encoded)
    assert excinfo.value.code == ERR_FIELD_MISSING
    assert excinfo.value.field == "language_identification"


def test_decoding_rejects_a_scalar_where_the_record_belongs() -> None:
    """A classifier recorded as a bare name is not enough to reproduce it."""
    encoded = encode_corpus_run_manifest(FILTERED)
    encoded["language_identification"] = "lid218e"
    with pytest.raises(FieldError) as excinfo:
        decode_corpus_run_manifest(encoded)
    assert excinfo.value.code == ERR_FIELD_TYPE
    assert excinfo.value.field == "language_identification"


def test_decoding_validates_inside_the_nested_record() -> None:
    """A malformed threshold inside the record names the inner field."""
    encoded = encode_corpus_run_manifest(FILTERED)
    encoded["language_identification"] = {
        "model_id": "lid218e",
        "weights_path": "/models/lid218e.bin",
        "weights_bytes": 1176355829,
        "threshold": 95,
        "script_aware": True,
    }
    with pytest.raises(FieldError) as excinfo:
        decode_corpus_run_manifest(encoded)
    assert excinfo.value.field == "threshold"


def test_decoding_rejects_a_negative_line_count() -> None:
    """A count below zero fails rather than being read as a total."""
    encoded = encode_corpus_run_manifest(FILTERED)
    encoded["lines_written"] = -3
    with pytest.raises(FieldError) as excinfo:
        decode_corpus_run_manifest(encoded)
    assert excinfo.value.field == "lines_written"
