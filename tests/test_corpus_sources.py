"""Tests for the corpus source registry.

The shipped ``corpora.yaml`` is decoded for real, so a malformed registry
fails here rather than at the start of a multi-gigabyte download. The
malformed documents below are written as YAML text and loaded through the
same path, not hand-built as dictionaries, so the loader is exercised end
to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turkic_translit.corpus.errors import (
    ERR_UNKNOWN_SOURCE,
    UnknownCorpusSourceError,
)
from turkic_translit.corpus.sources import (
    SOURCE_REGISTRY,
    decode_source_registry,
    decode_source_spec,
    encode_source_spec,
    get_source_spec,
    known_source_ids,
    load_source_registry,
)
from turkic_translit.validation import (
    ERR_FIELD_MISSING,
    ERR_FIELD_TYPE,
    FieldError,
)


def _write(tmp_path: Path, text: str) -> Path:
    """Write a registry document and return its path.

    Args:
        tmp_path: Directory to write into.
        text: YAML body of the registry.

    Returns:
        Path of the written file.
    """
    path = tmp_path / "corpora.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_registry_holds_both_sources() -> None:
    """The registry that ships with the package decodes to two sources."""
    assert known_source_ids() == ("oscar-2301", "wikipedia")


def test_oscar_entry_carries_its_dataset_name() -> None:
    """The OSCAR source names the Hugging Face dataset it streams."""
    spec = get_source_spec("oscar-2301")
    assert spec == {
        "source_id": "oscar-2301",
        "driver": "oscar",
        "license": "CC0-1.0",
        "hf_name": "oscar-corpus/OSCAR-2301",
    }


def test_wikipedia_entry_has_no_dataset_name() -> None:
    """The Wikipedia source derives its URL and so carries no dataset."""
    assert get_source_spec("wikipedia") == {
        "source_id": "wikipedia",
        "driver": "wikipedia",
        "license": "CC-BY-SA-3.0",
    }


def test_unknown_source_names_the_alternatives() -> None:
    """An unregistered id reports its code and every valid id."""
    with pytest.raises(UnknownCorpusSourceError) as excinfo:
        get_source_spec("oscar-2201")
    assert excinfo.value.code == ERR_UNKNOWN_SOURCE
    assert excinfo.value.known == ("oscar-2301", "wikipedia")


def test_each_source_round_trips_through_encode_decode() -> None:
    """Encoding a spec and decoding it again yields an equal spec."""
    for source_id, spec in SOURCE_REGISTRY.items():
        assert decode_source_spec(source_id, encode_source_spec(spec)) == spec


def test_oscar_entry_without_a_dataset_name_is_rejected(tmp_path: Path) -> None:
    """An OSCAR source with no dataset could never be streamed."""
    path = _write(tmp_path, "broken:\n  driver: oscar\n  license: CC0-1.0\n")
    with pytest.raises(FieldError) as excinfo:
        load_source_registry(path)
    assert excinfo.value.code == ERR_FIELD_MISSING
    assert excinfo.value.field == "hf_name"


def test_wikipedia_entry_with_a_dataset_name_is_rejected(tmp_path: Path) -> None:
    """A field the driver cannot use is refused, not silently ignored."""
    path = _write(
        tmp_path,
        "broken:\n  driver: wikipedia\n  license: CC-BY-SA-3.0\n  hf_name: nope\n",
    )
    with pytest.raises(FieldError) as excinfo:
        load_source_registry(path)
    assert excinfo.value.field == "hf_name"
    assert excinfo.value.code == ERR_FIELD_TYPE


def test_unknown_driver_lists_the_drivers_that_exist(tmp_path: Path) -> None:
    """A driver name with no reader behind it fails at load time."""
    path = _write(tmp_path, "broken:\n  driver: leipzig\n  license: CC-BY-4.0\n")
    with pytest.raises(FieldError) as excinfo:
        load_source_registry(path)
    assert excinfo.value.detail == "expected one of oscar, wikipedia, got 'leipzig'"


def test_entry_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    """A source whose body is a scalar names the offending source id."""
    path = _write(tmp_path, "broken: just-a-string\n")
    with pytest.raises(FieldError) as excinfo:
        load_source_registry(path)
    assert excinfo.value.field == "broken"
    assert excinfo.value.detail == "expected a mapping, got str"


def test_document_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    """A registry written as a list fails before any entry is read."""
    path = _write(tmp_path, "- oscar-2301\n- wikipedia\n")
    with pytest.raises(FieldError) as excinfo:
        load_source_registry(path)
    assert excinfo.value.field == "<registry>"
    assert excinfo.value.detail == "expected a mapping of source ids, got list"


def test_empty_source_id_is_rejected() -> None:
    """A blank key cannot name a source, so decoding refuses it."""
    with pytest.raises(FieldError) as excinfo:
        decode_source_registry({"  ": {"driver": "wikipedia", "license": "CC0-1.0"}})
    assert excinfo.value.field == "source_id"


def test_decoded_registry_preserves_document_order(tmp_path: Path) -> None:
    """Entries keep the order the document gave them."""
    path = _write(
        tmp_path,
        "zeta:\n  driver: wikipedia\n  license: CC-BY-SA-3.0\n"
        "alpha:\n  driver: wikipedia\n  license: CC-BY-SA-3.0\n",
    )
    assert tuple(load_source_registry(path)) == ("zeta", "alpha")
