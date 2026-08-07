"""Typed registry of the corpora this project can stream.

A source is described entirely by data: which driver reads it, what it is
licensed under, and — for Hugging Face datasets — which dataset holds it.
The description is a tagged union keyed on ``driver``, so the fields that
exist depend on the driver rather than on convention: an OSCAR source
always has a dataset name, and a Wikipedia source never does, and neither
fact is left to a caller to remember.

The registry is read from ``corpora.yaml`` at import and validated field
by field. A malformed entry fails when the package loads, naming the
source and the field, rather than surfacing as a confusing failure part
way through a multi-gigabyte download.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, TypedDict

import yaml

from turkic_translit.corpus.errors import UnknownCorpusSourceError
from turkic_translit.validation import (
    ERR_FIELD_TYPE,
    FieldError,
    require_absent,
    require_mapping,
    require_non_empty_str,
    require_one_of,
    require_present,
)

DRIVERS: Final[tuple[str, ...]] = ("oscar", "wikipedia")

SOURCE_REGISTRY_PATH: Final[Path] = Path(__file__).resolve().parent / "corpora.yaml"


class OscarSourceSpec(TypedDict):
    """A corpus streamed from a Hugging Face dataset.

    Attributes:
        source_id: Registry key, e.g. ``oscar-2301``.
        driver: Always ``oscar``; the tag that discriminates this union.
        license: SPDX identifier the corpus is published under.
        hf_name: Hugging Face dataset name, e.g.
            ``oscar-corpus/OSCAR-2301``.
    """

    source_id: str
    driver: Literal["oscar"]
    license: str
    hf_name: str


class WikipediaSourceSpec(TypedDict):
    """A corpus streamed from Wikimedia's own XML dumps.

    There is no dataset name because the dump URL is derived from the
    language code alone.

    Attributes:
        source_id: Registry key, e.g. ``wikipedia``.
        driver: Always ``wikipedia``; the tag that discriminates this
            union.
        license: SPDX identifier the corpus is published under.
    """

    source_id: str
    driver: Literal["wikipedia"]
    license: str


def decode_source_spec(
    source_id: str, source: Mapping[str, str | int | float | bool]
) -> OscarSourceSpec | WikipediaSourceSpec:
    """Validate one registry entry into a typed source specification.

    Args:
        source_id: Registry key the entry was filed under.
        source: Mapping of that entry's fields.

    Returns:
        The specification, whose type is fixed by its ``driver`` field.

    Raises:
        FieldError: If the driver is unrecognised, a required field is
            missing or malformed, or a field is present that the driver
            forbids.
    """
    checked_id = require_non_empty_str("source_id", source_id)
    license_name = require_non_empty_str("license", require_present("license", source))
    driver = require_one_of("driver", require_present("driver", source), DRIVERS)
    if driver == "oscar":
        return OscarSourceSpec(
            source_id=checked_id,
            driver="oscar",
            license=license_name,
            hf_name=require_non_empty_str("hf_name", require_present("hf_name", source)),
        )
    require_absent(
        "hf_name",
        source,
        "the wikipedia driver derives its dump URL from the language code",
    )
    return WikipediaSourceSpec(source_id=checked_id, driver="wikipedia", license=license_name)


def decode_source_registry(
    document: Mapping[
        str, str | int | float | bool | None | Mapping[str, str | int | float | bool]
    ],
) -> dict[str, OscarSourceSpec | WikipediaSourceSpec]:
    """Validate a whole registry document into typed specifications.

    Args:
        document: Mapping of source id to that source's fields, as read
            from YAML.

    Returns:
        The decoded registry, preserving document order.

    Raises:
        FieldError: If any entry is not a mapping, or fails field
            validation.
    """
    decoded: dict[str, OscarSourceSpec | WikipediaSourceSpec] = {}
    for source_id, entry in document.items():
        decoded[source_id] = decode_source_spec(source_id, require_mapping(source_id, entry))
    return decoded


def encode_source_spec(
    spec: OscarSourceSpec | WikipediaSourceSpec,
) -> dict[str, str]:
    """Render a source specification back to a plain mapping.

    The inverse of :func:`decode_source_spec`, minus the ``source_id``
    that the registry carries as the entry's key.

    Args:
        spec: The specification to encode.

    Returns:
        A mapping carrying exactly the fields this driver defines.
    """
    if spec["driver"] == "oscar":
        return {
            "driver": spec["driver"],
            "license": spec["license"],
            "hf_name": spec["hf_name"],
        }
    return {"driver": spec["driver"], "license": spec["license"]}


def load_source_registry(path: Path) -> dict[str, OscarSourceSpec | WikipediaSourceSpec]:
    """Read and validate a registry document from disk.

    Args:
        path: Location of the YAML registry.

    Returns:
        The decoded registry.

    Raises:
        FieldError: If the document is not a mapping of source ids, or
            any entry fails validation.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise FieldError(
            ERR_FIELD_TYPE,
            "<registry>",
            f"expected a mapping of source ids, got {type(document).__name__}",
        )
    return decode_source_registry(document)


SOURCE_REGISTRY: Final[Mapping[str, OscarSourceSpec | WikipediaSourceSpec]] = load_source_registry(
    SOURCE_REGISTRY_PATH
)


def known_source_ids() -> tuple[str, ...]:
    """List every registered source identifier in registry order.

    Returns:
        The registered identifiers.
    """
    return tuple(SOURCE_REGISTRY)


def get_source_spec(source_id: str) -> OscarSourceSpec | WikipediaSourceSpec:
    """Look up one source specification by identifier.

    Args:
        source_id: Registry key, e.g. ``oscar-2301``.

    Returns:
        The registered specification.

    Raises:
        UnknownCorpusSourceError: If no source is registered under that
            id.
    """
    spec = SOURCE_REGISTRY.get(source_id)
    if spec is None:
        raise UnknownCorpusSourceError(source_id, known_source_ids())
    return spec


__all__ = [
    "DRIVERS",
    "SOURCE_REGISTRY",
    "SOURCE_REGISTRY_PATH",
    "OscarSourceSpec",
    "WikipediaSourceSpec",
    "decode_source_registry",
    "decode_source_spec",
    "encode_source_spec",
    "get_source_spec",
    "known_source_ids",
    "load_source_registry",
]
