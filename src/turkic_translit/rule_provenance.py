"""The published description each rule file implements.

Every IPA rule file states, in its own header, the phonological
description its mappings come from. That statement used to be prose in a
comment: readable by a person, invisible to everything else, and
therefore unable to constrain anything. A test could claim a source the
rules never cited, and for most of this project's history several did.

The header is now a structured block, read here into a validated
:class:`RuleSource`. Two things follow from that. The library can answer
"what backs the Kazakh rules" as data rather than as a comment, which is
what a researcher inspecting this project needs. And a gold-standard test
can declare the source its expected values were taken from and have that
checked against what the rule file itself declares, so the provenance the
rules carry is inherited by the tests that verify them rather than
restated on trust.

Parsing is strict and total. A header missing a field, or carrying one
that is empty, fails at the boundary with a code naming the file and the
field, because a rule set whose provenance cannot be read is a rule set
whose claims cannot be checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

from turkic_translit import _test_hooks
from turkic_translit.rule_errors import (
    RuleSourceFieldMissingError,
    RuleSourceMalformedLineError,
)
from turkic_translit.validation import (
    require_int,
    require_non_empty_str,
    require_present,
)

FIELD_PREFIX: Final = "# Source-"
AUTHORS_FIELD: Final = "Authors"
YEAR_FIELD: Final = "Year"
TITLE_FIELD: Final = "Title"
CONTAINER_FIELD: Final = "Container"
IDENTIFIER_FIELD: Final = "Id"

REQUIRED_FIELDS: Final = (
    AUTHORS_FIELD,
    YEAR_FIELD,
    TITLE_FIELD,
    CONTAINER_FIELD,
    IDENTIFIER_FIELD,
)


class RuleSource(TypedDict):
    """The published description one rule file implements.

    Attributes:
        authors: Author list as the source itself gives it.
        year: Year of publication.
        title: Title of the work.
        container: Journal, series or publisher, with volume and pages
            where the source has them.
        identifier: Canonical resolvable identifier, a DOI URL where the
            work has a DOI and a permanent URL otherwise.
    """

    authors: str
    year: int
    title: str
    container: str
    identifier: str


def decode_rule_source(source: Mapping[str, str | int]) -> RuleSource:
    """Validate a loosely-typed mapping into a :class:`RuleSource`.

    Args:
        source: Mapping holding the five provenance fields.

    Returns:
        A fully validated source record.

    Raises:
        FieldError: If any field is missing, of the wrong type, or empty.
    """
    return RuleSource(
        authors=require_non_empty_str("authors", require_present("authors", source)),
        year=require_int("year", require_present("year", source)),
        title=require_non_empty_str("title", require_present("title", source)),
        container=require_non_empty_str("container", require_present("container", source)),
        identifier=require_non_empty_str("identifier", require_present("identifier", source)),
    )


def encode_rule_source(record: RuleSource) -> dict[str, str | int]:
    """Render a source record back to a plain mapping.

    The inverse of :func:`decode_rule_source`, used for manifest writing
    and for round-trip assertions.

    Args:
        record: The record to encode.

    Returns:
        A mapping carrying exactly the five provenance fields.
    """
    return {
        "authors": record["authors"],
        "year": record["year"],
        "title": record["title"],
        "container": record["container"],
        "identifier": record["identifier"],
    }


def parse_rule_source(text: str, origin: str) -> RuleSource:
    """Read the structured provenance block out of rule-file text.

    Only lines beginning ``# Source-`` are considered, so ordinary
    comments and the rules themselves are ignored. Every field named in
    :data:`REQUIRED_FIELDS` must be present exactly once.

    Args:
        text: Full contents of a rule file.
        origin: Name of the file, used in error messages.

    Returns:
        The validated source record the header declares.

    Raises:
        RuleSourceMalformedLineError: If a ``# Source-`` line carries no
            ``:`` separator, or names a field twice.
        RuleSourceFieldMissingError: If a required field is absent.
        FieldError: If a present field fails validation.
    """
    collected: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith(FIELD_PREFIX):
            continue
        body = line[len(FIELD_PREFIX) :]
        if ":" not in body:
            raise RuleSourceMalformedLineError(origin, line)
        name, _, value = body.partition(":")
        field = name.strip()
        if field in collected:
            raise RuleSourceMalformedLineError(origin, line)
        collected[field] = value.strip()

    for field in REQUIRED_FIELDS:
        if field not in collected:
            raise RuleSourceFieldMissingError(origin, field)

    year_text = collected[YEAR_FIELD]
    if not year_text.isdigit():
        raise RuleSourceMalformedLineError(origin, f"{FIELD_PREFIX}{YEAR_FIELD}: {year_text}")

    return decode_rule_source(
        {
            "authors": collected[AUTHORS_FIELD],
            "year": int(year_text),
            "title": collected[TITLE_FIELD],
            "container": collected[CONTAINER_FIELD],
            "identifier": collected[IDENTIFIER_FIELD],
        }
    )


def read_rule_source(path: Path) -> RuleSource:
    """Read one rule file's declared source from disk.

    Args:
        path: Path to a ``.rules`` file.

    Returns:
        The validated source record its header declares.

    Raises:
        RuleSourceMalformedLineError: If the header is malformed.
        RuleSourceFieldMissingError: If a required field is absent.
        FieldError: If a present field fails validation.
    """
    return parse_rule_source(_test_hooks.rule_text.read(path), path.name)


__all__ = [
    "AUTHORS_FIELD",
    "CONTAINER_FIELD",
    "FIELD_PREFIX",
    "IDENTIFIER_FIELD",
    "REQUIRED_FIELDS",
    "TITLE_FIELD",
    "YEAR_FIELD",
    "RuleSource",
    "decode_rule_source",
    "encode_rule_source",
    "parse_rule_source",
    "read_rule_source",
]
