"""Provenance parsing, including every way a header can fail to parse.

The rule files in this repository all carry a well-formed header, so the
happy path is exercised by every language test. What is not exercised
there is the failure side, and the failure side is the point: a rule set
whose provenance cannot be read is a rule set whose claims cannot be
checked, so the parser refuses rather than guessing.

Each failure is checked by its stable code rather than its message, so
the wording can improve without the test noticing and a caller can tell
one failure from another without reading prose.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from turkic_translit import _test_hooks
from turkic_translit.rule_errors import (
    ERR_SOURCE_FIELD_MISSING,
    ERR_SOURCE_LINE_MALFORMED,
    RuleSourceError,
    RuleSourceFieldMissingError,
    RuleSourceMalformedLineError,
)
from turkic_translit.rule_provenance import (
    REQUIRED_FIELDS,
    RuleSource,
    decode_rule_source,
    encode_rule_source,
    parse_rule_source,
    read_rule_source,
)
from turkic_translit.validation import FieldError

ORIGIN = "xx_ipa.rules"

WELL_FORMED = """\
# Example → IPA transliteration rules (xx_ipa.rules)
#
# Source-Authors: Doe, J. & Roe, R.
# Source-Year: 1999
# Source-Title: An Illustration of the IPA
# Source-Container: Journal of the International Phonetic Association 29(1): 1-4
# Source-Id: https://doi.org/10.1017/S0000000000000000
#
# An ordinary comment, which is not a Source- line and must be ignored.

a > ɑ ;
"""

EXPECTED = RuleSource(
    authors="Doe, J. & Roe, R.",
    year=1999,
    title="An Illustration of the IPA",
    container="Journal of the International Phonetic Association 29(1): 1-4",
    identifier="https://doi.org/10.1017/S0000000000000000",
)


def without_field(field: str) -> str:
    """Drop one provenance line from the well-formed header.

    Args:
        field: Name of the field to remove, as it appears after the
            ``# Source-`` prefix.

    Returns:
        The header text with that one line gone.
    """
    return "\n".join(
        line for line in WELL_FORMED.splitlines() if not line.startswith(f"# Source-{field}:")
    )


@pytest.fixture
def rule_text_of(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Serve one rule file's text from memory rather than from disk.

    Args:
        request: Provides the text to serve through ``request.param``.

    Yields:
        The path the text answers to, with the original reader restored
        afterwards.
    """
    path = Path("rules") / ORIGIN
    previous = _test_hooks.rule_text
    _test_hooks.rule_text = _test_hooks.MappingRuleText({path: str(request.param)})
    yield path
    _test_hooks.rule_text = previous


def test_a_well_formed_header_parses_to_its_five_fields() -> None:
    """Every field is read, stripped, and the year is an int."""
    assert parse_rule_source(WELL_FORMED, ORIGIN) == EXPECTED


def test_lines_that_are_not_provenance_are_ignored() -> None:
    """Ordinary comments and rules do not reach the parser's field map.

    Stated because the header sits among both, and a parser that read
    every comment would fail on the first one carrying a colon.
    """
    parsed = parse_rule_source(WELL_FORMED, ORIGIN)

    assert "ordinary" not in parsed["title"]
    assert parsed["title"] == "An Illustration of the IPA"


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_a_missing_field_names_itself_and_its_file(field: str) -> None:
    """Each required field fails on its own with a traceable code."""
    with pytest.raises(RuleSourceFieldMissingError) as raised:
        parse_rule_source(without_field(field), ORIGIN)

    assert raised.value.code == ERR_SOURCE_FIELD_MISSING
    assert raised.value.field == field
    assert raised.value.origin == ORIGIN
    assert ORIGIN in str(raised.value)


def test_a_provenance_line_with_no_separator_is_rejected() -> None:
    """A Source- line carrying no colon cannot be split into a field."""
    text = WELL_FORMED.replace("# Source-Year: 1999", "# Source-Year 1999")

    with pytest.raises(RuleSourceMalformedLineError) as raised:
        parse_rule_source(text, ORIGIN)

    assert raised.value.code == ERR_SOURCE_LINE_MALFORMED
    assert raised.value.line == "# Source-Year 1999"


def test_a_field_declared_twice_is_rejected() -> None:
    """Two lines for one field leave no way to know which is meant.

    Silently keeping the first or the last would make the header say
    something its writer did not choose.
    """
    text = WELL_FORMED.replace(
        "# Source-Year: 1999",
        "# Source-Year: 1999\n# Source-Year: 2001",
    )

    with pytest.raises(RuleSourceMalformedLineError) as raised:
        parse_rule_source(text, ORIGIN)

    assert raised.value.code == ERR_SOURCE_LINE_MALFORMED
    assert "2001" in raised.value.line


@pytest.mark.parametrize("year", ["nineteen ninety-nine", "199x", "", "1999a"])
def test_a_year_that_is_not_a_number_is_rejected(year: str) -> None:
    """The year is an int in the record, so it must parse as one."""
    text = WELL_FORMED.replace("# Source-Year: 1999", f"# Source-Year: {year}")

    with pytest.raises(RuleSourceMalformedLineError) as raised:
        parse_rule_source(text, ORIGIN)

    assert raised.value.code == ERR_SOURCE_LINE_MALFORMED


def test_a_field_present_but_empty_fails_validation() -> None:
    """An empty title is not a title, and the decoder says so."""
    text = WELL_FORMED.replace("# Source-Title: An Illustration of the IPA", "# Source-Title:")

    with pytest.raises(FieldError):
        parse_rule_source(text, ORIGIN)


def test_decode_and_encode_round_trip() -> None:
    """A record encodes to a mapping that decodes back to itself."""
    assert decode_rule_source(encode_rule_source(EXPECTED)) == EXPECTED


def test_encoding_carries_exactly_the_five_fields() -> None:
    """No field is dropped on the way out and none is invented."""
    encoded = encode_rule_source(EXPECTED)

    assert set(encoded) == {"authors", "year", "title", "container", "identifier"}
    assert encoded["year"] == 1999


@pytest.mark.parametrize("missing", ["authors", "year", "title", "container", "identifier"])
def test_decoding_a_mapping_short_of_a_field_fails(missing: str) -> None:
    """The decoder validates rather than trusting its caller."""
    incomplete = {
        key: value for key, value in encode_rule_source(EXPECTED).items() if key != missing
    }

    with pytest.raises(FieldError):
        decode_rule_source(incomplete)


def test_decoding_a_year_that_is_not_an_int_fails() -> None:
    """The year field is typed, and the type is checked at the boundary."""
    wrong = dict(encode_rule_source(EXPECTED))
    wrong["year"] = "1999"

    with pytest.raises(FieldError):
        decode_rule_source(wrong)


@pytest.mark.parametrize("rule_text_of", [WELL_FORMED], indirect=True)
def test_reading_from_the_seam_parses_what_the_reader_returns(rule_text_of: Path) -> None:
    """The disk-reading entry point goes through the injected reader."""
    assert read_rule_source(rule_text_of) == EXPECTED


@pytest.mark.parametrize("rule_text_of", [without_field("Id")], indirect=True)
def test_reading_a_headerless_file_raises_from_the_seam(rule_text_of: Path) -> None:
    """A failure in the text surfaces through the same call.

    The seam matters here: the failure must come from parsing what the
    reader returned, not from the file being absent.
    """
    with pytest.raises(RuleSourceFieldMissingError) as raised:
        read_rule_source(rule_text_of)

    assert raised.value.origin == ORIGIN


def test_the_two_failures_share_a_base_and_keep_distinct_codes() -> None:
    """A caller can catch both, and still tell them apart.

    Catching by the base class is the behaviour a caller depends on, so
    it is exercised rather than asserted about the class object. The
    codes are the stable part of the contract and are written as
    literals, so renumbering one is a visible change here.
    """
    with pytest.raises(RuleSourceError) as absent:
        parse_rule_source(without_field("Id"), ORIGIN)

    with pytest.raises(RuleSourceError) as unreadable:
        parse_rule_source(WELL_FORMED.replace("# Source-Year: 1999", "# Source-Year"), ORIGIN)

    assert absent.value.code == "TURKIC_RULESRC_001_FIELD_MISSING"
    assert unreadable.value.code == "TURKIC_RULESRC_002_LINE_MALFORMED"
    assert absent.value.code != unreadable.value.code


def test_a_failure_carries_both_its_code_and_a_human_message() -> None:
    """The string form is the code and then the explanation."""
    failure = RuleSourceFieldMissingError(ORIGIN, "Container")

    assert str(failure).startswith(f"{ERR_SOURCE_FIELD_MISSING}: ")
    assert failure.message.startswith(ORIGIN)
    assert "Container" in failure.message
