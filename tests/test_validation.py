"""Tests for the shared boundary validators.

Every check is exercised on both a value it accepts and a value it
rejects, and every rejection is pinned to its error code and field name,
because the codes are what a caller branches on.
"""

from __future__ import annotations

import pytest

from turkic_translit.validation import (
    ERR_FIELD_EMPTY,
    ERR_FIELD_MISSING,
    ERR_FIELD_RANGE,
    ERR_FIELD_TYPE,
    FieldError,
    require_absent,
    require_bool,
    require_float,
    require_int,
    require_mapping,
    require_non_empty_str,
    require_non_negative_int,
    require_one_of,
    require_optional_mapping,
    require_optional_non_empty_str,
    require_present,
    require_probability,
    require_str,
)


def test_present_returns_the_stored_value() -> None:
    """A key that exists yields its value unchanged."""
    assert require_present("a", {"a": "value"}) == "value"


def test_present_treats_an_explicit_null_as_present() -> None:
    """A field written as null is present, so optionality stays explicit."""
    assert require_present("a", {"a": None}) is None


def test_present_rejects_a_missing_key() -> None:
    """An absent key is reported with the missing code and its name."""
    with pytest.raises(FieldError) as excinfo:
        require_present("threshold", {"language": "uz"})
    assert excinfo.value.code == ERR_FIELD_MISSING
    assert excinfo.value.field == "threshold"


def test_str_accepts_an_empty_string() -> None:
    """``require_str`` checks the type only, so empty text passes."""
    assert require_str("a", "") == ""


def test_str_rejects_a_number() -> None:
    """A numeric value where text was declared fails with a type code."""
    with pytest.raises(FieldError) as excinfo:
        require_str("a", 7)
    assert excinfo.value.code == ERR_FIELD_TYPE
    assert excinfo.value.detail == "expected str, got int"


def test_non_empty_str_returns_text_with_content() -> None:
    """Text with content is returned unchanged, inner spacing intact."""
    assert require_non_empty_str("a", " uzn Latn ") == " uzn Latn "


def test_non_empty_str_rejects_whitespace_only_text() -> None:
    """Whitespace-only text fails with the empty code, not the type code."""
    with pytest.raises(FieldError) as excinfo:
        require_non_empty_str("model_id", "   ")
    assert excinfo.value.code == ERR_FIELD_EMPTY
    assert excinfo.value.field == "model_id"


def test_optional_non_empty_str_passes_null_through() -> None:
    """A null optional string stays null rather than becoming empty text."""
    assert require_optional_non_empty_str("filter_language", None) is None


def test_optional_non_empty_str_validates_a_present_value() -> None:
    """A present optional string is held to the non-empty rule."""
    assert require_optional_non_empty_str("filter_language", "uzn") == "uzn"


def test_optional_non_empty_str_rejects_empty_text() -> None:
    """An optional field written as empty text is an error, not a null."""
    with pytest.raises(FieldError) as excinfo:
        require_optional_non_empty_str("filter_language", "")
    assert excinfo.value.code == ERR_FIELD_EMPTY


def test_bool_returns_the_flag() -> None:
    """A genuine boolean is returned unchanged."""
    assert require_bool("script_aware", True) is True


def test_bool_rejects_the_string_yes() -> None:
    """A YAML ``yes`` that survived as text is not accepted as a flag."""
    with pytest.raises(FieldError) as excinfo:
        require_bool("script_aware", "yes")
    assert excinfo.value.code == ERR_FIELD_TYPE
    assert excinfo.value.detail == "expected bool, got str"


def test_int_returns_the_number() -> None:
    """An integer is returned unchanged."""
    assert require_int("lines", 12) == 12


def test_int_rejects_a_bool_despite_the_subclass() -> None:
    """``True`` is an ``int`` in Python but never a legitimate count."""
    with pytest.raises(FieldError) as excinfo:
        require_int("lines", True)
    assert excinfo.value.detail == "expected int, got bool"


def test_int_rejects_a_float() -> None:
    """A fractional value in a count field fails with a type code."""
    with pytest.raises(FieldError) as excinfo:
        require_int("lines", 1.5)
    assert excinfo.value.detail == "expected int, got float"


def test_non_negative_int_accepts_zero() -> None:
    """Zero lines written is a legitimate outcome, not an error."""
    assert require_non_negative_int("lines_written", 0) == 0


def test_non_negative_int_rejects_a_negative_count() -> None:
    """A negative count fails with the range code."""
    with pytest.raises(FieldError) as excinfo:
        require_non_negative_int("lines_written", -1)
    assert excinfo.value.code == ERR_FIELD_RANGE
    assert excinfo.value.detail == "must not be negative, got -1"


def test_float_widens_an_integer() -> None:
    """A threshold written as ``1`` is the same number as ``1.0``."""
    assert require_float("threshold", 1) == 1.0


def test_float_returns_a_fractional_value() -> None:
    """A fractional value passes through unchanged."""
    assert require_float("threshold", 0.95) == 0.95


def test_float_rejects_a_bool() -> None:
    """A boolean threshold is a malformed document, not the number one."""
    with pytest.raises(FieldError) as excinfo:
        require_float("threshold", True)
    assert excinfo.value.detail == "expected float, got bool"


def test_float_rejects_text() -> None:
    """A threshold left as a string fails rather than being parsed."""
    with pytest.raises(FieldError) as excinfo:
        require_float("threshold", "0.95")
    assert excinfo.value.detail == "expected float, got str"


def test_probability_accepts_both_ends_of_the_interval() -> None:
    """Zero and one are inside the interval and are kept as floats."""
    assert require_probability("threshold", 0) == 0.0
    assert require_probability("threshold", 1.0) == 1.0


def test_probability_rejects_a_negative_value() -> None:
    """A negative threshold fails with the range code."""
    with pytest.raises(FieldError) as excinfo:
        require_probability("threshold", -0.1)
    assert excinfo.value.code == ERR_FIELD_RANGE


def test_probability_rejects_a_percentage() -> None:
    """A threshold given as 95 rather than 0.95 is caught, not applied."""
    with pytest.raises(FieldError) as excinfo:
        require_probability("threshold", 95)
    assert excinfo.value.detail == "must lie between 0.0 and 1.0, got 95.0"


def test_one_of_returns_a_permitted_value() -> None:
    """A value from the vocabulary is returned unchanged."""
    assert require_one_of("driver", "oscar", ("oscar", "wikipedia")) == "oscar"


def test_one_of_lists_the_whole_vocabulary_on_failure() -> None:
    """A rejection names every value that would have been accepted."""
    with pytest.raises(FieldError) as excinfo:
        require_one_of("driver", "leipzig", ("oscar", "wikipedia"))
    assert excinfo.value.detail == "expected one of oscar, wikipedia, got 'leipzig'"


def test_one_of_rejects_a_non_string() -> None:
    """A non-string is rejected before membership is considered."""
    with pytest.raises(FieldError) as excinfo:
        require_one_of("driver", 3, ("oscar", "wikipedia"))
    assert excinfo.value.detail == "expected str, got int"


def test_mapping_returns_the_nested_document() -> None:
    """A nested mapping is returned ready for its own decoder."""
    assert require_mapping("lid", {"model_id": "lid218e"}) == {"model_id": "lid218e"}


def test_mapping_rejects_a_scalar() -> None:
    """A scalar where a nested document was declared fails with a type code."""
    with pytest.raises(FieldError) as excinfo:
        require_mapping("lid", "lid218e")
    assert excinfo.value.detail == "expected a mapping, got str"


def test_optional_mapping_passes_null_through() -> None:
    """A null nested document stays null rather than becoming empty."""
    assert require_optional_mapping("lid", None) is None


def test_optional_mapping_validates_a_present_value() -> None:
    """A present nested document is held to the mapping rule."""
    assert require_optional_mapping("lid", {"a": 1}) == {"a": 1}


def test_optional_mapping_rejects_a_scalar() -> None:
    """A scalar in an optional nested field is an error, not a null."""
    with pytest.raises(FieldError) as excinfo:
        require_optional_mapping("lid", 4)
    assert excinfo.value.code == ERR_FIELD_TYPE


def test_absent_accepts_a_document_without_the_field() -> None:
    """A document lacking the forbidden field passes, and is left alone."""
    document = {"driver": "wikipedia", "license": "CC-BY-SA-3.0"}
    require_absent("hf_name", document, "not applicable")
    assert sorted(document) == ["driver", "license"]


def test_absent_rejects_a_field_the_driver_forbids() -> None:
    """A surplus field is refused with the reason it cannot appear."""
    with pytest.raises(FieldError) as excinfo:
        require_absent("hf_name", {"hf_name": "x"}, "wikipedia derives its own URL")
    assert excinfo.value.detail == "must not be present: wikipedia derives its own URL"


def test_field_error_renders_code_field_and_detail() -> None:
    """The string form carries all three parts so logs stay greppable."""
    error = FieldError(ERR_FIELD_TYPE, "driver", "expected str, got int")
    assert str(error) == "TURKIC_FIELD_002_TYPE: field 'driver': expected str, got int"
