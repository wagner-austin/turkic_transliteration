"""Tests for how a run decides which lines it keeps.

The classifier is a real :class:`FastTextModel` answering from a table
and the filesystem probe is a real probe answering from a mapping, so
:func:`build_line_filter` runs its whole path — registry lookup,
resolution, load, record construction — without reading a gigabyte of
weights off disk.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from turkic_translit.corpus.filtering import (
    KeepEveryLine,
    LanguageLineFilter,
    LidFilterRequest,
    build_line_filter,
    decode_lid_filter_request,
    encode_lid_filter_request,
)
from turkic_translit.lid import _test_hooks
from turkic_translit.lid.classifier import LidClassifier
from turkic_translit.lid.errors import UnknownLidModelError
from turkic_translit.lid.locations import default_search_dirs
from turkic_translit.lid.registry import get_spec
from turkic_translit.validation import (
    ERR_FIELD_MISSING,
    ERR_FIELD_RANGE,
    FieldError,
)

ANSWERS = {
    "salom dunyo": [("__label__uzn_Latn", 0.99)],
    "marginal": [("__label__uzn_Latn", 0.80)],
    "privet mir": [("__label__rus_Cyrl", 0.99)],
}


@pytest.fixture
def installed_weights() -> Generator[None, None, None]:
    """Present ``lid218e`` as installed, and load a table-backed model.

    The weights are reported by a mapping probe at the first directory
    the resolver searches, so the production search order is exercised
    without a real model file.

    Yields:
        None, once, with the original hooks captured.
    """
    probe, loader = _test_hooks.probe, _test_hooks.model_loader
    _test_hooks.probe = _test_hooks.MappingFileProbe(
        {default_search_dirs()[0] / "lid218e.bin": 1176355829}
    )
    _test_hooks.model_loader = _test_hooks.FixedModelLoader(_test_hooks.TableFastTextModel(ANSWERS))
    yield
    _test_hooks.probe = probe
    _test_hooks.model_loader = loader


def test_filter_request_round_trips_through_encode_decode() -> None:
    """A request encodes and decodes back to an equal request."""
    original = LidFilterRequest(language="uzn", model_id="lid218e", threshold=0.95)
    assert decode_lid_filter_request(encode_lid_filter_request(original)) == original


def test_filter_request_rejects_a_threshold_given_as_a_percentage() -> None:
    """``--lid-threshold 95`` is caught at the boundary, not applied."""
    with pytest.raises(FieldError) as excinfo:
        decode_lid_filter_request({"language": "uzn", "model_id": "lid218e", "threshold": 95})
    assert excinfo.value.code == ERR_FIELD_RANGE
    assert excinfo.value.field == "threshold"


def test_filter_request_rejects_a_missing_model() -> None:
    """A request that names no classifier cannot be reproduced."""
    with pytest.raises(FieldError) as excinfo:
        decode_lid_filter_request({"language": "uzn", "threshold": 0.95})
    assert excinfo.value.code == ERR_FIELD_MISSING
    assert excinfo.value.field == "model_id"


def test_keep_every_line_keeps_what_it_is_given() -> None:
    """The unfiltered case is a filter that accepts, not a null to test."""
    assert KeepEveryLine().keeps("anything at all") is True


def test_language_filter_applies_both_label_and_threshold() -> None:
    """A line passes only when the label matches and confidence suffices."""
    line_filter = LanguageLineFilter(
        LidClassifier(get_spec("lid218e"), _test_hooks.TableFastTextModel(ANSWERS)),
        "uzn",
        0.95,
    )
    assert line_filter.keeps("salom dunyo") is True
    assert line_filter.keeps("marginal") is False
    assert line_filter.keeps("privet mir") is False


def test_no_request_yields_the_keep_everything_filter() -> None:
    """An unfiltered run loads no model, so there is no record to write."""
    line_filter, record = build_line_filter(None)
    assert record is None
    assert line_filter.keeps("salom dunyo") is True


def test_a_request_yields_a_filter_and_the_record_naming_its_weights(
    installed_weights: None,
) -> None:
    """The record identifies exactly the weights the filter is using."""
    line_filter, record = build_line_filter(
        LidFilterRequest(language="uzn", model_id="lid218e", threshold=0.95)
    )

    assert record == {
        "model_id": "lid218e",
        "weights_path": str(default_search_dirs()[0] / "lid218e.bin"),
        "weights_bytes": 1176355829,
        "threshold": 0.95,
        "script_aware": True,
    }
    assert line_filter.keeps("privet mir") is False


def test_an_unregistered_model_is_refused_before_any_download(
    installed_weights: None,
) -> None:
    """Naming a classifier that does not exist fails immediately."""
    with pytest.raises(UnknownLidModelError) as excinfo:
        build_line_filter(LidFilterRequest(language="uzn", model_id="lid999", threshold=0.95))
    assert excinfo.value.known == ("lid.176", "lid218e")
