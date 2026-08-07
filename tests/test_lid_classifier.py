"""Tests for classification and classifier construction.

:class:`~turkic_translit.lid._test_hooks.TableFastTextModel` is a real
implementation of the :class:`FastTextModel` protocol that answers from a
table, not a mock: it has no assertion helpers and records nothing. Every
test checks a returned value or a raised error code.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from turkic_translit.lid import _test_hooks
from turkic_translit.lid.classifier import (
    LidClassifier,
    LidPrediction,
    decode_lid_prediction,
    encode_lid_prediction,
)
from turkic_translit.lid.errors import (
    ERR_EMPTY_TEXT,
    ERR_LABEL_MALFORMED,
    ERR_MULTILINE_TEXT,
    EmptyClassificationTextError,
    LidLabelError,
    MultilineClassificationTextError,
)
from turkic_translit.lid.factory import (
    build_classifier,
    decode_lid_run_record,
    encode_lid_run_record,
)
from turkic_translit.lid.registry import get_spec
from turkic_translit.validation import ERR_FIELD_RANGE, FieldError


@pytest.fixture
def restore_hooks() -> Generator[None, None, None]:
    """Restore all three production hooks after a test rebinds them.

    Yields:
        None, once, with the originals captured.
    """
    probe, downloader, loader = (
        _test_hooks.probe,
        _test_hooks.downloader,
        _test_hooks.model_loader,
    )
    yield
    _test_hooks.probe = probe
    _test_hooks.downloader = downloader
    _test_hooks.model_loader = loader


def test_classify_strips_the_prefix_and_keeps_the_script() -> None:
    """A script-aware label survives prefix removal intact."""
    model = _test_hooks.TableFastTextModel({"salom": [("__label__uzn_Latn", 0.99)]})
    result = LidClassifier(get_spec("lid218e"), model).classify("salom")
    assert result == LidPrediction(label="uzn_Latn", probability=0.99)


def test_classify_removes_sentencepiece_markers() -> None:
    """Word markers are stripped before the model sees the text."""
    model = _test_hooks.TableFastTextModel({"salom dunyo": [("__label__uzn_Latn", 0.97)]})
    result = LidClassifier(get_spec("lid218e"), model).classify("▁salom ▁dunyo")
    assert result["label"] == "uzn_Latn"


def test_classify_rejects_empty_text() -> None:
    """Whitespace-only input raises rather than returning a sentinel."""
    classifier = LidClassifier(get_spec("lid.176"), _test_hooks.TableFastTextModel({}))
    with pytest.raises(EmptyClassificationTextError) as excinfo:
        classifier.classify("   ▁  ")
    assert excinfo.value.code == ERR_EMPTY_TEXT


def test_classify_rejects_a_label_without_the_declared_prefix() -> None:
    """A bare label means the loaded weights are not the declared model."""
    model = _test_hooks.TableFastTextModel({"salom": [("uzn_Latn", 0.99)]})
    with pytest.raises(LidLabelError) as excinfo:
        LidClassifier(get_spec("lid218e"), model).classify("salom")
    assert excinfo.value.code == ERR_LABEL_MALFORMED


def test_accepts_applies_both_language_and_threshold() -> None:
    """A line passes only when the label matches and confidence suffices."""
    model = _test_hooks.TableFastTextModel(
        {
            "keep": [("__label__uzn_Latn", 0.99)],
            "low": [("__label__uzn_Latn", 0.80)],
            "other": [("__label__rus_Cyrl", 0.99)],
        }
    )
    classifier = LidClassifier(get_spec("lid218e"), model)
    assert classifier.accepts("keep", "uzn", 0.95) is True
    assert classifier.accepts("low", "uzn", 0.95) is False
    assert classifier.accepts("other", "uzn", 0.95) is False


def test_encode_prediction_round_trips_the_fields() -> None:
    """Encoding a prediction preserves label and probability."""
    encoded = encode_lid_prediction(LidPrediction(label="uz", probability=0.5))
    assert encoded == {"label": "uz", "probability": 0.5}


def test_build_classifier_records_what_backed_the_run(restore_hooks: None, tmp_path: Path) -> None:
    """The run record names the model, path, size and threshold used."""
    weights = tmp_path / "lid218e.bin"
    weights.write_bytes(b"x" * 512)
    _test_hooks.model_loader = _test_hooks.FixedModelLoader(
        _test_hooks.TableFastTextModel({"salom": [("__label__uzn_Latn", 0.99)]})
    )

    classifier, record = build_classifier("lid218e", [tmp_path], tmp_path, 0.95)

    assert classifier.model_id == "lid218e"
    assert record["model_id"] == "lid218e"
    assert record["weights_path"] == str(weights)
    assert record["weights_bytes"] == 512
    assert record["threshold"] == 0.95
    assert record["script_aware"] is True


def test_run_record_encodes_to_a_manifest_mapping(restore_hooks: None, tmp_path: Path) -> None:
    """The record encodes to exactly the fields a manifest should carry."""
    weights = tmp_path / "lid.176.bin"
    weights.write_bytes(b"y" * 16)
    _test_hooks.model_loader = _test_hooks.FixedModelLoader(_test_hooks.TableFastTextModel({}))

    _classifier, record = build_classifier("lid.176", [tmp_path], tmp_path, 0.9)

    assert encode_lid_run_record(record) == {
        "model_id": "lid.176",
        "weights_path": str(weights),
        "weights_bytes": 16,
        "threshold": 0.9,
        "script_aware": False,
    }


def test_fasttext_loader_loads_a_real_model(tmp_path: Path) -> None:
    """The production loader loads a genuine fastText model from disk.

    A real model is trained here rather than stubbed, so the dynamic
    import and the narrowing assignment to :class:`FastTextModel` are
    exercised by running them. The corpus is tiny and the model trains
    in well under a second.
    """
    corpus = tmp_path / "train.txt"
    corpus.write_text(
        "\n".join(
            ["__label__uz salom dunyo qalaysiz"] * 20 + ["__label__tr merhaba dunya nasilsin"] * 20
        )
        + "\n",
        encoding="utf-8",
    )
    trainer = __import__("fasttext")
    model = trainer.train_supervised(input=str(corpus), epoch=5, minCount=1, thread=1, dim=10)
    weights = tmp_path / "tiny.bin"
    model.save_model(str(weights))

    loaded = _test_hooks.FastTextLoader().load(weights)
    labels, probabilities = loaded.predict("salom dunyo qalaysiz", 1)

    assert labels[0] == "__label__uz"
    assert probabilities[0] > 0.5


def test_prediction_round_trips_through_encode_decode() -> None:
    """Decoding an encoded prediction yields an equal prediction."""
    original = LidPrediction(label="uzn_Latn", probability=0.99)
    assert decode_lid_prediction(encode_lid_prediction(original)) == original


def test_decoding_a_prediction_rejects_an_out_of_range_probability() -> None:
    """A probability above one means the document is not a prediction."""
    with pytest.raises(FieldError) as excinfo:
        decode_lid_prediction({"label": "uzn_Latn", "probability": 1.5})
    assert excinfo.value.code == ERR_FIELD_RANGE


def test_run_record_round_trips_through_encode_decode(restore_hooks: None, tmp_path: Path) -> None:
    """A record written to a manifest decodes back to an equal record."""
    weights = tmp_path / "lid218e.bin"
    weights.write_bytes(b"z" * 64)
    _test_hooks.model_loader = _test_hooks.FixedModelLoader(_test_hooks.TableFastTextModel({}))

    _classifier, record = build_classifier("lid218e", [tmp_path], tmp_path, 0.95)

    assert decode_lid_run_record(encode_lid_run_record(record)) == record


def test_decoding_a_run_record_rejects_a_negative_size() -> None:
    """Weights cannot occupy a negative number of bytes."""
    with pytest.raises(FieldError) as excinfo:
        decode_lid_run_record(
            {
                "model_id": "lid218e",
                "weights_path": "/models/lid218e.bin",
                "weights_bytes": -1,
                "threshold": 0.95,
                "script_aware": True,
            }
        )
    assert excinfo.value.field == "weights_bytes"


def test_building_a_classifier_rejects_an_impossible_threshold(
    restore_hooks: None, tmp_path: Path
) -> None:
    """A threshold above one is refused before any weights are read."""
    _test_hooks.model_loader = _test_hooks.FixedModelLoader(_test_hooks.TableFastTextModel({}))
    with pytest.raises(FieldError) as excinfo:
        build_classifier("lid218e", [tmp_path], tmp_path, 1.5)
    assert excinfo.value.code == ERR_FIELD_RANGE


def test_classify_rejects_text_spanning_lines() -> None:
    """A newline would make fastText read a prefix and discard the rest."""
    classifier = LidClassifier(get_spec("lid.176"), _test_hooks.TableFastTextModel({}))
    with pytest.raises(MultilineClassificationTextError) as excinfo:
        classifier.classify("salom\ndunyo")
    assert excinfo.value.code == ERR_MULTILINE_TEXT


def test_known_labels_strips_the_prefix_from_every_label() -> None:
    """The vocabulary is reported in the same shape as a prediction."""
    model = _test_hooks.TableFastTextModel(
        {
            "a": [("__label__uzn_Latn", 0.9), ("__label__rus_Cyrl", 0.1)],
            "b": [("__label__uzn_Latn", 0.8)],
        }
    )
    classifier = LidClassifier(get_spec("lid218e"), model)
    assert classifier.known_labels() == ("uzn_Latn", "rus_Cyrl")


def test_classify_many_truncates_to_the_requested_count() -> None:
    """Asking for fewer predictions returns fewer, in rank order."""
    model = _test_hooks.TableFastTextModel(
        {
            "salom": [
                ("__label__uzn_Latn", 0.7),
                ("__label__tur_Latn", 0.2),
                ("__label__aze_Latn", 0.1),
            ]
        }
    )
    classifier = LidClassifier(get_spec("lid218e"), model)
    assert [p["label"] for p in classifier.classify_many("salom", 2)] == [
        "uzn_Latn",
        "tur_Latn",
    ]


def test_installed_classifier_loads_from_the_standard_locations(
    restore_hooks: None,
) -> None:
    """The convenience loader searches the project's own weight directories."""
    from turkic_translit.lid.factory import load_installed_classifier
    from turkic_translit.lid.locations import default_search_dirs

    _test_hooks.probe = _test_hooks.MappingFileProbe(
        {default_search_dirs()[0] / "lid.176.bin": 131266198}
    )
    _test_hooks.model_loader = _test_hooks.FixedModelLoader(
        _test_hooks.TableFastTextModel({"salom": [("__label__uz", 0.99)]})
    )

    classifier = load_installed_classifier("lid.176")

    assert classifier.model_id == "lid.176"
    assert classifier.classify("salom")["label"] == "uz"


def test_the_production_adapter_reports_a_real_model_vocabulary(
    tmp_path: Path,
) -> None:
    """The pybind adapter lists labels without touching an array library.

    Trained and read for real, so the native ``getLabels`` call and the
    prefix stripping above it are exercised by running them. This is the
    path that replaced fastText's own wrapper, whose ``predict`` ends in
    a NumPy call that NumPy 2 rejects.
    """
    corpus = tmp_path / "train.txt"
    corpus.write_text(
        "\n".join(
            ["__label__uz salom dunyo qalaysiz"] * 20 + ["__label__tr merhaba dunya nasilsin"] * 20
        )
        + "\n",
        encoding="utf-8",
    )
    trainer = __import__("fasttext")
    model = trainer.train_supervised(input=str(corpus), epoch=5, minCount=1, thread=1, dim=10)
    weights = tmp_path / "tiny.bin"
    model.save_model(str(weights))

    loaded = _test_hooks.FastTextLoader().load(weights)

    assert sorted(loaded.labels()) == ["__label__tr", "__label__uz"]


def test_the_table_model_reports_every_label_it_can_emit() -> None:
    """The in-memory model answers the vocabulary question too."""
    model = _test_hooks.TableFastTextModel(
        {"a": [("__label__uz", 0.9), ("__label__tr", 0.1)], "b": [("__label__uz", 0.8)]}
    )
    assert model.labels() == ["__label__uz", "__label__tr"]
