"""Tests for classification and classifier construction.

``ScriptedModel`` is a real implementation of the :class:`FastTextModel`
protocol that answers from a table, not a mock: it has no assertion
helpers and records nothing. Every test checks a returned value or a
raised error code.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from pathlib import Path

import pytest

from turkic_translit.lid import _test_hooks
from turkic_translit.lid.classifier import (
    LidClassifier,
    LidPrediction,
    encode_lid_prediction,
)
from turkic_translit.lid.errors import (
    ERR_EMPTY_TEXT,
    ERR_LABEL_MALFORMED,
    EmptyClassificationTextError,
    LidLabelError,
)
from turkic_translit.lid.factory import build_classifier, encode_lid_run_record
from turkic_translit.lid.registry import get_spec


class ScriptedModel:
    """Model answering from a fixed text-to-(label, probability) table.

    Args:
        answers: Mapping of exact input text to raw label and probability.
    """

    def __init__(self, answers: Mapping[str, tuple[str, float]]) -> None:
        """Store the answer table backing this model."""
        self._answers = dict(answers)

    def predict(self, text: str, k: int) -> tuple[Sequence[str], Sequence[float]]:
        """Return the scripted answer for ``text``.

        Args:
            text: Cleaned input line.
            k: Number of predictions requested; always 1 here.

        Returns:
            Parallel one-element sequences of label and probability.
        """
        label, probability = self._answers[text]
        return ([label] * k, [probability] * k)


class ScriptedLoader:
    """Loader returning one prepared model regardless of path.

    Args:
        model: The model to return from every load.
    """

    def __init__(self, model: ScriptedModel) -> None:
        """Store the model this loader hands out."""
        self._model = model

    def load(self, path: Path) -> ScriptedModel:
        """Return the prepared model.

        Args:
            path: Ignored; present to satisfy the protocol.

        Returns:
            The prepared model.
        """
        return self._model


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
    model = ScriptedModel({"salom": ("__label__uzn_Latn", 0.99)})
    result = LidClassifier(get_spec("lid218e"), model).classify("salom")
    assert result == LidPrediction(label="uzn_Latn", probability=0.99)


def test_classify_removes_sentencepiece_markers() -> None:
    """Word markers are stripped before the model sees the text."""
    model = ScriptedModel({"salom dunyo": ("__label__uzn_Latn", 0.97)})
    result = LidClassifier(get_spec("lid218e"), model).classify("▁salom ▁dunyo")
    assert result["label"] == "uzn_Latn"


def test_classify_rejects_empty_text() -> None:
    """Whitespace-only input raises rather than returning a sentinel."""
    classifier = LidClassifier(get_spec("lid.176"), ScriptedModel({}))
    with pytest.raises(EmptyClassificationTextError) as excinfo:
        classifier.classify("   ▁  ")
    assert excinfo.value.code == ERR_EMPTY_TEXT


def test_classify_rejects_a_label_without_the_declared_prefix() -> None:
    """A bare label means the loaded weights are not the declared model."""
    model = ScriptedModel({"salom": ("uzn_Latn", 0.99)})
    with pytest.raises(LidLabelError) as excinfo:
        LidClassifier(get_spec("lid218e"), model).classify("salom")
    assert excinfo.value.code == ERR_LABEL_MALFORMED


def test_accepts_applies_both_language_and_threshold() -> None:
    """A line passes only when the label matches and confidence suffices."""
    model = ScriptedModel(
        {
            "keep": ("__label__uzn_Latn", 0.99),
            "low": ("__label__uzn_Latn", 0.80),
            "other": ("__label__rus_Cyrl", 0.99),
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


def test_build_classifier_records_what_backed_the_run(
    restore_hooks: None, tmp_path: Path
) -> None:
    """The run record names the model, path, size and threshold used."""
    weights = tmp_path / "lid218e.bin"
    weights.write_bytes(b"x" * 512)
    _test_hooks.model_loader = ScriptedLoader(
        ScriptedModel({"salom": ("__label__uzn_Latn", 0.99)})
    )

    classifier, record = build_classifier("lid218e", [tmp_path], tmp_path, 0.95)

    assert classifier.model_id == "lid218e"
    assert record["model_id"] == "lid218e"
    assert record["weights_path"] == str(weights)
    assert record["weights_bytes"] == 512
    assert record["threshold"] == 0.95
    assert record["script_aware"] is True


def test_run_record_encodes_to_a_manifest_mapping(
    restore_hooks: None, tmp_path: Path
) -> None:
    """The record encodes to exactly the fields a manifest should carry."""
    weights = tmp_path / "lid.176.bin"
    weights.write_bytes(b"y" * 16)
    _test_hooks.model_loader = ScriptedLoader(ScriptedModel({}))

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
            ["__label__uz salom dunyo qalaysiz"] * 20
            + ["__label__tr merhaba dunya nasilsin"] * 20
        )
        + "\n",
        encoding="utf-8",
    )
    trainer = __import__("fasttext")
    model = trainer.train_supervised(
        input=str(corpus), epoch=5, minCount=1, thread=1, dim=10
    )
    weights = tmp_path / "tiny.bin"
    model.save_model(str(weights))

    loaded = _test_hooks.FastTextLoader().load(weights)
    labels, probabilities = loaded.predict("salom dunyo qalaysiz", 1)

    assert labels[0] == "__label__uz"
    assert probabilities[0] > 0.5
