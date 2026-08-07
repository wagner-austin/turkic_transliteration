"""Typed classification against a named language-identification model.

The classifier carries its specification, so a prediction is always
interpretable: the label is stripped using the prefix that model
declares, and a label that does not carry it is an error rather than a
silently mangled string.

Empty input raises. The previous implementation returned a sentinel
``("unknown", 0.0)``, which let a caller filter on a language that no
model can emit and made an empty line indistinguishable from a confident
misclassification. Deciding what an empty line means belongs to the
caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict

from turkic_translit.lid.errors import EmptyClassificationTextError
from turkic_translit.lid.spec import LidModelSpec, strip_label_prefix


class FastTextModel(Protocol):
    """The single fastText method this package depends on."""

    def predict(self, text: str, k: int) -> tuple[Sequence[str], Sequence[float]]:
        """Return the top ``k`` labels and their probabilities.

        Args:
            text: Single line of text to classify.
            k: Number of predictions to return.

        Returns:
            Parallel sequences of raw labels and probabilities.
        """
        ...


class LidPrediction(TypedDict):
    """One classification result.

    Attributes:
        label: Language tag with the model's prefix removed, e.g.
            ``uzn_Latn`` for a script-aware model or ``uz`` otherwise.
        probability: Model confidence in the range 0.0 to 1.0.
    """

    label: str
    probability: float


def encode_lid_prediction(prediction: LidPrediction) -> dict[str, str | float]:
    """Render a prediction as a plain mapping for manifest writing.

    Args:
        prediction: The prediction to encode.

    Returns:
        A mapping carrying the label and probability.
    """
    return {"label": prediction["label"], "probability": prediction["probability"]}


class LidClassifier:
    """A loaded model bound to the specification that describes it.

    Args:
        spec: Specification of the model held in ``model``.
        model: Loaded fastText model satisfying :class:`FastTextModel`.
    """

    def __init__(self, spec: LidModelSpec, model: FastTextModel) -> None:
        """Bind the specification and the loaded model together."""
        self._spec = spec
        self._model = model

    @property
    def model_id(self) -> str:
        """Identifier of the model backing this classifier.

        Returns:
            The registry key, e.g. ``lid218e``.
        """
        return self._spec["model_id"]

    def classify(self, text: str) -> LidPrediction:
        """Classify one line of text.

        Args:
            text: Line to classify. SentencePiece word markers are
                removed and surrounding whitespace stripped first.

        Returns:
            The top prediction, with the model's label prefix removed.

        Raises:
            EmptyClassificationTextError: If the text is empty once
                stripped.
            LidLabelError: If the model emits a label lacking the prefix
                its specification declares, which means the loaded
                weights are not the declared model.
        """
        cleaned = text.replace("▁", "").strip()
        if not cleaned:
            raise EmptyClassificationTextError()
        labels, probabilities = self._model.predict(cleaned, 1)
        return LidPrediction(
            label=strip_label_prefix(self._spec, labels[0]),
            probability=float(probabilities[0]),
        )

    def accepts(self, text: str, language: str, threshold: float) -> bool:
        """Report whether a line passes a language filter.

        Args:
            text: Line to classify.
            language: Expected label, matched as a prefix so that
                ``uzn`` accepts ``uzn_Latn`` from a script-aware model.
            threshold: Minimum probability required, e.g. 0.95.

        Returns:
            True when the top label starts with ``language`` and the
            probability meets ``threshold``.

        Raises:
            EmptyClassificationTextError: If the text is empty once
                stripped.
        """
        prediction = self.classify(text)
        return prediction["label"].startswith(language) and prediction["probability"] >= threshold


__all__ = [
    "FastTextModel",
    "LidClassifier",
    "LidPrediction",
    "encode_lid_prediction",
]
