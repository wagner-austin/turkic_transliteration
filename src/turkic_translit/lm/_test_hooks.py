"""Injection seam for the two effects the language-model commands have.

Production binds each hook to its real implementation at import time and
never rebinds it. Tests bind them to real implementations of the same
protocols that record what they were asked to do. Command code calls the
hooks unconditionally, so no branch exists purely to support testing.

Fine-tuning and perplexity scoring are seams because both load a
multi-gigabyte model and run it: neither can happen in a test, and both
are what the commands exist to arrange. Everything either command does
*around* them — parsing options, splitting language codes, building the
sentence stream, rendering the result — is then exercised for real.

The protocols are stated in terms of paths and sentences rather than
model objects, so binding one does not require constructing a model.

The module is private because the seam is internal to this package.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol


class Trainer(Protocol):
    """Fine-tunes a causal language model and saves it."""

    def fine_tune(
        self, base_model: str, epochs: int, sentences: Iterable[str], output_dir: str
    ) -> None:
        """Fine-tune a base model and save the result.

        Args:
            base_model: Local directory or Hub repository id to start from.
            epochs: Number of training epochs.
            sentences: Training text, consumed as it is read.
            output_dir: Directory the model and tokenizer are saved into,
                which holds a loadable model when this returns.

        Raises:
            OSError: If the base model cannot be read or the output
                directory cannot be written.
        """
        ...


class TransformersTrainer:
    """Trainer backed by :class:`~turkic_translit.lm.train.LMModel`."""

    def fine_tune(
        self, base_model: str, epochs: int, sentences: Iterable[str], output_dir: str
    ) -> None:
        """Fine-tune with Hugging Face and save to ``output_dir``.

        The import is deferred so that binding this hook does not pull in
        torch and transformers, which is what keeps a command that never
        trains — and a test that never trains — cheap to start.

        Args:
            base_model: Local directory or Hub repository id to start from.
            epochs: Number of training epochs.
            sentences: Training text, consumed as it is read.
            output_dir: Directory the model and tokenizer are saved into.

        Raises:
            OSError: If the base model cannot be read or the output
                directory cannot be written.
        """
        from turkic_translit.lm.train import LMModel

        LMModel.fresh(base_model, epochs=epochs, sentences=sentences, output_dir=output_dir)


class RecordingTrainer:
    """Trainer that records its instructions and saves a minimal model.

    A real implementation of :class:`Trainer`, not a mock: it satisfies
    the contract by leaving a ``config.json`` in the output directory,
    and holds no assertion helpers, so a test can only read the log it
    kept and the directory it wrote.
    """

    def __init__(self) -> None:
        """Start an empty training log."""
        self.runs: list[tuple[str, int, tuple[str, ...], str]] = []

    def fine_tune(
        self, base_model: str, epochs: int, sentences: Iterable[str], output_dir: str
    ) -> None:
        """Record the run and write a saved-model marker.

        The sentences are materialised here because the command hands
        over a generator, and a test that inspected it afterwards would
        otherwise find it already exhausted.

        Args:
            base_model: The model the command asked to start from.
            epochs: Number of epochs the command asked for.
            sentences: Training text, read to exhaustion.
            output_dir: Directory the command asked to save into.
        """
        self.runs.append((base_model, epochs, tuple(sentences), output_dir))
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")


class Evaluator(Protocol):
    """Scores a saved language model against held-out text."""

    def perplexity(self, model_path: str, sentences: Iterable[str]) -> float:
        """Return the model's sliding-window perplexity over the text.

        Args:
            model_path: Local directory or Hub repository id to load.
            sentences: Held-out text to score.

        Returns:
            The mean perplexity across the sentences.

        Raises:
            OSError: If the model cannot be read.
        """
        ...


class TransformersEvaluator:
    """Evaluator backed by :func:`~turkic_translit.lm.eval.cross_perplexity`."""

    def perplexity(self, model_path: str, sentences: Iterable[str]) -> float:
        """Load the model and score the text with it.

        The import is deferred for the same reason as in
        :class:`TransformersTrainer`.

        Args:
            model_path: Local directory or Hub repository id to load.
            sentences: Held-out text to score.

        Returns:
            The mean perplexity across the sentences.

        Raises:
            OSError: If the model cannot be read.
        """
        from turkic_translit.lm.eval import cross_perplexity
        from turkic_translit.lm.train import LMModel

        return cross_perplexity(LMModel.from_pretrained(model_path), sentences)


class FixedEvaluator:
    """Evaluator returning a stated score and recording what it scored.

    A real implementation of :class:`Evaluator`, not a mock: it holds no
    assertion helpers, so a test can only read the requests it logged.

    Args:
        score: The perplexity this evaluator reports for any request.
    """

    def __init__(self, score: float) -> None:
        """Store the score and start an empty request log."""
        self._score = score
        self.scored: list[tuple[str, tuple[str, ...]]] = []

    def perplexity(self, model_path: str, sentences: Iterable[str]) -> float:
        """Record the request and return the stated score.

        Args:
            model_path: The model the command asked to load.
            sentences: Held-out text, read to exhaustion so a test can
                inspect what the command streamed.

        Returns:
            The score given at construction.
        """
        self.scored.append((model_path, tuple(sentences)))
        return self._score


trainer: Trainer = TransformersTrainer()
evaluator: Evaluator = TransformersEvaluator()

__all__ = [
    "Evaluator",
    "FixedEvaluator",
    "RecordingTrainer",
    "Trainer",
    "TransformersEvaluator",
    "TransformersTrainer",
    "evaluator",
    "trainer",
]
