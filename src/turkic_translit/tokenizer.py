"""SentencePiece tokenisation and detokenisation.

The ``sentencepiece`` package ships no type information, so its surface is
narrowed here, at the one place this project touches it, by declaring a
protocol for the two methods used and assigning the loaded processor to
it. That replaces the two ``cast`` calls this module used to make on
every tokenise and detokenise.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from turkic_translit import _test_hooks

DEFAULT_MODEL_NAME = "turkic_model.model"
MODEL_PATH_VARIABLE = "TURKIC_TOKENIZER_MODEL"


class SentencePieceEncoder(Protocol):
    """The SentencePiece operations this project depends on."""

    def load(self, model_file: str) -> None:
        """Load a model from disk into this processor.

        Args:
            model_file: Path to the ``.model`` file.
        """
        ...

    def encode(self, input: str, out_type: type[str]) -> list[str]:
        """Split text into subword pieces.

        Args:
            input: Text to tokenise. Named to match the C++ binding.
            out_type: Requested element type; always ``str`` here.

        Returns:
            The pieces, in order.
        """
        ...

    def decode(self, input: list[str]) -> str:
        """Reassemble pieces into text.

        Args:
            input: Pieces to join. Named to match the C++ binding.

        Returns:
            The reconstructed text.
        """
        ...


class SentencePieceTrainer(Protocol):
    """SentencePiece's model trainer, as this project drives it."""

    def train(self, **arguments: str | int | float | bool | Sequence[str]) -> None:
        """Train a model and write it beside its vocabulary.

        Args:
            **arguments: Trainer options, passed through as named. The
                value type is the widest a SentencePiece option takes,
                so a caller may unpack a validated mapping into it.

        Raises:
            RuntimeError: If the corpus cannot support the requested
                vocabulary size, or an option is not recognised.
        """
        ...


def sentencepiece_trainer() -> SentencePieceTrainer:
    """Return SentencePiece's trainer, narrowed to what is used here.

    The package ships no type information, so it is imported at this one
    place and immediately bound to a protocol. Nothing downstream sees
    the untyped surface.

    Returns:
        The trainer class, as a :class:`SentencePieceTrainer`.
    """
    module = __import__("sentencepiece")
    trainer: SentencePieceTrainer = module.SentencePieceTrainer
    return trainer


def sentencepiece_processor(model_file: str) -> SentencePieceEncoder:
    """Build a SentencePiece processor with a model already loaded.

    The model is required rather than optional: a processor with no model
    cannot encode or decode anything, so handing one back would only move
    the failure to a later call that has less to say about it.

    Args:
        model_file: Path of the ``.model`` file to load.

    Returns:
        The processor, as a :class:`SentencePieceEncoder`.

    Raises:
        OSError: If the model file cannot be read.
    """
    module = __import__("sentencepiece")
    processor: SentencePieceEncoder = module.SentencePieceProcessor()
    processor.load(model_file)
    return processor


def default_model_path() -> Path:
    """Locate the SentencePiece model this project tokenises with.

    ``TURKIC_TOKENIZER_MODEL`` names the model when it is set. The model
    is not shipped — it has to be trained — and the only way to install
    one used to be to copy it inside the installed package directory,
    which is not somewhere a user can reasonably be asked to write.

    Returns:
        The configured path when one is named, otherwise the packaged
        location, whether or not a file is there.
    """
    configured = _test_hooks.environment.get(MODEL_PATH_VARIABLE)
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent / DEFAULT_MODEL_NAME


class TurkicTokenizer:
    """A loaded SentencePiece model.

    Args:
        model_path: Model to load, or ``None`` for the packaged model.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Load the SentencePiece model this tokeniser will use."""
        resolved = default_model_path() if model_path is None else Path(model_path)
        self.sp = sentencepiece_processor(str(resolved))

    def tokenize(self, text: str) -> list[str]:
        """Split text into subword pieces.

        Args:
            text: Text to tokenise.

        Returns:
            The pieces, in order.
        """
        return self.sp.encode(text, out_type=str)

    def detokenize(self, tokens: list[str]) -> str:
        """Reassemble pieces into text.

        Args:
            tokens: Pieces to join.

        Returns:
            The reconstructed text.
        """
        return self.sp.decode(tokens)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "MODEL_PATH_VARIABLE",
    "SentencePieceEncoder",
    "SentencePieceTrainer",
    "TurkicTokenizer",
    "default_model_path",
    "sentencepiece_processor",
    "sentencepiece_trainer",
]
