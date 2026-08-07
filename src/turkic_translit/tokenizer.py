"""SentencePiece tokenisation and detokenisation.

The ``sentencepiece`` package ships no type information, so its surface is
narrowed here, at the one place this project touches it, by declaring a
protocol for the two methods used and assigning the loaded processor to
it. That replaces the two ``cast`` calls this module used to make on
every tokenise and detokenise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

DEFAULT_MODEL_NAME = "turkic_model.model"


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


def default_model_path() -> Path:
    """Locate the SentencePiece model packaged with this project.

    Returns:
        Path of the packaged model, whether or not it exists.
    """
    return Path(__file__).resolve().parent / DEFAULT_MODEL_NAME


class TurkicTokenizer:
    """A loaded SentencePiece model.

    Args:
        model_path: Model to load, or ``None`` for the packaged model.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Load the SentencePiece model this tokeniser will use."""
        resolved = default_model_path() if model_path is None else Path(model_path)
        module = __import__("sentencepiece")
        processor: SentencePieceEncoder = module.SentencePieceProcessor()
        processor.load(str(resolved))
        self.sp = processor

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


__all__ = ["DEFAULT_MODEL_NAME", "SentencePieceEncoder", "TurkicTokenizer", "default_model_path"]
