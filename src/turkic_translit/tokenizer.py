import os
from typing import cast

import sentencepiece as spm


class TurkicTokenizer:
    """
    Wrapper for SentencePiece tokenization and detokenization.
    Loads a model from the given path or defaults to a packaged model.
    """

    def __init__(self, model_path: str | None = None) -> None:
        if model_path is None:
            # Default: look for model in the same directory as this file
            model_path = os.path.join(os.path.dirname(__file__), "turkic_model.model")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into subword units (list of strings)."""
        return cast(list[str], self.sp.encode(text, out_type=str))

    def detokenize(self, tokens: list[str]) -> str:
        """Reconstruct text from tokens (list of strings)."""
        return cast(str, self.sp.decode(tokens))
