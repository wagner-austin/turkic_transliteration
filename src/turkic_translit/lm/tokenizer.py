"""Tokenizer loading, with an optional shared SentencePiece override.

``AutoTokenizer`` is a factory, not a type: it returns a
``PreTrainedTokenizerBase``. Annotating the return as ``AutoTokenizer``
made every attribute access on the result an error, which is part of what
the file-level mypy suppression on this module was hiding.
"""

from __future__ import annotations

from pathlib import Path

# Concrete submodules, not the lazy package root: see the note in train.py.
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = ["load_tokenizer"]


def load_tokenizer(model_name: str, spm_override: str | None = None) -> PreTrainedTokenizerBase:
    """Load a tokenizer, optionally forcing a shared SentencePiece model.

    Args:
        model_name: Local directory or Hub repository id.
        spm_override: A ``.model`` file to load into the tokenizer's
            internal SentencePiece model, so that several languages share
            one sub-word vocabulary. ``None`` leaves the tokenizer as
            published.

    Returns:
        The loaded tokenizer.

    Raises:
        FileNotFoundError: If ``spm_override`` names a missing file.
        TypeError: If the tokenizer has no SentencePiece model to
            override, which means the override would silently do nothing.
    """
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if spm_override is None:
        return tokenizer

    spm_path = Path(spm_override)
    if not spm_path.exists():
        raise FileNotFoundError(spm_path)
    if not hasattr(tokenizer, "sp_model"):
        raise TypeError(f"{tokenizer.__class__.__name__} does not support SentencePiece override")
    tokenizer.sp_model.Load(str(spm_path))
    return tokenizer
