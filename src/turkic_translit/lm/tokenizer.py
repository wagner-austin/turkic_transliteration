# mypy: ignore-errors
"""Wrapper around 🤗 *AutoTokenizer* with optional shared SentencePiece override."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

__all__ = ["load_tokenizer"]


def load_tokenizer(model_name: str, spm_override: str | None = None) -> AutoTokenizer:
    """Return a *transformers.AutoTokenizer*.

    If *spm_override* points to a local ``.model`` file, we load this file into
    the tokenizer's internal *SentencePiece* model to guarantee **identical**
    sub-word vocabularies across languages.
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if spm_override is not None:
        spm_path = Path(spm_override)
        if not spm_path.exists():
            raise FileNotFoundError(spm_path)
        # Ensure tokenizer supports SentencePiece override
        if not hasattr(tok, "sp_model"):
            raise TypeError(
                f"{tok.__class__.__name__} does not support SentencePiece override"
            )
        tok.sp_model.Load(str(spm_path))
    return tok
