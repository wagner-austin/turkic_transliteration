"""Tokenizer loading, with an optional shared SentencePiece override.

``AutoTokenizer`` is a factory, not a type: it returns a
``PreTrainedTokenizerBase``. Annotating the return as ``AutoTokenizer``
made every attribute access on the result an error, which is part of what
the file-level mypy suppression on this module was hiding.

The fast implementation is used except when an override is asked for.
A fast tokenizer holds its vocabulary in the Rust ``tokenizers`` backend
and exposes no ``sp_model``, so the previous version — which always asked
for the fast one — could only ever reject the override it documents.
Asking for the SentencePiece-backed implementation is what makes
substituting a SentencePiece model possible at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = ["TokenizerLoader", "auto_tokenizer_loader", "load_tokenizer"]

# The attribute names belong to transformers, held as data so that the
# untyped surface is reached by name rather than by an import whose
# signature would then have to be exempted from strict checking.
_TOKENIZATION_AUTO = "transformers.models.auto.tokenization_auto"
_AUTO_TOKENIZER = "AutoTokenizer"
_FROM_PRETRAINED = "from_pretrained"


class TokenizerLoader(Protocol):
    """transformers' bound ``AutoTokenizer.from_pretrained``.

    ``AutoTokenizer`` is a factory whose ``from_pretrained`` carries no
    annotations, so calling it directly is a call into untyped code.
    Binding it here and stating its signature is what lets every caller
    be checked strictly, and the return type is the honest one:
    ``PreTrainedTokenizerBase``, not ``AutoTokenizer``, which is not a
    type any tokenizer is an instance of.
    """

    def __call__(self, model_name: str, use_fast: bool) -> PreTrainedTokenizerBase:
        """Load a published tokenizer.

        Args:
            model_name: Local directory or Hub repository id.
            use_fast: Whether to prefer the Rust-backed implementation.

        Returns:
            The loaded tokenizer.

        Raises:
            OSError: If the name resolves to no readable tokenizer.
        """
        ...


def auto_tokenizer_loader() -> TokenizerLoader:
    """Return transformers' tokenizer loader, narrowed to its signature.

    The concrete submodule is imported rather than the package root:
    transformers' root ``__init__`` is a lazy module whose re-exports
    vary by version, so the same name resolves to a placeholder under
    some releases and to the real class under others.

    Returns:
        The bound ``AutoTokenizer.from_pretrained``.
    """
    module = __import__(_TOKENIZATION_AUTO, fromlist=[_AUTO_TOKENIZER])
    loader: TokenizerLoader = getattr(getattr(module, _AUTO_TOKENIZER), _FROM_PRETRAINED)
    return loader


def load_tokenizer(model_name: str, spm_override: str | None = None) -> PreTrainedTokenizerBase:
    """Load a tokenizer, optionally forcing a shared SentencePiece model.

    Args:
        model_name: Local directory or Hub repository id.
        spm_override: A ``.model`` file to load into the tokenizer's
            internal SentencePiece model, so that several languages share
            one sub-word vocabulary. ``None`` leaves the tokenizer as
            published.

    Returns:
        The loaded tokenizer: the fast implementation ordinarily, and the
        SentencePiece-backed one when an override is given.

    Raises:
        FileNotFoundError: If ``spm_override`` names a missing file.
        TypeError: If the model's slow tokenizer is not SentencePiece
            backed, which means the override would silently do nothing.
    """
    load = auto_tokenizer_loader()
    if spm_override is None:
        return load(model_name, use_fast=True)

    spm_path = Path(spm_override)
    if not spm_path.exists():
        raise FileNotFoundError(spm_path)

    tokenizer = load(model_name, use_fast=False)
    if not hasattr(tokenizer, "sp_model"):
        raise TypeError(f"{tokenizer.__class__.__name__} does not support SentencePiece override")
    tokenizer.sp_model.Load(str(spm_path))
    return tokenizer
