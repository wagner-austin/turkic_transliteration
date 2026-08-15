"""Tokenizer loading.

``AutoTokenizer`` is a factory, not a type: it returns a
``PreTrainedTokenizerBase``. Annotating the return as ``AutoTokenizer``
made every attribute access on the result an error, which is part of what
the file-level mypy suppression on this module was hiding.

This module used to offer a second thing: substituting a shared
SentencePiece model into a loaded tokenizer, so that several languages
could train against one sub-word vocabulary. It worked by replacing the
tokenizer's ``sp_model``, and transformers 5 has no such attribute —
every tokenizer is backed by the Rust ``tokenizers`` library now, and
the documented replacements for the substitution accept the file and
silently keep the published vocabulary. A substitution that cannot be
verified is worse than none, since its whole purpose is a guarantee
about which vocabulary trained a model. Nothing called it: no console
script exposed it and no caller in this project or any other passed it.
"""

from __future__ import annotations

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


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a published tokenizer.

    Args:
        model_name: Local directory or Hub repository id.

    Returns:
        The loaded tokenizer.

    Raises:
        OSError: If the name resolves to no readable tokenizer.
    """
    return auto_tokenizer_loader()(model_name, use_fast=True)
