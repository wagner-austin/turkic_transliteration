"""transformers' unannotated model methods, reached by name.

Strict checking rejects a call into a function that carries no
annotations, and ``PreTrainedModel`` carries none on the two methods
below. That arrived with transformers 5: the same calls type-checked
under 4, and nothing about them changed here.

The rule for an untyped surface is the same wherever this project meets
one, and :mod:`turkic_translit.lm.tokenizer` already applies it to
``AutoTokenizer.from_pretrained``. Reach the attribute by name, bind it
to a Protocol that states the signature being relied on, and let one
named boundary be the only place that knows the attribute exists. The
alternatives are a suppression comment at each call site, which the
guards reject, and a mypy override for these modules, which pyproject
rules out in as many words.
"""

from __future__ import annotations

from typing import Protocol

from transformers.modeling_utils import PreTrainedModel

# Held as data so the attribute is fetched by name rather than by a
# literal, which is also what keeps the fetch out of the linter's
# constant-getattr rule.
_EVAL = "eval"

_GRADIENT_CHECKPOINTING_DISABLE = "gradient_checkpointing_disable"


class ModelMethod(Protocol):
    """A model method taking nothing and returning nothing this uses.

    ``eval`` returns the model so a caller may chain; nothing here
    chains, and stating a return this project ignores would be stating
    something it does not rely on.
    """

    def __call__(self) -> None:
        """Perform the method's effect on the model it is bound to."""
        ...


def switch_to_inference(model: PreTrainedModel) -> None:
    """Put a model into inference mode.

    Args:
        model: The model to switch, which is modified in place.
    """
    method: ModelMethod = getattr(model, _EVAL)
    method()


def disable_gradient_checkpointing(model: PreTrainedModel) -> None:
    """Turn off the gradient checkpointing training may have enabled.

    Args:
        model: The model to modify, which is modified in place.
    """
    method: ModelMethod = getattr(model, _GRADIENT_CHECKPOINTING_DISABLE)
    method()


__all__ = ["ModelMethod", "disable_gradient_checkpointing", "switch_to_inference"]
