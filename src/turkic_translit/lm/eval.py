"""Perplexity of a language model over held-out text.

The score is computed with the model that was passed in. The previous
version called Hugging Face ``evaluate``'s ``perplexity`` metric, which
takes a ``model_id`` and loads its own copy: the ``LMModel`` argument was
read only for its ``name_or_path`` and then discarded, so a freshly
fine-tuned model in memory was never the model being scored, and a model
whose name was not a Hub id or a local directory was scored as the
literal string ``"local"``. Scoring the model in hand removes that whole
class of mismatch, and drops the ``evaluate`` dependency with it.

Each sentence is scored independently and truncated to the model's own
maximum context. The reported figure is the exponential of the total
negative log-likelihood divided by the total number of predicted tokens,
so long sentences weigh more than short ones — the same quantity a
per-token perplexity is normally understood to be.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from .train import LMModel

__all__ = ["cross_perplexity"]

MIN_TOKENS_TO_SCORE = 2


def cross_perplexity(model: LMModel, sentences: Iterable[str]) -> float:
    """Return the model's per-token perplexity over the sentences.

    Args:
        model: The model to score with, together with its tokenizer.
        sentences: Held-out text. Each is truncated to the model's
            maximum context.

    Returns:
        The perplexity, which is at least 1.0.

    Raises:
        ValueError: If no sentence carried at least two tokens, and so
            nothing was predicted. Returning a perplexity for an empty
            evaluation would report a number that describes no text.
    """
    network = model.model
    tokenizer = model.tokenizer
    network.eval()
    device = next(network.parameters()).device

    total_nll = 0.0
    total_predicted = 0
    with torch.no_grad():
        for sentence in sentences:
            encoded = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
            token_ids = encoded["input_ids"]
            predicted = int(token_ids.shape[1]) - 1
            if predicted < MIN_TOKENS_TO_SCORE - 1:
                # A sentence of one token predicts nothing: the model is
                # only ever asked for the token after one it has seen.
                continue
            outputs = network(**encoded, labels=token_ids)
            total_nll += float(outputs.loss) * predicted
            total_predicted += predicted

    if total_predicted == 0:
        raise ValueError(
            "cannot report perplexity: no sentence carried the two tokens "
            "needed to predict anything"
        )
    return math.exp(total_nll / total_predicted)
