# mypy: ignore-errors
"""Evaluation helpers for language-model perplexity."""

from __future__ import annotations

from collections.abc import Iterable

import evaluate

from .train import LMModel

__all__ = ["cross_perplexity"]


def cross_perplexity(model: LMModel, sentences: Iterable[str]) -> float:
    """Return sliding-window perplexity of *model* over *sentences*.

    *sentences* may be any iterable of raw strings. We call
    🤗 *evaluate*'s ``perplexity`` metric which internally handles tokenisation.
    """
    ppl_metric = evaluate.load("perplexity", module_type="metric")
    txt = list(sentences)
    model_id = getattr(model.model, "name_or_path", None) or "local"
    res = ppl_metric.compute(model_id=model_id, add_start_token=True, predictions=txt)
    return float(sum(res["perplexities"]) / len(res["perplexities"]))
