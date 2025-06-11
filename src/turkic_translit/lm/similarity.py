# mypy: ignore-errors
"""Representation-level similarity metrics between two LMs."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .train import LMModel

__all__ = ["centred_cosine_matrix"]

# PEP 646 style generic ndarray type alias
ArrayF = np.ndarray[Any, np.dtype[np.floating]]

# Reuse the web_demo logger if configured; fallback to root.
logger = logging.getLogger("turkic_translit.web_demo")


def _embed(
    model: LMModel, sentences: Iterable[str], layer: int = -2
) -> ArrayF:  # noqa: D401
    """Return *L2*-normalised mean-pooled hidden states for *sentences*."""
    tok = model.tokenizer
    mdl = model.model

    vecs: list[np.ndarray] = []
    # Convert to list so we know total count for nicer progress.
    sent_list = list(sentences)
    total = len(sent_list)

    for i, sent in enumerate(sent_list, 1):
        ids = tok(sent, return_tensors="pt", truncation=True)
        device = next(mdl.parameters()).device
        ids = ids.to(device)
        with torch.no_grad():
            h = mdl(**ids, output_hidden_states=True).hidden_states[layer]
        vecs.append(h.mean(dim=1).cpu().numpy())

        if i % 50 == 0 or i == total:
            logger.info("[mutual] encoded %d/%d sentences", i, total)

    return normalize(np.vstack(vecs))


def centred_cosine_matrix(
    model_a: LMModel, model_b: LMModel, sentences: Iterable[str]
) -> float:
    """Return mean centred cosine similarity between *model_a* and *model_b*."""
    ea = _embed(model_a, sentences)
    eb = _embed(model_b, sentences)
    return float(cosine_similarity(ea, eb).mean())
