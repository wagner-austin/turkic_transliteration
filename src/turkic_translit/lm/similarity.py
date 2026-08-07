"""Representation-level similarity between two language models.

Embeddings are described by their concrete array type rather than by a
``np.ndarray[Any, ...]`` alias. NumPy's own generic parameters are the
shape and the dtype, and only the dtype is known here, so the shape slot
is left as the unparameterised ``tuple[int, ...]`` NumPy documents rather
than being papered over with ``Any``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

from turkic_translit.lm.train import LMModel

__all__ = ["centred_cosine_matrix"]

DEFAULT_HIDDEN_LAYER = -2

logger = logging.getLogger(__name__)


def _embed(
    model: LMModel, sentences: Iterable[str], layer: int = DEFAULT_HIDDEN_LAYER
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    """Mean-pool one hidden layer over each sentence and L2-normalise.

    Args:
        model: The model to embed with.
        sentences: Sentences to encode.
        layer: Which hidden layer to pool, counting from the output.

    Returns:
        One unit-length row per sentence.
    """
    tokenizer = model.tokenizer
    network = model.model
    device = next(network.parameters()).device

    vectors: list[np.ndarray[tuple[int, ...], np.dtype[np.floating]]] = []
    for sentence in tqdm(list(sentences), desc="[mutual] encoding", unit="sent"):
        encoded = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            hidden = network(**encoded, output_hidden_states=True).hidden_states[layer]
        vectors.append(hidden.mean(dim=1).cpu().numpy())

    return normalize(np.vstack(vectors))


def centred_cosine_matrix(
    model_a: LMModel, model_b: LMModel, sentences: Iterable[str]
) -> float:
    """Mean cosine similarity between two models' sentence embeddings.

    Args:
        model_a: First model.
        model_b: Second model.
        sentences: Sentences both models encode.

    Returns:
        The mean pairwise cosine similarity.
    """
    return float(
        cosine_similarity(_embed(model_a, sentences), _embed(model_b, sentences)).mean()
    )
