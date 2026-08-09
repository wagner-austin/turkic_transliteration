"""Representational similarity between two language models.

The measure is centred kernel alignment (CKA), defined by Kornblith,
Norouzi, Lee and Hinton, *Similarity of Neural Network Representations
Revisited*, ICML 2019 (arXiv:1905.00414), §"Comparing Similarity
Structures" and Table 1.

CKA answers a different question from perplexity. Perplexity asks how
well a model predicts another language's text; CKA asks whether two
models lay the same sentences out in the same relative arrangement. Two
models can agree closely on the second while differing on the first.

    HSIC(K, L) = tr(K H L H) / (n - 1)^2,  H = I - (1/n) 1 1^T
    CKA(K, L)  = HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L))

With a linear kernel this reduces to the form implemented here:

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)

on column-centred X and Y. The centring is what the "centred" in the
name refers to, and it is not optional: without it the index is kernel
alignment, which is not invariant to a shift of the representations.

Why not a plain cosine between the two models' embeddings. A network's
feature basis is arbitrary — two models trained separately, or the same
model trained from a different seed, place equivalent features in
unrelated coordinates — so a raw cosine reports how the two bases happen
to align rather than whether the models encode the same structure. CKA
is invariant to orthogonal transformation and to isotropic scaling,
which is exactly the set of changes that leave a representation's
content intact (Kornblith et al. 2019, Table 1).

It is deliberately *not* invariant to any invertible linear transform.
Kornblith et al. show that no index with that invariance can measure
anything meaningful once a representation has more features than there
are examples, which is the ordinary case for a transformer scored on a
few hundred sentences.

Known weakness, and the reason :func:`representation_similarity` requires
an explicit example count rather than accepting whatever it is handed:
Davari, Horoi, Natik, Lajoie, Wolf and Belilovsky, *Reliability of CKA as
a Similarity Measure in Deep Learning*, ICLR 2023 (arXiv:2210.16156),
§3, prove that translating a subset of the representations — in the
limit, a single outlier — drives linear CKA down sharply even when the
two representations are otherwise identical and the network's behaviour
is unchanged. A CKA score is therefore evidence about a chosen set of
examples, not a property of the two models alone.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import numpy as np
import torch
from tqdm import tqdm

from turkic_translit.lm.train import LMModel

__all__ = [
    "DEFAULT_BANDWIDTH_FRACTION",
    "DEFAULT_HIDDEN_LAYER",
    "ERR_CONSTANT_REPRESENTATION",
    "ERR_MISMATCHED_EXAMPLES",
    "ERR_TOO_FEW_EXAMPLES",
    "MINIMUM_EXAMPLES",
    "SimilarityError",
    "centre_columns",
    "embed_sentences",
    "linear_cka",
    "rbf_cka",
    "representation_similarity",
]

logger = logging.getLogger(__name__)

ERR_TOO_FEW_EXAMPLES: Final = "TURKIC_SIM_001_TOO_FEW_EXAMPLES"
ERR_MISMATCHED_EXAMPLES: Final = "TURKIC_SIM_002_MISMATCHED_EXAMPLES"
ERR_CONSTANT_REPRESENTATION: Final = "TURKIC_SIM_003_CONSTANT_REPRESENTATION"

# Two examples are the fewest that can be centred and still carry any
# structure: with one, centring leaves the zero vector.
MINIMUM_EXAMPLES: Final = 2

# The layer below the output. The last layer of a causal LM is shaped by
# the token-prediction objective, so the one beneath it is the usual
# choice for comparing what a model represents rather than what it emits.
DEFAULT_HIDDEN_LAYER: Final = -2

# Kornblith et al. 2019 set the RBF bandwidth as a fraction of the median
# distance between examples, which is what keeps RBF CKA invariant to
# isotropic scaling (Table 1, footnote).
DEFAULT_BANDWIDTH_FRACTION: Final = 0.8

Matrix = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


class SimilarityError(Exception):
    """Raised when a representation cannot be compared.

    Args:
        code: Stable error code from this module.
        message: Human-readable description naming the offending value.
    """

    def __init__(self, code: str, message: str) -> None:
        """Store the code and render ``code: message`` as the string form."""
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def centre_columns(features: Matrix) -> Matrix:
    """Subtract each feature's mean across examples.

    This is the centring in centred kernel alignment. It is applied to
    columns, not rows: the mean being removed is a feature's average over
    the examples, which is what makes the index blind to a constant shift
    of the whole representation.

    Args:
        features: One example per row, one feature per column.

    Returns:
        The same array with every column's mean at zero.
    """
    centred: Matrix = features - features.mean(axis=0, keepdims=True)
    return centred


def _require_comparable(first: Matrix, second: Matrix) -> None:
    """Reject representations that cannot be compared.

    Args:
        first: One example per row.
        second: The same examples, represented by the other model.

    Raises:
        SimilarityError: If the two describe different numbers of
            examples, if there are too few to centre, or if either is
            constant and so has no structure to align.
    """
    if first.shape[0] != second.shape[0]:
        raise SimilarityError(
            ERR_MISMATCHED_EXAMPLES,
            f"the two representations describe {first.shape[0]} and "
            f"{second.shape[0]} examples; CKA compares two views of the "
            f"same examples, so the counts must agree",
        )
    if first.shape[0] < MINIMUM_EXAMPLES:
        raise SimilarityError(
            ERR_TOO_FEW_EXAMPLES,
            f"cannot compare {first.shape[0]} example(s): centring leaves "
            f"the zero vector, and an alignment of nothing is undefined",
        )
    for name, matrix in (("first", first), ("second", second)):
        if not bool(np.any(centre_columns(matrix))):
            raise SimilarityError(
                ERR_CONSTANT_REPRESENTATION,
                f"the {name} representation is the same for every example, "
                f"so it has no structure for the other to align with",
            )


def linear_cka(first: Matrix, second: Matrix) -> float:
    """Compare two representations with linear CKA.

    Implements ``||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)`` on centred
    inputs — Kornblith et al. 2019, Table 1, "Linear CKA".

    Args:
        first: One example per row, one feature per column.
        second: The same examples as represented by the other model. The
            feature counts need not match; the example counts must.

    Returns:
        A score from 0.0 for no shared structure to 1.0 for
        representations related by an orthogonal transformation and an
        isotropic rescaling.

    Raises:
        SimilarityError: If the representations cannot be compared.
    """
    _require_comparable(first, second)
    left = centre_columns(first)
    right = centre_columns(second)

    cross = float(np.linalg.norm(right.T @ left, ord="fro") ** 2)
    left_norm = float(np.linalg.norm(left.T @ left, ord="fro"))
    right_norm = float(np.linalg.norm(right.T @ right, ord="fro"))
    return cross / (left_norm * right_norm)


def _rbf_kernel(features: Matrix, bandwidth_fraction: float) -> Matrix:
    """Build an RBF kernel matrix over the examples.

    The bandwidth is set to a fraction of the median pairwise distance,
    which is what makes the resulting index invariant to isotropic
    scaling (Kornblith et al. 2019, Table 1, footnote).

    Args:
        features: One example per row.
        bandwidth_fraction: Multiple of the median distance to use.

    Returns:
        The kernel matrix.

    Raises:
        SimilarityError: If the median distance between examples is
            zero, so no bandwidth exists. This does not require every
            example to coincide — it is enough that most pairs do, which
            happens when a majority of the sentences are represented
            identically.
    """
    squared = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=-1)
    median = float(np.median(np.sqrt(squared)))
    if median == 0.0:
        raise SimilarityError(
            ERR_CONSTANT_REPRESENTATION,
            "the median distance between examples is zero, so the RBF "
            "bandwidth is undefined; most of the examples are represented "
            "identically, even if not all of them are",
        )
    sigma = bandwidth_fraction * median
    kernel: Matrix = np.exp(-squared / (2.0 * sigma**2))
    return kernel


def _centred_gram(kernel: Matrix) -> Matrix:
    """Apply the centring matrix to both sides of a kernel matrix.

    Args:
        kernel: A kernel matrix over the examples.

    Returns:
        ``H K H`` for the centring matrix ``H = I - (1/n) 1 1^T``.
    """
    size = kernel.shape[0]
    centring = np.eye(size) - np.ones((size, size)) / size
    centred: Matrix = centring @ kernel @ centring
    return centred


def rbf_cka(
    first: Matrix,
    second: Matrix,
    bandwidth_fraction: float = DEFAULT_BANDWIDTH_FRACTION,
) -> float:
    """Compare two representations with RBF-kernel CKA.

    Implements ``tr(KHLH) / sqrt(tr(KHKH) tr(LHLH))`` — Kornblith et al.
    2019, Table 1, "RBF CKA". Kornblith et al. report that linear and RBF
    CKA agree across most of their experiments and use the linear form by
    default; this is here for the cases where they do not.

    Args:
        first: One example per row, one feature per column.
        second: The same examples as represented by the other model.
        bandwidth_fraction: Multiple of the median pairwise distance to
            use as the kernel bandwidth.

    Returns:
        A score from 0.0 to 1.0.

    Raises:
        SimilarityError: If the representations cannot be compared.
    """
    _require_comparable(first, second)
    left = _centred_gram(_rbf_kernel(first, bandwidth_fraction))
    right = _centred_gram(_rbf_kernel(second, bandwidth_fraction))

    cross = float(np.trace(left @ right))
    left_self = float(np.trace(left @ left))
    right_self = float(np.trace(right @ right))
    return cross / float(np.sqrt(left_self * right_self))


def embed_sentences(
    model: LMModel, sentences: Sequence[str], layer: int = DEFAULT_HIDDEN_LAYER
) -> Matrix:
    """Represent each sentence by mean-pooling one hidden layer.

    The rows are not normalised. CKA does its own centring, and scaling
    each row to unit length would rescale examples independently, which
    is a per-example transformation the index is not meant to be blind
    to.

    Args:
        model: The model to represent with.
        sentences: Sentences to encode, in a fixed order.
        layer: Which hidden layer to pool, counting from the output.

    Returns:
        One row per sentence, one column per hidden unit.
    """
    tokenizer = model.tokenizer
    network = model.model
    network.eval()
    device = next(network.parameters()).device

    rows: list[Matrix] = []
    with torch.no_grad():
        for sentence in tqdm(list(sentences), desc="[similarity] encoding", unit="sent"):
            encoded = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
            hidden = network(**encoded, output_hidden_states=True).hidden_states[layer]
            rows.append(hidden.mean(dim=1).cpu().numpy().astype(np.float64))
    stacked: Matrix = np.vstack(rows)
    return stacked


def representation_similarity(
    model_a: LMModel,
    model_b: LMModel,
    sentences: Sequence[str],
    layer: int = DEFAULT_HIDDEN_LAYER,
) -> float:
    """Report how alike two models' representations of the same text are.

    Both models encode the same sentences, in the same order, and the two
    resulting representations are compared with linear CKA.

    The sentences are a :class:`Sequence` rather than an
    :class:`Iterable` because both models must see the same ones: an
    iterator would be exhausted by the first model and leave the second
    with nothing.

    Args:
        model_a: First model.
        model_b: Second model.
        sentences: The text both models encode. A score describes these
            sentences as much as it describes the models — Davari et al.
            2023 show a single outlying example is enough to move it — so
            they should be a sample worth reporting alongside the number.
        layer: Which hidden layer to pool, counting from the output.

    Returns:
        A score from 0.0 to 1.0.

    Raises:
        SimilarityError: If fewer than :data:`MINIMUM_EXAMPLES` sentences
            were given, or either model represents them all alike.
    """
    logger.info("comparing representations over %d sentences", len(sentences))
    return linear_cka(
        embed_sentences(model_a, sentences, layer),
        embed_sentences(model_b, sentences, layer),
    )
