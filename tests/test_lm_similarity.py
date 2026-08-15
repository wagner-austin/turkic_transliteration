"""Tests for centred kernel alignment.

The invariances are the substance of the measure, so they are asserted
directly rather than inferred from a score: CKA must be blind to an
orthogonal change of basis and to isotropic scaling, and must *not* be
blind to an arbitrary invertible linear map. The last is what separates
it from CCA, and the reason Kornblith et al. proposed it.

The properties are checked on constructed matrices, where the
transformation applied is known exactly, and then confirmed end to end
on a real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.tiny_model import CORPUS, build_model, write_model_directory
from turkic_translit.lm.similarity import (
    ERR_CONSTANT_REPRESENTATION,
    ERR_MISMATCHED_EXAMPLES,
    ERR_TOO_FEW_EXAMPLES,
    Matrix,
    SimilarityError,
    centre_columns,
    embed_sentences,
    linear_cka,
    rbf_cka,
    representation_similarity,
)
from turkic_translit.lm.train import LMModel


def sample(rows: int, columns: int, seed: int) -> Matrix:
    """Draw a reproducible representation matrix.

    Args:
        rows: Number of examples.
        columns: Number of features.
        seed: Seed making the draw reproducible.

    Returns:
        One example per row.
    """
    drawn: Matrix = np.random.default_rng(seed).standard_normal((rows, columns))
    return drawn


def orthogonal(size: int, seed: int) -> Matrix:
    """Draw a random orthogonal matrix.

    Args:
        size: Side length of the square matrix.
        seed: Seed making the draw reproducible.

    Returns:
        A matrix ``Q`` with ``Q.T @ Q == I``.
    """
    factor, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((size, size)))
    orthonormal: Matrix = factor
    return orthonormal


def test_centring_puts_every_feature_mean_at_zero() -> None:
    """Centring is applied down the columns, not across the rows."""
    centred = centre_columns(np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]))

    assert centred.mean(axis=0).tolist() == [0.0, 0.0]
    assert centred[:, 0].tolist() == [-2.0, 0.0, 2.0]


def test_a_representation_is_perfectly_similar_to_itself() -> None:
    """The identity case is exactly one."""
    features = sample(20, 8, seed=0)

    assert linear_cka(features, features) == pytest.approx(1.0)


def test_similarity_survives_an_orthogonal_change_of_basis() -> None:
    """A rotated representation holds the same content.

    This is the property a plain cosine lacks, and the reason CKA is
    the right measure for comparing two separately trained models: their
    feature bases are unrelated.
    """
    features = sample(20, 8, seed=1)
    rotated = features @ orthogonal(8, seed=2)

    assert linear_cka(features, rotated) == pytest.approx(1.0)


def test_similarity_survives_isotropic_scaling() -> None:
    """Scaling every feature alike changes nothing."""
    features = sample(20, 8, seed=3)

    assert linear_cka(features, features * 7.5) == pytest.approx(1.0)


def test_similarity_survives_a_constant_shift() -> None:
    """Translating the whole representation changes nothing.

    This is what the centring buys; uncentred kernel alignment would
    report a different number here.
    """
    features = sample(20, 8, seed=4)

    assert linear_cka(features, features + 100.0) == pytest.approx(1.0)


def test_similarity_does_not_survive_an_arbitrary_linear_map() -> None:
    """CKA is deliberately not blind to a general invertible transform.

    An index with that invariance cannot say anything meaningful once a
    representation has more features than there are examples, which is
    the ordinary case for a transformer scored on a few hundred
    sentences.
    """
    features = sample(20, 8, seed=5)
    stretched = features @ np.diag([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01])

    assert linear_cka(features, stretched) < 0.99


def test_unrelated_representations_score_near_zero() -> None:
    """Independent draws share no structure to align."""
    score = linear_cka(sample(200, 8, seed=6), sample(200, 8, seed=7))

    assert 0.0 <= score < 0.2


def test_the_score_is_symmetric() -> None:
    """Which representation comes first does not matter."""
    first = sample(20, 8, seed=8)
    second = sample(20, 5, seed=9)

    assert linear_cka(first, second) == pytest.approx(linear_cka(second, first))


def test_representations_of_different_width_can_be_compared() -> None:
    """Feature counts may differ; only the example counts must agree."""
    features = sample(20, 8, seed=10)
    widened = np.hstack([features, features @ orthogonal(8, seed=11)])

    assert 0.0 <= linear_cka(features, widened) <= 1.0


def test_comparing_different_example_counts_is_rejected() -> None:
    """CKA compares two views of the same examples."""
    with pytest.raises(SimilarityError) as raised:
        linear_cka(sample(20, 8, seed=12), sample(19, 8, seed=13))

    assert raised.value.code == ERR_MISMATCHED_EXAMPLES


def test_comparing_a_single_example_is_rejected() -> None:
    """Centring one example leaves the zero vector."""
    with pytest.raises(SimilarityError) as raised:
        linear_cka(sample(1, 8, seed=14), sample(1, 8, seed=15))

    assert raised.value.code == ERR_TOO_FEW_EXAMPLES


def test_a_constant_representation_is_rejected() -> None:
    """A representation that is the same for every example aligns with nothing."""
    with pytest.raises(SimilarityError) as raised:
        linear_cka(np.ones((10, 4)), sample(10, 4, seed=16))

    assert raised.value.code == ERR_CONSTANT_REPRESENTATION


def test_a_single_outlier_moves_the_score() -> None:
    """One translated example is enough to change the answer.

    Davari et al. 2023 characterise this formally: translating a subset
    of the representations — in the limit a single point — drives linear
    CKA down even though the two representations are otherwise
    identical. Recorded here because a caller reading a CKA score needs
    to know it describes the chosen examples, not the models alone.
    """
    features = sample(50, 8, seed=17)
    with_outlier = features.copy()
    with_outlier[0] += 500.0

    assert linear_cka(features, with_outlier) < 0.5


def test_rbf_similarity_is_perfect_for_a_representation_with_itself() -> None:
    """The kernel form agrees with the linear one on the identity case."""
    features = sample(20, 8, seed=18)

    assert rbf_cka(features, features) == pytest.approx(1.0)


def test_rbf_similarity_survives_an_orthogonal_change_of_basis() -> None:
    """The kernel form shares the invariance that matters."""
    features = sample(20, 8, seed=19)

    assert rbf_cka(features, features @ orthogonal(8, seed=20)) == pytest.approx(1.0)


def test_rbf_similarity_survives_isotropic_scaling() -> None:
    """The median-distance bandwidth is what makes this hold."""
    features = sample(20, 8, seed=21)

    assert rbf_cka(features, features * 12.0) == pytest.approx(1.0)


def test_rbf_similarity_rejects_a_representation_with_no_spread() -> None:
    """With every example at one point there is no bandwidth to choose."""
    with pytest.raises(SimilarityError) as raised:
        rbf_cka(np.zeros((10, 4)), np.zeros((10, 4)))

    assert raised.value.code == ERR_CONSTANT_REPRESENTATION


def test_rbf_similarity_rejects_a_mostly_duplicated_representation() -> None:
    """A zero median needs most pairs to coincide, not all of them.

    Three of these four examples share a position, which puts a majority
    of the pairwise distances at zero and leaves the median there too —
    so the bandwidth is undefined even though the representation is not
    constant and passes the earlier check.
    """
    duplicated = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(SimilarityError, match="median distance between examples is zero"):
        rbf_cka(duplicated, sample(4, 2, seed=22))


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a loadable model directory once for the whole module.

    Returns:
        A directory holding a tokenizer and a tiny causal LM.
    """
    return write_model_directory(tmp_path_factory.mktemp("cka") / "model")


def test_a_model_is_perfectly_similar_to_itself(model_dir: Path) -> None:
    """Run end to end, a model compared with itself scores one.

    Args:
        model_dir: The written model directory.
    """
    model = LMModel.from_pretrained(str(model_dir))

    assert representation_similarity(model, model, CORPUS) == pytest.approx(1.0, abs=1e-6)


def test_unrelated_models_embed_the_same_text_differently(model_dir: Path) -> None:
    """Each model's own weights reach its representations, and CKA stays in range.

    This test used to assert that two unrelated models score strictly
    below one, which is not a property CKA has here: with six examples
    and hundreds of random dimensions, both similarity matrices centre
    to near-identity and the index sits at one up to float noise — the
    many-features-few-examples regime the module docstring cites. What
    the end-to-end path must guarantee is that the representations
    really come from each model's weights (a broken extraction would
    hand both models identical rows) and that the index respects its
    bounds up to that noise.

    The second model is built from a configuration rather than reloaded:
    ``PreTrainedModel.init_weights`` is a no-op on a loaded model, so
    reinitialising one would compare a model with itself.

    Args:
        model_dir: The written model directory.
    """
    model = LMModel.from_pretrained(str(model_dir))
    torch.manual_seed(20260814)
    other = LMModel(build_model(len(model.tokenizer)), model.tokenizer)

    ours = embed_sentences(model, CORPUS)
    theirs = embed_sentences(other, CORPUS)
    together = representation_similarity(model, other, CORPUS)

    assert float(np.abs(ours - theirs).max()) > 0.0
    assert 0.0 <= together <= 1.0 + 1e-9


def test_one_sentence_is_too_few_to_compare(model_dir: Path) -> None:
    """The end-to-end path rejects what the matrix path rejects.

    Args:
        model_dir: The written model directory.
    """
    model = LMModel.from_pretrained(str(model_dir))

    with pytest.raises(SimilarityError) as raised:
        representation_similarity(model, model, CORPUS[:1])

    assert raised.value.code == ERR_TOO_FEW_EXAMPLES


def test_embedding_gives_one_row_per_sentence(model_dir: Path) -> None:
    """The matrix handed to CKA has examples down the rows.

    Args:
        model_dir: The written model directory.
    """
    model = LMModel.from_pretrained(str(model_dir))

    embedded = embed_sentences(model, CORPUS)

    assert embedded.shape[0] == len(CORPUS)
    assert embedded.shape[1] == model.model.config.n_embd
