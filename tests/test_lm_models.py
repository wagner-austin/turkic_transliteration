"""Tests for loading, fine-tuning, scoring and comparing language models.

Every test runs a real GPT-2 — two layers, two heads, an
eight-dimensional embedding — built from a configuration and a tokenizer
trained in memory, so nothing is downloaded and nothing is substituted.
The models are small enough that training and scoring finish in seconds
while still exercising the same transformers code paths as a full run.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from tests.tiny_model import (
    CORPUS,
    PAD_TOKEN,
    write_model_directory,
)
from turkic_translit.lm import _test_hooks as lm_hooks
from turkic_translit.lm.eval import cross_perplexity
from turkic_translit.lm.tokenizer import load_tokenizer
from turkic_translit.lm.train import (
    LMModel,
    ensure_pad_token,
    model_dtype,
    use_half_precision,
)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a loadable model directory once for the whole module.

    Returns:
        A directory holding a tokenizer and a tiny causal LM.
    """
    return write_model_directory(tmp_path_factory.mktemp("tiny") / "model")


@pytest.fixture
def loaded(model_dir: Path) -> LMModel:
    """Load the tiny model through the production loader.

    Returns:
        The loaded model and tokenizer.
    """
    return LMModel.from_pretrained(str(model_dir))


def test_half_precision_is_used_only_with_cuda() -> None:
    """Training precision follows the hardware, both ways."""
    assert use_half_precision(True) is True
    assert use_half_precision(False) is False


def test_weights_load_as_half_only_with_cuda() -> None:
    """The load dtype follows the hardware, both ways."""
    assert model_dtype(True) == torch.float16
    assert model_dtype(False) == torch.float32


def test_a_tokenizer_without_a_pad_token_is_given_one(model_dir: Path) -> None:
    """The end-of-sequence token becomes the pad token."""
    tokenizer = load_tokenizer(str(model_dir))
    tokenizer.pad_token = None

    ensure_pad_token(tokenizer)

    assert tokenizer.pad_token == tokenizer.eos_token


def test_a_tokenizer_with_a_pad_token_keeps_it(model_dir: Path) -> None:
    """An existing pad token is left exactly as published."""
    tokenizer = load_tokenizer(str(model_dir))
    original = tokenizer.pad_token

    ensure_pad_token(tokenizer)

    assert tokenizer.pad_token == original


def test_loading_returns_a_usable_model_and_tokenizer(loaded: LMModel) -> None:
    """The loader returns a pair that can encode and run."""
    encoded = loaded.tokenizer("salem alem", return_tensors="pt")
    outputs = loaded.model(**encoded)

    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[2] == loaded.model.config.vocab_size


def test_loading_guarantees_a_pad_token(loaded: LMModel) -> None:
    """Batched work downstream can rely on the pad token being set."""
    assert loaded.tokenizer.pad_token == PAD_TOKEN
    assert loaded.tokenizer.pad_token_id == loaded.tokenizer.convert_tokens_to_ids(PAD_TOKEN)


def test_the_tokenizer_loads_as_published(model_dir: Path) -> None:
    """A saved tokenizer directory loads and encodes text."""
    tokenizer = load_tokenizer(str(model_dir))

    assert tokenizer("salem alem")["input_ids"]


def test_perplexity_is_finite_and_above_one(loaded: LMModel) -> None:
    """An untrained model scores a real, finite perplexity.

    An untrained model over a vocabulary of this size cannot do better
    than chance, so the score sits near the vocabulary size rather than
    near 1.0, and it must be a real number either way.
    """
    score = cross_perplexity(loaded, CORPUS)

    assert math.isfinite(score)
    assert 1.0 < score < loaded.model.config.vocab_size * 2


def test_perplexity_weighs_longer_sentences_more(loaded: LMModel) -> None:
    """Scoring a sentence twice is the same as scoring it once."""
    once = cross_perplexity(loaded, ["salem alem qazaq tili"])
    twice = cross_perplexity(loaded, ["salem alem qazaq tili"] * 2)

    assert twice == pytest.approx(once, rel=1e-6)


def test_a_sentence_too_short_to_predict_is_skipped(loaded: LMModel) -> None:
    """A one-token sentence contributes nothing to the score.

    Empty text encodes to the end-of-sequence token alone, and a single
    token is never predicted from anything.
    """
    assert len(loaded.tokenizer("", return_tensors="pt")["input_ids"][0]) == 1

    with_short = cross_perplexity(loaded, ["", "salem alem qazaq tili"])
    without = cross_perplexity(loaded, ["salem alem qazaq tili"])

    assert with_short == pytest.approx(without, rel=1e-6)


def test_scoring_nothing_is_reported_as_an_error(loaded: LMModel) -> None:
    """An evaluation with no predicted tokens has no perplexity."""
    with pytest.raises(ValueError, match="no sentence carried the two tokens"):
        cross_perplexity(loaded, [])


def test_fine_tuning_writes_a_loadable_model(model_dir: Path, tmp_path: Path) -> None:
    """Training runs for real and saves something the loader can read."""
    output = tmp_path / "fine-tuned"

    trained = LMModel.fresh(str(model_dir), epochs=1, sentences=iter(CORPUS), output_dir=output)

    assert trained.model.config.use_cache is True
    assert (output / "config.json").exists()
    reloaded = LMModel.from_pretrained(str(output))
    assert reloaded.model.config.vocab_size == trained.model.config.vocab_size


def test_fine_tuning_actually_learns_the_corpus(model_dir: Path, tmp_path: Path) -> None:
    """Training lowers the model's perplexity on the text it trained on.

    This is the test that distinguishes real training from a call that
    merely returns: an untouched model and a trained one cannot score the
    same text alike unless the weights moved. The learning rate is far
    above the default so that a two-layer model memorises six sentences
    decisively — at the default rate the improvement is real but within
    a margin that a different random start could swallow, which would
    make this test report the weather rather than the behaviour.
    """
    output = tmp_path / "learned"
    original = LMModel.from_pretrained(str(model_dir))
    before = cross_perplexity(original, CORPUS)

    LMModel.fresh(str(model_dir), lr=0.05, epochs=20, sentences=iter(CORPUS), output_dir=output)

    trained = LMModel.from_pretrained(str(output))
    after = cross_perplexity(trained, CORPUS)

    assert after < before / 2
    first_original = next(original.model.parameters()).detach().flatten()[0].item()
    first_trained = next(trained.model.parameters()).detach().flatten()[0].item()
    assert first_trained != first_original


def test_fine_tuning_caps_the_sentences_it_buffers(model_dir: Path, tmp_path: Path) -> None:
    """The corpus is read through an iterator, not materialised twice."""
    sentences: Iterator[str] = iter(CORPUS)

    LMModel.fresh(str(model_dir), epochs=1, sentences=sentences, output_dir=tmp_path / "out")

    assert list(sentences) == []


def test_the_production_trainer_hook_trains_and_saves(model_dir: Path, tmp_path: Path) -> None:
    """The hook production binds does what the protocol promises."""
    output = tmp_path / "hooked"

    lm_hooks.TransformersTrainer().fine_tune(str(model_dir), 1, iter(CORPUS), str(output))

    assert (output / "config.json").exists()
    assert LMModel.from_pretrained(str(output)).tokenizer.pad_token == PAD_TOKEN


def test_the_production_evaluator_hook_scores_a_saved_model(model_dir: Path) -> None:
    """The hook production binds loads the model and scores with it."""
    score = lm_hooks.TransformersEvaluator().perplexity(str(model_dir), CORPUS)

    assert score > 1.0
