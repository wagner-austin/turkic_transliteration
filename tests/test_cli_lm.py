"""Tests for the language-model training and evaluation commands.

Both commands are driven end to end: the corpus streamer answers from a
table, so the sentence stream each command builds is the real one, and
the trainer and evaluator are real implementations of the production
protocols that record what they were handed instead of loading a model.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from turkic_translit.cli.eval_lm import cli as eval_lm_cli
from turkic_translit.cli.train_lm import cli as train_lm_cli
from turkic_translit.cli.train_lm import parse_language_codes
from turkic_translit.corpus import _test_hooks as corpus_hooks
from turkic_translit.lm import _test_hooks as lm_hooks

CORPORA = {
    "kk": ["  салем  ", "", "әлем"],
    "ky": ["салам", "   ", "дүйнө"],
}


@pytest.fixture
def corpora() -> Iterator[None]:
    """Serve :data:`CORPORA` as the OSCAR configurations.

    Yields:
        None, once, with the original streamer captured.
    """
    original = corpus_hooks.dataset_texts
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer(CORPORA)
    yield
    corpus_hooks.dataset_texts = original


@pytest.fixture
def trainer() -> Iterator[lm_hooks.RecordingTrainer]:
    """Bind a trainer that records its instructions and saves a marker.

    Yields:
        The trainer the command ran against.
    """
    original = lm_hooks.trainer
    recording = lm_hooks.RecordingTrainer()
    lm_hooks.trainer = recording
    yield recording
    lm_hooks.trainer = original


@pytest.fixture
def evaluator() -> Iterator[lm_hooks.FixedEvaluator]:
    """Bind an evaluator reporting a fixed perplexity.

    Yields:
        The evaluator the command ran against.
    """
    original = lm_hooks.evaluator
    fixed = lm_hooks.FixedEvaluator(2.34)
    lm_hooks.evaluator = fixed
    yield fixed
    lm_hooks.evaluator = original


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("kk", ["kk"]),
        ("kk,ky", ["kk", "ky"]),
        (" kk , ky ", ["kk", "ky"]),
        ("kk,,ky", ["kk", "ky"]),
    ],
)
def test_language_codes_are_split_and_stripped(value: str, expected: list[str]) -> None:
    """Whitespace and empty entries never become a language code.

    Args:
        value: The ``--langs`` string as typed.
        expected: The codes it names.
    """
    assert parse_language_codes(value) == expected


@pytest.mark.parametrize("value", ["", "  ", ",", " , "])
def test_a_langs_value_naming_nothing_is_rejected(value: str) -> None:
    """Training on no language fails rather than producing an empty model.

    Args:
        value: A ``--langs`` string that names no code.
    """
    with pytest.raises(click.BadParameter, match="at least one ISO code"):
        parse_language_codes(value)


def test_train_lm_streams_every_language_into_one_run(
    corpora: None, trainer: lm_hooks.RecordingTrainer, tmp_path: Path
) -> None:
    """Sentences from each language arrive chained, normalised, and in order.

    Args:
        corpora: The bound corpus streamer.
        trainer: The bound trainer.
        tmp_path: Directory the model is saved into.
    """
    output = tmp_path / "model_out"
    result = CliRunner().invoke(
        train_lm_cli,
        [
            "--langs",
            "kk,ky",
            "--base-model",
            "hf/test",
            "--epochs",
            "1",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(trainer.runs) == 1
    base_model, epochs, sentences, output_dir = trainer.runs[0]
    assert base_model == "hf/test"
    assert epochs == 1
    assert sentences == ("салем", "әлем", "салам", "дүйнө")
    assert output_dir == str(output)


def test_train_lm_reports_where_it_saved(
    corpora: None, trainer: lm_hooks.RecordingTrainer, tmp_path: Path
) -> None:
    """The command names the saved model, which really is on disk.

    Args:
        corpora: The bound corpus streamer.
        trainer: The bound trainer.
        tmp_path: Directory the model is saved into.
    """
    output = tmp_path / "model_out"
    result = CliRunner().invoke(
        train_lm_cli, ["--langs", "kk", "--base-model", "hf/test", "--output-dir", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert str(output) in result.output
    assert (output / "config.json").exists()


def test_train_lm_uses_the_documented_default_epochs(
    corpora: None, trainer: lm_hooks.RecordingTrainer, tmp_path: Path
) -> None:
    """Omitting ``--epochs`` trains for the number the help advertises.

    Args:
        corpora: The bound corpus streamer.
        trainer: The bound trainer.
        tmp_path: Directory the model is saved into.
    """
    result = CliRunner().invoke(
        train_lm_cli, ["--langs", "kk", "--output-dir", str(tmp_path / "out")]
    )

    assert result.exit_code == 0, result.output
    _base_model, epochs, _sentences, _output_dir = trainer.runs[0]
    assert epochs == 3


def test_train_lm_rejects_a_langs_value_naming_nothing(
    corpora: None, trainer: lm_hooks.RecordingTrainer, tmp_path: Path
) -> None:
    """The command fails before training when no language is named.

    Args:
        corpora: The bound corpus streamer.
        trainer: The bound trainer.
        tmp_path: Directory that must stay empty.
    """
    result = CliRunner().invoke(
        train_lm_cli, ["--langs", " , ", "--output-dir", str(tmp_path / "out")]
    )

    assert result.exit_code == 2
    assert trainer.runs == []


def test_eval_lm_prints_the_perplexity_to_two_decimals(
    corpora: None, evaluator: lm_hooks.FixedEvaluator
) -> None:
    """The score is rendered exactly as the command documents.

    Args:
        corpora: The bound corpus streamer.
        evaluator: The bound evaluator.
    """
    result = CliRunner().invoke(
        eval_lm_cli, ["--model", "hf/test", "--eval-lang", "kk", "--sample", "10"]
    )

    # The stream's progress bar shares this stream, so the score is the
    # last thing written rather than the only thing.
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().endswith("2.34")


def test_eval_lm_scores_the_model_it_was_given(
    corpora: None, evaluator: lm_hooks.FixedEvaluator
) -> None:
    """The named model is scored against the normalised held-out text.

    Args:
        corpora: The bound corpus streamer.
        evaluator: The bound evaluator.
    """
    result = CliRunner().invoke(eval_lm_cli, ["--model", "hf/test", "--eval-lang", "ky"])

    assert result.exit_code == 0, result.output
    assert evaluator.scored == [("hf/test", ("салам", "дүйнө"))]


def test_eval_lm_honours_the_sample_cap(corpora: None, evaluator: lm_hooks.FixedEvaluator) -> None:
    """``--sample`` caps sentences scored, counting only non-blank ones.

    Args:
        corpora: The bound corpus streamer.
        evaluator: The bound evaluator.
    """
    result = CliRunner().invoke(
        eval_lm_cli, ["--model", "hf/test", "--eval-lang", "kk", "--sample", "1"]
    )

    assert result.exit_code == 0, result.output
    assert evaluator.scored == [("hf/test", ("салем",))]
