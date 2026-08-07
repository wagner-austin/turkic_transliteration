"""Unit tests for the LM CLI commands (PR 4)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from turkic_translit.cli.eval_lm import cli as eval_lm_cli

# Import CLI entry-points
from turkic_translit.cli.train_lm import cli as train_lm_cli

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Return a temporary directory path for model outputs."""
    out = tmp_path / "model_out"
    out.mkdir()
    return out


class _DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)

    def gradient_checkpointing_disable(self) -> None:
        pass


class _DummyLM(SimpleNamespace):
    """Stub mimicking LMModel return object."""

    model: _DummyModel = _DummyModel()
    tokenizer: str = "dummy"


# ---------------------------------------------------------------------------
# test: turkic-train-lm ------------------------------------------------------
@dataclass(frozen=True)
class _FreshCall:
    """One recorded call to ``LMModel.fresh``.

    Attributes:
        base_model: The model the CLI asked to start from.
        epochs: Number of epochs requested.
        sentences: Training text, materialised so it can be inspected
            after the CLI has consumed the iterator.
        output_dir: Where the CLI asked for the result to be saved.
    """

    base_model: str
    epochs: int
    sentences: tuple[str, ...]
    output_dir: str


# ---------------------------------------------------------------------------


def test_train_lm_quick(monkeypatch: pytest.MonkeyPatch, tmp_output_dir: Path) -> None:
    """CLI should call *LMModel.fresh* with expected arguments."""

    calls: list[_FreshCall] = []

    def _fake_fresh(
        base_model: str,
        *,
        epochs: int,
        sentences: Iterable[str],
        output_dir: str,
    ) -> _DummyLM:
        # Record the call, then create a file in output_dir to imitate
        # the model being saved.
        calls.append(
            _FreshCall(
                base_model=base_model,
                epochs=epochs,
                sentences=tuple(sentences),
                output_dir=output_dir,
            )
        )
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        (Path(output_dir) / "config.json").write_text("{}")
        return _DummyLM()

    monkeypatch.setattr("turkic_translit.cli.train_lm.LMModel.fresh", _fake_fresh)
    # DatasetStream yields sentences – patch to predictable iterable
    monkeypatch.setattr(
        "turkic_translit.cli.train_lm.DatasetStream",
        lambda *_a, **_kw: ["foo", "bar"],
    )

    runner = CliRunner()
    result = runner.invoke(
        train_lm_cli,
        [
            "--langs",
            "kk",
            "--base-model",
            "hf/test",
            "--epochs",
            "1",
            "--output-dir",
            str(tmp_output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls, "LMModel.fresh was not called"
    assert calls[0].base_model == "hf/test"
    assert calls[0].sentences == ("foo", "bar")
    assert calls[0].epochs == 1
    assert (tmp_output_dir / "config.json").exists()


# ---------------------------------------------------------------------------
# test: turkic-eval-lm -------------------------------------------------------
# ---------------------------------------------------------------------------


def test_eval_lm_quick(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI should print perplexity returned by *cross_perplexity*."""

    dummy_lm = _DummyLM()

    monkeypatch.setattr(
        "turkic_translit.cli.eval_lm.LMModel.from_pretrained",
        lambda *_a, **_kw: dummy_lm,
    )
    monkeypatch.setattr("turkic_translit.cli.eval_lm.cross_perplexity", lambda *_a, **_kw: 2.34)
    monkeypatch.setattr(
        "turkic_translit.cli.eval_lm.DatasetStream", lambda *_a, **_kw: ["x", "y", "z"]
    )

    runner = CliRunner()
    result = runner.invoke(
        eval_lm_cli,
        [
            "--model",
            "dummy",
            "--eval-lang",
            "kk",
            "--sample",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    # Output should contain the perplexity with two decimals (from f-string)
    assert "2.34" in result.output
