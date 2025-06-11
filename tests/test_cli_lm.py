"""Unit tests for the LM CLI commands (PR 4)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from turkic_translit.cli.eval_lm import cli as eval_lm_cli

# Import CLI entry-points
from turkic_translit.cli.train_lm import cli as train_lm_cli

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:  # noqa: D401 – imperative style
    """Return a temporary directory path for model outputs."""
    out = tmp_path / "model_out"
    out.mkdir()
    return out


class _DummyModel:  # noqa: D101 – minimal stub
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)

    def gradient_checkpointing_disable(self) -> None:  # noqa: D401 – stub
        pass


class _DummyLM(SimpleNamespace):
    """Stub mimicking LMModel return object."""

    model: _DummyModel = _DummyModel()
    tokenizer: str = "dummy"


# ---------------------------------------------------------------------------
# test: turkic-train-lm ------------------------------------------------------
# ---------------------------------------------------------------------------


def test_train_lm_quick(monkeypatch: pytest.MonkeyPatch, tmp_output_dir: Path) -> None:
    """CLI should call *LMModel.fresh* with expected arguments."""

    calls: list[dict[str, Any]] = []

    def _fake_fresh(
        base_model: str,
        *,
        epochs: int,
        sentences: Iterable[str],
        output_dir: str,
    ) -> _DummyLM:  # noqa: D401
        # Just record call args then create a dummy file in *output_dir* to
        # imitate model saving.
        calls.append(
            {
                "base_model": base_model,
                "epochs": epochs,
                "sentences": sentences,
                "output_dir": output_dir,
            }
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
    assert calls[0]["base_model"] == "hf/test"
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
    monkeypatch.setattr(
        "turkic_translit.cli.eval_lm.cross_perplexity", lambda *_a, **_kw: 2.34
    )
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
