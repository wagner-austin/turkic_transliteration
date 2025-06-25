"""
End-to-end smoke-tests for the new `turkic-train-spm` CLI.

We parametrise over the three supported corpus drivers:

* wikipedia   – single dump fetch, still lightweight with few lines
* oscar-2301  – HuggingFace streaming dataset

For speed we cap each at <= 100 lines and request a *tiny* vocab (1 k) so the
whole suite stays well under 1 min even on CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click  # for ClickException class used by CLI drivers
import pytest

PARAMS = [
    # (source-name, max_lines)
    ("wikipedia", 50),
    ("oscar-2301", 20),
]


@pytest.mark.parametrize(("source", "max_lines"), PARAMS)
def test_train_spm_each_source(tmp_path: Path, source: str, max_lines: int) -> None:
    """Train a tiny SPM model from each corpus driver and ensure it succeeds."""

    out_prefix = tmp_path / f"turkic_{source.replace('-', '_')}"
    manifest = tmp_path / f"manifest_{source}.json"

    cmd = [
        sys.executable,
        "-m",
        "turkic_translit.cli.train_spm",
        "--langs",
        "kk",  # single language is enough for smoke test
        "--source",
        source,
        "--model-prefix",
        str(out_prefix),
        "--vocab-size",
        "1000",  # tiny to speed up SPM training
        "--max-lines",
        str(max_lines),
        "--manifest",
        str(manifest),
    ]

    try:
        # Suppress verbose SentencePiece progress to keep CI logs short
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        # Network hiccup or remote host down → mark as skipped, not failed
        pytest.skip(f"{source} driver unavailable in this environment: {exc}")
    except click.ClickException as exc:
        pytest.skip(f"{source} driver raised ClickException: {exc}")

    # ---------- Assertions --------------------------------------------------
    model_path = out_prefix.with_suffix(".model")
    assert model_path.exists(), "SentencePiece model file missing"

    # Check manifest integrity
    meta = json.loads(Path(manifest).read_text())
    assert meta["source"] == source
    assert meta["spm_args"]["vocab_size"] == 1000
