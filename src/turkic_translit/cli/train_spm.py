#!/usr/bin/env python3
"""turkic-train-spm – download corpora & train a SentencePiece model in one go.

This command lets users fetch corpora (OSCAR, Wikipedia, Leipzig, …) and
immediately train a shared SentencePiece model – no intermediate files or
Python coding required.

It re-uses the existing *download_corpus* registry/drivers to stream text
language-by-language, normalises to NFC, and passes it to
`sentencepiece.SentencePieceTrainer`.
"""

from __future__ import annotations

import json
import tempfile
import unicodedata as ud
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import click
import sentencepiece as spm
from tqdm import tqdm

from .download_corpus import _DRIVERS, _REG  # re-use existing machinery

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _stream_lines(
    lang: str,
    src_name: str,
    max_lines: int | None,
    filter_langid: bool,
) -> Iterable[str]:
    """Yield NFC-normalised lines for *lang* from the *src_name* corpus."""
    cfg = _REG[src_name]
    driver = _DRIVERS[cfg["driver"]]
    itr = driver(lang, cfg, lang if filter_langid else None)
    for i, line in enumerate(itr, 1):
        if max_lines and i > max_lines:
            break
        yield ud.normalize("NFC", line)


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of *path* (efficiently via mmap)."""
    import hashlib
    import mmap

    h = hashlib.sha256()
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        h.update(mm)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


@click.command("train-spm")
@click.option("--langs", required=True, help="Comma-separated ISO codes (kk,ky,uz)")
@click.option("--source", default="oscar-2301", show_default=True)
@click.option("--model-prefix", default="spm/turkic", show_default=True)
@click.option("--vocab-size", default=12000, show_default=True, type=int)
@click.option("--model-type", default="unigram", show_default=True)
@click.option("--character-coverage", default=1.0, type=float, show_default=True)
@click.option("--user-symbols", default="", show_default=True)
@click.option(
    "--hard-vocab-limit",
    is_flag=True,
    default=False,
    help="Enforce hard vocab size limit (SentencePiece default).",
)
@click.option("--max-lines", type=int, default=None, help="Per-language cap")
@click.option(
    "--filter-langid/--no-filter-langid",
    default=False,
    show_default=True,
    help="Filter sentences by FastText LID for each language.",
)
@click.option("--input-sentence-size", type=int, default=None)
@click.option(
    "--manifest",
    type=click.Path(dir_okay=False, writable=True),
    help="Optional JSON manifest for reproducibility.",
)
def main(**kw: Any) -> None:  # noqa: D401 – short signature for Click
    """Download corpora and train a SentencePiece model (no code required)."""

    langs: list[str] = [code.strip() for code in kw["langs"].split(",") if code.strip()]
    if not langs:
        raise click.UsageError("--langs must list at least one ISO code")

    tmpdir = tempfile.TemporaryDirectory()
    txt_paths: list[str] = []

    # 1️⃣  Gather corpora ----------------------------------------------------- #
    for lang in langs:
        fname = Path(tmpdir.name) / f"{lang}.txt"
        with fname.open("w", encoding="utf8") as fh:
            for line in tqdm(
                _stream_lines(
                    lang,
                    kw["source"],
                    kw["max_lines"],
                    kw["filter_langid"],
                ),
                unit="line",
                desc=f"[{lang}]",
                total=kw["max_lines"],
            ):
                fh.write(line + "\n")
        txt_paths.append(str(fname))

    # 2️⃣  Train SentencePiece ---------------------------------------------- #
    user_syms = [s for s in kw["user_symbols"].split(",") if s]
    if not user_syms:  # auto-inject language tags unless overridden
        user_syms = [f"<lang_{lang}>" for lang in langs]

    # Ensure output directory exists
    Path(kw["model_prefix"]).parent.mkdir(parents=True, exist_ok=True)

    spm_args: dict[str, object] = {
        "input": ",".join(txt_paths),
        "model_prefix": kw["model_prefix"],
        "vocab_size": kw["vocab_size"],
        "model_type": kw["model_type"],
        "character_coverage": kw["character_coverage"],
        "user_defined_symbols": user_syms,
    }
    spm_args["hard_vocab_limit"] = kw["hard_vocab_limit"]

    if kw["input_sentence_size"] is not None:
        spm_args["input_sentence_size"] = kw["input_sentence_size"]

    click.echo("[+] Training SentencePiece – this may take a while …")
    spm.SentencePieceTrainer.train(**spm_args)
    click.secho(f"✓ Model at {kw['model_prefix']}.model", fg="green")

    # 3️⃣  Optional manifest ------------------------------------------------- #
    if kw["manifest"]:
        # Create manifest directory if needed
        Path(kw["manifest"]).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "langs": langs,
            "source": kw["source"],
            "txt_paths": txt_paths,
            "spm_args": spm_args,
            "sha256": _sha256(Path(f"{kw['model_prefix']}.model")),
        }
        Path(kw["manifest"]).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        click.echo(f"Manifest → {kw['manifest']}")

    # Cleanup temp dir (unless user interrupted) ---------------------------- #
    tmpdir.cleanup()


if __name__ == "__main__":  # pragma: no cover
    main()
