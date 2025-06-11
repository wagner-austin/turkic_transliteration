# mypy: ignore-errors
"""Thin wrapper around *transformers* causal LMs to keep model & tokenizer together."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .tokenizer import load_tokenizer

__all__ = ["LMModel"]


@dataclass
class LMModel:  # noqa: D101 – simple dataclass
    model: AutoModelForCausalLM
    tokenizer: PreTrainedTokenizerBase

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path: str) -> LMModel:
        """Load model **and** tokenizer from a local path or HF repo."""
        tok = load_tokenizer(path)
        # Ensure pad token exists for inference-time collation and evaluate
        if tok.pad_token_id is None:  # runtime safeguard for GPT-style models
            tok.pad_token = tok.eos_token or tok.bos_token or " "
        mdl = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
        return cls(mdl, tok)

    # NOTE: Ordering of keyword-only params matches the diff supplied by user.
    @classmethod
    def fresh(
        cls,
        base_model: str,
        *,
        lr: float = 5e-5,
        epochs: int = 3,
        sentences: Iterable[str],
        spm_override: str | None = None,
        output_dir: str | Path,
    ) -> LMModel:
        """Fine-tune *base_model* on *sentences* and save to *output_dir*."""
        tok = load_tokenizer(base_model, spm_override)

        # Ensure a pad token exists — required by HF *evaluate* perplexity metric
        # and common data collators. GPT-style models ship without one.
        if tok.pad_token_id is None:  # pragma: no cover – runtime guard
            # Some base models (e.g. GPT-style) ship without a *pad_token* which
            # leads to crashes in common data collators and HF *evaluate*.
            # We pick (in order): existing EOS, existing BOS, or a sentinel
            # fallback recognised by the Transformers stack (' ').
            tok.pad_token = tok.eos_token or tok.bos_token or " "

        mdl = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16
        )

        # ------------------------------------------------------------------
        # Build Dataset on-the-fly from python iterable
        # ------------------------------------------------------------------
        # Imported lazily to avoid heavyweight dependency at import-time.
        from datasets import Dataset  # type: ignore

        def _encode(batch):
            out = tok(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            out["labels"] = out["input_ids"]
            return out

        # Prevent memory blow-up by buffering at most 1M sentences.
        buf = list(islice(sentences, 1_000_000))
        ds = Dataset.from_dict({"text": buf}).map(_encode, batched=True)

        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=4,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            report_to=[],
        )

        trainer = Trainer(
            model=mdl,
            tokenizer=tok,
            args=args,
            train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        )
        trainer.train()

        # Make the model inference-friendly
        mdl.eval()
        mdl.gradient_checkpointing_disable()
        mdl.config.use_cache = True

        # Persist
        mdl.save_pretrained(str(output_dir))
        tok.save_pretrained(str(output_dir))

        return cls(mdl, tok)
