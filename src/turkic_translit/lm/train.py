"""Causal language models paired with the tokenizer they were built with.

The pad-token fallback below used a SIX-PER-EM SPACE (U+2005) rather than
an ordinary space, so a GPT-style model with neither an EOS nor a BOS
token would have been padded with an exotic Unicode space. Ruff found it
once this module stopped carrying a file-level mypy suppression.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import BatchEncoding

from .tokenizer import load_tokenizer

__all__ = ["LMModel"]


@dataclass
class LMModel:
    """A loaded causal LM together with its tokenizer.

    Attributes:
        model: The causal language model.
        tokenizer: The tokenizer the model was built with.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    @classmethod
    def from_pretrained(cls, path: str) -> LMModel:
        """Load a model and its tokenizer from a path or a Hub repo.

        Args:
            path: Local directory or Hugging Face repository id.

        Returns:
            The loaded pair.
        """
        tok = load_tokenizer(path)
        # Ensure pad token exists for inference-time collation and evaluate
        if tok.pad_token_id is None:  # runtime safeguard for GPT-style models
            tok.pad_token = tok.eos_token or tok.bos_token or " "
        mdl = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
        return cls(mdl, tok)

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
        """Fine-tune a base model on sentences and save the result.

        Args:
            base_model: Local directory or Hub repository id to start from.
            lr: Learning rate.
            epochs: Number of training epochs.
            sentences: Training text, read up to the buffer limit below.
            spm_override: SentencePiece model to substitute, or ``None``.
            output_dir: Directory to save the model and tokenizer into.

        Returns:
            The fine-tuned pair, already saved to ``output_dir``.
        """
        tok = load_tokenizer(base_model, spm_override)

        # Ensure a pad token exists — required by HF *evaluate* perplexity metric
        # and common data collators. GPT-style models ship without one.
        if tok.pad_token_id is None:
            # Some base models (e.g. GPT-style) ship without a *pad_token* which
            # leads to crashes in common data collators and HF *evaluate*.
            # We pick (in order): existing EOS, existing BOS, or a sentinel
            # fallback recognised by the Transformers stack (' ').
            tok.pad_token = tok.eos_token or tok.bos_token or " "

        mdl = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)

        # ------------------------------------------------------------------
        # Build Dataset on-the-fly from python iterable
        # ------------------------------------------------------------------
        # Imported lazily to avoid heavyweight dependency at import-time.
        from datasets import Dataset

        def _encode(batch: dict[str, list[str]]) -> BatchEncoding:
            """Tokenise one batch and use the inputs as labels.

            Args:
                batch: A mapping with a ``text`` key holding the batch.

            Returns:
                The encoding, with ``labels`` mirroring ``input_ids``.
            """
            out: BatchEncoding = tok(
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
