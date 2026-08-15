"""Causal language models paired with the tokenizer they were built with.

Precision is chosen from the hardware rather than written in. Both entry
points previously hardcoded half precision, and ``TrainingArguments(
fp16=True)`` is rejected outright by transformers on a machine with no
CUDA device, so :meth:`LMModel.fresh` could not run at all on CPU — which
is every machine this project's tests run on. The two decisions are pure
functions of one flag so that both answers are reachable, and the flag is
read once at the call site.

The pad-token rule is stated once and used by both entry points. GPT-style
models ship without a pad token, and the data collator and the perplexity
loop both need one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Imported from the concrete submodules rather than the package root.
# transformers' root __init__ is a lazy module whose re-exports vary by
# version, so `from transformers import Trainer` resolves to a placeholder
# under some releases and to the real class under others. The submodule
# paths are stable and give the real classes on every version.
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from turkic_translit.lm.model_calls import (
    disable_gradient_checkpointing,
    switch_to_inference,
)
from turkic_translit.lm.tokenizer import load_tokenizer

__all__ = [
    "MAX_BUFFERED_SENTENCES",
    "MAX_SEQUENCE_LENGTH",
    "LMModel",
    "TokenisedSentences",
    "ensure_pad_token",
    "model_dtype",
    "use_half_precision",
]

MAX_BUFFERED_SENTENCES = 1_000_000
MAX_SEQUENCE_LENGTH = 128
FALLBACK_PAD_TOKEN = " "


def use_half_precision(cuda_available: bool) -> bool:
    """Report whether training should run in half precision.

    Args:
        cuda_available: Whether a CUDA device is present.

    Returns:
        True only on CUDA. Transformers raises rather than downgrading
        when ``fp16`` is requested without one, so this is a correctness
        condition, not a performance preference.
    """
    return cuda_available


def model_dtype(cuda_available: bool) -> torch.dtype:
    """Choose the dtype to load model weights in.

    Args:
        cuda_available: Whether a CUDA device is present.

    Returns:
        Half precision on CUDA, single precision otherwise. CPU kernels
        for half precision are missing for several operations these
        models use, so loading half on CPU trades accuracy for nothing.
    """
    return torch.float16 if cuda_available else torch.float32


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    """Give the tokenizer a pad token if it has none.

    GPT-style models ship without one, which breaks the data collator and
    any batched evaluation. The end-of-sequence token is preferred, then
    the beginning-of-sequence token, then a literal space.

    Args:
        tokenizer: Tokenizer to inspect and, if necessary, modify.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or FALLBACK_PAD_TOKEN


class TokenisedSentences(Dataset[dict[str, list[int]]]):
    """Training examples, tokenised on demand.

    Replaces ``datasets.Dataset.from_dict(...).map(...)``. That package
    ships no type information and had to be reached through a typing
    exemption, and everything it was doing here — hold a list of strings,
    tokenise each one, hand the result to the trainer — is a torch
    ``Dataset``, and torch is typed. The package remains a dependency for
    corpus streaming, where it does work nothing else does.

    Tokenising in ``__getitem__`` rather than up front also means the
    encoded batch is never held for the whole corpus at once.

    Args:
        sentences: The training text, already materialised.
        tokenizer: Tokenizer to encode each sentence with.
    """

    def __init__(self, sentences: Sequence[str], tokenizer: PreTrainedTokenizerBase) -> None:
        """Store the corpus and the tokenizer that will encode it."""
        self._sentences = list(sentences)
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        """Report how many training examples there are.

        Returns:
            The number of sentences.
        """
        return len(self._sentences)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """Encode one sentence, using the input ids as the labels.

        Args:
            index: Position of the sentence to encode.

        Returns:
            The encoding, with ``labels`` mirroring ``input_ids``, which
            is what makes this causal language-model training.
        """
        encoded: BatchEncoding = self._tokenizer(
            self._sentences[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
        )
        input_ids: list[int] = encoded["input_ids"]
        attention_mask: list[int] = encoded["attention_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": list(input_ids),
        }


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
            The loaded pair, with a pad token guaranteed.

        Raises:
            OSError: If the path names neither a local model directory
                nor a reachable repository.
        """
        tokenizer = load_tokenizer(path)
        ensure_pad_token(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=model_dtype(torch.cuda.is_available())
        )
        return cls(model, tokenizer)

    @classmethod
    def fresh(
        cls,
        base_model: str,
        *,
        lr: float = 5e-5,
        epochs: int = 3,
        sentences: Iterable[str],
        output_dir: str | Path,
    ) -> LMModel:
        """Fine-tune a base model on sentences and save the result.

        Args:
            base_model: Local directory or Hub repository id to start from.
            lr: Learning rate.
            epochs: Number of training epochs.
            sentences: Training text, read up to
                :data:`MAX_BUFFERED_SENTENCES` so memory does not scale
                with the corpus.
            output_dir: Directory to save the model and tokenizer into.

        Returns:
            The fine-tuned pair, already saved to ``output_dir``.

        Raises:
            OSError: If the base model cannot be read or the output
                directory cannot be written.
        """
        cuda_available = torch.cuda.is_available()
        tokenizer = load_tokenizer(base_model)
        ensure_pad_token(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=model_dtype(cuda_available)
        )

        buffered = list(islice(sentences, MAX_BUFFERED_SENTENCES))
        dataset = TokenisedSentences(buffered, tokenizer)

        arguments = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=4,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=use_half_precision(cuda_available),
            # Pinned host memory exists to speed transfers to a device.
            # Left at its default of True it warns on every CPU run and
            # buys nothing, because there is nothing to transfer to.
            dataloader_pin_memory=cuda_available,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            args=arguments,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()

        # Make the model inference-friendly before it is written out.
        switch_to_inference(model)
        disable_gradient_checkpointing(model)
        model.config.use_cache = True

        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        return cls(model, tokenizer)
