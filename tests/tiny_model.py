"""A real but minimal causal language model, built offline.

The language-model code cannot be exercised against a stand-in: it calls
into torch and transformers at every step, and a double would prove only
that the double was called. What it can be exercised against is a real
GPT-2 that is small enough to build and train in a second — two layers,
two heads, an eight-dimensional embedding, and a SentencePiece
vocabulary trained here on a handful of sentences.

Nothing is downloaded. The vocabulary is trained on disk with the
``sentencepiece`` this project already depends on, and the model is
constructed from a configuration, so the whole fixture works with no
network and no Hugging Face cache.

This module is the one place transformers' unannotated constructors are
called. Everything it hands back is annotated, so the modules that use it
are checked as strictly as the rest of the tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from turkic_translit.tokenizer import sentencepiece_trainer

CORPUS = [
    "salem alem qazaq tili",
    "salam dune kyrgyz tili",
    "merhaba dunya turk dili",
    "salom dunyo ozbek tili",
    "yaxshimisiz uyghur tili",
    "hello world turkic transliteration",
]

# Must be at least the training pad width: `LMModel.fresh` pads every
# example to MAX_SEQUENCE_LENGTH, and a position past the end of the
# position embedding is an IndexError inside torch rather than a
# transformers-level message.
CONTEXT_LENGTH = 128
PAD_TOKEN = "<pad>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"

# Small enough to train on the corpus above, large enough for the
# alphabet plus SentencePiece's own control tokens.
SENTENCEPIECE_VOCAB_SIZE = 40

# transformers' model and configuration constructors carry no
# annotations. Binding each to a Protocol that states the arguments used
# here keeps the untyped surface inside this module.
_GPT2_TOKENIZATION = "transformers.models.gpt2.tokenization_gpt2"
_GPT2_TOKENIZER = "GPT2Tokenizer"


class ConfigBuilder(Protocol):
    """transformers' GPT-2 configuration constructor."""

    def __call__(
        self, vocab_size: int, n_positions: int, n_embd: int, n_layer: int, n_head: int
    ) -> PretrainedConfig:
        """Describe a GPT-2 of the given size.

        Args:
            vocab_size: Number of tokens the embedding covers.
            n_positions: Longest sequence the model accepts.
            n_embd: Width of the embedding.
            n_layer: Number of transformer blocks.
            n_head: Number of attention heads per block.

        Returns:
            The configuration.
        """
        ...


class ModelBuilder(Protocol):
    """transformers' GPT-2 causal language-model constructor."""

    def __call__(self, config: PretrainedConfig) -> PreTrainedModel:
        """Build an untrained model from a configuration.

        Args:
            config: The architecture to instantiate.

        Returns:
            The model, with randomly initialised weights.
        """
        ...


class ByteLevelTokenizerBuilder(Protocol):
    """transformers' byte-level BPE tokenizer constructor."""

    def __call__(
        self,
        vocab_file: str,
        merges_file: str,
        unk_token: str,
        bos_token: str,
        eos_token: str,
    ) -> PreTrainedTokenizerBase:
        """Build a tokenizer from a vocabulary and a merge table.

        Args:
            vocab_file: Path of the ``vocab.json`` file.
            merges_file: Path of the ``merges.txt`` file.
            unk_token: Token standing for anything out of vocabulary.
            bos_token: Token marking the start of a sequence.
            eos_token: Token marking the end of a sequence.

        Returns:
            The tokenizer.
        """
        ...


# Each unannotated constructor is narrowed once, here, rather than at
# every call site. The annotation is what performs the narrowing, so it
# is load bearing rather than decorative — but one of them is enough.
# T5Tokenizer needs none: transformers annotates that one.
build_config: ConfigBuilder = GPT2Config
build_network: ModelBuilder = GPT2LMHeadModel
build_byte_level_tokenizer: ByteLevelTokenizerBuilder = getattr(
    __import__(_GPT2_TOKENIZATION, fromlist=[_GPT2_TOKENIZER]), _GPT2_TOKENIZER
)


def build_tokenizer(vocab_file: Path) -> PreTrainedTokenizerBase:
    """Wrap a trained SentencePiece model as a transformers tokenizer.

    Args:
        vocab_file: A ``.model`` file from :func:`write_sentencepiece_model`.

    Returns:
        A tokenizer with an end-of-text token and a pad token, ready to
        be saved beside a model.
    """
    return T5Tokenizer(
        vocab_file=str(vocab_file),
        eos_token=END_TOKEN,
        unk_token=UNKNOWN_TOKEN,
        pad_token=PAD_TOKEN,
        model_max_length=CONTEXT_LENGTH,
        legacy=False,
    )


def build_model(vocab_size: int) -> PreTrainedModel:
    """Construct an untrained GPT-2 small enough to run in a test.

    Args:
        vocab_size: Size of the tokenizer's vocabulary, which the
            embedding must match.

    Returns:
        The model, with randomly initialised weights.
    """
    return build_network(
        build_config(
            vocab_size=vocab_size,
            n_positions=CONTEXT_LENGTH,
            n_embd=8,
            n_layer=2,
            n_head=2,
        )
    )


def write_sentencepiece_model(target: Path, prefix: str = "spiece") -> Path:
    """Train a small SentencePiece model and return its ``.model`` file.

    Args:
        target: Directory to train into. Created if absent.
        prefix: Base name for the generated ``.model`` and ``.vocab``.

    Returns:
        Path of the trained ``.model`` file.
    """
    target.mkdir(parents=True, exist_ok=True)
    corpus = target / f"{prefix}-corpus.txt"
    corpus.write_text(
        "\n".join(f"{line} {index}" for index in range(60) for line in CORPUS) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sentencepiece_trainer().train(
        input=str(corpus),
        model_prefix=str(target / prefix),
        vocab_size=SENTENCEPIECE_VOCAB_SIZE,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=3,
    )
    return target / f"{prefix}.model"


def write_sentencepiece_tokenizer_directory(target: Path) -> Path:
    """Save a SentencePiece-backed tokenizer, as a slow tokenizer needs.

    ``T5Tokenizer`` is the SentencePiece-backed implementation used here
    because it exposes ``sp_model``, which is the attribute the override
    path in :func:`turkic_translit.lm.tokenizer.load_tokenizer` replaces.

    Args:
        target: Directory to write into. Created if absent.

    Returns:
        The directory, now loadable by ``AutoTokenizer``.
    """
    target.mkdir(parents=True, exist_ok=True)
    build_tokenizer(write_sentencepiece_model(target)).save_pretrained(str(target))
    return target


def write_byte_level_tokenizer_directory(target: Path) -> Path:
    """Save a tokenizer that is not SentencePiece backed.

    GPT-2's tokenizer is byte-level BPE and exposes no ``sp_model``,
    which is what makes it the case that must reject a SentencePiece
    override rather than silently ignoring it. Its two vocabulary files
    are written here rather than downloaded.

    Args:
        target: Directory to write into. Created if absent.

    Returns:
        The directory, now loadable by ``AutoTokenizer``.
    """
    target.mkdir(parents=True, exist_ok=True)
    vocabulary = {END_TOKEN: 0, **{chr(code): code - 96 for code in range(97, 123)}}
    (target / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    (target / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8", newline="\n")
    tokenizer = build_byte_level_tokenizer(
        vocab_file=str(target / "vocab.json"),
        merges_file=str(target / "merges.txt"),
        unk_token=END_TOKEN,
        bos_token=END_TOKEN,
        eos_token=END_TOKEN,
    )
    tokenizer.save_pretrained(str(target))
    return target


def write_model_directory(target: Path) -> Path:
    """Save a tokenizer and a model into a directory, as the Hub would.

    The result is a directory ``AutoTokenizer.from_pretrained`` and
    ``AutoModelForCausalLM.from_pretrained`` both load, which is what the
    production loaders are given.

    Args:
        target: Directory to write into. Created if absent.

    Returns:
        The directory, now holding a loadable model and tokenizer.
    """
    target.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(write_sentencepiece_model(target))
    model = build_model(len(tokenizer))
    tokenizer.save_pretrained(str(target))
    model.save_pretrained(str(target))
    return target
