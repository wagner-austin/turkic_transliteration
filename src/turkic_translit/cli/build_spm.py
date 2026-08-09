#!/usr/bin/env python3
import click

from turkic_translit.tokenizer import sentencepiece_trainer


@click.command()
@click.option("--input", required=True, help="Comma-separated files for training.")
@click.option(
    "--model-prefix",
    default="spm/turkic12k",
    show_default=True,
    help="Prefix for output model files.",
)
@click.option("--vocab-size", default=12000, show_default=True, type=int, help="Vocabulary size.")
@click.option(
    "--model-type",
    default="unigram",
    show_default=True,
    help="Model type (unigram, bpe, char, word).",
)
@click.option(
    "--character-coverage",
    default=1.0,
    show_default=True,
    type=float,
    help="Amount of characters covered by the model.",
)
@click.option(
    "--user-symbols",
    default="<lang_kk>,<lang_ky>",
    show_default=True,
    help="Comma-separated user-defined symbols.",
)
def main(
    input: str,
    model_prefix: str,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    user_symbols: str,
) -> None:
    """Train a SentencePiece model for Turkic transliteration.

    Args:
        input: Comma-separated corpus files to train on.
        model_prefix: Base path for the ``.model`` and ``.vocab`` output.
        vocab_size: Number of pieces to learn. SentencePiece rejects a
            size the corpus cannot support, naming the largest that fits.
        model_type: Algorithm to use — ``unigram``, ``bpe``, ``char`` or
            ``word``.
        character_coverage: Fraction of the corpus's characters the
            vocabulary must cover.
        user_symbols: Comma-separated symbols to reserve as whole pieces,
            such as the per-language tags this project prefixes lines
            with.

    Raises:
        RuntimeError: If SentencePiece rejects the requested vocabulary
            size or cannot read the corpus.
    """
    user_symbols_list = [s.strip() for s in user_symbols.split(",") if s.strip()]
    sentencepiece_trainer().train(
        input=input,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        normalization_rule_name="nfkc",
        user_defined_symbols=user_symbols_list,
    )
    click.echo(f"SentencePiece model saved at {model_prefix}.model")
