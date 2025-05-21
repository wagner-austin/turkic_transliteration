#!/usr/bin/env python3
import click
import sentencepiece as spm


@click.command()
@click.option("--input", required=True, help="Comma-separated files for training.")
@click.option(
    "--model-prefix",
    default="spm/turkic12k",
    show_default=True,
    help="Prefix for output model files.",
)
@click.option(
    "--vocab-size", default=12000, show_default=True, type=int, help="Vocabulary size."
)
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
    """Train a SentencePiece model for Turkic transliteration."""
    user_symbols_list = [s.strip() for s in user_symbols.split(",") if s.strip()]
    spm.SentencePieceTrainer.train(
        input=input,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        normalization_rule_name="nfkc",
        user_defined_symbols=user_symbols_list,
    )
    click.echo(f"SentencePiece model saved at {model_prefix}.model")


if __name__ == "__main__":  # pragma: no cover
    main()
