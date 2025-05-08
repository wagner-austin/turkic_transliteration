#!/usr/bin/env python3
import sentencepiece as spm, argparse, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="comma-separated files")
ap.add_argument("--model_prefix", default="spm/turkic12k")
ap.add_argument("--vocab_size", type=int, default=12000)
ap.add_argument("--model_type", default="unigram")
args = ap.parse_args()

spm.SentencePieceTrainer.train(
    input=args.input,
    model_prefix=args.model_prefix,
    vocab_size=args.vocab_size,
    model_type=args.model_type,
    character_coverage=1.0,
    normalization_rule_name="nfkc",
    user_defined_symbols=["<lang_kk>", "<lang_ky>"],
)
print(f"SentencePiece model saved at {args.model_prefix}.model")
