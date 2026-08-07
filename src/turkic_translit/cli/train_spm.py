"""``turkic-train-spm`` — stream corpora and train a SentencePiece model.

Corpora are gathered through :func:`turkic_translit.corpus.run.download_corpus`,
one file per language, so each language's corpus arrives with its own
manifest naming the filter that produced it. Those manifests are embedded
in the model's manifest, which makes a trained vocabulary traceable to
the exact classifier and threshold that shaped its input.

``--lid-model`` is what turns filtering on. There is no boolean switch,
because "filter by language" without naming the classifier is the
ambiguity that made this project's earlier models irreproducible: two
classifiers keep different lines at the same threshold, and nothing
recorded which had been used.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import click
import sentencepiece as spm

from turkic_translit.corpus.filtering import LidFilterRequest
from turkic_translit.corpus.manifest import (
    CorpusRunManifest,
    encode_corpus_run_manifest,
)
from turkic_translit.corpus.run import download_corpus
from turkic_translit.corpus.sources import known_source_ids
from turkic_translit.lid.registry import known_model_ids

DEFAULT_SOURCE: str = "oscar-2301"
DEFAULT_THRESHOLD: float = 0.95
_HASH_CHUNK_BYTES: int = 1 << 20


def sha256_of_file(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in chunks.

    Args:
        path: File to digest.

    Returns:
        The digest as lowercase hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def gather_corpora(
    languages: tuple[str, ...],
    source_id: str,
    max_lines: int | None,
    lid_model: str | None,
    lid_threshold: float,
    directory: Path,
) -> list[tuple[Path, CorpusRunManifest]]:
    """Download one corpus per language into ``directory``.

    Each language is filtered against its own code, so ``kk`` keeps
    Kazakh and ``ky`` keeps Kyrgyz, using the one classifier named for
    the whole run.

    Args:
        languages: Language codes to gather, in the order requested.
        source_id: Registry key of the corpus to stream.
        max_lines: Per-language cap on lines kept, or ``None``.
        lid_model: Classifier to filter with, or ``None`` for no filter.
        lid_threshold: Minimum probability required to keep a line.
        directory: Directory to write the per-language corpora into.

    Returns:
        Each corpus path paired with the manifest describing its run.
    """
    gathered: list[tuple[Path, CorpusRunManifest]] = []
    token = os.getenv("HF_TOKEN")
    for language in languages:
        lid_filter = (
            None
            if lid_model is None
            else LidFilterRequest(language=language, model_id=lid_model, threshold=lid_threshold)
        )
        path = directory / f"{language}.txt"
        manifest, _manifest_path = download_corpus(
            source_id, language, path, max_lines, token, lid_filter
        )
        gathered.append((path, manifest))
    return gathered


def build_trainer_arguments(
    corpus_paths: list[Path],
    model_prefix: str,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    user_symbols: tuple[str, ...],
    hard_vocab_limit: bool,
    input_sentence_size: int | None,
) -> dict[str, str | int | float | bool | list[str]]:
    """Assemble the argument mapping SentencePiece is trained with.

    Args:
        corpus_paths: Per-language corpus files to train on.
        model_prefix: Path prefix for the ``.model`` and ``.vocab`` files.
        vocab_size: Target vocabulary size.
        model_type: SentencePiece algorithm, e.g. ``unigram``.
        character_coverage: Fraction of characters the model must cover.
        user_symbols: Symbols reserved verbatim in the vocabulary.
        hard_vocab_limit: Whether the vocabulary size is a hard limit.
        input_sentence_size: Sentence sample size, or ``None`` for all.

    Returns:
        The trainer arguments, recorded verbatim in the manifest.
    """
    arguments: dict[str, str | int | float | bool | list[str]] = {
        "input": ",".join(str(path) for path in corpus_paths),
        "model_prefix": model_prefix,
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "user_defined_symbols": list(user_symbols),
        "hard_vocab_limit": hard_vocab_limit,
    }
    if input_sentence_size is not None:
        arguments["input_sentence_size"] = input_sentence_size
    return arguments


def build_manifest_document(
    languages: tuple[str, ...],
    source_id: str,
    corpora: list[tuple[Path, CorpusRunManifest]],
    trainer_arguments: dict[str, str | int | float | bool | list[str]],
    model_path: Path,
) -> dict[
    str,
    str
    | list[str]
    | dict[str, str | int | float | bool | list[str]]
    | list[dict[str, str | int | None | dict[str, str | int | float | bool]]],
]:
    """Assemble the JSON document describing a training run.

    Args:
        languages: Language codes that were gathered.
        source_id: Registry key of the corpus they came from.
        corpora: Each corpus path with the manifest describing its run.
        trainer_arguments: Exactly what SentencePiece was called with.
        model_path: The trained ``.model`` file.

    Returns:
        A mapping ready to be written as JSON. Each corpus manifest is
        embedded whole, so the classifier that filtered the training text
        is recoverable from the model's manifest alone.
    """
    return {
        "languages": list(languages),
        "source_id": source_id,
        "spm_args": trainer_arguments,
        "model_sha256": sha256_of_file(model_path),
        "corpora": [encode_corpus_run_manifest(manifest) for _path, manifest in corpora],
    }


@click.command("train-spm")
@click.option("--langs", required=True, help="Comma-separated codes, e.g. kk,ky,uz")
@click.option(
    "--source",
    type=click.Choice(known_source_ids()),
    default=DEFAULT_SOURCE,
    show_default=True,
)
@click.option("--model-prefix", default="spm/turkic", show_default=True)
@click.option("--vocab-size", default=12000, show_default=True, type=int)
@click.option("--model-type", default="unigram", show_default=True)
@click.option("--character-coverage", default=1.0, type=float, show_default=True)
@click.option(
    "--user-symbols",
    default="",
    show_default=True,
    help="Comma-separated symbols; defaults to one language tag per language",
)
@click.option(
    "--hard-vocab-limit",
    is_flag=True,
    default=False,
    help="Enforce the vocabulary size as a hard limit",
)
@click.option("--max-lines", type=int, default=None, help="Per-language cap")
@click.option(
    "--lid-model",
    type=click.Choice(known_model_ids()),
    default=None,
    help="Filter each language with this classifier; omit to keep every line",
)
@click.option(
    "--lid-threshold",
    type=float,
    default=DEFAULT_THRESHOLD,
    show_default=True,
    help="Minimum probability a line must reach to be kept",
)
@click.option("--input-sentence-size", type=int, default=None)
@click.option(
    "--manifest",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Where to write the JSON manifest describing this run",
)
def main(
    langs: str,
    source: str,
    model_prefix: str,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    user_symbols: str,
    hard_vocab_limit: bool,
    max_lines: int | None,
    lid_model: str | None,
    lid_threshold: float,
    input_sentence_size: int | None,
    manifest: str | None,
) -> None:
    """Stream corpora and train a SentencePiece model over them.

    Args:
        langs: Comma-separated language codes to train on.
        source: Registry key of the corpus to stream.
        model_prefix: Path prefix for the model and vocabulary files.
        vocab_size: Target vocabulary size.
        model_type: SentencePiece algorithm, e.g. ``unigram``.
        character_coverage: Fraction of characters the model must cover.
        user_symbols: Comma-separated reserved symbols; when empty, one
            ``<lang_xx>`` tag is reserved per language.
        hard_vocab_limit: Whether the vocabulary size is a hard limit.
        max_lines: Per-language cap on lines kept, or ``None``.
        lid_model: Classifier to filter each language with, or ``None``.
        lid_threshold: Minimum probability required to keep a line.
        input_sentence_size: Sentence sample size, or ``None`` for all.
        manifest: Where to write the run manifest, or ``None`` to skip.

    Raises:
        click.UsageError: If ``--langs`` names no language.
    """
    languages = tuple(code.strip() for code in langs.split(",") if code.strip())
    if not languages:
        raise click.UsageError("--langs must list at least one language code")

    symbols = tuple(symbol for symbol in user_symbols.split(",") if symbol)
    if not symbols:
        symbols = tuple(f"<lang_{language}>" for language in languages)

    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as workspace:
        corpora = gather_corpora(
            languages, source, max_lines, lid_model, lid_threshold, Path(workspace)
        )
        trainer_arguments = build_trainer_arguments(
            [path for path, _manifest in corpora],
            model_prefix,
            vocab_size,
            model_type,
            character_coverage,
            symbols,
            hard_vocab_limit,
            input_sentence_size,
        )
        click.echo("Training SentencePiece; this may take a while.")
        spm.SentencePieceTrainer.train(**trainer_arguments)
        model_path = Path(f"{model_prefix}.model")
        click.secho(f"Model at {model_path}", fg="green")

        if manifest is not None:
            document = build_manifest_document(
                languages, source, corpora, trainer_arguments, model_path
            )
            manifest_path = Path(manifest)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            click.echo(f"Manifest at {manifest_path}")

