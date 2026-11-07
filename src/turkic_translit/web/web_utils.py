"""Helpers behind the web UI's transliteration, filtering and training tabs.

The download directory and its janitor are in :mod:`.downloads`, the
browser's log panel in :mod:`.ui_log`, and corpus streaming in
:mod:`.corpus_download`; they were part of this module until 2026-11-04.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

import pandas as pd

from turkic_translit.lang_utils import pretty_lang
from turkic_translit.tokenizer import (
    MODEL_PATH_VARIABLE,
    default_model_path,
    sentencepiece_trainer,
)

from ..corpus.normalize import language_folds, normalize_source_line
from ..error_service import (
    set_correlation_id,
    set_request_context,
)
from ..lang_filter import is_russian_token
from ..lid.errors import LidError
from ..lid.factory import load_installed_classifier
from ..pipeline import TurkicTransliterationPipeline

log = logging.getLogger(__name__)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class NamedFile(Protocol):
    """A Gradio file upload, which this module only reads a path from.

    The path is declared read-only because that is all this module does
    with it. A mutable attribute would exclude any immutable stand-in
    while granting a capability nothing here uses.
    """

    @property
    def name(self) -> str:
        """Return the path the upload's contents were written to.

        Returns:
            An absolute path on the local filesystem.
        """
        ...


def labelise(codes: list[str]) -> list[tuple[str, str]]:
    """Pair each language code with the name a dropdown should show.

    Args:
        codes: ISO language codes, in the order to offer them.

    Returns:
        Label and value pairs, in the same order.
    """
    return [(pretty_lang(c), c) for c in codes]


def _make_pipeline() -> TurkicTransliterationPipeline:
    """Build the pipeline the transliteration helpers share.

    Returns:
        A pipeline over the default language-identification model.
    """
    log.info("Instantiating TurkicTransliterationPipeline singleton")
    return TurkicTransliterationPipeline()


_lazy_pipeline = functools.lru_cache(maxsize=1)(_make_pipeline)

# The language-identification model is named explicitly; there is no
# preference order to drift. Loading it costs a 126 MB read, so the
# result is held for the life of the process. The cache is public
# because it is the reason a rebound loader hook would otherwise go
# unnoticed: whoever rebinds the hook clears this.
LANGUAGE_MODEL_ID = "lid.176"
installed_classifier = functools.lru_cache(maxsize=1)(load_installed_classifier)


def direct_transliterate(
    text: str, lang: str, include_arabic: bool, out_fmt: str
) -> tuple[str, str]:
    """Transliterate text to Latin or IPA with the language's rules.

    Each line is normalised first, as the downloader and the command line
    normalise theirs, so pasted text gives the IPA the same text gives
    anywhere else in the package.

    Args:
        text: Input in the language's native orthography.
        lang: ISO 639-1 language code.
        include_arabic: Whether to fold Arabic-script tokens to Latin
            before applying the target rules.
        out_fmt: ``latin`` or ``ipa``, in any casing.

    Returns:
        The transliteration and a Markdown line reporting the input and
        output sizes in bytes.

    Raises:
        ValueError: If the format is neither ``latin`` nor ``ipa``, or
            the language has no rules for the one requested.
    """
    from ..core import to_ipa, to_latin  # Import from main package

    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="direct_transliterate", lang=lang, out_fmt=out_fmt)

    fmt = out_fmt.lower()
    if fmt not in {"latin", "ipa"}:
        raise ValueError(f"out_fmt must be 'latin' or 'ipa', got {out_fmt!r}")
    folds = language_folds(lang)
    source = "\n".join(normalize_source_line(line, folds) for line in text.split("\n"))
    result = to_latin(source, lang, include_arabic) if fmt == "latin" else to_ipa(source, lang)
    stats_markdown = (
        f"**Bytes** — Cyrillic : {len(text.encode('utf8'))}, "
        f"{fmt.upper()} : {len(result.encode('utf8'))}"
    )
    return result, stats_markdown


def pipeline_transliterate(text: str, mode: str) -> tuple[str, str]:
    """Transliterate text through the tokenising pipeline.

    Unlike :func:`direct_transliterate`, this splits the text into
    tokens and identifies each one's language first, so mixed-language
    input is transliterated by the rules of the language each token is
    actually in.

    Args:
        text: Input text, which may mix languages.
        mode: ``latin`` or ``ipa``, in any casing.

    Returns:
        The transliteration and a Markdown line reporting its length.

    Raises:
        ValueError: If the mode is neither ``latin`` nor ``ipa``.
    """
    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="pipeline_transliterate", mode=mode)

    mode = mode.lower()
    if mode not in {"latin", "ipa"}:
        raise ValueError(f"mode must be 'latin' or 'ipa', got {mode!r}")
    pipeline = _lazy_pipeline()
    pipeline.mode = mode
    result = pipeline.process(text)
    stats_markdown = f"**{len(result)} chars**"
    return result, stats_markdown


def token_table_markdown(text: str) -> str:
    """Tokenise text and tabulate each token's predicted language.

    Args:
        text: Text to tokenise.

    Returns:
        A Markdown table, or a message explaining that the SentencePiece
        model has not been trained yet. That model is not shipped, so its
        absence is an ordinary first-run state rather than a defect, and
        the message says how to produce it.
    """
    set_correlation_id()
    set_request_context(action="token_table", sample=len(text))

    if not default_model_path().is_file():
        return (
            "**⚠️ Tokenizer model file missing**\n\n"
            f"`{default_model_path()}` is required for tokenization.\n"
            "Train one with the `turkic-build-spm` command, then point "
            f"`{MODEL_PATH_VARIABLE}` at it."
        )

    pipeline = _lazy_pipeline()
    tokens = pipeline.tokenizer.tokenize(text)
    languages = pipeline.predict_tokens(tokens)
    table: str = pd.DataFrame({"Token": tokens, "Lang": languages}).to_markdown(index=False)
    return table


def mask_russian(
    text: str, thr: float, min_len: int, *, margin: float = 0.10, debug: bool = False
) -> str:
    """
    Replace Russian tokens with <RU> using the shared heuristic.

    Args:
        text: Text to process
        thr: Confidence threshold for Russian detection
        min_len: Minimum token length to consider
        margin: Maximum margin for accepting RU when not the top label
        debug: Whether to include debug information in output

    Returns:
        Masked text with <RU> replacing Russian tokens
    """
    set_correlation_id()
    set_request_context(action="mask_russian", thr=thr, min_len=min_len)

    try:
        lid = installed_classifier(LANGUAGE_MODEL_ID)
    except LidError as exc:
        log.warning("language-identification model unavailable: %s", exc)
        return (
            "**⚠️ Language-identification model unavailable**\n\n"
            f"The Russian filter needs the `{LANGUAGE_MODEL_ID}` model.\n\n"
            f"Error: {exc!s}"
        )

    masked: list[str] = []
    reports: list[dict[str, str | bool | float]] = []
    for token in text.strip().split():
        is_russian = is_russian_token(
            token, thr=thr, min_len=min_len, lid=lid, stoplist=None, margin=margin
        )
        masked.append("<RU>" if is_russian else token)
        if debug:
            top = lid.classify(token.lower())
            reports.append(
                {
                    "tok": token,
                    "ru": is_russian,
                    "winner": top["label"],
                    "conf": top["probability"],
                }
            )

    out = " ".join(masked)
    if debug:
        out += "\n\n<!--debug " + json.dumps(reports, ensure_ascii=False) + " -->"
    return _ANSI.sub("", out)


def median_levenshtein(file_lat: NamedFile, file_ipa: NamedFile, sample: int | None = None) -> str:
    """Report the median edit distance between two aligned files.

    Args:
        file_lat: Upload holding the Latin transliteration, one line per
            sentence.
        file_ipa: Upload holding the IPA transliteration, aligned line
            for line with ``file_lat``.
        sample: Lines to compare, or ``None`` to use the default cap.

    Returns:
        The distance rendered as ``Median distance: 0.1234``.

    Raises:
        ValueError: If either upload names no path, which means the
            browser sent a file reference the UI never wrote.
        OSError: If either path cannot be read.
    """
    # Correlation for this user action
    set_correlation_id()
    set_request_context(action="median_lev", sample=sample or 0)

    from .. import sanity  # Import from main package, not web subpackage

    if file_lat.name == "" or file_ipa.name == "":
        raise ValueError("Both file_lat and file_ipa must have a .name attribute")
    if sample is not None:
        value = sanity.median_lev(file_lat.name, file_ipa.name, sample=sample)
    else:
        value = sanity.median_lev(file_lat.name, file_ipa.name)
    return f"Median distance: {value:.4f}"


def train_sentencepiece_model(
    input_text: str,
    training_file: NamedFile | None = None,
    vocab_size: int = 12000,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
    user_symbols: str = "<lang_kk>,<lang_ky>",
) -> tuple[str, str]:
    """
    Train a SentencePiece model using provided text and parameters.

    Args:
        input_text: Text content to use for training
        training_file: Optional file object to use for training (must have .name attribute)
        vocab_size: Largest vocabulary to learn; a text that supports
            fewer pieces yields fewer, and the report says so
        model_type: SentencePiece model type (unigram, bpe, char, word)
        character_coverage: Character coverage ratio
        user_symbols: Comma-separated list of user-defined symbols

    Returns:
        Tuple of (output model file path, info markdown string)

    Raises:
        ValueError: If neither input_text nor training_file is provided
    """
    if not input_text.strip() and not training_file:
        raise ValueError("Either input text or training file must be provided")

    # Create a temporary directory for training data and model files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        training_data_path = temp_dir_path / "training_data.txt"
        model_prefix = temp_dir_path / "spm_model"

        # Write input text to training file if provided
        input_files = []

        if input_text.strip():
            with open(training_data_path, "w", encoding="utf-8") as f:
                f.write(input_text.strip() + "\n")
            input_files.append(str(training_data_path))

        # If a file was uploaded, use its path directly for SentencePiece
        # training, which avoids loading large files into memory.
        if training_file is not None and training_file.name != "":
            input_files.append(training_file.name)

        # Parse user symbols
        user_symbols_list = [s.strip() for s in user_symbols.split(",") if s.strip()]

        # Train the model with all input files
        # This approach is more memory-efficient for large files
        sentencepiece_trainer().train(
            input=",".join(input_files),  # SentencePiece accepts comma-separated file paths
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            normalization_rule_name="nfkc",
            user_defined_symbols=user_symbols_list,
            # The requested size is a ceiling, not a demand. SentencePiece
            # 0.2.3 refuses a size the corpus cannot fill ("Vocabulary
            # size too high (50). Please set it to a value <= 48") for
            # every model type, so the demo's default of 12000 failed on
            # any pasted text; as a soft limit it learns what the corpus
            # supports, and the report below says how many that was.
            hard_vocab_limit=False,
            # Additional parameters that help with large corpus files
            input_sentence_size=10000000,  # Process up to 10M sentences (plenty for most use cases)
            shuffle_input_sentence=True,  # Shuffle for better training outcome
            num_threads=os.cpu_count() or 4,  # Use multiple threads for faster processing
        )

        # Path to the output model file
        model_file_path = str(model_prefix) + ".model"
        vocab_file_path = str(model_prefix) + ".vocab"

        # Count vocab items for stats
        vocab_count = 0
        with open(vocab_file_path, encoding="utf-8") as vocab_file:
            for _ in vocab_file:
                vocab_count += 1

        # Get model file size
        model_size_bytes = os.path.getsize(model_file_path)
        model_size_kb = model_size_bytes / 1024

        # Copy model to a more permanent location for download
        output_model_path = (
            Path(tempfile.gettempdir()) / f"turkic_sp_model_{vocab_size}_{model_type}.model"
        )
        with open(model_file_path, "rb") as src, open(output_model_path, "wb") as dst:
            dst.write(src.read())

        fitted_note = (
            f" (of {vocab_size} requested; the training text supports no more)"
            if vocab_count < vocab_size
            else ""
        )

        # Create info message
        info_md = f"""### Model Training Complete

**Model Statistics:**
- Vocabulary Size: {vocab_count} tokens{fitted_note}
- Model Type: {model_type}
- Character Coverage: {character_coverage}
- Model File Size: {model_size_kb:.2f} KB

You can download the model file below. To use this model with the Turkic Transliteration toolkit, 
place it in the appropriate location for your application.

For the default tokenizer, rename it to `turkic_model.model` and place it in the 
same directory as the `tokenizer.py` file.
"""

        return str(output_model_path), info_md


__all__ = [
    "direct_transliterate",
    "labelise",
    "mask_russian",
    "median_levenshtein",
    "pipeline_transliterate",
    "token_table_markdown",
    "train_sentencepiece_model",
]
