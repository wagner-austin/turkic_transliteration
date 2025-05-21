from __future__ import annotations

import functools
import logging
import os
import tempfile
import typing as t
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # for static checkers only
    from ..pipeline import TurkicTransliterationPipeline  # Import from main package


def _make_pipeline() -> TurkicTransliterationPipeline:
    from ..pipeline import TurkicTransliterationPipeline  # Import from main package

    log.info("Instantiating TurkicTransliterationPipeline singleton")
    return TurkicTransliterationPipeline()


_lazy_pipeline = functools.lru_cache(maxsize=1)(_make_pipeline)


def direct_transliterate(
    text: str, lang: str, include_arabic: bool, out_fmt: str
) -> tuple[str, str]:
    """
    Transliterate text directly to Latin or IPA.
    Usage: direct_transliterate('сәлем', 'kk', False, 'latin')
    Returns: (result, stats_markdown)
    Raises: ValueError if out_fmt is invalid.
    """
    from ..core import to_ipa, to_latin  # Import from main package

    fmt = out_fmt.lower()
    if fmt not in {"latin", "ipa"}:
        raise ValueError(f"out_fmt must be 'latin' or 'ipa', got {out_fmt!r}")
    if fmt == "latin":
        result = to_latin(text, lang, include_arabic)
    else:
        result = to_ipa(text, lang)
    stats_markdown = (
        f"**Bytes** — Cyrillic : {len(text.encode('utf8'))}, "
        f"{fmt.upper()} : {len(result.encode('utf8'))}"
    )
    return result, stats_markdown


def pipeline_transliterate(text: str, mode: str) -> tuple[str, str]:
    """
    Transliterate text using the pipeline (mode: 'latin' or 'ipa').
    Usage: pipeline_transliterate('сәлем', 'ipa')
    Returns: (result, stats_markdown)
    Raises: ValueError if mode is invalid (passed to pipeline).
    """
    mode = mode.lower()
    if mode not in {"latin", "ipa"}:
        raise ValueError(f"mode must be 'latin' or 'ipa', got {mode!r}")
    pipeline = _lazy_pipeline()
    pipeline.mode = mode
    result = pipeline.process(text)
    stats_markdown = f"**{len(result)} chars**"
    return result, stats_markdown


# Soft import pandas at module level

# Soft import pandas - using TYPE_CHECKING pattern to provide type hints
# without requiring pandas at import time
if TYPE_CHECKING:
    import pandas as pd
else:
    # Runtime pandas import with fallback
    try:
        import pandas as pd
    except ModuleNotFoundError:  # pragma: no cover
        pd: ModuleType | None = None


def token_table_markdown(text: str) -> str:
    """
    Tokenize text and return a markdown table of tokens and language predictions.
    Usage: token_table_markdown('сәлем әлем!')
    Returns: markdown string
    Raises: ImportError if pandas is not installed.
    """
    if pd is None:
        raise ImportError(
            "install turkic-transliterate[ui] to use token_table_markdown"
        )

    try:
        pipeline = _lazy_pipeline()
        tokens = pipeline.tokenizer.tokenize(text)
        langs = pipeline.langid.predict_tokens(tokens)
        df = pd.DataFrame({"Token": tokens, "Lang": langs})
        markdown_table: str = df.to_markdown(index=False)
        return markdown_table
    except OSError as e:
        if "turkic_model.model" in str(e):
            return (
                "**⚠️ Tokenizer model file missing**\n\n"
                "The `turkic_model.model` file is required for tokenization.\n"
                "Please train a model with `turkic-build-spm` command\n"
                "and place it in the package directory."
            )
        # Other OSError not related to missing model
        return f"**Error loading tokenizer:** {e}"


def mask_russian(text: str, thr: float, min_len: int) -> str:
    """
    Mask Russian tokens in text using CLI runner.
    Usage: mask_russian('сәлем привет', 0.8, 2)
    Returns: masked text (str)
    Raises: RuntimeError if CLI invocation fails.
    """
    from pathlib import Path

    from click.testing import CliRunner

    from ..cli.filter_russian import main as _filter_ru_main  # Import from main package

    try:
        # Try to ensure the model is available, downloading if necessary
        from ..model_utils import ensure_fasttext_model  # Import from main package

        model_path = ensure_fasttext_model()
        log.info(f"Using FastText model at {model_path}")
    except Exception as e:
        log.warning(f"Failed to automatically download FastText model: {e}")
        # Check if language identification model exists in standard locations
        home_lid = Path.home() / "lid.176.ftz"
        pkg_dir = Path(__file__).parent
        pkg_lid = pkg_dir / "lid.176.ftz"

        if not home_lid.exists() and not pkg_lid.exists():
            log.warning(
                f"FastText language identification model missing. Need lid.176.ftz in {home_lid} or {pkg_lid}"
            )
            return (
                "**⚠️ Attempting to download language identification model...**\n\n"
                "The Russian filter feature requires the FastText language identification model.\n"
                "Automatic download failed. You can manually download `lid.176.ftz` from https://fasttext.cc/docs/en/language-identification.html\n"
                f"and place it in your home directory ({home_lid}) or package directory ({pkg_lid})\n\n"
                f"Error: {str(e)}"
            )

    # Set up a temporary log level to avoid logs in output
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.ERROR)  # Temporarily suppress INFO logs

    try:
        runner = CliRunner()
        result = runner.invoke(
            _filter_ru_main,
            ["--mode", "mask", "--thr", str(thr), "--min-len", str(min_len)],
            input=text,
        )

        if result.exit_code != 0:
            if "No such file or directory: 'lid.176.ftz'" in result.output:
                return (
                    "**⚠️ Language identification model not found**\n\n"
                    "The Russian filter feature requires the FastText language identification model.\n"
                    "Please download `lid.176.ftz` from https://fasttext.cc/docs/en/language-identification.html\n"
                    f"and place it in your home directory ({home_lid}) or package directory ({pkg_lid})"
                )
            # For other errors, show a clean error message
            return f"**Error masking Russian text**: {result.output.strip()}"

        # Strip log messages from output - only keep the last line
        output_lines = result.output.strip().split("\n")
        return output_lines[-1] if output_lines else ""
    finally:
        # Restore original log level
        root_logger.setLevel(original_level)


def median_levenshtein(
    file_lat: t.Any, file_ipa: t.Any, sample: int | None = None
) -> str:
    """
    Compute median Levenshtein distance between two files (accepts any objects with .name attribute).
    Usage: median_levenshtein(NamedTuple('F', [('name', str)])('lat.txt'), NamedTuple('F', [('name', str)])('ipa.txt'))
    Returns: formatted string prefixed with 'Median distance: ...'.
    Example: 'Median distance: 0.1234'
    Raises: ValueError if file objects are missing .name.
    """
    from .. import sanity  # Import from main package, not web subpackage

    lat_path = getattr(file_lat, "name", None)
    ipa_path = getattr(file_ipa, "name", None)
    if not lat_path or not ipa_path:
        raise ValueError("Both file_lat and file_ipa must have a .name attribute")
    if sample is not None:
        value = sanity.median_lev(lat_path, ipa_path, sample=sample)
    else:
        value = sanity.median_lev(lat_path, ipa_path)
    return f"Median distance: {value:.4f}"


def train_sentencepiece_model(
    input_text: str,
    training_file: t.Any = None,
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
        vocab_size: Size of the vocabulary to create
        model_type: SentencePiece model type (unigram, bpe, char, word)
        character_coverage: Character coverage ratio
        user_symbols: Comma-separated list of user-defined symbols

    Returns:
        Tuple of (output model file path, info markdown string)

    Raises:
        ValueError: If neither input_text nor training_file is provided
        ImportError: If sentencepiece is not installed
    """
    try:
        import sentencepiece as spm
    except ImportError as err:
        raise ImportError(
            "SentencePiece is required for model training. Please install with: pip install sentencepiece"
        ) from err

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

        # If a file was uploaded, use its path directly for SentencePiece training
        # This avoids loading large files into memory
        if training_file:
            file_path = getattr(training_file, "name", None)
            if file_path:
                input_files.append(file_path)

        # Parse user symbols
        user_symbols_list = [s.strip() for s in user_symbols.split(",") if s.strip()]

        # Train the model with all input files
        # This approach is more memory-efficient for large files
        spm.SentencePieceTrainer.train(
            input=",".join(
                input_files
            ),  # SentencePiece accepts comma-separated file paths
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            normalization_rule_name="nfkc",
            user_defined_symbols=user_symbols_list,
            # Additional parameters that help with large corpus files
            input_sentence_size=10000000,  # Process up to 10M sentences (plenty for most use cases)
            shuffle_input_sentence=True,  # Shuffle for better training outcome
            num_threads=os.cpu_count()
            or 4,  # Use multiple threads for faster processing
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
            Path(tempfile.gettempdir())
            / f"turkic_sp_model_{vocab_size}_{model_type}.model"
        )
        with open(model_file_path, "rb") as src, open(output_model_path, "wb") as dst:
            dst.write(src.read())

        # Create info message
        info_md = f"""### Model Training Complete

**Model Statistics:**
- Vocabulary Size: {vocab_count} tokens
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
    "pipeline_transliterate",
    "token_table_markdown",
    "mask_russian",
    "median_levenshtein",
    "train_sentencepiece_model",
]
