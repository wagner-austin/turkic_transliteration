from __future__ import annotations

import functools
import logging
import typing as t
from types import ModuleType
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # for static checkers only
    from .pipeline import TurkicTransliterationPipeline


def _make_pipeline() -> TurkicTransliterationPipeline:
    from .pipeline import TurkicTransliterationPipeline  # runtime import

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
    from .core import to_ipa, to_latin

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

    from .cli.filter_russian import main as _filter_ru_main

    # Check if language identification model exists
    home_lid = Path.home() / "lid.176.ftz"
    # Check in current directory and package directory
    pkg_dir = Path(__file__).parent
    pkg_lid = pkg_dir / "lid.176.ftz"

    if not home_lid.exists() and not pkg_lid.exists():
        log.warning(
            f"FastText language identification model missing. Need lid.176.ftz in {home_lid} or {pkg_lid}"
        )
        return (
            "**⚠️ Language identification model missing**\n\n"
            "The Russian filter feature requires the FastText language identification model.\n"
            "Please download `lid.176.ftz` from https://fasttext.cc/docs/en/language-identification.html\n"
            f"and place it in your home directory ({home_lid}) or package directory ({pkg_lid})"
        )

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
        raise RuntimeError(result.output)
    return str(result.output).rstrip("\n")


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
    from . import sanity

    lat_path = getattr(file_lat, "name", None)
    ipa_path = getattr(file_ipa, "name", None)
    if not lat_path or not ipa_path:
        raise ValueError("Both file_lat and file_ipa must have a .name attribute")
    if sample is not None:
        value = sanity.median_lev(lat_path, ipa_path, sample=sample)
    else:
        value = sanity.median_lev(lat_path, ipa_path)
    return f"Median distance: {value:.4f}"


__all__ = [
    "direct_transliterate",
    "pipeline_transliterate",
    "token_table_markdown",
    "mask_russian",
    "median_levenshtein",
]
