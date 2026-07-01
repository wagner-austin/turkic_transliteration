"""``turkic-translit translit`` — file-to-file IPA / Latin transliteration.

Delegates to :mod:`turkic_translit.core`. The set of language codes
accepted by ``--lang`` is discovered dynamically from the rules
directory via :func:`core.get_supported_languages`, so adding a new
``<lang>_ipa.rules`` file automatically exposes that language through
the CLI without further code changes.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import ExitStack, nullcontext
from typing import TextIO

import click

from ..core import get_supported_languages, to_ipa, to_latin

_logger = logging.getLogger(__name__)


def _open_input(stack: ExitStack, path: str, encoding: str) -> TextIO:
    """Open ``path`` for reading, or return ``sys.stdin`` when ``path`` is ``-``.

    The returned stream is registered with ``stack`` so callers do not
    need a separate ``finally`` clause.

    Args:
        stack: The active :class:`contextlib.ExitStack` that owns the
            returned stream.
        path: A filesystem path, or the sentinel ``"-"`` for stdin.
        encoding: Text encoding to use when ``path`` names a real file.

    Returns:
        A text-mode readable stream registered with ``stack``.
    """
    if path == "-":
        return stack.enter_context(nullcontext(sys.stdin))
    return stack.enter_context(open(path, encoding=encoding))


def _open_output(stack: ExitStack, path: str | None, encoding: str) -> TextIO | None:
    """Open ``path`` for writing.

    Returns ``sys.stdout`` for ``"-"``, an opened file for a real path,
    and ``None`` when ``path`` itself is ``None`` (mode not requested).

    Args:
        stack: The active :class:`contextlib.ExitStack` that owns the
            returned stream.
        path: A filesystem path, ``"-"`` for stdout, or ``None`` to
            signal that this output mode was not requested.
        encoding: Text encoding to use when ``path`` names a real file.

    Returns:
        A text-mode writable stream, or ``None`` when ``path`` is
        ``None``.
    """
    if path is None:
        return None
    if path == "-":
        return stack.enter_context(nullcontext(sys.stdout))
    return stack.enter_context(open(path, "w", encoding=encoding))


def _stream_transliteration(
    input_stream: TextIO,
    latin_output: TextIO | None,
    ipa_output: TextIO | None,
    lang: str,
    arabic: bool,
) -> int:
    """Read ``input_stream`` line-by-line and emit Latin / IPA output.

    Reads one line at a time so behaviour is streaming rather than
    buffered — the entire input need not fit in memory. Callers pass
    ``None`` for a mode they do not want emitted; that mode's
    per-line transliteration is skipped entirely (never computed).

    Args:
        input_stream: File-like object yielding orthographic text lines.
        latin_output: Destination for Latin output, or ``None`` to skip
            Latin transliteration entirely.
        ipa_output: Destination for IPA output, or ``None`` to skip IPA
            transliteration entirely.
        lang: ISO 639-1 language code understood by :func:`core.to_ipa`
            and :func:`core.to_latin`.
        arabic: When True, pre-pass Arabic script through the
            ``ar_lat.rules`` transliterator before the target rule set
            applies. Only meaningful when ``latin_output`` is not
            ``None``.

    Returns:
        The number of input lines consumed.
    """
    lines_consumed = 0
    for line in input_stream:
        stripped = line.rstrip("\n")
        if latin_output is not None:
            latin_output.write(to_latin(stripped, lang, arabic) + "\n")
        if ipa_output is not None:
            ipa_output.write(to_ipa(stripped, lang) + "\n")
        lines_consumed += 1
    return lines_consumed


def _validate_output_selection(
    lang: str, latin_path: str | None, ipa_path: str | None
) -> None:
    """Reject argument combinations the language's rules cannot satisfy.

    Args:
        lang: The requested language code.
        latin_path: The value of ``--out-latin`` (or ``None`` when the
            option was omitted).
        ipa_path: The value of ``--out-ipa`` (or ``None`` when the
            option was omitted).

    Raises:
        click.UsageError: When neither output was requested, or when
            the requested output is not supported by the language's
            rules directory contents.
    """
    if latin_path is None and ipa_path is None:
        raise click.UsageError(
            "at least one of --out-latin or --out-ipa must be specified"
        )

    supported = get_supported_languages()
    modes = supported.get(lang, [])
    if latin_path is not None and "latin" not in modes:
        latin_langs = sorted(
            code for code, fmts in supported.items() if "latin" in fmts
        )
        raise click.UsageError(
            f"language '{lang}' has no Latin rules; "
            f"omit --out-latin or choose a language with Latin rules: "
            f"{', '.join(latin_langs)}"
        )
    if ipa_path is not None and "ipa" not in modes:
        ipa_langs = sorted(code for code, fmts in supported.items() if "ipa" in fmts)
        raise click.UsageError(
            f"language '{lang}' has no IPA rules; "
            f"omit --out-ipa or choose a language with IPA rules: "
            f"{', '.join(ipa_langs)}"
        )


def _supported_lang_choices() -> list[str]:
    """Return the sorted list of language codes exposed by the rules directory.

    Returns:
        Every ISO 639-1 code the ``rules/`` directory advertises,
        sorted alphabetically. Both IPA-only and dual-mode languages
        are included; per-mode filtering is deferred to
        :func:`_validate_output_selection`.
    """
    return sorted(get_supported_languages().keys())


@click.command(name="translit")
@click.option(
    "--lang",
    required=True,
    type=click.Choice(_supported_lang_choices(), case_sensitive=False),
    help="ISO 639-1 language code (discovered from the rules directory).",
)
@click.option(
    "--in",
    "input_path",
    default="-",
    show_default=True,
    help="Input file path, or '-' for stdin.",
)
@click.option(
    "--out-latin",
    "latin_path",
    default=None,
    help="Latin output path, or '-' for stdout. Omit to skip Latin output.",
)
@click.option(
    "--out-ipa",
    "ipa_path",
    default=None,
    help="IPA output path, or '-' for stdout. Omit to skip IPA output.",
)
@click.option(
    "--arabic",
    is_flag=True,
    help="Pre-pass Arabic script through ar_lat.rules before the target rules.",
)
@click.option(
    "--benchmark",
    is_flag=True,
    help="Log throughput statistics at the end of the run.",
)
def translit(
    lang: str,
    input_path: str,
    latin_path: str | None,
    ipa_path: str | None,
    arabic: bool,
    benchmark: bool,
) -> None:
    """Transliterate ORTHOGRAPHY -> IPA and/or Latin using per-language rules.

    Uses UTF-8 with a BOM on Windows so downstream editors correctly
    detect the encoding; UTF-8 without BOM on other platforms.

    Args:
        lang: ISO 639-1 language code selected by ``--lang``.
        input_path: Value of ``--in``.
        latin_path: Value of ``--out-latin`` or ``None`` when omitted.
        ipa_path: Value of ``--out-ipa`` or ``None`` when omitted.
        arabic: Whether the ``--arabic`` flag was set.
        benchmark: Whether the ``--benchmark`` flag was set.

    Raises:
        click.UsageError: When the requested output selection cannot
            be satisfied by the language's rules.
        UnicodeDecodeError: Propagated from the input stream when it
            contains bytes that are not valid in the chosen encoding.
    """
    _validate_output_selection(lang, latin_path, ipa_path)
    encoding = "utf-8-sig" if sys.platform == "win32" else "utf-8"

    _logger.info(
        "Starting transliteration: lang=%s, input=%s, "
        "out_latin=%s, out_ipa=%s, arabic=%s",
        lang,
        input_path,
        latin_path,
        ipa_path,
        arabic,
    )

    with ExitStack() as stack:
        input_stream = _open_input(stack, input_path, encoding)
        latin_stream = _open_output(stack, latin_path, encoding)
        ipa_stream = _open_output(stack, ipa_path, encoding)
        start = time.time()
        n = _stream_transliteration(
            input_stream, latin_stream, ipa_stream, lang, arabic
        )
        elapsed = time.time() - start

    if benchmark:
        rate = n / elapsed if elapsed > 0 else 0.0
        _logger.info("Benchmark: %d lines in %.2fs (%.0f lines/s)", n, elapsed, rate)
    _logger.info("Transliteration complete.")
