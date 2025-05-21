import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext
from typing import Optional, TextIO

from .core import to_ipa, to_latin
from .logging_config import setup as _log_setup

# Initialize logger
log = logging.getLogger(__name__)


def process_lines(
    input_stream: TextIO,
    latin_output: TextIO,
    ipa_output: Optional[TextIO],
    lang: str,
    arabic: bool,
    progress_bar=None,
) -> int:
    """Process each line from the input and write to outputs.

    Args:
        input_stream: Input file or stdin object
        latin_output: File object for Latin output
        ipa_output: File object for IPA output (or None if not needed)
        lang: Language code ('kk' or 'ky')
        arabic: Whether to transliterate Arabic script too
        progress_bar: Optional tqdm progress bar

    Returns:
        Number of lines processed
    """
    n = 0
    for line in input_stream:
        # Transliterate to Latin
        lat = to_latin(line.rstrip("\n"), lang, arabic)
        latin_output.write(lat + "\n")

        # Transliterate to IPA if requested
        if ipa_output:
            ipa = to_ipa(line.rstrip("\n"), lang)
            ipa_output.write(ipa + "\n")

        n += 1
        if progress_bar:
            progress_bar.update(1)

    return n


def transliterate_file(
    input_path: str,
    latin_output_path: str,
    ipa_output_path: Optional[str],
    lang: str,
    arabic: bool,
    encoding: str,
) -> int:
    """Transliterate a file with proper context management for all files.

    Args:
        input_path: Path to input file or "-" for stdin
        latin_output_path: Path to Latin output file or "-" for stdout
        ipa_output_path: Path to IPA output file or None if not needed
        lang: Language code ('kk' or 'ky')
        arabic: Whether to transliterate Arabic script too
        encoding: Encoding to use for file operations

    Returns:
        Number of lines processed
    """
    # Timer for benchmarking
    start = time.time()

    # Progress bar setup - only used for file inputs on TTY
    progress_bar = None
    is_tty_output = sys.stderr.isatty()
    is_file_input = input_path != "-"

    if is_tty_output and is_file_input:
        try:
            from tqdm import tqdm

            # Count lines for progress bar (only for file input)
            with open(input_path, encoding=encoding) as f:
                total_lines = sum(1 for _ in f)

            progress_bar = tqdm(total=total_lines, unit="lines")
            log.debug("Using tqdm progress bar for %d lines", total_lines)
        except ImportError:
            log.debug("tqdm not available, falling back to basic processing")

    # Set up file handling with context managers
    n = 0
    try:
        # Use context managers properly to satisfy linter
        # For input file
        if input_path == "-":
            # Use stdin directly
            with nullcontext(sys.stdin) as input_file:
                # For Latin output
                if latin_output_path == "-":
                    # Use stdout directly
                    with nullcontext(sys.stdout) as latin_output:
                        # For IPA output
                        if ipa_output_path:
                            with open(
                                ipa_output_path, "w", encoding=encoding
                            ) as ipa_output:
                                return process_lines(
                                    input_file,
                                    latin_output,
                                    ipa_output,
                                    lang,
                                    arabic,
                                    progress_bar,
                                )
                        else:
                            return process_lines(
                                input_file,
                                latin_output,
                                None,
                                lang,
                                arabic,
                                progress_bar,
                            )
                else:
                    # Latin output to file
                    with open(
                        latin_output_path, "w", encoding=encoding
                    ) as latin_output:
                        # For IPA output
                        if ipa_output_path:
                            with open(
                                ipa_output_path, "w", encoding=encoding
                            ) as ipa_output:
                                return process_lines(
                                    input_file,
                                    latin_output,
                                    ipa_output,
                                    lang,
                                    arabic,
                                    progress_bar,
                                )
                        else:
                            return process_lines(
                                input_file,
                                latin_output,
                                None,
                                lang,
                                arabic,
                                progress_bar,
                            )
        else:
            # Input from file
            with open(input_path, encoding=encoding) as input_file:
                # For Latin output
                if latin_output_path == "-":
                    # Use stdout directly
                    with nullcontext(sys.stdout) as latin_output:
                        # For IPA output
                        if ipa_output_path:
                            with open(
                                ipa_output_path, "w", encoding=encoding
                            ) as ipa_output:
                                return process_lines(
                                    input_file,
                                    latin_output,
                                    ipa_output,
                                    lang,
                                    arabic,
                                    progress_bar,
                                )
                        else:
                            return process_lines(
                                input_file,
                                latin_output,
                                None,
                                lang,
                                arabic,
                                progress_bar,
                            )
                else:
                    # Latin output to file
                    with open(
                        latin_output_path, "w", encoding=encoding
                    ) as latin_output:
                        # For IPA output
                        if ipa_output_path:
                            with open(
                                ipa_output_path, "w", encoding=encoding
                            ) as ipa_output:
                                return process_lines(
                                    input_file,
                                    latin_output,
                                    ipa_output,
                                    lang,
                                    arabic,
                                    progress_bar,
                                )
                        else:
                            return process_lines(
                                input_file,
                                latin_output,
                                None,
                                lang,
                                arabic,
                                progress_bar,
                            )

    finally:
        # Always close the progress bar if it exists
        if progress_bar:
            progress_bar.close()

    # Log completion
    log.info(
        f"Finished writing {n} lines to {latin_output_path if latin_output_path != '-' else 'stdout'}"
        + (f" and {ipa_output_path}" if ipa_output_path else "")
    )

    # Log performance statistics
    elapsed = time.time() - start
    log.debug(
        "Processed %d lines in %.2fs (%.0f lines/s)",
        n,
        elapsed,
        n / elapsed if elapsed > 0 else 0,
    )

    return n


def main() -> None:
    """Main CLI entry point for Turkic transliteration."""
    import platform

    # Check for Windows Python 3.12+ which has issues with PyICU
    if platform.system() == "Windows" and sys.version_info >= (3, 12):
        sys.stderr.write(
            "[turkic-transliterate] ERROR: PyICU might have issues on Windows for Python 3.12+!\n"
            "If you encounter problems, please create a virtual environment with Python 3.11:\n\n"
            "    py -3.11 -m venv turkic311\n"
            "    turkic311\\Scripts\\activate\n"
            "    pip install turkic-transliterate\n"
            "    turkic-pyicu-install\n\n"
            "See the README for more details.\n"
        )

    # Parse arguments
    ap = argparse.ArgumentParser(description="Turkic transliteration")
    ap.add_argument("--lang", required=True, choices=["kk", "ky"])
    ap.add_argument("--ipa", action="store_true", help="produce IPA")
    ap.add_argument(
        "--arabic", action="store_true", help="also transliterate Arabic script"
    )
    ap.add_argument("--in", dest="inp", default="-", help="Input file (default: stdin)")
    ap.add_argument(
        "--out_latin", default="-", help="Latin output file (default: stdout)"
    )
    ap.add_argument("--out_ipa", help="IPA output file (required if --ipa is used)")
    ap.add_argument(
        "--benchmark", action="store_true", help="Display performance metrics"
    )
    ap.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set logging level (default: info)",
    )
    args = ap.parse_args()

    # Configure logging
    os.environ["TURKIC_LOG_LEVEL"] = args.log_level.upper()
    _log_setup()

    # Report configuration
    outputs = ["latin"]
    if args.ipa:
        outputs.append("ipa")
    outputs_markup = ", ".join(f"[magenta]{o}[/]" for o in outputs)
    log.info(
        f"Starting transliteration: lang={args.lang}, input={args.inp}, "
        f"outputs={outputs_markup}, out_latin={args.out_latin}, "
        f"out_ipa={args.out_ipa}, arabic={args.arabic}"
    )

    # Validate arguments
    if args.ipa and not args.out_ipa:
        ap.error("--ipa requires --out_ipa")

    # Use UTF-8-sig for Windows to include BOM for proper encoding support
    encoding = "utf-8-sig" if sys.platform == "win32" else "utf-8"

    # Process input/output
    try:
        n = transliterate_file(
            args.inp,
            args.out_latin,
            args.out_ipa if args.ipa else None,
            args.lang,
            args.arabic,
            encoding,
        )
    except UnicodeDecodeError as e:
        sys.stderr.write(f"Encoding error: {e}\n")
        sys.stderr.write(
            "If you're on Windows, make sure your input file is properly encoded in UTF-8.\n"
        )
        sys.exit(1)

    # Performance benchmark reporting (higher visibility if --benchmark is used)
    if args.benchmark:
        log.info(
            "Benchmark: Processed %d lines with settings: lang=%s, ipa=%s, arabic=%s",
            n,
            args.lang,
            args.ipa,
            args.arabic,
        )

    log.info("Transliteration complete.")


# This is the entry point when the module is run directly
if __name__ == "__main__":
    main()
