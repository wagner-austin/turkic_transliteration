import sys, argparse, pathlib, time, os, logging
from .core import to_latin, to_ipa
from .logging_config import setup as _log_setup

# Initialize logger
log = logging.getLogger(__name__)

def main() -> None:
    ap = argparse.ArgumentParser(description="Turkic transliteration")
    ap.add_argument("--lang", required=True, choices=["kk", "ky"])
    ap.add_argument("--ipa", action="store_true", help="produce IPA")
    ap.add_argument("--arabic", action="store_true", help="also transliterate Arabic script")
    ap.add_argument("--in", dest="inp", default="-")
    ap.add_argument("--out_latin", default="-")
    ap.add_argument("--out_ipa")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--log-level", choices=["debug", "info", "warning", "error", "critical"],
                    default="info",
                    help="Set logging level (default: info)")
    args = ap.parse_args()

    # Always set log level from args at the start (first runtime line)
    os.environ["TURKIC_LOG_LEVEL"] = args.log_level.upper()
    _log_setup()
    
    outputs = ["latin"]
    if args.ipa:
        outputs.append("ipa")
    # Use Rich markup for output modes (magenta)
    outputs_markup = ", ".join(f"[magenta]{o}[/]" for o in outputs)
    log.info(
        f"Starting transliteration: lang={args.lang}, input={args.inp}, outputs={outputs_markup}, "
        f"out_latin={args.out_latin}, out_ipa={args.out_ipa}, arabic={args.arabic}, benchmark={args.benchmark}"
    )
    
    # Use UTF-8-sig for Windows to include BOM for proper encoding support
    encoding = "utf-8-sig" if sys.platform == "win32" else "utf-8"
    
    try:
        fin  = sys.stdin  if args.inp  == "-" else open(args.inp, encoding=encoding)
        fo_l = sys.stdout if args.out_latin == "-" else open(args.out_latin, "w", encoding=encoding)
        fo_i = None
        if args.ipa:
            if not args.out_ipa:
                ap.error("--ipa requires --out_ipa")
            fo_i = open(args.out_ipa, "w", encoding=encoding)
    except UnicodeDecodeError as e:
        sys.stderr.write(f"Encoding error: {e}\n")
        sys.stderr.write("If you're on Windows, make sure your input file is properly encoded in UTF-8.\n")
        sys.exit(1)

    start = time.time()
    n = 0
    
    # Try to use tqdm for a progress bar if available and if we're in a TTY
    use_progress_bar = False
    pbar = None
    
    # Check if we should use a progress bar (stderr is a TTY and input is not stdin)
    is_tty_output = sys.stderr.isatty()
    is_file_input = args.inp != "-"
    
    if is_tty_output and is_file_input:
        try:
            from tqdm import tqdm
            # Count the number of lines in the input file for the progress bar
            total_lines = sum(1 for _ in fin)
            fin.seek(0)  # Reset file pointer
            pbar = tqdm(total=total_lines, unit="lines")
            use_progress_bar = True
            log.debug("Using tqdm progress bar for %d lines", total_lines)
        except ImportError:
            log.debug("tqdm not available, falling back to basic processing")
    
    # Process lines
    for line in fin:
        lat = to_latin(line.rstrip("\n"), args.lang, args.arabic)
        fo_l.write(lat + "\n")
        if fo_i:
            fo_i.write(to_ipa(line.rstrip("\n"), args.lang) + "\n")
        n += 1
        if use_progress_bar and pbar:
            pbar.update(1)
    
    log.info(f"Finished writing {n} lines to {args.out_latin if args.out_latin != '-' else 'stdout'}" + (f" and {args.out_ipa}" if args.ipa else ""))
    
    # Close progress bar if used
    if use_progress_bar and pbar:
        pbar.close()
    
    elapsed = time.time() - start
    # Always log processing statistics, but at different levels based on benchmark flag
    if args.benchmark:
        log.info("Processed %d lines in %.2fs (%.0f lines/s)", n, elapsed, n/elapsed if elapsed > 0 else 0)
    else:
        log.debug("Processed %d lines in %.2fs (%.0f lines/s)", n, elapsed, n/elapsed if elapsed > 0 else 0)
    log.info("Transliteration complete.")
    
    # Clean up file handles
    if fin is not sys.stdin:
        fin.close()
    if fo_l is not sys.stdout:
        fo_l.close()
    if fo_i:
        fo_i.close()

# This is the entry point when the module is run directly
if __name__ == "__main__":
    main()
