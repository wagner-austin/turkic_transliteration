import sys, argparse, pathlib, time, os
from .core import to_latin, to_ipa

def main() -> None:
    ap = argparse.ArgumentParser(description="Turkic transliteration")
    ap.add_argument("--lang", required=True, choices=["kk", "ky"])
    ap.add_argument("--ipa", action="store_true", help="produce IPA")
    ap.add_argument("--arabic", action="store_true", help="also transliterate Arabic script")
    ap.add_argument("--in", dest="inp", default="-")
    ap.add_argument("--out_latin", default="-")
    ap.add_argument("--out_ipa")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

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
    for line in fin:
        lat = to_latin(line.rstrip("\n"), args.lang, args.arabic)
        fo_l.write(lat + "\n")
        if fo_i:
            fo_i.write(to_ipa(line.rstrip("\n"), args.lang) + "\n")
        n += 1
    elapsed = time.time() - start
    if args.benchmark:
        sys.stderr.write(f"{n} lines → {elapsed:.2f}s ({n/elapsed:.0f} lines/s)\n")
    
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
