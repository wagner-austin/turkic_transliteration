import sys, argparse, pathlib, time
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

    fin  = sys.stdin  if args.inp  == "-" else open(args.inp,  encoding="utf8")
    fo_l = sys.stdout if args.out_latin == "-" else open(args.out_latin, "w", encoding="utf8")
    fo_i = None
    if args.ipa:
        if not args.out_ipa:
            ap.error("--ipa requires --out_ipa")
        fo_i = open(args.out_ipa, "w", encoding="utf8")

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
        sys.stderr.write(f"{n} lines  {elapsed:.2f}s ({n/elapsed:.0f} lines/s)\n")
