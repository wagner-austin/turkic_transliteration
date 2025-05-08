#!/usr/bin/env python3
import argparse, sys, fasttext

lid = fasttext.load_model("lid.176.ftz")
UZ_CORE = set()  # optionally load core-vocab file

def is_ru(tok, thr):
    lbl, conf = lid.predict(tok.lower(), k=1)
    return lbl[0] == "__label__ru" and conf[0] >= thr and tok.lower() not in UZ_CORE

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["drop","mask"], default="drop")
ap.add_argument("--thr", type=float, default=0.8)
ap.add_argument("--min_len", type=int, default=3)
args = ap.parse_args()

for line in sys.stdin:
    out=[]
    for tok in line.strip().split():
        if len(tok) < args.min_len or not is_ru(tok, args.thr):
            out.append(tok)
        elif args.mode == "mask":
            out.append("<RU>")
    print(" ".join(out))
