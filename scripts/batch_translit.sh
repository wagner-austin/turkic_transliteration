#!/usr/bin/env bash
set -euo pipefail
for lang in kk ky; do
  python -m turkic_translit.cli \
    --lang "$lang" --arabic \
    --in "corpora/${lang}_raw.txt" \
    --out_latin "corpora/${lang}_lat.txt"
done
