# turkic_transliterate

Deterministic Latin + IPA transliteration for Kazakh and Kyrgyz, with ready-made
SentencePiece training scripts and Russian-token filtering.

## Quick install
pip install turkic_transliterate

## CLI
turkic-translit --lang kk --in text.txt --out_latin kk_lat.txt --ipa --out_ipa kk_ipa.txt

## Build tokenizer
python scripts/build_spm.py --input corpora/kk_lat.txt,corpora/ky_lat.txt --model_prefix spm/turkic12k

## Filter Russian from Uzbek
cat uz_raw.txt | python scripts/filter_russian.py --mode drop > uz_clean.txt
