# turkic_transliterate

Deterministic Latin + IPA transliteration for Kazakh and Kyrgyz, with ready-made
SentencePiece training scripts and Russian-token filtering.

## Quick install
```bash
conda env create -f env.yml
conda activate turkic
python verification.py     # all green ✔
```

### Pip installation
You can also install using pip with optional extras:

```bash
pip install -e .[dev,ui,winlid]
```

- `dev`: Development tools (pytest, black, ruff)
- `ui`: Web interface (gradio)
- `winlid`: Windows-specific language identification (fasttext-wheel)

## Windows & PyICU
On Windows with Python 3.12+, you'll need to manually install the PyICU wheel. We've created a helper script to make this easy:

```bash
python scripts/get_pyicu_wheel.py
```

This will automatically download and install the correct PyICU wheel for your Python version.

## CLI
turkic-translit --lang kk --in text.txt --out_latin kk_lat.txt --ipa --out_ipa kk_ipa.txt

## Build tokenizer
python scripts/build_spm.py --input corpora/kk_lat.txt,corpora/ky_lat.txt --model_prefix spm/turkic12k

## Filter Russian from Uzbek
cat uz_raw.txt | python scripts/filter_russian.py --mode drop > uz_clean.txt

