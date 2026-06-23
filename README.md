---
title: Turkic Transliteration Demo
emoji: 🌖
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Transliteration of Turkic languages into Latin and IPA
---

# turkic_transliterate

Deterministic Latin and IPA transliteration for Turkic languages, plus helper
utilities for corpus building, tokenizer/LM training, and Russian-token
filtering. Languages are discovered dynamically from rule files in
`src/turkic_translit/rules/`, so support grows by adding a `.rules` file.

**Supported languages** (verified by the test suite): Azerbaijani, Finnish,
Kazakh, Kyrgyz, Turkish, Uyghur, and Uzbek (both Cyrillic and Latin input).
IPA output is available for all of them; Latin output covers Kazakh and Kyrgyz
(plus embedded Arabic-script handling). The batch `turkic-translit` CLI is
currently limited to `kk`/`ky`; the library API (`to_ipa`/`to_latin`) and the
web demo cover the full set.

## Install

This project uses **Poetry**. Python **3.9+**.

```bash
# Simplest: the dev setup script (also installs the Windows PyICU wheel)
python scripts/setup_dev.py

# Or with Poetry directly
poetry install

# Or with pip (editable, with dev + web-UI extras)
pip install -e .[dev,ui]        # add ,winlid on Windows for the fasttext wheel
```

Optional extras: `dev` (ruff, mypy, pytest) · `ui` (Gradio demo) ·
`winlid` (Windows fasttext wheel for language ID) · `sentry` (error reporting).

### Windows & PyICU

PyPI rules prevent the correct Windows PyICU wheel from installing automatically
during `pip install`. After installing, Windows users run:

```bash
turkic-pyicu-install
```

which fetches the right PyICU wheel for your Python version. `scripts/setup_dev.py`
does this for you.

### Package names

- Import path: `turkic_translit`
- PyPI distributable: `turkic_transliterate`
- Primary CLI entry point: `turkic-translit`

## Development

The Makefile wraps the common tasks (all via Poetry):

```bash
make check   # lint + test (the full gate)
make lint    # ruff check --fix, ruff format, mypy --strict
make test    # pytest
make web     # launch the Gradio web UI
make help    # list all targets
```

On Windows without GNU Make, use the PowerShell wrapper:

```powershell
./scripts/run.ps1 lint
./scripts/run.ps1 test
./scripts/run.ps1 web
./scripts/run.ps1 help
```

Type-checking is `mypy --strict` (config in `mypy.ini`, scoped to `src/`).
Formatting and linting are both handled by **ruff** (`ruff format` replaced
black).

## Command-line tools

Console entry points defined in `pyproject.toml`:

| Command | Purpose |
|---------|---------|
| `turkic-translit` | Transliterate text to Latin and/or IPA (`--lang kk\|ky`) |
| `turkic-filter-russian` | Drop or mask Russian tokens from a stream |
| `turkic-download-corpus` | Download/prepare OSCAR corpora |
| `turkic-build-spm` / `turkic-train-spm` | Train a SentencePiece tokenizer |
| `turkic-train-lm` / `turkic-eval-lm` | Train / evaluate a language model |
| `turkic-leven` | Levenshtein-based comparison utility |
| `turkic-web` | Launch the Gradio web demo |
| `turkic-pyicu-install` | Install the correct PyICU wheel (Windows) |

### `turkic-translit` usage

```bash
turkic-translit --lang kk --in text.txt --out_latin kk_lat.txt \
    --ipa --out_ipa kk_ipa.txt --arabic --log-level debug
```

- `--lang` — `kk` or `ky`
- `--ipa` — emit IPA alongside Latin
- `--arabic` — also transliterate embedded Arabic script
- `--benchmark` — print throughput statistics
- `--log-level` — debug | info | warning | error | critical (default: info)

### Tokenizer training

```bash
turkic-build-spm --input corpora/kk_lat.txt,corpora/ky_lat.txt \
    --model_prefix spm/turkic12k --vocab_size 12000
```

### Filtering Russian tokens

```bash
cat uz_raw.txt | turkic-filter-russian --mode drop > uz_clean.txt
```

## Demos via `turkic_tools.py`

```bash
python turkic_tools.py [command]
```

- `web` — launch the Gradio web interface
- `demo` — simple CLI demo
- `full-demo` — comprehensive multi-language demo
- `help` — list commands

## Project Organization

- `src/turkic_translit/` — core package (`core.py`, `transliterate.py`,
  `rules/`, `cli/`, `web/`, `lm/`, language-ID and filtering modules)
- `data/` — sample data and language resources
- `docs/` — documentation and setup/troubleshooting guides
- `scripts/` — dev + release utilities (`setup_dev.py`, `run.ps1`, `release/`)
- `vendor/` — pre-built PyICU wheels for Windows
- `tests/` — test suite (per-language IPA coverage for all supported languages)
- `cronjob/` — scheduled-task assets for the hosted demo
- `app.py` — Hugging Face Space entry point for the web UI

## FastText language-identification model

Russian-token filtering and language detection use FastText's `lid.176.bin`.
The model is **not** committed (too large); it is downloaded automatically from
the official Facebook AI link on first use and cached in the package directory —
no manual step needed for pip installs, Hugging Face Spaces, or CI. Manual
source if needed:
https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

## Logging & error reporting

Central logging supports structured JSON with correlation IDs. Control with
`TURKIC_LOG_LEVEL` (DEBUG…ERROR) and `TURKIC_LOG_FORMAT=json|rich` (default
json); libraries can call `turkic_translit.logging_config.setup()`. Optional
Sentry integration via `TURKIC_SENTRY_DSN` (+ `TURKIC_ENV`,
`TURKIC_SENTRY_TRACES`); install with `pip install turkic-translit[sentry]`.

All I/O is UTF-8; on Windows a BOM is written when piping to files to avoid
encoding issues.

## Relationship to the LSTM project

This repo is the upstream data pipeline for the Turkic mutual-intelligibility
LSTM experiments (github.com/wagner-austin/LSTM): `turkic-download-corpus` plus
the IPA transliteration rules produce the per-language IPA corpora that project
trains on.

## License

Apache-2.0
