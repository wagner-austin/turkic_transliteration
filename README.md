# Turkic Transliteration

**Try it:** <https://huggingface.co/spaces/AustinWagner/turkic-transliteration-demo>

*(This repo is the deployment source for that demo. The Space's card lives at
`.github/hf-space/README.md`; a push to `main` writes it, `app.py` and the
pinned requirement onto the Space. This README used to carry a second copy of
that card, which is how the Space came to install a Gradio the package no
longer accepted.)*

Deterministic Latin and IPA transliteration for Turkic languages, plus helper
utilities for corpus building, tokenizer/LM training, and Russian-token
filtering.

The design decision worth noting: **languages are discovered dynamically from
rule files** in `src/turkic_translit/rules/`, so adding a language means adding a
`.rules` file — no code change, and the CLI, the library API (`to_ipa`/`to_latin`)
and the web demo all pick it up at once.

**Supported languages** (verified by the test suite): Azerbaijani, Finnish,
Kazakh, Kyrgyz, Turkish, Uyghur, and Uzbek (both Cyrillic and Latin input).
IPA output is available for all of them; Latin output covers Kazakh and
Kyrgyz (Cyrillic → Latin, with optional embedded Arabic-script handling)
and Turkish (diacritic folding to ASCII).

## Install

Python **3.10–3.14**, whichever platform: `pyicu-wheels` publishes the ICU
extension for all of them.

```bash
pip install turkic-translit
turkic-translit web          # or any of the commands below
```

Nothing else: ICU arrives with it. That was not true before 0.5.6 — PyICU
publishes no wheels on any platform, so an install had to compile against ICU
headers, and this project shipped a Windows wheel fetcher to work around it.
The dependency is `pyicu-wheels`, which publishes the same extension prebuilt
for Linux, macOS and Windows.

To work on the project rather than use it:

```bash
git clone https://github.com/wagner-austin/turkic_transliteration.git
cd turkic_transliteration
pip install -e .[dev]        # or: python scripts/setup_dev.py
turkic-translit web
```

Optional extras: `dev` (ruff, mypy, pytest) · `examples` (Flask, Streamlit,
JupyterLab) · `sentry` (error reporting).

### Package names

- Import path: `turkic_translit`
- PyPI distributable: `turkic-translit`
- Primary CLI entry point: `turkic-translit`

## Development

The Makefile wraps the common tasks (all via Poetry):

```bash
make check   # lint + test (the full gate)
make lint    # guards, then ruff check --fix + ruff format, then strict mypy
make guard   # just the guard rules (scripts/guards/)
make test    # pytest with 100% statement+branch coverage enforced
make web     # launch the Gradio web UI
```

On Windows without GNU Make: `./scripts/run.ps1 <target>`.

Linting, type-checking (strict mypy) and the guard rules all cover `src`,
`tests` and `scripts`. The guards ban weak typing (`Any`, casts, ignores),
silent exception handling, and weak or fake tests — including
transliteration tests that never compare output against an expected value.

## Validation

- Every IPA rule file declares its source in machine-readable `# Source-*`
  header fields, and a test module that pins expected output must name the
  source it inherits (`INHERITS_SOURCE`); both are guard-enforced, as is the
  ban on expectations computed by the code under test.
- Expected values are transcriptions printed in the cited descriptions
  (journal Illustrations, reference-grammar chapters). Each language is
  covered letter-exhaustively plus word sets from a second, independent
  description; Uyghur is additionally checked against the MFA dictionary's
  22,630 pronunciations.
- Departures from a source's printed notation are declared in two tables —
  `NOTATION` (different glyph, same phoneme) and `DECLARED_SIMPLIFICATIONS`
  (a real distinction the rules drop) — each entry carrying a reason. A
  central test fails any entry no printed datum exercises, and a
  cross-language test requires shared letters to agree across languages
  after corpus harmonization unless a contrast is declared with a citation.

## Command-line tools

Console entry points defined in `pyproject.toml`:

| Command | Purpose |
|---------|---------|
| `turkic-translit` | Transliterate text to Latin and/or IPA |
| `turkic-translit web` | Launch the Gradio web demo |
| `turkic-filter-russian` | Drop or mask Russian tokens from a stream |
| `turkic-download-corpus` | Download/prepare OSCAR corpora |
| `turkic-clean-corpus` | Clean and harmonize downloaded corpora |
| `turkic-build-spm` / `turkic-train-spm` | Train a SentencePiece tokenizer |
| `turkic-train-lm` / `turkic-eval-lm` | Train / evaluate a language model |
| `turkic-leven` | Levenshtein-based comparison utility |

### `turkic-translit translit` usage

```bash
turkic-translit translit --lang kk --in text.txt \
    --out-latin kk_lat.txt --out-ipa kk_ipa.txt --arabic
```

- `--lang` — any code advertised by the rules directory; IPA is available
  for `az`, `fi`, `kk`, `ky`, `tr`, `ug`, `uz` and `uzc` (Uzbek Cyrillic)
- `--in` — input file, or `-` for stdin (default: `-`)
- `--out-latin` — Latin output path, or `-` for stdout. Omit to skip.
  Only meaningful for languages with `<lang>_lat.rules` (`kk`, `ky`, `tr`).
- `--out-ipa` — IPA output path, or `-` for stdout. Omit to skip.
- `--arabic` — also transliterate embedded Arabic script (Latin mode only)
- `--benchmark` — log throughput statistics on completion

At least one of `--out-latin` / `--out-ipa` must be specified.
`--log-level` is set on the parent group: `turkic-translit --log-level debug translit ...`.

### Tokenizer training

```bash
turkic-build-spm --input corpora/kk_lat.txt,corpora/ky_lat.txt \
    --model_prefix spm/turkic12k --vocab_size 12000
```

### Filtering Russian tokens

```bash
cat uz_raw.txt | turkic-filter-russian --mode drop > uz_clean.txt
```


## Project Organization

- `src/turkic_translit/` — core package (`core.py`, `transliterate.py`,
  `rules/`, `cli/`, `web/`, `lm/`, language-ID and filtering modules)
- `data/` — sample data and language resources
- `docs/` — documentation and guides
- `scripts/` — dev + release utilities (`setup_dev.py`, `run.ps1`, `release/`)
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
