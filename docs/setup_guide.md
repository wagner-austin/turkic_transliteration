# Setup Guide for Turkic Transliteration

This document provides detailed setup instructions for the Turkic Transliteration package.

## Environment Setup

This project uses **Poetry** and supports **Python 3.9+**. The quickest path is
the dev setup script, which also installs the Windows PyICU wheel:

```bash
python scripts/setup_dev.py
```

Or set up with Poetry directly:

```bash
poetry install          # creates the virtualenv and installs everything
poetry run pytest       # verify (all tests should pass)
```

## PyICU Installation on Windows

Windows users need to manually install the correct PyICU wheel:

1. Download the appropriate wheel for your Python version:
   - For Python 3.10: [pyicu-2.15-cp310-cp310-win_amd64.whl](https://github.com/cgohlke/pyicu-build/releases/download/v2.15/pyicu-2.15-cp310-cp310-win_amd64.whl)
   - For Python 3.11: [pyicu-2.15-cp311-cp311-win_amd64.whl](https://github.com/cgohlke/pyicu-build/releases/download/v2.15/pyicu-2.15-cp311-cp311-win_amd64.whl)
   - For Python 3.12: [pyicu-2.15-cp312-cp312-win_amd64.whl](https://github.com/cgohlke/pyicu-build/releases/download/v2.15/pyicu-2.15-cp312-cp312-win_amd64.whl)
   - For Python 3.13: [pyicu-2.15-cp313-cp313-win_amd64.whl](https://github.com/cgohlke/pyicu-build/releases/download/v2.15/pyicu-2.15-cp313-cp313-win_amd64.whl)

2. Install the wheel:
   ```bash
   pip install ./pyicu-2.15-cpXXX-cpXXX-win_amd64.whl
   ```
   (Replace XXX with your Python version)

3. Alternatively, use the helper script provided with the package:
   ```bash
   turkic-pyicu-install
   ```

## Package Installation

```bash
# Install the package with development and UI tools
pip install -e .[dev,ui]

# On Windows, if you need fasttext-wheel for language ID:
pip install -e .[dev,ui,winlid]

# Optional: error service (Sentry)
pip install -e .[sentry]
```

## Required Dependencies

The core dependencies will be installed automatically with the package, but you can install them manually if needed:

```bash
pip install sentencepiece rapidfuzz
pip install numpy pybind11 wheel
pip install fasttext-wheel
pip install python-json-logger
```

`epitran` and `typing-extensions` were listed here but are not imported by this
project and have been removed from its dependencies. IPA is produced by the ICU
rule files in `rules/`, through PyICU.

## Cleaning Up

If you need to remove the Poetry virtualenv:

```bash
poetry env remove --all
```
