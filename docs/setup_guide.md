# Setup Guide for Turkic Transliteration

This document provides detailed setup instructions for the Turkic Transliteration package.

## Environment Setup

Python **3.10–3.14**, any platform. To use the package:

```bash
pip install turkic-translit
turkic-translit web
```

To work on it:

```bash
git clone https://github.com/wagner-austin/turkic_transliteration.git
cd turkic_transliteration
pip install -e .[dev]     # or: python scripts/setup_dev.py
make check                # the full gate: guards, ruff, mypy, tests
```

Optional extras: `dev` (ruff, mypy, pytest) · `examples` (Flask, Streamlit,
JupyterLab) · `sentry` (error reporting).

## ICU

ICU is a dependency and installs with the package. It is taken as
`pyicu-wheels`, which publishes the extension prebuilt for Linux, macOS and
Windows.

This used to be the hardest part of installing the project. PyICU itself
publishes no wheels — sdist only, every platform — so an install compiled
against ICU headers the machine usually lacked, and Windows had no workable
path at all. That is why the project once shipped a wheel fetcher, a vendored
wheel directory, and two troubleshooting guides. All of it is gone as of 0.5.6.

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
