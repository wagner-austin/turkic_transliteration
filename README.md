turkic\_transliterate
Deterministic Latin and IPA transliteration for Kazakh and Kyrgyz, plus helper utilities for tokenizer training and Russian-token filtering.

Quick install

1. Install Miniconda or Anaconda (recommended).
2. Clone the repo and create the environment:
   conda env create -f env.yml
3. Activate the environment:
   conda activate turkic
4. Run the verification tests:
   python -m pytest      (all tests should pass)

Python compatibility
• Works on CPython 3.10 and 3.11.
• CPython 3.12+ is supported everywhere except on Windows until official PyICU wheels are available; see “Windows & PyICU” below.

Package names
• Runtime import path:  turkic\_translit
• Distributable name on PyPI:  turkic\_transliterate
• Command-line entry point:  turkic-translit

## Developer Setup

For the simplest developer setup experience, run the setup script:

```bash
python scripts/setup_dev.py
```

This script will:
1. Install the package with all development dependencies
2. Set up PyICU on Windows automatically
3. Verify that development tools are working properly

### Manual Installation

Alternatively, install with pip:

```bash
pip install -e .[dev,ui]        # add ,winlid on Windows if you need fasttext-wheel
```

### Development Tools

#### Linux/macOS/Windows with GNU Make

If you have GNU Make installed, you can use the Makefile for common tasks:

```bash
make lint       # Run linting (ruff, black, mypy)
make format     # Auto-format code
make test       # Run tests
make web        # Launch the web UI
make help       # Show all available commands
```

#### Windows

**Option 1: Install GNU Make using Chocolatey (Recommended)**

Install GNU Make using Chocolatey (requires admin privileges):

```powershell
# In an Admin PowerShell window
choco install make
```

After installation, you can use the same `make` commands as on Linux/macOS.

**Option 2: Use the PowerShell Script Alternative**

If you prefer not to install Chocolatey or GNU Make, use the PowerShell script:

```powershell
./scripts/run.ps1 lint       # Run linting
./scripts/run.ps1 format     # Auto-format code
./scripts/run.ps1 test       # Run tests
./scripts/run.ps1 web        # Launch the web UI
./scripts/run.ps1 help       # Show all available commands
```

Optional extras
dev   → black, ruff, pytest
ui    → gradio web demo
winlid (Windows only) → fasttext-wheel for language ID

Windows & PyICU

**Important:** Due to PyPI rules, the correct PyICU wheel for Windows cannot be installed automatically during pip install. After installing this package with pip, Windows users must run the helper script to install the appropriate PyICU wheel:

    turkic-pyicu-install

This script will download and install the correct PyICU wheel from Christoph Gohlke’s repository based on your Python version. See the script for details.

Command-line usage
turkic-translit --lang kk --in text.txt --out\_latin kk\_lat.txt --ipa --out\_ipa kk\_ipa.txt --arabic --log-level debug
• --lang            kk or ky
• --ipa             emit IPA alongside Latin
• --arabic          also transliterate embedded Arabic script
• --benchmark       print throughput statistics
• --log-level       debug | info | warning | error | critical (default: info)

Logging
The central logging setup uses Rich for colour when available.
Set TURKIC\_LOG\_LEVEL or pass --log-level to the CLI.
Fallback to standard logging when Rich is absent.

# Project Organization

The project is organized into the following directories:

- `src/turkic_translit/` - Core source code for the package
- `examples/` - Example scripts showing how to use the package
  - `examples/web/` - Web interface for demonstrating transliteration features
- `data/` - Sample data files and language resources
- `docs/` - Documentation and reference materials
- `scripts/` - Utility scripts for development and release
  - `scripts/release/` - Scripts for building and publishing packages
- `vendor/pyicu/` - Pre-built PyICU wheels for Windows
- `tests/` - Test suite for the package

## Using the Examples

Use the main entry point script to run examples:

```bash
python turkic_tools.py [command]
```

Available commands:
- `web` - Launch the Gradio web interface for real-time transliteration
- `demo` - Run the simple CLI demo
- `full-demo` - Run the comprehensive demo with multiple languages
- `help` - Display available commands

Tokenizer training example
turkic-build-spm --input corpora/kk\_lat.txt,corpora/ky\_lat.txt --model\_prefix spm/turkic12k --vocab\_size 12000

Filtering Russian tokens from Uzbek
cat uz\_raw\.txt | turkic-filter-russian --mode drop > uz\_clean.txt

Developer checklist
black .
ruff check .
pytest -q

All code is UTF-8-only; on Windows a BOM is written when piping to files to avoid encoding issues.

License
Apache-2.0

### Type-checking

```bash
pip install mypy
mypy --strict .
```

The included mypy.ini restricts analysis to the src/ tree and skips
build/, dist/, virtual-env and egg directories so duplicate-module
errors do not occur even if you build wheels locally.
