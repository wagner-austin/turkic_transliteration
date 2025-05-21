# Scripts Directory

This directory contains utility scripts for project maintenance.

## Development Setup

- `setup_dev.py` - Sets up a development environment with all required dependencies

### Usage

```bash
python scripts/setup_dev.py
```

This script will install the package with all development dependencies, set up PyICU on Windows if needed, and verify that all development tools are working properly.

## Release Scripts

The `release/` directory contains scripts for building and publishing the package to PyPI:

- `release.ps1` - PowerShell script for Windows users
- `release.sh` - Bash script for Unix/Linux/Mac users

### Usage

#### Windows:

```powershell
cd scripts/release
powershell -ExecutionPolicy Bypass -File release.ps1
```

#### Unix/Linux/MacOS:

```bash
cd scripts/release
bash release.sh
```

Both scripts will:
1. Clean up previous build artifacts
2. Run tests to verify the package
3. Build the package
4. Upload it to PyPI (requires PyPI credentials)
