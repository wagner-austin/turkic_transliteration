# Scripts Directory

This directory contains utility scripts for project maintenance.

## Development Setup

- `setup_dev.py` - Sets up a development environment with all required dependencies

### Usage

```bash
python scripts/setup_dev.py
```

This script installs the package in editable mode with all development dependencies and reports which development tools are on PATH.

To see structured logs during development:

```bash
export TURKIC_LOG_LEVEL=DEBUG
export TURKIC_LOG_FORMAT=json   # or 'rich'
```

## Releasing

There is no release script. A push to `main` runs
`.github/workflows/release.yml`, which publishes the version
`pyproject.toml` declares if PyPI does not already have it, through PyPI's
trusted publishing, and tags the commit `v<version>`. To release, raise the
version in `pyproject.toml`, add its section to `CHANGELOG.md`, and push.
