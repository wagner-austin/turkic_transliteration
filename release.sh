#!/usr/bin/env bash
# release.sh - Automate PyPI release for turkic-transliterate
# Usage: bash release.sh
set -euo pipefail

# Remove previous build artifacts
echo "Removing previous build artifacts..."
rm -rf dist build
find . -type d -name '*.egg-info' -exec rm -rf {} +

echo "Checking for PyPI credentials..."
if [[ -z "${TWINE_USERNAME:-}" || -z "${TWINE_PASSWORD:-}" ]]; then
  if [[ -f "$HOME/.pypirc" ]]; then
    echo "Loading PyPI credentials from $HOME/.pypirc..."
    export TWINE_USERNAME=$(awk '/\[pypi\]/{flag=1;next}/\[.*\]/{flag=0}flag && /username/{print $3}' $HOME/.pypirc | head -n1)
    export TWINE_PASSWORD=$(awk '/\[pypi\]/{flag=1;next}/\[.*\]/{flag=0}flag && /password/{print $3}' $HOME/.pypirc | head -n1)
  fi
fi

if [[ -z "${TWINE_USERNAME:-}" ]]; then
  echo "Error: TWINE_USERNAME is not set. Please set it or configure your .pypirc file." >&2
  exit 1
fi
if [[ -z "${TWINE_PASSWORD:-}" ]]; then
  echo "Error: TWINE_PASSWORD is not set. Please set it or configure your .pypirc file." >&2
  exit 1
fi

echo "Running tests..."
pytest || { echo "Tests failed. Aborting release."; exit 1; }

echo "Building package..."
python -m build

echo "Uploading package to PyPI..."
twine upload dist/*

echo "Release process completed successfully."
