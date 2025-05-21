# Vendor Directory

This directory contains third-party dependencies and resources that are distributed with the Turkic Transliteration package.

## Contents

### PyICU Wheels for Windows

The `pyicu/` directory contains pre-built PyICU wheels for Windows, which are used by the `turkic-pyicu-install` tool to provide a seamless installation experience for Windows users. These wheels are from Christoph Gohlke's unofficial Windows binaries collection.

Currently included wheels:
- Python 3.11: `pyicu-2.15-cp311-cp311-win_amd64.whl`
- Python 3.12: `pyicu-2.15-cp312-cp312-win_amd64.whl`
- Python 3.13: `pyicu-2.15-cp313-cp313-win_amd64.whl`

## Why Include These Files?

Windows users often face difficulties installing PyICU due to the lack of official wheels and the complex build process required. By including these pre-built wheels, we simplify the installation process for Windows users.

The `turkic-pyicu-install` command will automatically detect these local wheels before attempting to download from the internet.

## Usage

End users should not need to interact with these files directly. Instead, they should use the `turkic-pyicu-install` command provided by this package.
