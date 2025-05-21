# Windows PyICU Installation Guide

This document explains why PyICU installation fails on Windows and how to resolve it.

## Why PyICU Installation Fails on Windows

PyICU requires the International Components for Unicode (ICU) libraries to be present on your system. On Windows, these are not typically available by default, and building them from source is complex.

Key challenges:
1. PyICU depends on the C/C++ ICU libraries
2. Building these libraries on Windows requires a proper compiler setup
3. The build process can be error-prone due to compiler, path, and environment variables

## Solution: Pre-built Wheels

The simplest solution is to use pre-built wheels from a trusted source. This project includes wheels for Python 3.10-3.13 from Christoph Gohlke's unofficial Windows binaries collection.

### Manual Installation

1. Identify your Python version:
   ```bash
   python --version
   ```

2. Install the matching wheel:
   ```bash
   pip install ./pyicu-2.15-cpXXX-cpXXX-win_amd64.whl
   ```
   (Replace XXX with your Python version)

### Automated Installation

This package includes a helper script to automate the process:

```bash
turkic-pyicu-install
```

This script:
1. Detects your Python version
2. Locates the appropriate wheel
3. Installs it for you automatically

## Verification

To verify the installation worked correctly:

```python
import icu
print(icu.ICU_VERSION)
```

This should print the ICU version without any errors.

## Troubleshooting

If you encounter issues after installing the wheel:

1. Ensure you're using the correct wheel for your Python version
2. Try uninstalling and reinstalling any PyICU package
3. Check for conflicts with other ICU-related packages
4. Make sure your PATH environment variable doesn't have conflicting library paths
