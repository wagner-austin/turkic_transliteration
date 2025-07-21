# Windows Installation Troubleshooting Guide

This document describes the challenges faced when installing turkic_transliteration on Windows and the solutions implemented.

## Issues Encountered and Solutions

### 1. PyICU Installation
**Problem**: PyICU doesn't provide official Windows wheels on PyPI, causing installation failures.

**Root Cause**: The poetry.lock file initially referenced local wheel files that didn't exist:
```
vendor/pyicu/pyicu-2.15-cp311-cp311-win_amd64.whl
```

**Solution Implemented**: 
- Regenerated poetry.lock with `poetry lock --regenerate`
- Created `turkic-pyicu-install` command that downloads wheels from GitHub
- Updated Makefile to automatically run PyICU installer after Poetry install
- The installer now checks for existing PyICU installation and downloads appropriate wheels for Python 3.10-3.13

### 2. BeautifulSoup4 Long Path Error
**Problem**: BeautifulSoup4 installation fails with "FileNotFoundError" on Windows.

**Error**: 
```
FileNotFoundError: [Errno 2] No such file or directory: 
'C:\\Users\\...\\bs4\\tests\\fuzz\\clusterfuzz-testcase-minimized-bs4_fuzzer-4670634698080256.testcase'
```

**Root Cause**: Windows has a 260 character path limit. BeautifulSoup4 has test files with very long names that, combined with Poetry's deep virtualenv paths, exceed this limit.

**Dependencies pulling in BeautifulSoup4**:
- `wikipedia` (direct dependency)
- `nbconvert` (via Jupyter-related packages)

**Solution Implemented**:
- Created `poetry.toml` with `virtualenvs.in-project = true`
- This creates `.venv` in the project directory instead of deep AppData paths
- Reduces path length by ~100 characters, preventing the 260 character limit error
- The Makefile now automatically clears Poetry cache to avoid corrupted installations

### 3. NumPy Build Error (Python 3.13)
**Problem**: NumPy 1.26.4 fails to build on Windows with Python 3.13.

**Error**: 
```
ERROR: Compiler cl cannot compile programs.
```

**Root Cause**: NumPy 1.26.4 doesn't have pre-built wheels for Python 3.13 on Windows and requires a C compiler. Python 3.13 is very new (released October 2024) and many packages haven't released wheels yet.

**Why we can't just update to NumPy 2.x**:
- fasttext-wheel is incompatible with NumPy 2.0 (C++ API breaking changes)
- The test suite explicitly checks and skips tests if NumPy 2.x is detected
- Other dependencies (epitran, panphon) haven't been tested with NumPy 2.0

### 4. Windows Encoding Issues
**Problem**: Various libraries (panphon, transformers, datasets, evaluate) fail with encoding errors on Windows.

**Root Cause**: Windows defaults to system locale encoding instead of UTF-8, causing issues when reading data files.

**Solution Implemented**:
- Enhanced `patches.py` to intercept file operations from problematic libraries
- Automatically adds UTF-8 encoding when these libraries open text files
- Added PYTHONUTF8=1 environment variable for additional safety
- Patches are applied early in the import process via `__init__.py`

## Solutions for One-Tap Installation

### Option 1: Restrict Python version in pyproject.toml (Recommended)
Add to pyproject.toml:
```toml
[tool.poetry.dependencies]
python = ">=3.9,<3.13"
```
This prevents installation on Python 3.13+ and gives users a clear error message.

### Option 2: Add Python version check to Makefile
Update the pre-install target to check Python version:
```makefile
pre-install:
	@python -c "import sys; v=sys.version_info; exit(0 if v.major==3 and v.minor<=12 else 1)" || \
	(echo "ERROR: Python 3.13+ detected. Please use Python 3.12 or earlier." && exit 1)
```

### Option 3: Use .python-version file
Create a `.python-version` file in the repo root:
```
3.12.8
```
Users with pyenv installed will automatically use the correct Python version.

### Option 4: Provide pre-built wheels
Include NumPy 1.26.4 wheels for Python 3.13 in the vendor directory (like PyICU approach).

### Option 5: Wait for ecosystem to catch up
Monitor fasttext-wheel for NumPy 2.0 support, then update all dependencies together.

## Current Status

### Issues Resolved:
1. **PyICU**: ✅ SOLVED
   - Regenerated poetry.lock to remove local wheel references
   - Makefile now auto-runs `turkic-pyicu-install` after Poetry install
   - The installer downloads wheels from GitHub for Python 3.10-3.13
   - Automatically detects if PyICU is already installed

2. **BeautifulSoup4**: ✅ SOLVED
   - Created `poetry.toml` with in-project virtualenvs
   - Shortens paths to avoid Windows 260 character limit
   - Successfully installs when using `.venv` instead of deep AppData paths
   - Makefile clears Poetry cache to prevent corrupted installations

3. **Windows Encoding**: ✅ SOLVED
   - Enhanced patches.py to handle encoding for multiple libraries
   - Automatically applies UTF-8 encoding for file operations
   - Patches are loaded early in the import process

4. **NumPy on Python 3.13**: ❌ NOT SOLVED
   - NumPy 1.26.4 has no wheels for Python 3.13
   - Requires updating to NumPy 2.x or using Python 3.12

### Changes Made:
1. **Makefile updates**:
   - `check-environment` target for Python version validation
   - `check-poetry` target for automatic Poetry installation
   - `pre-install` target that clears Poetry cache and checks for long paths
   - Automatic PyICU installation for Windows after Poetry install
   - `clean-all` target to fully reset the environment
   - Error recovery with cache clearing on installation failures

2. **poetry.toml created**:
   - Configures in-project virtual environments
   - Prevents path length issues

3. **Python code patches**:
   - Enhanced `patches.py` for broader library coverage
   - Moved patches import to top of `__init__.py`
   - Added encoding fix to `pyicu_install.py`

## One-Tap Installation Goal

The goal is to make `make web` and `make check` work from a clean clone without manual intervention.

### Current State:
- ✅ Works on Python 3.9, 3.10, 3.11 and 3.12 on Windows
- ❌ Fails on Python 3.13 due to NumPy compatibility
- ✅ Works on macOS/Linux (all Python versions)

## Recommendations for Users

### For Windows Users:
1. **Use Python 3.9-3.12** (3.12 recommended) - Download from https://www.python.org/downloads/
2. Clone the repository and run:
   ```bash
   make web    # For the web interface
   make check  # For linting and tests
   ```
3. The installation will:
   - Automatically install Poetry if not present
   - Configure virtual environment in project directory
   - Install PyICU from pre-built wheels
   - Handle encoding issues automatically
4. If you must use Python 3.13, you'll need Visual Studio Build Tools to compile NumPy

### For macOS/Linux Users:
No special steps needed - the standard installation process should work with any Python version.

## Future Improvements

1. Consider pinning to Python versions with better Windows wheel support
2. Investigate replacing dependencies that cause issues (e.g., find wikipedia alternative)
3. Add CI/CD testing for Windows to catch these issues earlier
4. Consider providing pre-built Docker images as an alternative