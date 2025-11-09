.PHONY: clean lint test build docs examples

# Default action when running `make` without arguments
all: lint test

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	python -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('dist') + glob.glob('build') + glob.glob('*.egg-info') + glob.glob('.coverage') + glob.glob('htmlcov') + glob.glob('.pytest_cache') + glob.glob('.ruff_cache') + glob.glob('.mypy_cache') + glob.glob('**/__pycache__', recursive=True) if os.path.exists(p)]; [os.remove(p) for p in glob.glob('**/*.pyc', recursive=True) if os.path.exists(p)]"
	@echo "Clean completed successfully!"

# Deep clean including Poetry environment
clean-all: clean
	@echo "Removing Poetry virtual environment..."
	-poetry env remove --all
ifeq ($(OS),Windows_NT)
	@echo "Clearing Poetry cache..."
	@powershell -Command "if (Test-Path '$$env:LOCALAPPDATA\pypoetry\Cache') { Remove-Item -Path '$$env:LOCALAPPDATA\pypoetry\Cache' -Recurse -Force }"
endif

# Run linting tools
lint: install
	poetry run ruff check --fix .
	poetry run ruff format
	poetry run mypy --strict
	
# Check and validate environment
check-environment:
	@echo "Checking Python version..."
	@python -c "import sys; v=sys.version_info; supported=(v.major==3 and 9<=v.minor<=12); status='supported' if supported else 'NOT SUPPORTED'; print(f'[OK] Python {v.major}.{v.minor}.{v.micro} detected - {status}'); exit(0 if supported else 1)" || (echo "ERROR: Python 3.9-3.12 required" && exit 1)
	@echo "Checking for virtual environment..."
	@python -c "exec('''import os, sys, shutil, subprocess\nvenv_path = '.venv'\nvenv_python = os.path.join(venv_path, 'Scripts', 'python.exe') if sys.platform == 'win32' else os.path.join(venv_path, 'bin', 'python')\nif not os.path.exists(venv_path):\n    print('[OK] No virtual environment found (will be created during install)')\nelif not os.path.exists(venv_python):\n    print('[WARN] Incomplete virtual environment found. Removing...')\n    shutil.rmtree(venv_path, ignore_errors=True)\n    print('  Cleaned up incomplete environment')\nelse:\n    try:\n        result = subprocess.run([venv_python, '-I', '-W', 'ignore', '-c', 'import sys, encodings'], capture_output=True, text=True)\n        if result.returncode != 0:\n            print('[WARN] Virtual environment is corrupted (missing core modules). Removing...')\n            if result.stderr and ('ModuleNotFoundError' in result.stderr or 'Fatal Python error' in result.stderr):\n                lines = result.stderr.splitlines()\n                if lines:\n                    print(f'  Error: {lines[0]}')\n            shutil.rmtree(venv_path, ignore_errors=True)\n            print('  Cleaned up corrupted environment')\n        else:\n            print('[OK] Virtual environment exists and is healthy')\n    except Exception as e:\n        print(f'[WARN] Virtual environment check failed: {e}. Removing...')\n        shutil.rmtree(venv_path, ignore_errors=True)\n        print('  Cleaned up problematic environment')''')"

# Check if Poetry is installed and install if missing
check-poetry: check-environment
ifeq ($(OS),Windows_NT)
	@powershell -Command "if (!(Get-Command poetry -ErrorAction SilentlyContinue)) { Write-Host 'Poetry not found. Installing Poetry...'; (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -; Write-Host 'Please restart your terminal and run make again' } else { Write-Host 'Poetry is already installed.' }"
else
	@if ! command -v poetry &> /dev/null; then \
		echo "Poetry not found. Installing Poetry..."; \
		if command -v curl &> /dev/null; then \
			curl -sSL https://install.python-poetry.org | python3 -; \
		elif command -v wget &> /dev/null; then \
			wget -qO- https://install.python-poetry.org | python3 -; \
		else \
			echo "ERROR: Neither curl nor wget found. Please install curl or wget."; \
			exit 1; \
		fi; \
		echo "Please restart your terminal and run make again"; \
	else \
		echo "Poetry is already installed."; \
	fi
endif

# Pre-install: Handle environment setup and common issues
pre-install: check-poetry
	@echo "Configuring Poetry..."
	-@poetry config virtualenvs.in-project true
	@echo "Checking for corrupted poetry.lock..."
	@python -c "import os; os.remove('poetry.lock') if os.path.exists('poetry.lock') and os.path.getsize('poetry.lock')==0 else None"
ifeq ($(OS),Windows_NT)
	@echo "Preparing Windows environment..."
	@powershell -Command "if ((Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -ErrorAction SilentlyContinue).LongPathsEnabled -ne 1) { Write-Host 'WARNING: Long paths not enabled. Some packages may fail to install.' }"
	@echo "Clearing Poetry cache to avoid installation issues..."
	@powershell -Command "if (Test-Path '$$env:LOCALAPPDATA\pypoetry\Cache\artifacts') { Remove-Item -Path '$$env:LOCALAPPDATA\pypoetry\Cache\artifacts' -Recurse -Force }"
endif
	@echo "Ensuring poetry.lock exists..."
	@python -c "import os, subprocess, sys; exec(\"\"\"if not os.path.exists('poetry.lock'):\n    print('Creating poetry.lock file...')\n    result = subprocess.run(['poetry', 'lock'], capture_output=True, text=True)\n    if result.returncode != 0:\n        print('Failed to create lock file:', result.stderr)\n        print('Clearing cache and retrying...')\n        subprocess.run(['poetry', 'cache', 'clear', 'pypi', '--all', '--no-interaction'], capture_output=True)\n        result = subprocess.run(['poetry', 'lock'], capture_output=True, text=True)\n        if result.returncode != 0:\n            print('ERROR: Failed to create lock file:', result.stderr)\n            sys.exit(1)\"\"\")"

# Install all dependencies (corpus extras)
lock: check-poetry
	@echo "Locking dependencies (poetry.lock)..."
	@poetry lock

install: pre-install lock
	@echo "Installing dependencies..."
	@poetry install --extras corpus --extras dev --no-ansi || \
		(echo "Installation failed. Attempting recovery..." && \
		echo "Removing virtual environment..." && \
		-@poetry env remove --all --quiet && \
		echo "Clearing Poetry cache..." && \
		poetry cache clear pypi --all --no-interaction && \
		echo "Retrying installation..." && \
		poetry install --extras corpus --extras dev --no-ansi)
ifeq ($(OS),Windows_NT)
	@echo "Installing PyICU for Windows..."
	-@poetry run python -m turkic_translit.cli.pyicu_install
endif
	@echo "Verifying installation..."
	@poetry run python -c "import turkic_tools; print('[OK] Installation successful!')" || (echo "ERROR: Installation verification failed" && exit 1)

# Run tests
test: install
	poetry run pytest -rsxv

check: lint test

# Build distributable package
build: clean install
	poetry run python -m build

# Run the web UI with DEBUG logs
debug: install
	poetry run python turkic_tools.py web

# Run the web UI example
web: debug
	poetry run python turkic_tools.py web

# Run the web UI example
run: web

# Test corpus download with progress
test-corpus: install
	poetry run turkic-translit --log-level debug download-corpus download --source oscar-2301 --lang tr --out test_corpus.txt --max-lines 1000

savecode:
	savecode . --skip tests	web --ext py toml

savecode-web:
	savecode . --skip tests --ext py toml

savecode-test:
	savecode . --ext py toml


# Show help
help:
	@echo "Available commands:"
	@echo "  make clean      - Remove build artifacts and cache directories"
	@echo "  make lint       - Run linting checks"
	@echo "  make format     - Auto-format code"
	@echo "  make test       - Run tests"
	@echo "  make build      - Build distribution package"
		@echo "  make web        - Run the web UI example"
	@echo "  make lock       - Refresh poetry.lock"
	@echo "  make run-debug  - Run the web UI with TURKIC_LOG_LEVEL=DEBUG"
	@echo "  make demo       - Run the simple demo"
	@echo "  make full-demo  - Run the comprehensive demo"
