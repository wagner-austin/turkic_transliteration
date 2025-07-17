.PHONY: clean lint test build docs examples

# Default action when running `make` without arguments
all: lint test

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	python -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('dist') + glob.glob('build') + glob.glob('*.egg-info') + glob.glob('.coverage') + glob.glob('htmlcov') + glob.glob('.pytest_cache') + glob.glob('.ruff_cache') + glob.glob('.mypy_cache') + glob.glob('**/__pycache__', recursive=True) if os.path.exists(p)]; [os.remove(p) for p in glob.glob('**/*.pyc', recursive=True) if os.path.exists(p)]"
	@echo "Clean completed successfully!"

# Run linting tools
lint:
	poetry run ruff check --fix .
	poetry run ruff format
	poetry run mypy --strict .
	
# Install all dependencies (corpus extras)
install:
	poetry lock
	poetry install --extras corpus --extras dev

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
	poetry run turkic-translit download-corpus download --source oscar-2301 --lang tr --out test_corpus.txt --max-lines 1000 -v

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
	@echo "  make run-debug  - Run the web UI with PYTHONLOGLEVEL=DEBUG"
	@echo "  make demo       - Run the simple demo"
	@echo "  make full-demo  - Run the comprehensive demo"
