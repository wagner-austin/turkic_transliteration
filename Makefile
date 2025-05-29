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
	ruff check --fix .
	ruff format
	black .
	mypy --strict . --exclude build

# Auto-format code
format:
	#ruff format
	ruff check --fix .
	black .

# Run tests
test:
	python -m pytest

# Build distributable package
build: clean
	python -m build

# Run the web UI example
web:
	python turkic_tools.py web

# Run the simple demo
demo:
	python turkic_tools.py demo

# Run the comprehensive demo
full-demo:
	python turkic_tools.py full-demo

# Show help
help:
	@echo "Available commands:"
	@echo "  make clean      - Remove build artifacts and cache directories"
	@echo "  make lint       - Run linting checks"
	@echo "  make format     - Auto-format code"
	@echo "  make test       - Run tests"
	@echo "  make build      - Build distribution package"
	@echo "  make web        - Run the web UI example"
	@echo "  make demo       - Run the simple demo"
	@echo "  make full-demo  - Run the comprehensive demo"
