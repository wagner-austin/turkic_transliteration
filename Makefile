.PHONY: clean lint test build docs examples

# Default action when running `make` without arguments
all: lint test

# Clean build artifacts
clean:
	rm -rf dist build *.egg-info .coverage htmlcov .pytest_cache .ruff_cache .mypy_cache __pycache__
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Run linting tools
lint:
	-ruff check src tests examples
	-black --check src tests examples
	-mypy --strict .

# Auto-format code
format:
	black src tests examples
	ruff check --fix src tests examples

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
