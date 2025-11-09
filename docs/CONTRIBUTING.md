# Contributing to Turkic Transliteration

Thank you for your interest in contributing to the Turkic Transliteration project! This document provides guidelines to help you contribute effectively.

## Project Structure

The project follows a standard Python package structure:

```
.
├── data/             # Sample data and language resources
├── docs/             # Documentation
├── examples/         # Example scripts and applications
│   └── web/          # Web interface examples
├── src/              # Source code
│   └── turkic_translit/  # Main package
│       └── cli/      # Command-line tools
├── tests/            # Test suite
└── vendor/           # Third-party dependencies
```

## Development Guidelines

### Scripts and CLI Tools

**IMPORTANT:** All runnable scripts and CLI tools must live in `src/turkic_translit/cli/`.

- Never add runnable code, scripts, or entry points outside the `turkic_translit/cli` package
- Register new CLI tools in pyproject.toml under the `[project.scripts]` section

### Code Style

- Use Black for formatting: `black .`
- Use Ruff for linting: `ruff check .`
- Write type hints and use mypy: `mypy --strict .`

### Logging & Errors

- Do not call `logging.basicConfig` or modify the root logger. Entrypoints should call `turkic_translit.logging_config.setup()`.
- Acquire module loggers via `logging.getLogger(__name__)` and use appropriate levels (debug/info/warning/error).
- Use `logger.exception(...)` at error boundaries to capture stack traces.
- When adding UI-facing errors, use `turkic_translit.error_service.error_response()` to standardize payloads.
- Correlation IDs and request context are available via `set_correlation_id()` / `set_request_context()` in `turkic_translit.error_service`.

### Testing

- Write tests for all new functionality
- Run tests with pytest: `pytest`
- All tests should pass before submitting a PR

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure they pass
5. Submit a pull request

## Sample Data

If you need to add sample data, place it in the `data/` directory, with appropriate subdirectories for organization.

## Examples

If you're adding example code, place it in the `examples/` directory. For web-based examples, use the `examples/web/` subdirectory.
