# Contributing to Turkic Transliteration

Thank you for your interest in contributing to the Turkic Transliteration project! This document provides guidelines to help you contribute effectively.

## Project Structure

The project follows a standard Python package structure:

```
.
├── data/             # Sample data and language resources
├── docs/             # Documentation
├── src/              # Source code
│   └── turkic_translit/  # Main package
│       ├── cli/      # Command-line tools
│       ├── web/      # Gradio web UI
│       └── rules/    # Per-language transliteration rule files
├── tests/            # Test suite
├── scripts/          # Dev and release utilities
└── vendor/           # Pre-built PyICU wheels (Windows)
```

## Development Guidelines

### Scripts and CLI Tools

**IMPORTANT:** All runnable scripts and CLI tools must live in `src/turkic_translit/cli/`.

- Never add runnable code, scripts, or entry points outside the `turkic_translit/cli` package
- Register new CLI tools in pyproject.toml under the `[project.scripts]` section

### Code Style

- Format and lint with Ruff: `ruff format` and `ruff check --fix .`
- Write type hints and use mypy: `mypy --strict`
- `make lint` runs all three together; `make check` runs lint + tests.

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

## Adding a language

Transliteration languages are discovered dynamically from rule files in
`src/turkic_translit/rules/` (e.g. `tr_ipa.rules`, `kk_lat.rules`). To add a
language, add its `{lang}_{ipa,lat}.rules` file there; see
`src/turkic_translit/rules/README.md`. Add per-language tests under `tests/`.

## Demos

Demo entry points live in `turkic_tools.py` (`web`, `demo`, `full-demo`) and
the Gradio UI in `src/turkic_translit/web/`. There is no separate `examples/`
directory — all runnable code belongs in `src/turkic_translit/`.
