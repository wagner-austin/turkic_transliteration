# Web UI Documentation

This document covers the web interface for the Turkic Transliteration Suite.

## Tabs Overview

The current UI includes:
- Download Corpus
- Train Tokenizer (SentencePiece)
- Transliterate to IPA/Latin
- Mutual Intelligibility (LM analysis)

Removed tabs:
- ID Token Language
- Filter Russian
- Levenshtein Distance

These features still exist in the library/CLI where applicable, but the web UI no longer exposes them.

## Logging and Errors

- Logging is centralized. Control verbosity with `TURKIC_LOG_LEVEL` (DEBUG, INFO, WARNING, ERROR).
- Default log format is JSON with correlation IDs; change with `TURKIC_LOG_FORMAT=rich` for human‑readable output.
- Each user action adds a correlation ID and minimal request context (e.g., action name, language).
- Errors surfaced in the UI are standardized and include the correlation ID for cross‑referencing logs.
- Optional Sentry integration is available via `pip install turkic-translit[sentry]` and `TURKIC_SENTRY_DSN`.
