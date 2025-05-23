# Changelog

All notable changes to the Turkic Transliteration project will be documented in this file.

## [Unreleased]

### Added
* Web UI now shows a confidence table for each Russian-masking run.
  Implementation lives in `web_demo._to_md_table`.
* `mask_russian` now strips ANSI colour escapes at source, so its
  output is always display-safe.
