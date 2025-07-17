# Changelog

All notable changes to the Turkic Transliteration project will be documented in this file.

## [0.3.0] - 2025-01-17

### Added
* Turkish language support for both IPA and Latin (ASCII-fold) transliteration
* Dynamic language detection system - automatically discovers languages from rules files
* File upload functionality in the transliteration tab
* Download button for transliterated output
* Corpus preview with copy button in the download tab
* Transliteration option for corpus downloads
* README.md in rules directory explaining how to add new languages
* Web UI now shows a confidence table for each Russian-masking run.
  Implementation lives in `web_demo._to_md_table`.
* `mask_russian` now strips ANSI colour escapes at source, so its
  output is always display-safe.

### Changed
* Moved shared text input to individual tabs to reduce UI clutter
* Improved FastText confidence threshold description in corpus download
* Renamed rules files from `*_lat2023.rules` to `*_lat.rules` for consistency
* Enhanced corpus download with better logging and progress reporting

### Fixed
* Corpus download now properly limits lines when filtering is enabled
* FastText model corruption detection and automatic re-download
* Test updates to handle new Turkish support
