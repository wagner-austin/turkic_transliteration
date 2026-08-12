# Changelog

All notable changes to the Turkic Transliteration project will be documented in this file.

## [0.4.0] - 2026-08-11

The rule files changed behaviour in this release. A corpus transliterated
with 0.3.x is **not** byte-identical to one transliterated with 0.4.0, so
anything built from the older output should record which version produced
it. See "Rule corrections" below for what moved and by how much.

### Rule corrections

Each rule file cites a published phonological description. Every file was
read against the source it names, one language at a time, and six
mismatches were found and fixed. Measured against the affected sites in
this project's own OSCAR corpora:

* **Turkish** soft g was lengthening the preceding vowel in every
  position. Zimmer & Orgun (1992, p. 44) give lengthening only when word-final
  or before a consonant, and "phonetically zero" between vowels. 56,566 of
  the 77,279 length marks in the Turkish corpus sat before a vowel.
* **Uzbek (Latin)** `<yo>` emitted the close vowel while plain `<o>`
  correctly emitted the open one. Ido (2025, p. 154) gives *yol* with the
  open vowel. 45,999 sites.
* **Uzbek (Cyrillic)** `<ж>` produced a voiced stop tied to a voiceless
  fricative, which is not a producible segment. Now the voiced
  postalveolar affricate. `<ё>` carried the same vowel defect as `<yo>`.
* **Finnish** wrote the velar nasal long in every environment. Suomi,
  Toivanen & Ylitalo (2008) transcribe *Englanti* with a short nasal and
  *kengän* and *tango* with a long one, and the conditioning follows from
  the phonotactics they state. 701 sites.
* **Finnish** `<gn>` was unhandled, so *kognitio* got a velar plosive
  where the source has the nasal.
* **Finnish** mapped `<sh>` to the postalveolar fricative, which no
  statement in the cited source licenses — the grapheme for that phoneme
  is `<š>`. It deleted the sibilant from every compound with a
  morpheme-boundary *s* before *h*: *keskushallinto*, *kuningashuone*,
  *rakennushanke*.
* **Kyrgyz** wrote its affricates with the precomposed ligatures
  withdrawn from the IPA while every other language used a tie bar, so
  one phoneme was one character in Kyrgyz and three elsewhere. Notation
  only; now the tie bar.

Kazakh, Uyghur and Azerbaijani required no correction.

### Added

* `turkic-clean-corpus`: the corpus-cleaning stage — symbol
  harmonisation, junk filtering, deduplication and size equalisation —
  now ships with the tool rather than living beside the experiments that
  used it. Verified to reproduce this project's seven published corpora
  byte for byte. The run report names the language whose surviving text
  was smallest, since that language sets the budget every other corpus is
  truncated to.
* The symbol map ships as package data, with a verdict and a citation on
  every row, including the rows that record a contrast deliberately kept.
* Machine-readable provenance: each `*_ipa.rules` header declares its
  source in parseable fields, readable through
  `turkic_translit.rule_provenance`. A guard fails the build when a rule
  file declares no source, or when a test pins expected output without
  naming the rule file's source it inherits.
* Typed language identification: an explicit model choice, a run record
  naming the classifier that filtered a corpus, and `ensure_lid_model` as
  a distinct operation rather than a silent fallback.

### Changed

* Input is NFC-normalised before transliteration. Decomposed input
  previously matched no rule, so its base letter was rewritten while the
  combining mark survived.
* The outbound User-Agent reads the version from package metadata. It was
  a literal, so a release bump left it announcing the previous version.
* Rule files no longer carry macros or commented-out rules that nothing
  referenced. The Turkish velarised lateral is a phonemic contrast these
  rules drop; that is now stated as a decision rather than left as a
  disabled rule.

### Fixed

* NumPy 2 support, by removing the reason for the pin rather than raising it.
* Declared dependencies match the imported ones in both directions.

### Testing

* `make check` clears install, guards, Ruff, strict Mypy and the suite at
  100% statement and branch coverage.
* The seven per-language transliteration tests each declare the source
  they inherit, checked against the rule file's own declaration.
* Passage tests that generated their expected output by running the
  transliterator and then compared against it — so they passed under
  every implementation — are gone. Their text now feeds inventory
  conformance checks, which can fail.

## [0.3.9] - 2026-07-01

* Click `translit` subcommand, lazy PyICU import, three-surface deployment.

## [0.3.8] - 2026-05-19

* Corpus tab shows the full fastText language list; the IPA checkbox
  disables for languages without rule files.

## [0.3.7] - 2026-05-19

* UI trimmed to IPA-only, unused tabs dropped, Kazakh `У` → `w` synced.

## [0.3.1] - [0.3.6]

Not recorded at the time. These releases have no changelog entry, and one
written now would be reconstruction rather than record; `git log` is the
only account of them.

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
