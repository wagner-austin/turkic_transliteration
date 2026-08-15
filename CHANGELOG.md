# Changelog

All notable changes to the Turkic Transliteration project will be documented in this file.

## [0.5.2] - 2026-08-14

### Fixed

**The web demo could not start.** Building the interface lists the
OSCAR-2301 corpus's configurations, which makes `datasets` fetch and
execute that dataset's own loading script, and that script's first
third-party import is `zstandard`. Nothing under `src/` imports it, so
0.5.0's dependency audit — which kept what an import statement named —
dropped it, and an install with no other source of `zstandard` raised
`ModuleNotFoundError` before it could serve a page.

Development environments never saw it, because something else in the
lock brings `zstandard` along. The Hugging Face Space, which installs
this package and nothing else, saw nothing but it.

The declaration is back, alongside `accelerate`, which is undeclarable
by the same import-statement rule for the same reason: needed at
runtime by something this package does, named in no import here.

## [0.5.1] - 2026-08-12

### Added

* `turkic-clean-corpus --harmonize-dir/--harmonize-output-dir`: rewrite
  evaluation texts with the symbol map and nothing else — no filtering,
  no deduplication, no equalisation — so files whose line structure must
  survive (perception passages with section headers and markers) can
  share the training corpora's symbol space. The corpus-cleaning pair
  becomes optional; give either pair or both, and a half-given pair is
  refused. Verified byte-identical against the LSTM experiment's own
  snippet harmonisation across all seven perception files, which lets
  that experiment retire its private copy of the cleaning pipeline.

## [0.5.0] - 2026-08-12

0.4.0 was tagged but never published to PyPI, so anyone upgrading from
0.3.9 receives both releases' changes at once. Read the 0.4.0 section
below as well.

### Fixed a regression shipped in 0.3.7, 0.3.8 and 0.3.9

**Kyrgyz and Uzbek Cyrillic deleted vowels.** Anyone who installed the
tool after 2026-05-20 should upgrade and regenerate.

```
0.3.9   тау  ->  twu        0.5.0   тау  ->  tɑw
        ауа  ->  wuɑ                ауа  ->  ɑwɑ
        маек ->  mjeek              маек ->  mɑjek
```

The glide and iotation rules were written with the right-context bracket
where the left one was meant, so rather than rewriting the letter they
named they replaced the vowel in front of it. In 0.3.6 the same rules
tested against a set of IPA vowels, which the Cyrillic input could never
match, so they did nothing at all; a later change to the Cyrillic set
made them match, and they began to fire. Every vowel before `<е>` in
Uzbek Cyrillic collapsed to the same three characters, so `ае`, `ие`,
`ое`, `уе` and five more all produced `jee`.

A left context is matched against text that has already been
transliterated, so these sets have to name IPA. That is now stated in
each file and enforced: a left context containing a character of the
source script fails the build.

### Rule corrections

* **Kyrgyz** `<ж>` was the fricative. McCollum (2020) Table 3 supplies
  two roots containing it, *жыл* 'year' and *жол* 'road', and prints the
  affricate in both. Table 1 does list the fricative, but as a phoneme of
  the language rather than as the value of this letter — it is what the
  letter is in Russian. Kazakh is where the fricative is correct, so the
  error also made the two languages alike on a segment that separates
  them.
* **Kazakh** `<у>` was the glide everywhere. McCollum & Chen (2021) give
  /w/ in the consonant chart, with *уақ* and *тау*, and also /uw/ in the
  eleven-vowel inventory on p. 281, with *ту* 'flag'. One letter spells
  both, so an unconditional rule cannot serve it: the glide everywhere
  left *су* 'water' as `[sw]`, a syllable with no vowel in it. It is now
  the glide next to a vowel and the vowel elsewhere, collapsed as the
  file already collapses /ij/ and /ie/.
* **Kyrgyz** `<ц>` still emitted a withdrawn precomposed ligature after
  the others were converted in 0.4.0, because the test named four of them
  by hand and this was the fifth. The symbol map gains the matching row,
  since corpora published before now carry it.
* **Seventeen Cyrillic letters** had no rule in the file for a language
  whose corpus contains them, so they passed through as Cyrillic into
  output that is supposed to be IPA. Each now takes the value this
  project's own sourced rule file gives it for the language that owns the
  letter; where Kazakh and Kyrgyz disagree the choice is written down
  rather than made silently.

### Added

* Guards stated over the whole rule set rather than as lists kept by
  hand, since a hand-kept list is what let the fifth ligature through:
  the withdrawn ligatures as a codepoint range, the Cyrillic check as
  coverage of the union of every alphabet in the project, and a left
  context may not name the source script. One needs no source to
  adjudicate and would have caught the Kazakh defect on its own — no word
  may lose its only vowel.

### Note for anyone with data from an earlier version

A corpus or transcription made with 0.3.x or 0.4.0 is not reproducible
from 0.5.0. Record the version **and** a hash of the rule files used:
during 0.3.7 to 0.3.9 the repository and the published release disagreed,
so the version number alone did not identify the rules.

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
