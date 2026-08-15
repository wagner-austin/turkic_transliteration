# Changelog

All notable changes to the Turkic Transliteration project will be documented in this file.

## [0.5.9] - 2026-08-15

### Fixed

**`turkic-clean-corpus` discarded almost every Turkish line.** The
sanitiser runs on text the symbol map has already rewritten, but it
compared that text against the raw emitted set — what the rules produce
*before* the map. Turkish rules emit `a` and the map rewrites it to `ɑ`,
so every harmonised Turkish token carried a character the filter did not
recognise and was dropped whole as foreign material. On a forty-line
Turkish corpus: one line kept, four hundred tokens dropped, forty-eight
characters written.

`harmonized_emitted` builds the set the sanitiser actually needs — the
map's image of what the rules emit — and `clean_corpora` uses it. A test
pins the case: `ɑ` is absent from `emitted_characters("tr")`, present in
`harmonized_emitted("tr", rules)`, and a Turkish line survives cleaning
with no tokens dropped.

The emitted set also now collects the letters an apostrophe
transliterates *to*, which is how the Uzbek tutuq belgisi (`ba'zi` →
`baʔzi`) survives cleaning while a bare apostrophe stays foreign in the
languages that have no apostrophe letter.

Both changes existed on the corpus-rebuild branch, where the corpora
were built, and both were missed when 0.5.8 took only that branch
commit's rule file. 0.5.7 and 0.5.8 carry the defect; anyone who ran
`turkic-clean-corpus` under either should rebuild.

## [0.5.8] - 2026-08-15

### Fixed

**Latin Uzbek left the glottal stop untransliterated.** An apostrophe
that the `oʻ` and `gʻ` rules did not consume is the tutuq belgisi, and
it passed through into the output as a typographic mark rather than a
phoneme: `ba’zi` gave `ba’zi`. Cyrillic Uzbek has always given the same
sound its value — `uzc_ipa.rules` maps `ъ > ʔ` — so one language's two
orthographies disagreed about a word they spell the same way, and the
Latin side emitted a character no phoneme inventory contains. Ido (2025)
p. 154 records the segment as /ʔ/ with minimal phonemic load, limited to
Arabic loanwords written with an apostrophe after a letter. Both now
give `baʔzi`.

The rule had been fixed on the corpus-rebuild branch, so the corpora and
models built there were correct while the released package was not — a
divergence the manifest's rule fingerprint reports rather than leaves to
be discovered. With this merged, all eight rule files match the digests
recorded in the 2026-08-15 corpus manifest.

## [0.5.7] - 2026-08-15

### Fixed

**Kazakh and Kyrgyz wrote two glyphs for one low vowel.** `Я` and `я`
mapped to `ja` with an ASCII `a`, while plain `А` and `а` map to `ɑ`, so
a native word like Kazakh аяқ or Kyrgyz аял came out carrying both. A
character-level model reads that split as a phonological contrast the
language does not have, and it is invisible to inventory and script
checks because both glyphs are legal IPA — the same shape as the
train/eval notation defect recorded earlier for Cyrillic u. Both files
now write `jɑ`, no rule in either emits ASCII `a`, and a hygiene test
requires every iotated letter to equal the glide plus the plain vowel.

**`ar_lat.rules` had three defects behind an uncited header.** The file
now cites Duval & Janbaz (2006), the Latin-Script Uyghur standard whose
Table 3 it implements, in machine-readable `Source-*` fields. The
apostrophe that keeps `n`+`g`, `s`+`h` and `ng`+`h` from reading as
digraphs was missing; the hamza carrier was not distinguished across its
three positions (after a consonant and word-finally it is an apostrophe,
between vowels it is nothing); and the rule for `ﻉ` had been swallowed
by the rule for `ﺀ`, because a single apostrophe opens a quoted literal
in ICU rule syntax rather than emitting a character.

### Changed

**Cleaning sanitises against what each language's rules can emit.** It
replaced anything outside one fixed character list, which could not know
what a given language actually produces, so quoted foreign material
survived as fragments and punctuation survived as corpus style.
`corpus/inventory.py` derives each language's emitted set by running its
rules over every letter, every letter pair and every multi-character
head against every letter — the sweep the hygiene tests already used, so
the two cannot drift. A token carrying a letter the rules can never emit
is dropped whole; everything else they cannot emit becomes a space.

The manifest gains the counts that produces and a SHA-256 fingerprint of
every rule file and of the symbol map the run applied, so a corpus can
be checked against the rules that exist now rather than assumed to match
them. `turkic-clean-corpus` prints the two new counts per language.

### Added

* A daily request to the Hugging Face Space, so it does not reach the
  48-hour idle timeout and make its next visitor wait for a wake-up.

## [0.5.6] - 2026-08-15

### Changed

**`pip install turkic-translit` now installs something that works.** It
did not before, on any operating system. PyICU — which compiles every
rule file and produces every transliteration this project performs —
publishes no wheels at all, only sdists, so installing it meant
compiling against ICU headers the machine usually lacked: `libicu-dev`
on Linux, Homebrew's `icu4c` on macOS, and on Windows a wheel fetched
from a third-party build server by a console script shipped for the
purpose.

The dependency is now `pyicu-wheels`, which publishes the same extension
prebuilt for Linux, macOS and Windows across Python 3.10–3.14. Verified
on a clean Windows virtual environment with no compiler and no ICU
headers: one `pip install`, then `to_ipa("құс", "kk")` returns `qʊs`.

So the whole install is:

```bash
pip install turkic-translit
turkic-translit web
```

**The web demo is a subcommand.** `turkic-translit --help` listed seven
commands and never mentioned that a web interface existed, while the
demo was reachable three separate ways — a `turkic-web` script, a
`turkic_tools.py web` runner, and a `make web` target wrapping the
second. It is now `turkic-translit web`, listed where a new user looks.

### Removed

* `turkic-pyicu-install` and the module behind it, `turkic_translit.wheels`
  (the wheel-selection logic), the release-index and installer hooks in
  `turkic_translit._test_hooks`, and `core.py`'s table of per-platform
  build instructions. None of it has anything left to do.
* `turkic_tools.py`. Its `web` command duplicated the console script and
  its other two, `demo` and `full-demo`, had pointed at an `examples/`
  directory deleted in `da1eb62` — so they failed with a missing-file
  error, while the README and the contributing guide went on documenting
  them.
* The `turkic-web` console script, replaced by the subcommand.
* The empty `winlid` extra, which installed nothing while the README
  described it as providing the Windows FastText wheel.
* `docs/windows_pyicu_guide.md` and
  `docs/windows-installation-troubleshooting.md`, 250 lines about a
  problem that no longer exists, and `vendor/`, which held Windows PyICU
  wheels for the same reason.

### Fixed

* The README, `docs/setup_guide.md` and `docs/index.md` documented a
  `ui` extra that has never existed, an install command using it, and
  `turkic_transliterate` as the PyPI name — which is the deprecated
  redirect package, not this one.

* **Uzbek `ngʻ` and `yoʻ` were read as a digraph plus a stray mark.**
  The apostrophe exists only as part of the letters `oʻ` and `gʻ`, so
  `ngʻ` is `n` + `gʻ` (Cyrillic н-ғ, қўнғиз) and `yoʻ` is `y` + `oʻ`
  (Cyrillic й-ў, йўл); the rules for the apostrophe-bearing letters now
  stand ahead of the digraphs that would otherwise take the letters
  apart. A seam sweep over every letter pair and every multi-character
  rule head against every letter now runs for all eight rule files.

* **Three classes of raw-web character reached the published corpora.**
  An invisible format character inside a word blocked contextual rules
  and later split the word; an Arabic presentation form spelled a native
  Uyghur letter as a display codepoint the rules do not name, so whole
  native words passed through untransliterated; and a page authored in
  Windows-1254 but decoded as Windows-1252 arrived with Turkish `ı`,
  `ş`, `ğ` and `İ` swapped for `ý`, `þ`, `ð` and `Ý`. Source text is now
  normalised before transliteration: format characters and presentation
  forms fall to Unicode itself, and the codepage swaps to `folds.csv`, a
  table of verified repairs carrying the same columns, reader and
  citation discipline as the symbol map.

## [0.5.5] - 2026-08-14

### Removed

**`load_tokenizer`'s SentencePiece override, and `LMModel.fresh`'s
`spm_override` argument.** They substituted a shared SentencePiece model
into a loaded tokenizer, so that several languages could train against
one sub-word vocabulary, by replacing the tokenizer's `sp_model`.

transformers 5 has no such attribute: every tokenizer is backed by the
Rust `tokenizers` library now, including the ones that were previously
"slow". Its documented replacements do not substitute anything —
`AutoTokenizer.from_pretrained(vocab=…)` and direct construction with
`vocab_file=` both accept the file and keep the published vocabulary,
which was verified against a shared model that segments differently:

```
shared vocabulary:     40 pieces
published vocabulary: 104 pieces
constructed with vocab_file=<shared> → 104 pieces
```

A substitution whose only purpose is a guarantee about which vocabulary
trained a model is worse than useless when it fails silently, and
nothing called this one: no console script exposed it, and no caller
existed in this project or any other.

**Sub-word tokenisation is unaffected.** `turkic-build-spm` and
`turkic-train-spm` still train SentencePiece models,
`turkic_translit.tokenizer` still tokenises with them, and fine-tuning
still uses the base model's own published sub-word vocabulary.

### Changed

* **transformers 5**, which the removal above unblocked, and with it
  **gradio 6.24** and **huggingface-hub 1.x**. Gradio 6.18 onwards
  requires hub 1.2 or newer, while transformers 4 capped it below 1.0,
  so the demo had been held at gradio 6.17.3 by a dependency of the
  language-model tooling it never loads.
* `lm/model_calls.py` states the two `PreTrainedModel` methods that
  transformers 5 leaves unannotated — `eval` and
  `gradient_checkpointing_disable` — as a Protocol reached by name, the
  same treatment `lm/tokenizer.py` already gives
  `AutoTokenizer.from_pretrained`. Strict checking rejects a call into
  unannotated code, and this project permits neither suppression
  comments nor per-module type-checker overrides.

## [0.5.4] - 2026-08-14

### Removed

**The `datasets` dependency, and the `corpus` extra with it.** Listing
OSCAR's languages used to download and execute the dataset's own loading
script — that is how `datasets` reads a script-published corpus — so
drawing a dropdown ran third-party code and inherited its dependencies.
That is how `zstandard` reached this project without ever being
declared, and it would have stopped working entirely under `datasets` 4,
which removes loading scripts.

The layout that script describes is public and simple: one directory per
language of Zstandard-compressed JSON lines. `corpus/hub.py` states the
naming rules and `corpus/_test_hooks.py` reads them, so languages come
from the repository's file listing — served without a credential, though
the shards themselves are gated — and text comes from the shards,
decompressed as it streams. Eleven transitive packages left with
`datasets`, including the whole `aiohttp` stack.

`pyarrow` was then the only thing left in the `corpus` extra and nothing
imports it, so `pip install turkic-translit[corpus]` is now just
`pip install turkic-translit`.

### Fixed

* A refused corpus read reaches the interface as this package's own
  `CorpusStreamError` rather than as `urllib.error.HTTPError`. The web
  tab reports the former and would have shown a visitor a traceback for
  the latter.
* **The Hugging Face Space served an unthemed page.** Gradio 6 takes the
  theme and stylesheet at `launch`, and only the server hook passed
  them; the Space's entry point built the interface and launched it
  itself, so inputs rendered near-invisible against their own
  background. The entry point now calls the same `main()` the
  `turkic-web` console script calls.
* The stylesheet's hardcoded `#ddd`, `#555` and `#666` became Gradio
  theme variables, so borders and secondary text are legible in dark
  mode as well as light.

### Changed

* The Transliterate tab is two columns: input, upload, language and the
  button on the left, output on the right — the arrangement the corpus
  tab already used. Previously the input box spanned the top at double
  width beside an empty column, and the output sat in a narrow strip
  below it.
* The page header was a title, a subtitle restating it, and a paragraph
  explaining that tabs are tabs; it is now one heading and one sentence.
  The footer points at the rule files and their sources instead of
  repeating the title.

## [0.5.3] - 2026-08-14

### Fixed

**Typing in the web demo wrote a file per keystroke.** Every keystroke
ran the transliteration handler, and that handler saved a timestamped
copy whenever its output passed fifty characters, so typing a paragraph
left a hundred files behind. Typing now transliterates and returns;
the Transliterate button writes the file. The fifty-character threshold
is gone rather than adjusted — it made the short answer the harder one
to keep.

### Changed

* The demo opens on **Transliterate to IPA** rather than on the corpus
  downloader.
* Every language the tab offers now has an example, and each example is
  a word this project already checks against a published description of
  that language (Ragagnin, Karlsson, Abish, McCollum, Routledge, Ido,
  and the Montreal Forced Aligner dictionary). Previously two languages
  had examples, and the Kazakh one was Russian text.
* The language selector names its languages instead of listing bare
  codes, the IPA output has a copy button, and the download slot stays
  hidden until there is a file in it — all three matching what the
  corpus tab already did.

### Release process

`git push` is now the whole release. The publish workflow runs on a push
to `main`, publishes whatever version `pyproject.toml` declares if PyPI
does not have it, and tags the commit it published. Hand-made tags were
a second chance to get a release wrong.

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
