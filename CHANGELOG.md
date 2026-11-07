# Changelog

All notable changes to the Turkic Transliteration project will be documented in this file.

## [Unreleased]

### Fixed

**Cleaning a corpus no longer needs PyTorch.** The `turkic_translit.cli`
package imported every subcommand when it was imported, and
`turkic-clean-corpus` lives in it, so the cleaner imported `torch` and
`transformers` through `eval-lm` before reading its arguments; on a host
whose `torch` could not load its native library it could not print its
own `--help`. The group now imports a subcommand's module when the
command is looked up (`LazyCommandGroup`, `COMMANDS`, `load_command`), so
`translit` and `turkic-clean-corpus` load neither, which
`tests/test_cli_lazy_imports.py` reads from a fresh interpreter's
`sys.modules`, while `eval-lm` still loads both. The unused re-export of
`train_spm` from the package is gone.

**A download filter that can keep nothing fails instead of writing an
empty corpus.** On 2026-11-07 `turkic-download-corpus download --lang kk
--filter-langid kk --lid-model lid218e` read all 338,811 Kazakh OSCAR
lines in thirteen minutes, kept none, and exited 0 with an empty file,
because lid218e labels Kazakh `kaz_Cyrl` and the filter matches its code as
a prefix. Before streaming, the filter now refuses a code no label of its
model starts with (`TURKIC_LID_010_LANGUAGE_NOT_EMITTED`), naming the
model's labels for the same language under its ISO 639-3 code
(`three_letter_code`, `LidClassifier.labels_matching`). A filtered run
that keeps none of the lines it saw removes the empty file, writes no
manifest and raises `TURKIC_CORPUS_010_FILTER_KEPT_NOTHING`. The README
documents the download command and which code each model takes.

**The demo's SentencePiece training works under the sentencepiece a pip
install gets.** `pyproject` admits `sentencepiece>=0.2.0` and the lockfile
pinned 0.2.0, so the fleet tested a version no new install receives.
Under 0.2.3, measured on 2026-11-07, a vocabulary size the training text
cannot fill is refused for every model type ("Vocabulary size too high
(12000). Please set it to a value <= ..."), so the demo's default of 12000
failed on any pasted text. The web trainer now passes
`hard_vocab_limit=False`, learning what the text supports (48 pieces of
50 unigram, 194 of 12000 BPE on the test's text), and its summary states
the learned size beside the requested one. The lockfile now pins 0.2.3,
the version pip resolves. `build-spm` still refuses an unfillable size, as
its help says, and `train-spm` keeps its `--hard-vocab-limit` flag.

**Kazakh typed in pre-Unicode fonts keeps its words.** Those fonts put
Kazakh letters on the Windows-1251 slots of Macedonian and Serbian ones,
so web text spells қ ғ ң as ќ ѓ њ, and some writes ң as the en-with-tail
the Kyrgyz folds already repair. None of these letters is in the Kazakh
rules, so such words reached the IPA untransliterated and the cleaner
dropped them. Eight new `kk` rows in `corpus/folds.csv` repair them on
download, each measured on the 2026-11-07 OSCAR kk download: spelled
with the Kazakh letter, 1,774 of 1,901 ќ tokens, 486 of 552 ѓ tokens, 572
of 635 њ tokens and 400 of 431 ӊ tokens are words the same corpus already
writes that way. The fonts' stand-ins for ә are left alone: є reads as ә
in 336 tokens and as ғ in 296, and ј and ѕ split between ә, ң and і, so
no single letter is right for them. Corpora downloaded before this carry
the old folds digest in their manifests.

## [0.5.11] - 2026-11-07

### Added

**Real failures for the real seams.** `tests/test_real_seam_failures.py`
runs the production implementations against failures the world
produces, each measured before it was asserted: the model downloader
against a real HTTP server answering 404 (an `HTTPError`, and no file or
`.part` left behind), the fastText loader against a file that is not a
model, the Gradio server against a port another socket holds, and the
Sentry reporter against a malformed DSN; the language-model trainer and
evaluator against a model path that does not exist (`OSError`) and a
directory holding no tokenizer (`ValueError`); and the terminal prompt
against a closed standard input (`EOFError`). Measuring the empty
directory found the hooks' docstrings promising `OSError` for it, so
they, `LMModel` and `load_tokenizer` now name the `ValueError` too.

**`make test` says which language-identification tests ran.** It passes
`-rs`, so a run without the real fastText weights ends with
`SKIPPED [1] ...: lid.176 weights are not installed` and
`SKIPPED [8] ...: lid218e weights are not installed`, and
`tests/conftest.py` adds a line per model, `LID lid.176: 0 ran against
real weights, 1 skipped` and `LID lid218e: 0 ran against real weights,
8 skipped` there, read from markers the tests carry, so a run with the
weights says how many ran as well. `tests/test_lid_summary.py` copies that
conftest beside marked tests that pass, skip and fail, runs real pytest
over them with two xdist workers, and requires the exact lines, so the
count itself is tested and not only printed.

### Changed

**`turkic-clean-corpus` cleans only IPA the installed rules wrote.** The
cleaner drops a token carrying a letter the installed rules cannot emit,
reasoning that such a letter came through untransliterated from foreign
text. That holds only for IPA those rules wrote. On 2026-11-07 a Kazakh
OSCAR file transliterated before 0.5.7, which spells ‹я› as `ja` with an
ASCII `a` and ‹щ› as `ɕː`, went through the 0.5.10 cleaner and lost about
54,700 Kazakh tokens such as `sɑjasɑt` and `sijaqtə`, and about 1,100 with
‹щ›, as foreign, with nothing to say why.

`turkic-translit translit --out-ipa <file>` now writes
`<file>.manifest.json` beside the IPA, naming the language, the rule file,
its `rule_digest` and the SHA-256 of the IPA written
(`corpus/transliteration_record.py`). Before reading or writing anything,
the cleaner refuses a corpus whose record is missing
(`TURKIC_CORPUS_006_TRANSLITERATION_RECORD_MISSING`), names another
language (`_007_`), names other rules than the installed ones (`_008_`,
with both digests), or describes other bytes (`_009_`). IPA written to
standard output carries no record and cannot be cleaned. A corpus
transliterated by an older release is transliterated again from its
source text. `rule_digest` moved from `corpus.clean` to
`turkic_translit.digests`, where the writer and the cleaner both read it.

**No Python file is longer than 600 lines, and the guard enforces it.**
The new `file-length` rule (`scripts/guards/file_length_rules.py`) fails
`make lint` for any file in `src`, `tests` or `scripts` past 600 lines,
naming it and its count, with no allow-list. Six files were past it and
were split by concern, their code moved unchanged: `scripts/guard.py`
into the runner plus `guards/typing_rules.py` and
`guards/exception_rules.py`; `guards/test_quality_rules.py` into itself,
`patching_rules.py` and `translit_test_rules.py`; `web/web_utils.py` into
itself, `downloads.py`, `ui_log.py` and `corpus_download.py`; and the
tests beside them, with the shared `write_test_module` fixture in
`tests/conftest.py` and the shared corpus builder in
`tests/corpus_clean_helpers.py`. Nothing re-exports an old location; every
importer names the new one. The whole guard is run over a long file in
each of `src`, `tests` and `scripts`, and a test asks make which commands
`lint` and `check` run, so a rule that skipped a directory, or a lint that
stopped running the guard, fails a test.

**`make check` writes nothing.** `make lint` ran `ruff check --fix` and
`ruff format`, so the check rewrote the tree it was checking: on
2026-11-04 it changed an expression in a test on the very run that
reported the suite green, a run that therefore passed on code the commit
did not hold. Lint now runs `ruff check` and `ruff format --check`, and
fails on anything Ruff would change; the new `make format` is the one
target that rewrites. `tests/test_makefile_targets.py` asks real make,
with `-n`, which ruff commands `lint`, `check` and `format` would run, so
a `--fix` restored to lint fails a test instead of passing on a tree that
happens to be clean.

**The download janitor stops on a filesystem error.** `sweep_once`
caught `OSError`, logged it, and the janitor swept again an interval
later, on the grounds that a file the browser held open could not be
deleted. That is Windows behaviour; where deleting an open file
succeeds, an error here is a real fault, and retrying it every interval
only hid it. `sweep_once` is gone: `run_janitor` calls
`purge_expired_downloads` directly, and an error ends the janitor thread
at `threading.excepthook`, which reports it with its traceback; a test
starts the real thread on a blocked directory and reads the error at that
hook, named `cron-janitor`.

**The guard refuses a handler that logs and carries on.** A specific
`except` passed the exceptions rule if it only logged, so the old
`sweep_once` shape, logging a sweep's error and sweeping again, would have
passed it. Such a handler must now re-raise, or log and return a value
built from the error it caught (bound with `as`, used directly or through
a name assigned from it), the way a Gradio handler returns the message
the page shows. One that logs and falls through, or returns nothing, is
reported as `except-logs-and-continues`; one that logs and returns a
default the failure did not produce, as `sweep_once` returned 0, is
reported as `except-returns-a-default`, and naming the error in the log
call does not excuse it. The rule now reads each handler from the parsed
module, as an `ast.ExceptHandler`, rather than scanning lines. The line
scanner missed the old `sweep_once` body when its header carried a
comment after the colon, when its types spanned lines, and when the word
`raise` stood in a body comment or the log message; in the tree a comment
is gone and only a `raise` statement counts. A handler's body is its own
statements, so a return after the `try` statement is not the handler's,
and a `raise` inside a function defined in the handler does not count
either. A logger may be named `logging`, `log` or `logger`, with any
leading underscores. Every way a handler can end is judged, read
through `if`, loops, `with`, `try` and `match`: a `raise` anywhere in
the body used to excuse all of it, so a handler re-raising only some
errors could log and return 0 for a held file, and a broad one raising
only when fatal could log and carry on. Each ending must now re-raise,
or, in a specific handler, return a value built from the error, and a
`break` or `continue` out of a handler is carrying on. Two handlers in `src` returned such a
default and now return the error: the reachability probe answers
`unreachable_reason(url)`, `None` or the request's error, in place of a
bare False, so `download-corpus doctor` names each unreachable source
with its cause; and the corpus tab's unreachable-source label carries the
error it caught.

### Removed

**The Windows code.** `scripts/run.ps1`, the PowerShell copy of the
Makefile targets that 0.5.10 still kept, is gone, and so is
`scripts/release/`: `release.ps1` and `release.sh` uploaded with twine by
hand, under the deprecated `turkic-transliterate` name, while the release
has been a push to `main` since 0.5.3, so they were a second release path
that no longer described the first. `turkic-translit translit` wrote a
byte-order mark when run on Windows and now writes the same plain UTF-8
everywhere, pinned by a test that reads the output file's bytes; the
README, `scripts/README.md`, the Makefile and `scripts/setup_dev.py` no
longer give Windows instructions.

Four more Windows provisions are gone with it. The `make` dev
dependency, listed as the make command for Windows, was PyPI's `make`
0.1.6, a jinja2 project-template tool that never supplied GNU Make; it
and its own dependency `jinja2-time` leave the lock file. The guard
reader decoded `utf-8-sig` to absorb the byte-order mark Windows editors
write; it now reads plain UTF-8 and refuses a file that opens with a
mark, which would otherwise begin its first line with a zero-width
no-break space and slip past every rule matching on column one.
`sitecustomize.py` and `tests/__init__.py` no longer force `PYTHONUTF8`,
which existed for the Windows console's codepage; every file this
project reads or writes names its encoding. And `scripts/_test_hooks.py`
no longer offers the interpreter's platform name: `setup_dev.py` asked
for it only to bootstrap PyICU on Windows, and has not asked since that
step went in b19ccfd. `.editorconfig` drops its `[*.{ps1,sh}]` section,
since the repository tracks neither kind of script.

**A guard keeps it out.** `scripts/guards/windows_rules.py` adds the
`windows` rule, which reads every file git tracks, this changelog
excepted, and fails `make lint` on a file carrying the `.ps1` extension
and on any line naming PowerShell, that extension (dotted or bare, as in
an editor glob), Chocolatey or winget. The guard therefore runs only in
a git work tree, which the fleet's one-commit export of a checked commit
also is.

### Fixed

**Fine-tuning runs on a GPU.** `LMModel.fresh` loaded the base model in
half precision on CUDA and then asked the Trainer for `fp16`, which is
mixed precision: float32 master weights under half-precision arithmetic.
The gradient scaler refuses gradients of weights already loaded in half,
so every fine-tune on a CUDA machine stopped at its first optimiser step
with `ValueError: Attempting to unscale FP16 gradients`. Training now
loads float32 on every device and the trained model is float32, which a
test asserts; loading a model for inference still uses half precision on
CUDA. The suite had passed only because the machines it ran on had no
GPU, and the first run on one (fleet job 34236407 on sedona) failed
its four fine-tuning tests.

## [0.5.10] - 2026-11-04

This release carries the rule fixes made since 0.5.9, which never reached
PyPI because the version was not raised when they were pushed: among
them Kazakh, Kyrgyz and Cyrillic Uzbek ‹щ› now give `ʃː` where 0.5.9
gives `ɕː` (5c06eef), and the Russian rules' palatalisation, soft sign
and pre-1918 letters. Install 0.5.10 before building a corpus.

### Fixed

**`make check` passed only on a machine that had run the suite before.**
Run on a fresh install for the first time on 2026-11-04 (pendragon,
NixOS), it failed in four places, each on state a long-used checkout had
accumulated and a new one lacks:

- Mypy reported `attr-defined` on every Gradio event listener
  (`Button.click`, `Textbox.change`): Gradio adds those methods at
  runtime and writes `.pyi` stubs describing them into its own install
  directory on first import, so they existed only once something had
  imported Gradio. `web/events.py` reaches the listeners through a
  Protocol instead, and `tests/test_web_events.py` registers handlers on
  real components through it.
- `web_demo._model_check` found the language-identification model
  through the `lid` probe and then sized it with `Path.stat()`, so the
  tests that bind the probe passed only where the 131 MB model had been
  downloaded into the source tree. It now sizes it through the probe, as
  `lid.factory` does.
- `test_representations_of_different_width_can_be_compared` asserted
  that a CKA score whose exact value is 1 was at most 1.0; NumPy's BLAS
  on that machine returned `1.0000000000000002`. It now asserts the
  exact value, within rounding.
- panphon 0.22's `segment.py` has an invalid escape in a docstring,
  which Python warns about while compiling it; under the suite's
  warnings-as-errors that was a `SyntaxError` wherever panphon had not
  already been compiled. The warning is ignored for that one file, by
  path, since a compile-time warning names its file rather than its
  package.

**A rule file's fingerprint depended on the checkout's line endings.**
`rules_fingerprint` hashed each rule file's raw bytes, and
`.gitattributes` declared `*.rules text` without a line ending, so git
gave the files CRLF on Windows and LF on Linux and one commit
fingerprinted two ways. The transliterator never saw the difference:
`core` reads rule files with `read_text`, which turns CRLF into LF
before ICU compiles them, so no corpus was transliterated wrongly. What
broke was the provenance check: a Linux rerun of the commit a Windows
manifest named read as different rules.

The new `rule_digest` hashes the decoded text, which is what ICU
compiles, so every checkout of a commit gives one digest, the one an LF
checkout's bytes already gave. `.gitattributes` now declares
`*.rules text eol=lf`, so every checkout holds LF bytes as well.
`tests/test_rule_digest.py` writes both forms of `kk_ipa.rules` and
pins them to one digest.

**Manifests written before this change on Windows carry the CRLF
digest.** The LSTM corpora `rebuild_2026-09/corpora_clean_v4`, `v5` and
`v6` record `kk_ipa.rules` as
`c2d4eeec01032ba6f7fc1cb38c27c86513e8f8fd4d1818d9a146a1e60948dc26`. That
is the CRLF form of commit 5c06eef's `kk_ipa.rules`, whose LF form, and
so its digest under this change, is
`2b69fda41a28786b75c449184197b5209531b59cc23561226113380ac1f9e9c8`. To
match an old Windows manifest, hash the rule file's bytes with CRLF line
endings; to match a Linux one, or any manifest written from now on, use
`rule_digest`.

### Changed

**The Makefile runs under a POSIX shell.** It ran every recipe through
`powershell.exe`, so `make check` could not run on a Linux machine.
Windows users keep `scripts/run.ps1`, which runs the same targets.

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
