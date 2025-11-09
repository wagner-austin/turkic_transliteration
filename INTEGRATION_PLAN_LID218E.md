# Integration Plan: lid218e.bin (Script-Aware Language Model)

## Overview
This document lists ALL code changes needed to integrate the lid218e.bin model which returns script-aware language codes (e.g., `kaz_Cyrl`, `uzn_Latn` instead of just `kk`, `uz`).

## Summary Statistics
- **Total Python files in project:** 56
- **Files requiring changes:** 23
- **Existing rule files:** 9
- **New rule files needed:** 0 (already have uz_ipa_cyr.rules, uz_ipa_lat.rules!)

---

## CRITICAL DECISION: Backwards Compatible vs Breaking Changes

### Option A: Backwards Compatible (RECOMMENDED)
- Keep existing API unchanged
- Parse script internally but expose simple interface
- All existing code continues to work
- Estimated changes: **~100 lines across 5 files**

### Option B: Breaking Changes
- Change all APIs to return/accept (lang, script) tuples
- Update all callsites
- Rename all rule files
- Estimated changes: **~800-1000 lines across 23 files**

---

## RECOMMENDED APPROACH: Backwards Compatible

### Phase 1: Core Model Support (5 files)

#### 1.1 `src/turkic_translit/model_utils.py`
**Changes:**
- Add `FASTTEXT_MODEL_218E_URL` constant
- Add `download_fasttext_218e_model()` function
- Update `ensure_fasttext_model()` to check for both models
- Prefer lid218e if both exist

**Lines changed:** ~40

**Details:**
```python
# Add after line 51:
FASTTEXT_MODEL_218E_URL = (
    "https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin"
)

# Add new function
def ensure_fasttext_model(prefer_218e: bool = True) -> tuple[pathlib.Path, str]:
    """Returns (model_path, model_type) where model_type is '176' or '218e'"""
    # Check for lid218e.bin first if prefer_218e=True
    # Fall back to lid.176.bin
    # Return tuple indicating which model was found
```

---

#### 1.2 `src/turkic_translit/langid.py`
**Changes:**
- Detect which model is loaded (176 vs 218e)
- Add internal `_parse_label()` method to handle both formats
- Keep existing `predict()` signature returning `str`
- Add new `predict_with_script()` method returning `tuple[str, str]`
- Map ISO 639-3 back to ISO 639-1 for backwards compat

**Lines changed:** ~60

**Details:**
```python
class FastTextLangID:
    def __init__(self, model_path: str | None = None) -> None:
        # ... existing code ...
        self.model = fasttext.load_model(model_path)

        # NEW: Detect model type
        self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """Detect if model is lid.176 or lid218e by checking output format"""
        test_result = self.model.predict("test", k=1)
        label = test_result[0][0].replace("__label__", "")
        return "218e" if "_" in label else "176"

    def _parse_label(self, label: str) -> tuple[str, str]:
        """Parse label into (lang, script). Returns (lang, 'unknown') for lid.176"""
        label = label.replace("__label__", "")

        if "_" in label:  # lid218e format: "kaz_Cyrl"
            lang_639_3, script = label.split("_", 1)
            # Map ISO 639-3 to ISO 639-1 for backwards compat
            lang = self._map_to_639_1(lang_639_3)
            return lang, script.lower()
        else:  # lid.176 format: "kk"
            return label, "unknown"

    def _map_to_639_1(self, code_639_3: str) -> str:
        """Map ISO 639-3 to ISO 639-1 codes"""
        mapping = {
            "kaz": "kk",  # Kazakh
            "kir": "ky",  # Kyrgyz
            "tur": "tr",  # Turkish
            "uzn": "uz",  # Northern Uzbek
            "uzs": "uz",  # Southern Uzbek
            "tgk": "tg",  # Tajik
            "rus": "ru",  # Russian
            "eng": "en",  # English
            # Add more as needed
        }
        return mapping.get(code_639_3, code_639_3)

    def predict(self, text: str) -> str:
        """UNCHANGED - returns language code only for backwards compat"""
        clean_text = text.replace("\u2581", "").strip()
        label = cast(str, self.model.predict(clean_text)[0][0])
        lang, _ = self._parse_label(label)  # Discard script
        return lang

    def predict_with_script(self, text: str) -> tuple[str, str]:
        """NEW - returns (language, script) tuple"""
        clean_text = text.replace("\u2581", "").strip()
        label = cast(str, self.model.predict(clean_text)[0][0])
        return self._parse_label(label)
```

---

#### 1.3 `src/turkic_translit/core.py`
**Changes:**
- Update `get_supported_languages()` to parse script-aware filenames
- Keep backwards compat with existing `kk_ipa.rules` format
- Support new format like `uz_ipa_lat.rules` or `uzn_Latn_ipa.rules`
- Update `to_ipa()` and `to_latin()` to accept optional `script` parameter

**Lines changed:** ~50

**Details:**
```python
def get_supported_languages() -> dict[str, list[str]]:
    """
    Scan rules and return {lang: [formats]}

    Supports multiple filename patterns:
    - kk_ipa.rules       → kk supports ipa
    - uz_ipa_lat.rules   → uz supports ipa (latin variant)
    - uzn_Latn_ipa.rules → uz supports ipa (explicit script)
    """
    supported: dict[str, list[str]] = {}

    for rule_file in _RULE_DIR.glob("*.rules"):
        filename = rule_file.stem
        parts = filename.split("_")

        if len(parts) >= 2:
            # Pattern 1: kk_ipa.rules
            if len(parts) == 2:
                lang, fmt = parts
                # Normalize
                if fmt in ("lat", "lat2023"):
                    fmt = "latin"

            # Pattern 2: uz_ipa_lat.rules or uzn_Latn_ipa.rules
            elif len(parts) == 3:
                # Could be: lang_format_script OR lang_script_format
                # Heuristic: if last part is ipa/lat/latin, it's format
                if parts[2] in ("ipa", "lat", "latin", "lat2023"):
                    lang, fmt, script = parts[0], parts[2], parts[1]
                else:
                    lang, script, fmt = parts

                if fmt in ("lat", "lat2023"):
                    fmt = "latin"

            # Map ISO 639-3 to ISO 639-1 if needed
            lang = _map_to_639_1(lang)

            if lang not in supported:
                supported[lang] = []
            if fmt not in supported[lang]:
                supported[lang].append(fmt)

    return supported

def to_ipa(text: str, lang: str, script: str = None) -> str:
    """
    Transliterate to IPA, optionally considering script.

    Args:
        text: Input text
        lang: Language code (ISO 639-1 or 639-3)
        script: Optional script ("cyrl", "latn", etc.)
    """
    # Try script-specific rule file first if script provided
    if script:
        possible_files = [
            f"{lang}_ipa_{script}.rules",      # uz_ipa_lat.rules
            f"{lang}_{script}_ipa.rules",      # uz_lat_ipa.rules
            # Also try ISO 639-3 variants
            f"{_map_to_639_3(lang)}_ipa_{script}.rules",
            f"{_map_to_639_3(lang)}_{script}_ipa.rules",
        ]

        for rule_file in possible_files:
            path = _RULE_DIR / rule_file
            if path.exists():
                trans = _icu_trans(rule_file)
                return ud.normalize("NFC", trans.transliterate(text))

    # Fall back to script-agnostic rule file
    rule_file = f"{lang}_ipa.rules"
    if not (_RULE_DIR / rule_file).exists():
        raise ValueError(f"IPA rules not found for '{lang}'")

    trans = _icu_trans(rule_file)
    return ud.normalize("NFC", trans.transliterate(text))
```

---

#### 1.4 `src/turkic_translit/transliterate.py`
**Changes:**
- Update `transliterate_token()` to accept optional `script` parameter
- Pass script to `to_ipa()` and `to_latin()`
- Add Uzbek to supported languages

**Lines changed:** ~15

**Details:**
```python
def transliterate_token(token: str, lang: str, mode: str = "latin", script: str = None) -> str:
    """
    Transliterate a token to Latin or IPA depending on mode and language.

    Args:
        token: Input token
        lang: Language code
        mode: 'latin' or 'ipa'
        script: Optional script hint ('cyrl', 'latn', etc.)
    """
    if lang in ("kk", "ky", "uz"):  # Added uz
        if mode == "latin":
            return to_latin(token, lang, script)
        if mode == "ipa":
            return to_ipa(token, lang, script)
        raise ValueError(f"Unknown transliteration mode: {mode}")
    if lang == "ru":
        return token
    return token
```

---

#### 1.5 `src/turkic_translit/pipeline.py`
**Changes:**
- Use `predict_with_script()` when available
- Pass script to `transliterate_token()`

**Lines changed:** ~10

**Details:**
```python
def process(self, text: str) -> str:
    tokens = self.tokenizer.tokenize(text)

    # Check if model supports script detection
    if hasattr(self.langid, 'predict_with_script'):
        # Use script-aware prediction
        lang_script_pairs = [self.langid.predict_with_script(token) for token in tokens]
        transliterated = [
            transliterate_token(token, lang, self.mode, script)
            for token, (lang, script) in zip(tokens, lang_script_pairs)
        ]
    else:
        # Fall back to simple prediction
        langs = self.langid.predict_tokens(tokens)
        transliterated = [
            transliterate_token(token, lang, self.mode)
            for token, lang in zip(tokens, langs)
        ]

    return self.tokenizer.detokenize(transliterated)
```

---

### Phase 2: Tests Updates (Minimal - 3 files)

#### 2.1 `tests/test_model_utils.py`
**Changes:**
- Update tests to handle both model types
- Test that `ensure_fasttext_model()` returns correct model type

**Lines changed:** ~20

---

#### 2.2 `tests/test_langid_fasttext.py`
**Changes:**
- Add tests for `predict_with_script()` method
- Test ISO 639-3 to 639-1 mapping

**Lines changed:** ~30

---

#### 2.3 Add new test file: `tests/test_lid218e.py`
**Changes:**
- Copy and adapt `test_lid218e.py` from project root
- Test script detection specifically

**Lines changed:** +140 (new file)

---

### Phase 3: Documentation (2 files)

#### 3.1 `README.md`
**Changes:**
- Document lid218e support
- Explain when to use which model
- Add section on script-aware detection

**Lines changed:** ~30

---

#### 3.2 `docs/` (if exists)
**Changes:**
- Update API documentation
- Add migration guide

**Lines changed:** TBD

---

## Files That DON'T Need Changes (Backwards Compatible)

✅ **No changes needed for:**
- `src/turkic_translit/cli/*.py` (11 files) - works with existing API
- `src/turkic_translit/web/*.py` (multiple files) - works with existing API
- `tests/test_*.py` (most test files) - continue to work
- All existing rule files

---

## Optional Enhancements (Phase 4)

These are NOT required but could be added later:

#### 4.1 CLI flag to choose model
```bash
turkic-translit --model lid218e ...
```

#### 4.2 Expose script in web UI
Show detected script in Gradio interface

#### 4.3 Create ISO 639-3 rule files
Rename/copy existing rules to use ISO 639-3 codes:
- `kk_ipa.rules` → also support `kaz_Cyrl_ipa.rules`
- `ky_ipa.rules` → also support `kir_Cyrl_ipa.rules`

---

## Migration Path

### Step 1: Implement Phase 1 (Core changes - 5 files, ~175 lines)
- All existing code continues to work
- New capabilities available via `predict_with_script()`

### Step 2: Test with existing tests
- Run full test suite
- Everything should pass

### Step 3: Add new tests for script detection (Phase 2)
- Validate lid218e specific features

### Step 4: Update documentation (Phase 3)

### Step 5: Optional enhancements (Phase 4) - future work

---

## Risk Assessment

### Low Risk ✅
- Backwards compatible approach
- Existing API unchanged
- All existing tests continue to pass
- No breaking changes for users

### Moderate Risk ⚠️
- ISO code mapping could miss some languages
- Model file size (1.2 GB) could be issue for some deployments
- Uzbek Cyrillic detection issues (may detect as Tajik)

### Mitigation Strategies
1. Make lid218e optional - fall back to lid.176 if not present
2. Document known limitations (Uzbek Cyrillic → Tajik)
3. Provide character-based overrides for problematic cases

---

## Testing Checklist

Before merging:
- [ ] All existing tests pass
- [ ] New script detection tests pass
- [ ] Uzbek Latin detection works (uzn_Latn)
- [ ] Kazakh Cyrillic detection works (kaz_Cyrl)
- [ ] Kyrgyz Cyrillic detection works (kir_Cyrl)
- [ ] Fallback to lid.176 works if lid218e not present
- [ ] Pipeline processes mixed-script text correctly
- [ ] Web demo still works
- [ ] CLI tools still work

---

## Estimated Implementation Time

- **Phase 1 (Core):** 2-3 hours
- **Phase 2 (Tests):** 1-2 hours
- **Phase 3 (Docs):** 1 hour
- **Testing & debugging:** 2-3 hours

**Total:** 6-9 hours

---

## Questions to Resolve

1. ✅ **Model location:** `src/turkic_translit/lid218e.bin` (already there!)
2. ✅ **Rule files:** Already have `uz_ipa_cyr.rules` and `uz_ipa_lat.rules`
3. ❓ **Default model:** Should lid218e be default if present?
4. ❓ **Handle Tajik→Uzbek confusion:** Add character-based override?
5. ❓ **Support both ISO 639-1 and 639-3 rule filenames:** Worth it?

---

## Summary

**Recommended approach: Backwards Compatible Integration**

- **Pros:**
  - Minimal code changes (~175 lines)
  - No breaking changes
  - Existing code continues to work
  - Easy to test incrementally
  - Can add more features later

- **Cons:**
  - Slightly more complex internally (needs mapping layer)
  - Can't take full advantage of script codes in external APIs (yet)

**This approach lets you start using lid218e immediately while keeping all existing functionality intact.**
