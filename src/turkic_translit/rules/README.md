# Adding New Language Support

To add transliteration support for a new language, simply add rules files to this directory following the naming convention:

## File Naming Convention

- **IPA rules**: `{lang_code}_ipa.rules`
- **Latin rules**: `{lang_code}_lat.rules` or `{lang_code}_lat2023.rules` or `{lang_code}_latin.rules`

Where `{lang_code}` is the ISO 639-1 language code (e.g., `tr` for Turkish, `kk` for Kazakh).

## Example

To add Uzbek support:
1. Create `uz_ipa.rules` for IPA transliteration
2. Create `uz_lat.rules` for Latin transliteration
3. The language will automatically appear in the UI!

## Rules File Format

Rules files use ICU transliterator syntax. See existing files for examples:
- `kk_ipa.rules` - Kazakh to IPA
- `tr_ipa.rules` - Turkish to IPA
- `kk_lat2023.rules` - Kazakh to Latin

## Testing

After adding new rules files:
1. Restart the web UI
2. The new language should appear in the language selection dropdown
3. Test with sample text to ensure rules work correctly

No code changes needed - the system automatically detects new rules files!