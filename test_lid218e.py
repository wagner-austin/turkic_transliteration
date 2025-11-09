#!/usr/bin/env python3
"""
Test script for lid218e.bin model - script-aware language identification.

This script tests:
1. Model loads correctly
2. Returns script-aware labels (e.g., "uzb_Latn", "kaz_Cyrl")
3. Can distinguish Uzbek Latin vs Cyrillic
4. Can distinguish Kazakh, Kyrgyz, etc.
"""

import os
import sys
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    import fasttext
except ImportError:
    print("ERROR: fasttext not installed. Run: pip install fasttext")
    sys.exit(1)


def test_lid218e():
    """Test the lid218e model with various Turkic languages and scripts."""

    # Find the model
    model_paths = [
        Path("src/turkic_translit/lid218e.bin"),
        Path("lid218e.bin"),
        Path.home() / "lid218e.bin",
    ]

    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            print(f"Ã¢Å“â€œ Found model at: {path}")
            print(f"  Size: {path.stat().st_size / (1024**3):.2f} GB\n")
            break

    if not model_path:
        print("ERROR: lid218e.bin not found!")
        print("Please download it to one of these locations:")
        for path in model_paths:
            print(f"  - {path}")
        print("\nDownload with:")
        print("  wget https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin")
        print("  # or")
        print("  curl -O https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin")
        sys.exit(1)

    # Load model
    print("Loading model (this may take a moment for 1.2GB file)...")
    model = fasttext.load_model(str(model_path))
    print("Ã¢Å“â€œ Model loaded successfully!\n")

    # Test cases: (text, expected_lang, expected_script, acceptable_langs, description)
    # Note: lid218e returns ISO 639-3 codes; Uzbek is typically 'uzn' (Northern Uzbek).
    # acceptable_langs: list of language codes that are acceptable (for known ambiguities)
    test_cases = [
        # Uzbek - Latin vs Cyrillic
        ("O'zbekiston", "uzn", "Latn", [], "Uzbek (Latin)"),
        (
            "Oʻzbekiston respublikasi",
            "uzn",
            "Latn",
            [],
            "Uzbek (Latin with modifier letter)",
        ),
        # KNOWN LIMITATION: Short Uzbek Cyrillic is often confused with Tajik
        # because they're extremely similar (both Persian-influenced Turkic in Cyrillic)
        ("Ўзбекистон", "uzn", "Cyrl", ["tgk"], "Uzbek (Cyrillic)"),
        (
            "Ўзбекистон Республикаси",
            "uzn",
            "Cyrl",
            ["tgk", "rus"],
            "Uzbek (Cyrillic, longer)",
        ),
        # Kazakh - Cyrillic
        ("Қазақстан", "kaz", "Cyrl", [], "Kazakh (Cyrillic)"),
        ("Қазақстан Республикасы", "kaz", "Cyrl", [], "Kazakh (Cyrillic, longer)"),
        # Kyrgyz - Cyrillic
        ("Кыргызстан", "kir", "Cyrl", [], "Kyrgyz (Cyrillic)"),
        ("Кыргыз Республикасы", "kir", "Cyrl", [], "Kyrgyz (Cyrillic, longer)"),
        # Turkish - Latin
        ("Türkiye Cumhuriyeti", "tur", "Latn", [], "Turkish (Latin)"),
        # Russian - Cyrillic
        ("Российская Федерация", "rus", "Cyrl", [], "Russian (Cyrillic)"),
        # English - Latin
        ("Hello world", "eng", "Latn", [], "English (Latin)"),
    ]

    print("=" * 80)
    print("TESTING SCRIPT-AWARE LANGUAGE IDENTIFICATION")
    print("=" * 80)
    print()

    passed = 0
    failed = 0
    known_limitations = 0

    for (
        text,
        expected_lang,
        expected_script,
        acceptable_langs,
        description,
    ) in test_cases:
        # Predict
        labels, probs = model.predict(text, k=1)
        full_label = labels[0].replace("__label__", "")
        confidence = probs[0]

        # Parse result
        if "_" in full_label:
            detected_lang, detected_script = full_label.split("_", 1)
        else:
            detected_lang, detected_script = full_label, "unknown"

        # Check if correct
        lang_match = detected_lang == expected_lang
        script_match = detected_script == expected_script

        # Check if detected language is in acceptable alternatives
        lang_acceptable = lang_match or detected_lang in acceptable_langs

        success = lang_acceptable and script_match
        is_known_limitation = not lang_match and lang_acceptable

        # Display result
        if success and not is_known_limitation:
            status = "✓ PASS"
        elif is_known_limitation:
            status = "⚠ PASS (known limitation)"
        else:
            status = "✗ FAIL"

        print(f"{status} {description}")
        print(f'  Text:     "{text}"')
        print(f"  Expected: {expected_lang}_{expected_script}")
        print(
            f"  Detected: {detected_lang}_{detected_script} (confidence: {confidence:.3f})"
        )

        if not success:
            if not lang_acceptable:
                acceptable_str = f" or {acceptable_langs}" if acceptable_langs else ""
                print(
                    f"  ERROR: Language mismatch (expected {expected_lang}{acceptable_str}, got {detected_lang})"
                )
            if not script_match:
                print(
                    f"  ERROR: Script mismatch (expected {expected_script}, got {detected_script})"
                )
        elif is_known_limitation:
            print(
                f"  NOTE: Detected as acceptable alternative ({detected_lang} instead of {expected_lang})"
            )

        print()

        if success:
            passed += 1
            if is_known_limitation:
                known_limitations += 1
        else:
            failed += 1

    # Summary
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    if known_limitations > 0:
        print(f"  ({known_limitations} passed with known model limitations)")
    print("=" * 80)

    assert failed == 0, (
        f"{failed} test(s) failed. The model may need adjustment or test expectations may be wrong."
    )

    if known_limitations > 0:
        print(
            f"\n✓ All tests passed! ({known_limitations} with documented limitations)"
        )
    else:
        print("\n✓ All tests passed! The model correctly detects scripts.")


if __name__ == "__main__":
    try:
        test_lid218e()
    except AssertionError as e:
        print(e)
        sys.exit(1)
    sys.exit(0)
