# This file was moved from the project root to examples/
# Usage: pip install turkic-transliterate[examples] to ensure dependencies

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")  # Ensure local import for demonstration
    from turkic_translit.pipeline import TurkicTransliterationPipeline
    print("Demo: Transliterate sample text.")
    pipeline = TurkicTransliterationPipeline()
    text = "Привет, сәлем!"
    print("Input:", text)
    print("Output:", pipeline.process(text))
