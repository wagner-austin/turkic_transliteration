# This file was moved from the project root to examples/
# Usage: pip install turkic-transliterate[examples] to ensure dependencies

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from turkic_translit.pipeline import TurkicTransliterationPipeline
    pipeline = TurkicTransliterationPipeline()
    print(pipeline.process("сәлем әлем!"))
