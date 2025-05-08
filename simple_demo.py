"""
Simple demo to test the transliteration functionality visually with explicit imports.
"""

# Import directly from the core module with explicit imports
import sys
from turkic_translit.core import to_latin, to_ipa

def main():
    # Sample text in Kazakh Cyrillic
    text_kk = "Қазақ тілі - Түркі тілдерінің бірі."
    
    print("=" * 60)
    print("TURKIC TRANSLITERATION DEMO")
    print("=" * 60)
    
    # Kazakh
    print("\nKAZAKH TRANSLITERATION EXAMPLE:")
    print(f"Original (Cyrillic): {text_kk}")
    
    try:
        # Try Latin transliteration
        latin = to_latin(text_kk, "kk")
        print(f"Latin script:      {latin}")
    except Exception as e:
        print(f"Latin error: {e}")
    
    try:
        # Try IPA transliteration
        ipa = to_ipa(text_kk, "kk")
        print(f"IPA pronunciation: {ipa}")
    except Exception as e:
        print(f"IPA error: {e}")

if __name__ == "__main__":
    main()
