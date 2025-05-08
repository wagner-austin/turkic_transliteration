"""
Demo script to show the transliteration functionality visually.
"""
from turkic_translit.core import to_latin, to_ipa

# Sample texts in Kazakh and Kyrgyz Cyrillic
sample_texts = {
    "kk": "Қазақ тілі - Түркі тілдерінің бірі. Бұл тілде шамамен 15 миллион адам сөйлейді.",
    "ky": "Кыргыз тили - Түрк тилдеринин бири. Бул тилде болжол менен 6 миллион адам сүйлөйт."
}

def demo_transliteration():
    print("=" * 80)
    print("TURKIC TRANSLITERATION DEMO")
    print("=" * 80)
    
    for lang, text in sample_texts.items():
        print(f"\n{lang.upper()} (Kazakh/Kyrgyz) TRANSLITERATION EXAMPLE:")
        print("-" * 50)
        
        # Original Cyrillic text
        print(f"ORIGINAL (Cyrillic): {text}")
        
        # Latin transliteration
        latin = to_latin(text, lang)
        print(f"LATIN SCRIPT:        {latin}")
        
        # IPA transliteration
        ipa = to_ipa(text, lang)
        print(f"IPA PRONUNCIATION:   {ipa}")
        
        print("-" * 50)

if __name__ == "__main__":
    demo_transliteration()
