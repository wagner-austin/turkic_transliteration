"""
Demo script to show the transliteration functionality visually.
"""

import logging
from turkic_translit.logging_config import setup

from turkic_translit.core import to_latin, to_ipa

setup("INFO")

# Set up logger for this module
logger = logging.getLogger(__name__)

# Sample texts in Kazakh and Kyrgyz Cyrillic
sample_texts = {
    "kk": "Қазақ тілі - Түркі тілдерінің бірі. Бұл тілде шамамен 15 миллион адам сөйлейді.",
    "ky": "Кыргыз тили - Түрк тилдеринин бири. Бул тилде болжол менен 6 миллион адам сүйлөйт.",
}


def demo_transliteration() -> None:
    logger.info("=" * 80)
    logger.info("TURKIC TRANSLITERATION DEMO")
    logger.info("=" * 80)

    for lang, text in sample_texts.items():
        logger.info("\n%s (Kazakh/Kyrgyz) TRANSLITERATION EXAMPLE:", lang.upper())
        logger.info("-" * 50)

        # Original Cyrillic text
        logger.info("ORIGINAL (Cyrillic): %s", text)

        # Latin transliteration
        latin = to_latin(text, lang)
        logger.info("LATIN SCRIPT:        %s", latin)

        # IPA transliteration
        ipa = to_ipa(text, lang)
        logger.info("IPA PRONUNCIATION:   %s", ipa)

        logger.info("-" * 50)


if __name__ == "__main__":
    demo_transliteration()
