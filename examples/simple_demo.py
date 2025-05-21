"""
Simple demo to test the transliteration functionality visually with explicit imports.
"""

# Import logging framework first
import logging
import os
import pathlib
import sys

# Add the parent project directory to the path when running as an example
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Set log level via environment variable before importing setup
os.environ["TURKIC_LOG_LEVEL"] = "INFO"

from turkic_translit.core import to_ipa, to_latin
from turkic_translit.logging_config import setup

# Call setup without arguments - it reads from environment variables
setup()

# Set up logger for this module
logger = logging.getLogger(__name__)


def main() -> None:
    # Sample text in Kazakh Cyrillic
    text_kk = "Қазақ тілі - Түркі тілдерінің бірі."

    logger.info("=" * 60)
    logger.info("TURKIC TRANSLITERATION DEMO")
    logger.info("=" * 60)

    # Kazakh
    logger.info("\nKAZAKH TRANSLITERATION EXAMPLE:")
    logger.info("Original (Cyrillic): %s", text_kk)

    try:
        # Try Latin transliteration
        latin = to_latin(text_kk, "kk")
        logger.info("Latin script:      %s", latin)
    except Exception as e:
        logger.error("Latin error: %s", e)

    try:
        # Try IPA transliteration
        ipa = to_ipa(text_kk, "kk")
        logger.info("IPA pronunciation: %s", ipa)
    except Exception as e:
        logger.error("IPA error: %s", e)


if __name__ == "__main__":
    main()
