import logging
import os
from typing import cast

import fasttext

from .model_utils import ensure_fasttext_model

logger = logging.getLogger(__name__)


class FastTextLangID:
    """
    Wrapper for fastText language identification.
    Loads a model from the given path or defaults to a packaged model.
    If the model doesn't exist, it will be downloaded automatically.
    """

    def __init__(self, model_path: str | None = None) -> None:
        if model_path is None:
            try:
                # Load the full model which is more accurate
                model_path = str(ensure_fasttext_model())
                logger.info(f"Using FastText model from: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to download FastText model: {e}")
                # Try to find a model in the package directory
                model_path = os.path.join(os.path.dirname(__file__), "lid.176.bin")
                logger.info(f"Attempting to use model at: {model_path}")

        self.model = fasttext.load_model(model_path)

    def predict_with_prob(self, text: str) -> tuple[str, float]:
        """Return (language, probability) for the top FastText prediction."""
        clean = text.replace("\u2581", "").strip()
        if not clean:
            logger.warning("Empty text passed to predict_with_prob")
            return "unknown", 0.0
        labels, probs = self.model.predict(clean, k=1)
        lang = cast(str, labels[0]).replace("__label__", "")
        # Log suspicious results
        if lang == "en" and float(probs[0]) == 0.25001001358032227:
            logger.debug(
                f"Suspicious FastText result for '{clean[:50]}...': lang={lang}, prob={probs[0]}"
            )
            # Try with k=5 to see what other predictions it has
            labels5, probs5 = self.model.predict(clean, k=5)
            logger.debug(f"Top 5 predictions: {list(zip(labels5, probs5))}")
        return lang, float(probs[0])

    def predict(self, text: str) -> str:
        # Remove SentencePiece underline and whitespace
        clean_text = text.replace("\u2581", "").strip()
        label = cast(str, self.model.predict(clean_text)[0][0])  # returns Any
        return label.replace("__label__", "")

    def predict_tokens(self, tokens: list[str]) -> list[str]:
        """
        Predict language for a list of tokens. Returns a list of language codes.
        """
        return cast(list[str], [self.predict(token) for token in tokens])
