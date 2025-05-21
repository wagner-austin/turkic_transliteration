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
                # First try to load the compressed model which is much smaller
                model_path = str(ensure_fasttext_model())
                logger.info(f"Using FastText model from: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to download FastText model: {e}")
                # Fallback to bin file if it exists
                model_path = os.path.join(os.path.dirname(__file__), "lid.176.bin")
                logger.info(f"Falling back to bin model at: {model_path}")

        self.model = fasttext.load_model(model_path)

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
