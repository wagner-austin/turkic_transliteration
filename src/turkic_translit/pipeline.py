"""Tokenise, identify each token's language, then transliterate.

The language of a token is decided by an explicitly named classifier, not
by whichever weights happened to be resolvable, so two runs of this
pipeline over the same text agree.
"""

from __future__ import annotations

from turkic_translit.lid.classifier import LidClassifier
from turkic_translit.lid.factory import load_installed_classifier
from turkic_translit.tokenizer import TurkicTokenizer
from turkic_translit.transliterate import transliterate_token

DEFAULT_LANGUAGE_MODEL_ID = "lid.176"


class TurkicTransliterationPipeline:
    """Orchestrates tokenisation, language identification and transliteration.

    Args:
        sp_model_path: SentencePiece model to tokenise with, or ``None``
            for the packaged model.
        model_id: Registry key of the classifier deciding each token's
            language.
        mode: Either ``latin`` or ``ipa``.
    """

    def __init__(
        self,
        sp_model_path: str | None = None,
        model_id: str = DEFAULT_LANGUAGE_MODEL_ID,
        mode: str = "latin",
    ) -> None:
        """Load the tokeniser and the named classifier."""
        self.tokenizer = TurkicTokenizer(sp_model_path)
        self.classifier: LidClassifier = load_installed_classifier(model_id)
        self.mode = mode

    def predict_tokens(self, tokens: list[str]) -> list[str]:
        """Identify the language of each token.

        A SentencePiece piece that carries only the word-boundary marker
        has no text to classify. It is reported as an empty language
        rather than being passed to the classifier, which would raise.

        Args:
            tokens: Tokens as the tokeniser produced them.

        Returns:
            One language label per token, in the same order.
        """
        labels: list[str] = []
        for token in tokens:
            stripped = token.replace("▁", "").strip()
            labels.append("" if stripped == "" else self.classifier.classify(stripped)["label"])
        return labels

    def process(self, text: str) -> str:
        """Transliterate ``text`` token by token.

        Args:
            text: The text to transliterate.

        Returns:
            The transliterated text, detokenised.
        """
        tokens = self.tokenizer.tokenize(text)
        languages = self.predict_tokens(tokens)
        transliterated = [
            transliterate_token(token, language, self.mode)
            for token, language in zip(tokens, languages, strict=False)
        ]
        return self.tokenizer.detokenize(transliterated)


__all__ = ["DEFAULT_LANGUAGE_MODEL_ID", "TurkicTransliterationPipeline"]
