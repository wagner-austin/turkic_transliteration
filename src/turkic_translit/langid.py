import fasttext
import os

class FastTextLangID:
    """
    Wrapper for fastText language identification.
    Loads a model from the given path or defaults to a packaged model.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # Default: look for model in the same directory as this file
            model_path = os.path.join(os.path.dirname(__file__), 'lid.176.bin')
        self.model = fasttext.load_model(model_path)

    def predict(self, text):
        # Remove SentencePiece underline and whitespace
        clean_text = text.replace('▁', '').strip()
        label = self.model.predict(clean_text)[0][0]  # e.g. '__label__ru'
        return label.replace('__label__', '')

    def predict_tokens(self, tokens):
        """
        Predict language for a list of tokens. Returns a list of language codes.
        """
        return [self.predict(token) for token in tokens]
