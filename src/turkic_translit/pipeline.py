from .langid import FastTextLangID
from .tokenizer import TurkicTokenizer
from .transliterate import transliterate_token


class TurkicTransliterationPipeline:
    """
    Orchestrates tokenization, language identification, and transliteration.
    Now includes tokenization, language identification, and transliteration.
    """

    def __init__(
        self,
        sp_model_path: str | None = None,
        ft_model_path: str | None = None,
        mode: str = "latin",
    ) -> None:
        self.tokenizer = TurkicTokenizer(sp_model_path)
        self.langid = FastTextLangID(ft_model_path)
        self.mode = mode  # 'latin' or 'ipa'

    def process(self, text: str) -> str:
        """
        Tokenizes text, predicts language for each token, transliterates, and detokenizes.
        Returns the final transliterated string.
        """
        tokens = self.tokenizer.tokenize(text)
        langs = self.langid.predict_tokens(tokens)
        transliterated = [
            transliterate_token(token, lang, self.mode)
            for token, lang in zip(tokens, langs)
        ]
        return self.tokenizer.detokenize(transliterated)
