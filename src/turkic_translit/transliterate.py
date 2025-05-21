from .core import to_ipa, to_latin


def transliterate_token(token: str, lang: str, mode: str = "latin") -> str:
    """
    Transliterate a token to Latin or IPA depending on mode and language.
    For Russian, returns the token unchanged (or you can add logic).
    """
    if lang in ("kk", "ky"):
        if mode == "latin":
            return to_latin(token, lang)
        if mode == "ipa":
            return to_ipa(token, lang)
        raise ValueError(f"Unknown transliteration mode: {mode}")
    if lang == "ru":
        # Optionally, add Russian transliteration logic here
        return token
    return token
