from .core import get_supported_languages, to_ipa, to_latin


def transliterate_token(token: str, lang: str, mode: str = "latin") -> str:
    """Transliterate a token to Latin or IPA.

    Delegates to :func:`core.to_ipa` or :func:`core.to_latin` for any
    ``(lang, mode)`` combination supported by the rules directory (discovered
    dynamically via :func:`core.get_supported_languages`). Returns ``token``
    unchanged for Russian or any ``(lang, mode)`` combination without a rule
    file — this preserves the pipeline pass-through behaviour needed when
    tokens in a mixed-language stream fall outside the target set.

    Args:
        token: The input token in its native orthography.
        lang: ISO 639-1 language code (e.g. ``"kk"``, ``"tr"``, ``"fi"``).
        mode: Either ``"latin"`` or ``"ipa"``.

    Returns:
        The transliterated token, or ``token`` unchanged when no matching
        rule file is available.

    Raises:
        ValueError: If ``mode`` is not ``"latin"`` or ``"ipa"``.
    """
    if mode not in ("latin", "ipa"):
        raise ValueError(f"Unknown transliteration mode: {mode}")
    if lang == "ru":
        return token
    supported = get_supported_languages()
    if lang in supported and mode in supported[lang]:
        return to_ipa(token, lang) if mode == "ipa" else to_latin(token, lang)
    return token
