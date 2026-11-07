"""Language utility helpers shared across UI and CLI modules."""

from __future__ import annotations

from functools import lru_cache

import pycountry

__all__ = ["pretty_lang", "three_letter_code"]

_OVERRIDES = {
    "bh": "Bhojpuri",
}


@lru_cache(maxsize=512)
def pretty_lang(code: str) -> str:
    """Return a human-friendly label such as ``Persian (fa)``.

    Args:
        code: ISO 639-1 or 639-3 language code.

    Returns:
        The language name followed by the code, or the bare code when
        the code is unknown or carries no name. pycountry is a declared
        dependency and its ``get`` returns ``None`` rather than raising,
        so an unknown code is an ordinary result, not a failure.
    """
    if code in _OVERRIDES:
        return f"{_OVERRIDES[code]} ({code})"

    record = pycountry.languages.get(alpha_2=code) or pycountry.languages.get(alpha_3=code)
    if record is None:
        return code
    name = getattr(record, "name", "").strip()
    return f"{name} ({code})" if name else code


def three_letter_code(code: str) -> str | None:
    """Return the ISO 639-3 code for an ISO 639-1 code, e.g. ``kaz`` for ``kk``.

    Script-aware classifiers such as lid218e label languages by their
    three-letter code (``kaz_Cyrl``), while the corpus sources and the
    rules use the two-letter one, so a filter asked for ``kk`` under such
    a model needs this to say what to ask for instead.

    Args:
        code: ISO 639-1 language code.

    Returns:
        The three-letter code, or ``None`` when pycountry has no
        two-letter entry for ``code``; its ``get`` returns ``None`` rather
        than raising, so an unknown code is an ordinary result.
    """
    record = pycountry.languages.get(alpha_2=code)
    if record is None:
        return None
    alpha_3: str = record.alpha_3
    return alpha_3
