"""Public API for Latin and IPA transliteration."""

try:
    import icu  # noqa: F401
except ImportError as e:  # PyICU wheel is still missing
    raise RuntimeError(
        "PyICU missing. On Windows run:\n"
        "  turkic-pyicu-install\n"
        "or manually install a wheel from "
        "https://github.com/cgohlke/pyicu-build/releases ."
    ) from e

import unicodedata as ud
from functools import lru_cache
from pathlib import Path

_RULE_DIR = Path(__file__).with_suffix("").parent / "rules"


@lru_cache
def _icu_trans(name: str) -> icu.Transliterator:
    txt = (_RULE_DIR / name).read_text(encoding="utf8")
    return icu.Transliterator.createFromRules(name, txt, 0)


def to_latin(text: str, lang: str, include_arabic: bool = False) -> str:
    if lang not in ("kk", "ky"):
        raise ValueError("lang must be 'kk' or 'ky'")
    rule = f"{lang}_lat2023.rules"
    trans = _icu_trans(rule)
    if include_arabic:
        ar = _icu_trans("ar_lat.rules")
        text = ar.transliterate(text)
    out = trans.transliterate(text)
    return ud.normalize("NFC", out)


def to_ipa(text: str, lang: str) -> str:
    trans = _icu_trans(f"{lang}_ipa.rules")
    return ud.normalize("NFC", trans.transliterate(text))
