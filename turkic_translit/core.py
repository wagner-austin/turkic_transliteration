"""Public API for Latin and IPA transliteration."""
from functools import lru_cache
import unicodedata as ud
from pathlib import Path
import icu
import epitran

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

@lru_cache
def _epitran(lang: str):
    return epitran.Epitran({"kk": "kaz-Cyrl", "ky": "kir-Cyrl"}[lang])

def to_ipa(text: str, lang: str) -> str:
    epi = _epitran(lang)
    # first, ICU fixes OOV Cyrillic (Щ, hard sign, etc.)
    pre = _icu_trans(f"{lang}_ipa.rules").transliterate(text)
    ipa = epi.transliterate(pre)
    return ud.normalize("NFC", ipa)
