"""Public API for Latin and IPA transliteration."""

import sys

try:
    import icu  # noqa: F401
except ImportError as e:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    platform = sys.platform

    install_instructions = {
        "win32": (
            "On Windows, run:\n"
            "  turkic-pyicu-install\n"
            "or manually install a wheel from "
            "https://github.com/cgohlke/pyicu-build/releases ."
        ),
        "linux": (
            "On Debian/Ubuntu, run:\n"
            "  sudo apt-get install -y libicu-dev\n"
            "In a Hugging Face Space, add 'libicu-dev' to your packages.txt.\n"
            "Then, reinstall the package."
        ),
        "darwin": (
            "On macOS, run:\n"
            "  brew install icu4c\n"
            "Then, reinstall the package with CFLAGS from brew."
        ),
    }
    instruction = install_instructions.get(
        platform, "Please install the ICU C++ libraries for your platform."
    )
    raise RuntimeError(
        f"PyICU missing on Python {py_ver} ({platform}).\n\n{instruction}"
    ) from e

import unicodedata as ud
from functools import lru_cache
from pathlib import Path

_RULE_DIR = Path(__file__).with_suffix("").parent / "rules"


@lru_cache
def get_supported_languages() -> dict[str, list[str]]:
    """Dynamically detect supported languages and their available formats.

    Returns a dict like: {'kk': ['latin', 'ipa'], 'ky': ['latin', 'ipa'], 'tr': ['ipa']}
    """
    supported: dict[str, list[str]] = {}

    # Scan the rules directory for available files
    for rule_file in _RULE_DIR.glob("*.rules"):
        filename = rule_file.stem

        # Parse filename pattern: lang_format.rules
        if "_" in filename:
            parts = filename.split("_", 1)
            if len(parts) == 2:
                lang, fmt = parts

                # Normalize format names
                if fmt == "lat2023" or fmt == "lat":
                    fmt = "latin"

                if lang not in supported:
                    supported[lang] = []
                if fmt not in supported[lang]:
                    supported[lang].append(fmt)

    return supported


@lru_cache
def _icu_trans(name: str) -> icu.Transliterator:
    txt = (_RULE_DIR / name).read_text(encoding="utf8")
    return icu.Transliterator.createFromRules(name, txt, 0)


def to_latin(text: str, lang: str, include_arabic: bool = False) -> str:
    supported = get_supported_languages()

    if lang not in supported or "latin" not in supported[lang]:
        available = [
            lang_code for lang_code, fmts in supported.items() if "latin" in fmts
        ]
        raise ValueError(
            f"Latin transliteration not supported for '{lang}'. "
            f"Available languages: {', '.join(sorted(available))}"
        )

    # Try different possible rule file names
    possible_rules = [
        f"{lang}_lat2023.rules",
        f"{lang}_lat.rules",
        f"{lang}_latin.rules",
    ]
    rule_file = None

    for rule in possible_rules:
        if (_RULE_DIR / rule).exists():
            rule_file = rule
            break

    if not rule_file:
        raise ValueError(f"No Latin rules file found for language '{lang}'")

    trans = _icu_trans(rule_file)
    if include_arabic:
        ar = _icu_trans("ar_lat.rules")
        text = ar.transliterate(text)
    out = trans.transliterate(text)
    return ud.normalize("NFC", out)


def to_ipa(text: str, lang: str) -> str:
    supported = get_supported_languages()

    if lang not in supported or "ipa" not in supported[lang]:
        available = [
            lang_code for lang_code, fmts in supported.items() if "ipa" in fmts
        ]
        raise ValueError(
            f"IPA transliteration not supported for '{lang}'. "
            f"Available languages: {', '.join(sorted(available))}"
        )

    rule_file = f"{lang}_ipa.rules"
    if not (_RULE_DIR / rule_file).exists():
        raise ValueError(f"IPA rules file not found for language '{lang}'")

    trans = _icu_trans(rule_file)
    return ud.normalize("NFC", trans.transliterate(text))
