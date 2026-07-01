"""Public API for Latin and IPA transliteration.

PyICU is loaded at call time, not at import time. That lets
``import turkic_translit`` (and ``import turkic_translit.core``)
succeed in environments where PyICU has not yet been installed —
notably the ``turkic-pyicu-install`` console script, which needs to
be importable in order to *install* PyICU into the current
environment. A missing PyICU is reported by :func:`_require_icu` on
first use of a function that actually needs it, with a message that
names the caller's platform and the exact install command.
"""

from __future__ import annotations

import sys
import unicodedata as ud
from functools import lru_cache
from pathlib import Path
from typing import Any

_RULE_DIR = Path(__file__).with_suffix("").parent / "rules"

_INSTALL_INSTRUCTIONS: dict[str, str] = {
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


def _require_icu() -> Any:
    """Import and return the PyICU ``icu`` module.

    Deferred to call time so ``import turkic_translit`` succeeds even
    when PyICU is missing. This is what lets the
    ``turkic-pyicu-install`` console script bootstrap PyICU without
    requiring PyICU to already be installed.

    Returns:
        The imported ``icu`` module (a C extension shipping without
        type stubs — hence ``Any``).

    Raises:
        RuntimeError: When PyICU cannot be imported. The exception
            message names the current Python version and platform and
            gives the exact command to install PyICU on that
            platform.
    """
    try:
        import icu
    except ImportError as e:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        platform = sys.platform
        instruction = _INSTALL_INSTRUCTIONS.get(
            platform, "Please install the ICU C++ libraries for your platform."
        )
        raise RuntimeError(
            f"PyICU missing on Python {py_ver} ({platform}).\n\n{instruction}"
        ) from e
    return icu


@lru_cache
def get_supported_languages() -> dict[str, list[str]]:
    """Dynamically detect supported languages and their available formats.

    Reads the rules directory only; does not require PyICU. This lets
    the CLI report its language coverage even in environments where
    PyICU is being bootstrapped.

    Returns:
        A dict mapping each ISO 639-1 language code advertised by the
        rules directory to a list of the formats available for it —
        for example ``{"kk": ["latin", "ipa"], "tr": ["ipa"]}``.
    """
    supported: dict[str, list[str]] = {}

    for rule_file in _RULE_DIR.glob("*.rules"):
        filename = rule_file.stem

        if "_" in filename:
            parts = filename.split("_", 1)
            if len(parts) == 2:
                lang, fmt = parts

                if fmt == "lat2023" or fmt == "lat":
                    fmt = "latin"

                if lang not in supported:
                    supported[lang] = []
                if fmt not in supported[lang]:
                    supported[lang].append(fmt)

    return supported


@lru_cache
def _icu_trans(name: str) -> Any:
    """Load ``name`` from the rules directory and compile it via PyICU.

    Args:
        name: The rule-file basename (e.g. ``"kk_ipa.rules"``).

    Returns:
        An ``icu.Transliterator`` compiled from the rule file. The
        return type is annotated ``Any`` because PyICU is a C
        extension without published type stubs; the object is only
        used via its ``transliterate(text)`` method.

    Raises:
        RuntimeError: Propagated from :func:`_require_icu` when
            PyICU is not installed.
        FileNotFoundError: When the rule file does not exist under
            :data:`_RULE_DIR`.
    """
    icu = _require_icu()
    txt = (_RULE_DIR / name).read_text(encoding="utf8")
    return icu.Transliterator.createFromRules(name, txt, 0)


def to_latin(text: str, lang: str, include_arabic: bool = False) -> str:
    """Transliterate ``text`` to Latin script using the ``lang`` rules.

    Args:
        text: The input string in the language's native orthography.
        lang: ISO 639-1 language code. Must be a key of
            :func:`get_supported_languages` for which the value
            contains ``"latin"``.
        include_arabic: When ``True``, pre-passes ``text`` through
            ``ar_lat.rules`` before applying the target rule set.
            Useful for input streams that mix Arabic-script tokens
            (proper names, loanwords) into a Latin-target corpus.

    Returns:
        The Latin transliteration of ``text``, NFC-normalized.

    Raises:
        ValueError: When ``lang`` has no Latin rule file or no
            ``<lang>_lat*.rules`` file is present.
        RuntimeError: Propagated from :func:`_require_icu` when
            PyICU is not installed.
    """
    supported = get_supported_languages()

    if lang not in supported or "latin" not in supported[lang]:
        available = [code for code, fmts in supported.items() if "latin" in fmts]
        raise ValueError(
            f"Latin transliteration not supported for '{lang}'. "
            f"Available languages: {', '.join(sorted(available))}"
        )

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
    """Transliterate ``text`` to broad phonemic IPA using the ``lang`` rules.

    Args:
        text: The input string in the language's native orthography.
        lang: ISO 639-1 language code. Must be a key of
            :func:`get_supported_languages` for which the value
            contains ``"ipa"``.

    Returns:
        The IPA transliteration of ``text``, NFC-normalized.

    Raises:
        ValueError: When ``lang`` has no IPA rule file.
        RuntimeError: Propagated from :func:`_require_icu` when
            PyICU is not installed.
    """
    supported = get_supported_languages()

    if lang not in supported or "ipa" not in supported[lang]:
        available = [code for code, fmts in supported.items() if "ipa" in fmts]
        raise ValueError(
            f"IPA transliteration not supported for '{lang}'. "
            f"Available languages: {', '.join(sorted(available))}"
        )

    rule_file = f"{lang}_ipa.rules"
    if not (_RULE_DIR / rule_file).exists():
        raise ValueError(f"IPA rules file not found for language '{lang}'")

    trans = _icu_trans(rule_file)
    return ud.normalize("NFC", trans.transliterate(text))
