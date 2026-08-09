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

from turkic_translit import _test_hooks
from turkic_translit._test_hooks import IcuTransliterator, RuleCompiler

_RULE_DIR = Path(__file__).with_suffix("").parent / "rules"

# ICU's UTRANS_FORWARD: apply the rules left to right, as written.
FORWARD = 0

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
        "On macOS, run:\n  brew install icu4c\nThen, reinstall the package with CFLAGS from brew."
    ),
}


def missing_icu_message(python_version: str, platform: str) -> str:
    """Explain how to install PyICU on one platform.

    Args:
        python_version: Version the caller is running, e.g. ``3.11``.
        platform: Value of ``sys.platform``.

    Returns:
        The message, naming the platform's install command, or generic
        advice for a platform with no recorded command.
    """
    instruction = _INSTALL_INSTRUCTIONS.get(
        platform, "Please install the ICU C++ libraries for your platform."
    )
    return f"PyICU missing on Python {python_version} ({platform}).\n\n{instruction}"


def _require_icu() -> RuleCompiler:
    """Return PyICU's rule compiler, or explain how to install PyICU.

    Deferred to call time so ``import turkic_translit`` succeeds even
    when PyICU is missing. This is what lets the
    ``turkic-pyicu-install`` console script bootstrap PyICU without
    requiring PyICU to already be installed.

    Returns:
        ICU's bound rule-compiling classmethod.

    Raises:
        RuntimeError: When PyICU cannot be imported. The exception
            message names the current Python version and platform and
            gives the exact command to install PyICU on that
            platform.
    """
    try:
        return _test_hooks.icu.rule_compiler()
    except ImportError as exc:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(missing_icu_message(version, sys.platform)) from exc


# The rule-file suffixes that provide each output format, most preferred
# first. Both the language listing and the file lookup read this table, so
# adding a spelling is one edit and the two can never disagree.
FORMAT_SPELLINGS: dict[str, tuple[str, ...]] = {
    "latin": ("lat2023", "lat", "latin"),
    "ipa": ("ipa",),
}


def rule_file_for(lang: str, fmt: str, directory: Path = _RULE_DIR) -> str | None:
    """Name the rule file providing one format for one language.

    Args:
        lang: ISO 639-1 language code.
        fmt: Output format, ``latin`` or ``ipa``.
        directory: Directory to look in.

    Returns:
        The file's basename, or ``None`` when the language has no rules
        for that format.
    """
    for spelling in FORMAT_SPELLINGS[fmt]:
        candidate = f"{lang}_{spelling}.rules"
        if (directory / candidate).exists():
            return candidate
    return None


def languages_offering(fmt: str) -> list[str]:
    """List every language this project can produce ``fmt`` for.

    Args:
        fmt: Output format, ``latin`` or ``ipa``.

    Returns:
        The language codes, sorted, for use in an error message.
    """
    return sorted(code for code, fmts in get_supported_languages().items() if fmt in fmts)


def scan_rule_directory(directory: Path) -> dict[str, list[str]]:
    """List the languages and formats a rules directory advertises.

    A rule file is named ``<lang>_<fmt>.rules``. The two spellings of the
    Latin rule set, ``lat`` and ``lat2023``, are both advertised as
    ``latin``, so a caller asks for one name rather than knowing which
    revision shipped. A file whose name carries no format is not a rule
    set this function can describe, and is passed over.

    The directory is a parameter so that the naming rules can be
    exercised against a directory built for the purpose. The previous
    version read the packaged directory directly, which meant the
    unnamed-format case could not be reached at all.

    Args:
        directory: Directory to scan for ``*.rules`` files.

    Returns:
        A mapping of language code to the formats available for it, e.g.
        ``{"kk": ["ipa", "latin"], "az": ["ipa"]}``.
    """
    supported: dict[str, list[str]] = {}

    spelled_as = {
        spelling: fmt for fmt, spellings in FORMAT_SPELLINGS.items() for spelling in spellings
    }

    for rule_file in sorted(directory.glob("*.rules")):
        if "_" not in rule_file.stem:
            continue
        lang, spelling = rule_file.stem.split("_", 1)
        fmt = spelled_as.get(spelling, spelling)
        formats = supported.setdefault(lang, [])
        if fmt not in formats:
            formats.append(fmt)

    return supported


@lru_cache
def get_supported_languages() -> dict[str, list[str]]:
    """Dynamically detect supported languages and their available formats.

    Reads the packaged rules directory only; does not require PyICU. This
    lets the CLI report its language coverage even in environments where
    PyICU is being bootstrapped.

    Returns:
        A dict mapping each ISO 639-1 language code advertised by the
        rules directory to a list of the formats available for it —
        for example ``{"kk": ["ipa", "latin"], "az": ["ipa"]}``.
    """
    return scan_rule_directory(_RULE_DIR)


@lru_cache
def _icu_trans(name: str) -> IcuTransliterator:
    """Load ``name`` from the rules directory and compile it via PyICU.

    Args:
        name: The rule-file basename (e.g. ``"kk_ipa.rules"``).

    Returns:
        An ``icu.Transliterator`` compiled from the rule file.

    Raises:
        RuntimeError: Propagated from :func:`_require_icu` when
            PyICU is not installed.
        FileNotFoundError: When the rule file does not exist under
            :data:`_RULE_DIR`.
    """
    compile_rules = _require_icu()
    rules = (_RULE_DIR / name).read_text(encoding="utf8")
    return compile_rules(name, rules, FORWARD)


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
    rule_file = rule_file_for(lang, "latin")
    if rule_file is None:
        raise ValueError(
            f"Latin transliteration not supported for '{lang}'. "
            f"Available languages: {', '.join(languages_offering('latin'))}"
        )

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
    rule_file = rule_file_for(lang, "ipa")
    if rule_file is None:
        raise ValueError(
            f"IPA transliteration not supported for '{lang}'. "
            f"Available languages: {', '.join(languages_offering('ipa'))}"
        )

    trans = _icu_trans(rule_file)
    return ud.normalize("NFC", trans.transliterate(text))
