"""What each language's rules can read and what they can emit.

The source alphabets are the orthographies the rule files implement; the
Uyghur letters are read out of the rule file itself, so a newly mapped
letter is covered without this module being edited. The emitted set is
derived by running the rules over every letter, every letter pair, and
every multi-character rule head against every letter — the same sweep
the hygiene tests use — so it cannot drift from the rules the way a
hand-kept inventory would.

The emitted set is what separates transcription from passthrough: a
letter the rules can never produce, found in transliterated output, got
there by passing through untransliterated, which is how quoted foreign
material reads. The cleaner drops such tokens rather than shredding
them character by character.
"""

from __future__ import annotations

import unicodedata as ud
from functools import cache, lru_cache
from typing import Final

from turkic_translit.core import _RULE_DIR, to_ipa

APOSTROPHES: Final = ("'", "ʻ", "’", "ʼ")

_CYRILLIC_LETTERS: Final = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"  # Russian
    "әғқңөұүһі"  # Kazakh
    "ңөүҥѳ"  # Kyrgyz
    "ўқғҳ"  # Uzbek
)


def _uyghur_letters() -> str:
    """The Arabic letters the Uyghur rule file maps.

    Returns:
        Every Arabic letter appearing in the rule file, sorted.
    """
    source = (_RULE_DIR / "ug_ipa.rules").read_text(encoding="utf-8")
    return "".join(sorted({c for c in source if ud.name(c, "").startswith("ARABIC LETTER")}))


@lru_cache(maxsize=1)
def source_alphabets() -> dict[str, str]:
    """Each language's source alphabet, covering every rule file.

    Returns:
        Language code to its letters.
    """
    return {
        "kk": _CYRILLIC_LETTERS,
        "ky": _CYRILLIC_LETTERS,
        "uzc": _CYRILLIC_LETTERS,
        "az": "abcçdeəfgğhxıijkqlmnoöprsştuüvyz",
        "tr": "abcçdefgğhıijklmnoöprsştuüvyz",
        "uz": "abdefghijklmnopqrstuvxyz",
        "fi": "abcdefghijklmnopqrstuvwxyzåäö",
        "ug": _uyghur_letters(),
    }


def heads_from_source(source: str) -> list[str]:
    """The multi-character left-hand sides a rule text declares.

    A ``$Apo`` reference expands to every apostrophe variant; heads
    using contexts or other macros are skipped, because a bare
    concatenation cannot exercise them faithfully.

    Args:
        source: The full text of a rule file.

    Returns:
        Every literal multi-character head, expanded.
    """
    heads: list[str] = []
    for line in source.splitlines():
        rule = line.split("#")[0]
        if ">" not in rule or "{" in rule or "}" in rule:
            continue
        left = rule.split(">")[0].strip()
        if left.endswith("$Apo"):
            base = left.removesuffix("$Apo").strip()
            if base and "$" not in base:
                heads.extend(base + mark for mark in APOSTROPHES)
            continue
        if left and "$" not in left and len(left) > 1:
            heads.append(left)
    return heads


def multi_char_rule_heads(lang: str) -> list[str]:
    """The multi-character heads of one language's rule file.

    Read from the file itself, so a newly added digraph is swept
    without this module being edited.

    Args:
        lang: The language whose rule file is read.

    Returns:
        Every literal multi-character head, expanded.
    """
    return heads_from_source((_RULE_DIR / f"{lang}_ipa.rules").read_text(encoding="utf-8"))


def seam_inputs(lang: str) -> list[str]:
    """Letter pairs plus every letter set against every multi-char head.

    Pairs trigger digraph and context rules against each other; the head
    combinations put each multi-character rule directly before and after
    every letter, which is where a shorter rule can steal a character
    from the rule behind it.

    Args:
        lang: The language the inputs are for.

    Returns:
        The input strings to sweep.
    """
    letters = source_alphabets()[lang]
    heads = multi_char_rule_heads(lang)
    pairs = [a + b for a in letters for b in letters]
    before = [a + head for a in letters for head in heads]
    after = [head + a for head in heads for a in letters]
    return pairs + before + after


@cache
def emitted_characters(lang: str) -> frozenset[str]:
    """Every character the language's rules can produce.

    Derived by transliterating the full seam sweep, so contextual and
    digraph outputs are included alongside the single-letter values.

    Args:
        lang: The language whose emitted set is wanted.

    Returns:
        The characters, as a frozen set.
    """
    letters = source_alphabets()[lang]
    out: set[str] = set()
    for letter in letters:
        out.update(to_ipa(letter, lang))
    for text in seam_inputs(lang):
        out.update(to_ipa(text, lang))
    return frozenset(out)


__all__ = [
    "APOSTROPHES",
    "emitted_characters",
    "heads_from_source",
    "multi_char_rule_heads",
    "seam_inputs",
    "source_alphabets",
]
