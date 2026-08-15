"""Uyghur rules checked against the Montreal Forced Aligner's dictionary.

The article ug_ipa.rules cites grounds its inventory but prints no
keyword chart, so for a long time Uyghur was the least-verified rule
file: inventory conformance and nothing else. The MFA project's
uyghur_cv dictionary supplies what the article does not — 22,630
independent word pronunciations, built by different people for a
different purpose. On 2026-08-13 the full dictionary agreed with these
rules on all entries but one, after the two symbol conventions below.

This file pins a deterministic sample of that comparison: every 45th
word of the sorted dictionary, chosen by stride so the sample cannot
favour agreeable words. If a rule change moves any of these 503 words
off the dictionary's pronunciation, the change has to explain itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

INHERITS_SOURCE = "https://doi.org/10.5334/labphon.239"

SAMPLE = Path(__file__).parent / "data" / "uyghur_mfa_sample.tsv"

# The two transcription conventions separating MFA's notation from this
# project's, applied to MFA's side. Both are convention, not phonology:
# the same letters, the same phonemes, different glyph choices.
NOTATION: tuple[tuple[str, str, str], ...] = (
    ("a", "ɑ", "MFA writes the low back vowel as plain a; McCollum and this project use ɑ"),
    ("ɛ", "æ", "MFA writes the front low vowel as ɛ; McCollum, and kk/az here, use æ"),
)

# The one word of 22,630 where the dictionary and the rules disagree
# beyond convention: MFA epenthesizes an initial vowel in a loanword
# whose orthography is itself anomalous (hamza-less initial cluster).
KNOWN_DIVERGENT: frozenset[str] = frozenset({"ئففىكىت"})


def sample_rows() -> list[tuple[str, tuple[str, ...]]]:
    """The pinned sample: word and its dictionary pronunciations.

    Returns:
        Pairs of orthographic word and the MFA pronunciations for it.
    """
    rows = []
    for line in SAMPLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        word, prons = line.split("\t", 1)
        rows.append((word, tuple(p.strip() for p in prons.split(" | "))))
    return rows


ROWS = sample_rows()


def as_this_project_writes_it(phones: str) -> str:
    """Rewrite an MFA pronunciation in this project's conventions.

    Args:
        phones: Space-separated MFA phones.

    Returns:
        The same pronunciation as a plain string in our glyphs.
    """
    text = phones.replace(" ", "")
    for mfa, ours, _reason in NOTATION:
        text = text.replace(mfa, ours)
    return text


def test_the_sample_is_the_size_the_stride_produces() -> None:
    """503 words: 22,630 sorted entries at a stride of 45."""
    assert len(ROWS) == 503


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """The rule file still names the article whose inventory it implements."""
    declared = read_rule_source(_RULE_DIR / "ug_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "McCollum, A. G."


@pytest.mark.parametrize(("word", "prons"), ROWS, ids=[w for w, _ in ROWS])
def test_word_matches_a_dictionary_pronunciation(word: str, prons: tuple[str, ...]) -> None:
    """Each sampled word transliterates to a pronunciation MFA lists.

    The glottal stop is stripped from our side: the rules write hamza as
    a glottal stop, MFA does not transcribe it, and the corpus pipeline
    strips it as well (see the symbol map's decided-artifact row).

    Args:
        word: The word in Perso-Arabic orthography.
        prons: Every pronunciation the dictionary lists for it.
    """
    if word in KNOWN_DIVERGENT:
        pytest.skip("the one known epenthesis divergence of 22,630 entries")

    ours = to_ipa(word, "ug").replace("ʔ", "")
    expected = {as_this_project_writes_it(p) for p in prons}

    assert ours in expected, f"{word}: ours {ours!r} not among {sorted(expected)!r}"
