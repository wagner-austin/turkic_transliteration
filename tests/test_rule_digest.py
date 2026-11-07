"""Rule fingerprints name the rules, not the checkout that delivered them.

A cleaning manifest records a digest per rule file so a corpus can be
matched to the rules that built it. The digests used to be taken over the
raw bytes, and git gives a ``text`` file CRLF line endings on Windows and
LF on Linux, so one commit's rules fingerprinted two ways: the v4 to v6
corpora built on Windows recorded ``c2d4eeec...`` for 5c06eef's
``kk_ipa.rules``, which a Linux checkout of the same commit hashes to
``2b69fda4...``. Every case here writes real files and hashes them through
the real functions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from turkic_translit.core import _RULE_DIR
from turkic_translit.corpus.clean import rules_fingerprint
from turkic_translit.digests import rule_digest

PACKAGED = _RULE_DIR / "kk_ipa.rules"


def test_the_packaged_rules_carry_linux_line_endings() -> None:
    """``.gitattributes`` pins the rule files to LF in every checkout.

    The other cases build the Windows form from these bytes, so they are
    only a comparison of two line-ending forms while this one holds.
    """
    raw = PACKAGED.read_bytes()

    assert b"\r" not in raw
    assert raw.count(b"\n") > 100


def test_windows_and_linux_line_endings_give_one_digest(tmp_path: Path) -> None:
    """The same rules hash the same whichever line endings the file has."""
    linux = tmp_path / "linux.rules"
    windows = tmp_path / "windows.rules"
    linux.write_bytes(PACKAGED.read_bytes())
    windows.write_bytes(PACKAGED.read_bytes().replace(b"\n", b"\r\n"))

    assert windows.read_bytes() != linux.read_bytes()
    assert rule_digest(windows) == rule_digest(linux)
    assert rule_digest(linux) == hashlib.sha256(PACKAGED.read_bytes()).hexdigest()


def test_a_changed_rule_changes_the_digest(tmp_path: Path) -> None:
    """Normalising line endings does not make different rules collide."""
    original = tmp_path / "original.rules"
    changed = tmp_path / "changed.rules"
    text = PACKAGED.read_text(encoding="utf-8")
    original.write_text(text, encoding="utf-8")
    changed.write_text(text.replace("щ > ʃː", "щ > ɕː"), encoding="utf-8")

    assert changed.read_text(encoding="utf-8") != text
    assert rule_digest(changed) != rule_digest(original)


def test_the_run_report_uses_the_line_ending_free_digest() -> None:
    """``rules_fingerprint`` records exactly what ``rule_digest`` computes."""
    fingerprint = rules_fingerprint(("kk", "ky"), ())

    assert fingerprint["kk_ipa.rules"] == rule_digest(PACKAGED)
    assert fingerprint["ky_ipa.rules"] == rule_digest(_RULE_DIR / "ky_ipa.rules")


def test_a_missing_rule_file_is_an_error(tmp_path: Path) -> None:
    """A rule file that is not there raises rather than hashing nothing."""
    with pytest.raises(FileNotFoundError):
        rule_digest(tmp_path / "absent.rules")
