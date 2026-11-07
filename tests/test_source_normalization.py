"""Every path from source text to the rules applies the same normalisation.

Until 2026-11-07 only the downloader stripped format characters, applied
NFKC and the per-language folds; ``turkic-translit translit`` and the web
demo handed raw text to the rules. So a Kazakh line typed in a legacy font,
which spells қ ғ ң as Macedonian and Serbian ќ ѓ њ, gave Kazakh IPA from
a download and untransliterated Cyrillic from the command line. Each case
runs a real entry point and compares it with the same entry point given the
line's Unicode spelling, so the expected value is the package's own answer
for correctly written text, not a string copied from the code under test.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from turkic_translit.cli.translit import translit
from turkic_translit.corpus.normalize import language_folds, normalize_source_line
from turkic_translit.web.web_utils import direct_transliterate

LEGACY = "Ќазаќстан Республикасыныњ жоѓары оќу орны, оныӊ Ѓылыми кеңесі"
UNICODE = "Қазақстан Республикасының жоғары оқу орны, оның Ғылыми кеңесі"
SOFT_HYPHEN = "­"


def _translit_ipa(tmp_path: Path, name: str, text: str) -> str:
    """Run ``turkic-translit translit`` over one Kazakh line.

    Args:
        tmp_path: Directory to work in.
        name: Stem for this run's files.
        text: The line to transliterate.

    Returns:
        The IPA file's text.
    """
    source = tmp_path / f"{name}.txt"
    source.write_text(text + "\n", encoding="utf-8")
    out = tmp_path / f"{name}_ipa.txt"
    result = CliRunner().invoke(
        translit, ["--lang", "kk", "--in", str(source), "--out-ipa", str(out)]
    )
    assert result.exit_code == 0, result.output
    return out.read_text(encoding="utf-8")


def test_the_command_line_gives_legacy_font_kazakh_its_unicode_ipa(tmp_path: Path) -> None:
    """translit folds ќ ѓ њ ӊ as the downloader does."""
    legacy = _translit_ipa(tmp_path, "legacy", LEGACY)

    assert legacy == _translit_ipa(tmp_path, "unicode", UNICODE)
    assert not any(letter in legacy for letter in "ќѓњӊЌЃ")


def test_the_command_line_strips_a_format_character_inside_a_word(tmp_path: Path) -> None:
    """A soft hyphen no longer splits саясат into two untransliterated halves."""
    assert _translit_ipa(tmp_path, "hyphen", f"сая{SOFT_HYPHEN}сат") == _translit_ipa(
        tmp_path, "plain", "саясат"
    )


def test_the_web_demo_gives_legacy_font_kazakh_its_unicode_ipa() -> None:
    """The direct tab normalises each pasted line and keeps the line breaks."""
    legacy, _stats = direct_transliterate(f"{LEGACY}\n{LEGACY}", "kk", False, "ipa")
    unicode, _stats = direct_transliterate(f"{UNICODE}\n{UNICODE}", "kk", False, "ipa")

    assert legacy == unicode
    assert legacy.count("\n") == 1


def test_a_language_with_no_folds_is_only_unicode_normalised() -> None:
    """Azerbaijani has no fold rows: NFKC and format stripping still apply."""
    folds = language_folds("az")

    assert folds == {}
    assert normalize_source_line(f"  ﬁ{SOFT_HYPHEN}x  y ", folds) == "fix y"
