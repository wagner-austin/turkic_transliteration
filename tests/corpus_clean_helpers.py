"""Corpora and a builder shared by the corpus-cleaning test suites.

``test_corpus_clean.py`` tests the cleaning functions and
``test_clean_corpus_cli.py`` the command that drives them; both build
raw corpora the same way, so the line and the builder live here once.
They were in ``test_corpus_clean.py`` until 2026-11-04, when that file
was split under the 600-line limit, and moved here unchanged.
"""

from __future__ import annotations

from pathlib import Path

# Long enough to clear the default minimum line length, and written
# entirely in characters both the Kazakh and the Kyrgyz rules can emit,
# so the token filter keeps every word for either language.
LONG = "sɑlɑmɑt bɑlɑ ʒoldɑ turɑt eki kun ʒɑj bolup"


def write_corpora(root: Path, corpora: dict[str, list[str]]) -> Path:
    """Write one raw corpus per language and return their directory.

    Args:
        root: Directory to build under.
        corpora: Language code to that language's raw lines.

    Returns:
        The directory the corpora were written to.
    """
    raw = root / "raw"
    raw.mkdir()
    for language, lines in corpora.items():
        (raw / f"oscar_{language}_ipa.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw
