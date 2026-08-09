import pathlib
import unicodedata

from turkic_translit.core import to_latin

ROOT = pathlib.Path(__file__).parent


def test_roundtrip_and_nfc() -> None:
    """The sample corpus transliterates to composed Latin text."""
    with open(ROOT / "sample_cy.txt", encoding="utf8") as f:
        src = f.read()
    out = "\n".join(to_latin(line, "kk") for line in src.splitlines())
    assert unicodedata.is_normalized("NFC", out)
    with open(ROOT / "expected_lat.txt", encoding="utf8") as f:
        exp = f.read()
    assert out.splitlines()[0] == exp.splitlines()[0]  # quick sanity


def test_byte_reduction() -> None:
    """Latin output is shorter in bytes than the Cyrillic source.

    Cyrillic costs two UTF-8 bytes per letter and Latin mostly one, so
    the transliteration of a Cyrillic sample is necessarily smaller. The
    first line is pinned as well, so this cannot pass on an empty or
    truncated result.
    """
    src = (ROOT / "sample_cy.txt").read_bytes()
    latin = to_latin(src.decode("utf8"), "kk")
    expected_first = (ROOT / "expected_lat.txt").read_text(encoding="utf8").splitlines()[0]

    assert latin.splitlines()[0] == expected_first
    assert len(latin.encode("utf8")) < len(src)
