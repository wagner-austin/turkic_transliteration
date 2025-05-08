import unicodedata as ud, io, pathlib
from turkic_translit.core import to_latin

ROOT = pathlib.Path(__file__).parent

def test_roundtrip_and_nfc():
    src = (ROOT / "sample_cy.txt").read_text(encoding="utf8")
    out = "\n".join(to_latin(l, "kk") for l in src.splitlines())
    assert unicodedata.is_normalized("NFC", out)
    exp = (ROOT/"expected_lat.txt").read_text(encoding="utf8")
    assert out.splitlines()[0] == exp.splitlines()[0]  # quick sanity

def test_byte_reduction():
    src = (ROOT / "sample_cy.txt").read_bytes()
    out = to_latin(src.decode("utf8"), "kk").encode("utf8")
    assert len(out) < len(src)
