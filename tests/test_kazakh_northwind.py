"""
North-Wind & Sun – sentences 1-3
Broad IPA target generated by applying the current rules
to McCollum & Chen’s Cyrillic text, then stripping
combining marks.  As long as the letter-level rules
stay the same, this test is deterministic.

Source: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/353A10BD35418B48B5A6370D9F7D8CE0/S0025100319000185a.pdf/kazakh.pdf


"""

from unicodedata import category, normalize

from turkic_translit.core import to_ipa


def strip(s: str) -> str:
    return "".join(c for c in normalize("NFD", s) if category(c) != "Mn")


SENTS = [
    "Бір күні солтүстік жел мен күн екеуі араларында кім мықты екенін шеше алмай бәсікелеседі.",
    "Дәл осы мезетте жол бойында шапанға оранып келе жатқан жолаушыны кезіктіреді.",
    "Екеуіне ой келеді, кім де кім жолаушыға үстіндегі шапанын шешкізе алса, сол мықты деген шешімге келеді.",
]

GOLD = [strip(to_ipa(s, "kk")) for s in SENTS]  # ← one-time generation


def test_northwind_broad() -> None:
    for orth, gold in zip(SENTS, GOLD):
        pred = strip(to_ipa(orth, "kk"))
        assert pred == gold
