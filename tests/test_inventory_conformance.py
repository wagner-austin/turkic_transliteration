"""Transliteration emits only segments the cited description allows.

This is the check that needs no gold value and so cannot be circular. A
published phonological description states which segments a language has;
running text through the rules must not produce anything else. Whatever
the right transcription of a given word is, a symbol outside the
inventory is wrong for every word.

It is also the check that would have caught the two defects found while
this file was being written. A decomposed soft g used to leave a
combining breve in the output, and a breve is in no Turkic inventory. A
rule emitting a symbol its source never lists fails here on the first
sentence of real text.

The inventories are read from each source's own consonant and vowel
charts, adjusted only by the deviations the rule file declares.
"""

from __future__ import annotations

import unicodedata as ud

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

INHERITS_SOURCE = {
    "tr": "https://doi.org/10.1017/S0025100300004588",
    "kk": "https://doi.org/10.1017/S0025100319000185",
    "fi": "https://urn.fi/URN:ISBN:9789514289842",
    "az": "https://doi.org/10.1017/S0025100317000184",
}

TIE_BAR = "͡"
LENGTH_MARK = "ː"

# Segments every language's output may carry regardless of inventory:
# the tie bar binds an affricate, the length mark is quantity rather
# than a segment.
SUPRASEGMENTAL = frozenset({TIE_BAR, LENGTH_MARK})

# Turkish, from Zimmer & Orgun's consonant chart (p. 43) and vowel list
# (p. 44), written in this project's notation: the palatal series and the
# velar fricative are absent, one lateral rather than two, and the front
# rounded mid vowel takes the close-mid symbol.
# The affricates are written with the tie bar over the two components,
# each of which is already listed, so they add no segment of their own.
TURKISH_INVENTORY = frozenset(
    "pbtdkɡmnfvszʃʒhɾjl"  # consonants
    "iyeøaɯuo"  # vowels
)

TURKISH_SENTENCES: tuple[str, ...] = (
    "Kuzey rüzgarı ile güneş, hangisinin daha güçlü olduğu konusunda tartışıyorlardı.",
    "O sırada oradan sıcak bir paltoya bürünmüş bir yolcu geçmekteydi.",
    "Yolcunun paltosunu ilk çıkartmaya kim başarırsa, onun daha güçlü sayılacağı "
    "konusunda anlaştılar.",
    "Bunun üzerine kuzey rüzgarı olanca gücüyle esmeye başladı.",
    "Ne var ki, rüzgar sertleştikçe yolcu da paltosuna daha çok sarındı ve sonunda "
    "kuzey rüzgarı esmekten vazgeçti.",
    "Sonra güneş sıcak bir şekilde parladı ve yolcu az sonra paltosunu çıkardı.",
    "Böylece kuzey rüzgarı, güneşin ikisi arasında daha güçlü olduğunu kabul etmek zorunda kaldı.",
)


def segments_of(text: str) -> set[str]:
    """Collect the phonological segments in transliterated output.

    Punctuation, digits and spaces come through from the input and say
    nothing about the rules, so only letters and the marks that attach
    to them are considered.

    Args:
        text: IPA output from a transliteration.

    Returns:
        The distinct segment characters the output contains.
    """
    return {char for char in text if char.isalpha() or ud.category(char) == "Mn"}


# Kazakh, from McCollum & Chen's consonant chart (p. 277), the
# non-native segments their text names on the same page, and the
# eleven-vowel inventory of p. 281 with its two diphthongs written as the
# single vowels these rules produce. The front rounded mid vowel there is
# the barred o, not the slashed one; confirmed against the page by a
# second reader, the two being easy to confuse.
KAZAKH_INVENTORY = frozenset(
    "pbtdkɡqmnŋrszʃʒχʁwjl"  # consonants
    "fvh"  # non-native, p. 277
    "ɑoəʊæeɵɪʏiu"  # vowels, p. 281
)

KAZAKH_SENTENCES: tuple[str, ...] = (
    "Бір күні солтүстік жел мен күн екеуі араларында кім мықты екенін шеше алмай бәсікелеседі.",
    "Дәл осы мезетте жол бойында шапанға оранып келе жатқан жолаушыны кезіктіреді.",
    "Екеуіне ой келеді, кім де кім жолаушыға үстіндегі шапанын шешкізе алса, сол мықты "
    "деген шешімге келеді.",
    "Солтүстік жел бар күшімен жел үрлей бастайды, ол қатты үрлеген сайын жолаушы "
    "шапанына орана түседі.",
)

# Finnish, from the vowel inventory of p. 20 and the consonant groups of
# Figure 2 (p. 25). The book declines to state a single consonant count
# because the paradigm varies by speaker; the set below is the maximum
# paradigm, groups (1) to (5), or 17 phonemes. The dental place the book
# marks on the coronal plosives is left off here, as the rule file
# declares.
FINNISH_INVENTORY = frozenset(
    "ptkshmnlrʋj"  # group (1), common to all varieties
    "ŋdfbɡʃ"  # groups (2) to (5), by decreasing generality
    "ieyøæɑou"  # vowels, p. 20
)

# Runeberg's Maamme in Cajander's Finnish translation, first stanza. A
# native text with no loan stratum, so every segment it produces should
# be a Finnish one.
FINNISH_SENTENCES: tuple[str, ...] = (
    "Oi maamme, Suomi, synnyinmaa, soi, sana kultainen!",
    "Ei laaksoa, ei kukkulaa, ei vettä rantaa rakkaampaa,",
    "kuin kotimaa tää pohjoinen, maa kallis isien!",
)

# Azerbaijani, from Mokari & Werner's consonant chart (p. 2) and the nine
# vowels their text states on p. 3. The chart describes Tabriz and these
# rules target Baku, so it is wider than what they emit in two places:
# the alveolar affricate pair and the palatal voiceless plosive. The rule
# file records that gap and the language's own test pins it; here the
# chart is taken as published, which is the weaker and safer direction.
AZERBAIJANI_INVENTORY = frozenset(
    "pbtdcɟkɡmnɾfvszʃʒxɣhlj"  # consonants, p. 2
    "æɑoeœɯuiy"  # vowels, p. 3
)

AZERBAIJANI_SENTENCES: tuple[str, ...] = (
    "Şimal yeli ilə Günəş mübahisə edirdilər ki, hansı daha güclüdür.",
    "O zaman bir isti əbaya bürünmüş səyahətçi oradan keçirdi.",
    "O zaman Şimal yeli bəcərdiyi qədər əsməyə başladı.",
    "Amma hər nə qədər artıq əsdikcə, səyahətçi də əbasını dərəsinə bürüyürdü.",
    "Sonra Günəş isti şəfəqlənməyə başladı və bilavasitə səyahətçi əbasını çıxartdı.",
)

INVENTORIES = {
    "tr": (TURKISH_INVENTORY, TURKISH_SENTENCES),
    "kk": (KAZAKH_INVENTORY, KAZAKH_SENTENCES),
    "fi": (FINNISH_INVENTORY, FINNISH_SENTENCES),
    "az": (AZERBAIJANI_INVENTORY, AZERBAIJANI_SENTENCES),
}


@pytest.mark.parametrize("language", sorted(INHERITS_SOURCE))
def test_the_declared_source_is_the_one_the_rules_cite(language: str) -> None:
    """Each inventory below comes from the paper that language's rules name."""
    declared = read_rule_source(_RULE_DIR / f"{language}_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE[language]


@pytest.mark.parametrize(
    ("language", "sentence"),
    [
        (language, sentence)
        for language, (_inventory, sentences) in sorted(INVENTORIES.items())
        for sentence in sentences
    ],
)
def test_output_stays_inside_the_published_inventory(language: str, sentence: str) -> None:
    """Real text produces no segment the language's source does not list."""
    inventory, _sentences = INVENTORIES[language]
    produced = segments_of(to_ipa(sentence, language))

    assert produced <= inventory | SUPRASEGMENTAL
    assert produced, "a sentence of real text should produce segments"


def test_a_segment_outside_the_inventory_is_detected() -> None:
    """The check fails when output carries a segment the source omits.

    Stated directly so the conformance test cannot quietly become
    vacuous: a combining breve, the exact residue the normalisation
    defect used to leave behind, is not in any Turkic inventory.
    """
    intruder = segments_of("daɡ̆")

    assert not intruder <= TURKISH_INVENTORY | SUPRASEGMENTAL


def test_every_inventory_symbol_is_a_segment() -> None:
    """The declared inventory holds segments, not punctuation or spacing."""
    for symbol in TURKISH_INVENTORY:
        assert symbol.isalpha(), f"{symbol!r} is not a segment"
