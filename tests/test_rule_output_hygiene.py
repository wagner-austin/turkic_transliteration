"""Properties every IPA rule file must hold, checked for all of them at once.

Two defects reached the published corpora because the tests that should
have caught them were written as hand-kept lists. One test named four
withdrawn ligatures and Kyrgyz kept emitting a fifth for <ц>. Nothing at
all checked that a rule file covers the letters its corpus contains, so
five Cyrillic letters passed through Kyrgyz untransliterated and ten
through Uzbek Cyrillic, landing raw in files that are supposed to be IPA.

Both are properties of the whole rule set rather than of one language, and
neither needs a source to adjudicate: a withdrawn symbol is withdrawn for
every language, and Cyrillic is not IPA in any of them. So they are stated
here over every IPA rule file, derived from the files present rather than
from a list kept by hand, and a new language inherits both the moment it
is added.
"""

from __future__ import annotations

import re
import unicodedata as ud

import pytest

from turkic_translit.core import _RULE_DIR, to_ipa
from turkic_translit.rule_provenance import read_rule_source

# U+02A3..U+02A8, the affricate ligatures the IPA withdrew in 1989 in
# favour of two symbols joined by a tie bar. Written as a range so the
# block cannot be under-enumerated the way the previous list was.
WITHDRAWN_LIGATURES = tuple(chr(cp) for cp in range(0x02A3, 0x02A9))

# Every letter used by the Cyrillic orthographies in this project, plus the
# Russian letters that appear in loans and in mixed-script corpus text. A
# rule file for a Cyrillic language must map all of them to something,
# whether or not the letter belongs to that language's own alphabet.
CYRILLIC_LETTERS = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"  # Russian
    "әғқңөұүһі"  # Kazakh
    "ңөүҥѳ"  # Kyrgyz
    "ўқғҳ"  # Uzbek
)

CYRILLIC_LANGUAGES = ("kk", "ky", "uzc")

# The property tests need no source to adjudicate, but the contextual
# pins below state expected strings, and those inherit from the sources
# the exercised rule files declare. The binding is read from the files
# themselves, so it cannot drift from them.
INHERITS_SOURCE = {
    lang: read_rule_source(_RULE_DIR / f"{lang}_ipa.rules")["identifier"]
    for lang in CYRILLIC_LANGUAGES
}


def ipa_rule_languages() -> list[str]:
    """Every language with an IPA rule file, taken from the directory.

    Returns:
        The language codes, so a newly added rule file is covered
        without this list being edited.
    """
    return sorted(path.name.removesuffix("_ipa.rules") for path in _RULE_DIR.glob("*_ipa.rules"))


LANGUAGES = ipa_rule_languages()


def test_every_expected_language_has_an_ipa_rule_file() -> None:
    """The discovery above finds the rule files, rather than masking their loss.

    A test that derives its own parameters reports success when it finds
    nothing, so the set it discovers is pinned here.
    """
    assert set(LANGUAGES) == {"az", "fi", "kk", "ky", "tr", "ug", "uz", "uzc"}


@pytest.mark.parametrize("lang", CYRILLIC_LANGUAGES)
def test_no_cyrillic_letter_survives_into_the_output(lang: str) -> None:
    """Every Cyrillic letter maps to something; none passes through raw.

    A letter with no rule is emitted unchanged, so it lands in the corpus
    as Cyrillic among IPA. It is then a character the model sees in one
    language and never in another, which reads as a difference between
    the languages rather than as the gap in coverage it is.

    Args:
        lang: The Cyrillic-script language whose rule file is checked.
    """
    leaked = {
        letter: to_ipa(letter, lang)
        for letter in CYRILLIC_LETTERS
        if any(ud.name(c, "").startswith("CYRILLIC") for c in to_ipa(letter, lang))
    }

    assert leaked == {}, f"{lang} emits Cyrillic for {leaked}"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_no_withdrawn_affricate_ligature_is_emitted(lang: str) -> None:
    """Affricates use a tie bar, so one phoneme costs the same everywhere.

    A precomposed ligature is one character where the tie-bar spelling is
    three. A character-level model comparing languages reads that purely
    notational difference as a difference between the languages, so the
    convention has to hold across the whole rule set rather than in the
    files someone remembered to check.

    Comments are stripped before the check, because a comment may quote a
    source that prints the ligature, as the Kyrgyz file quotes Table 3.
    What must not carry one is a rule.

    Args:
        lang: The language whose rule file is checked.
    """
    source = (_RULE_DIR / f"{lang}_ipa.rules").read_text(encoding="utf-8")
    rules = "\n".join(line.split("#")[0] for line in source.splitlines())
    offending = [ligature for ligature in WITHDRAWN_LIGATURES if ligature in rules]

    assert not offending, f"{lang}_ipa.rules contains withdrawn ligatures {offending}"


def resolve_macros(source: str, text: str) -> str:
    """Expand ``$Name`` references using the file's own definitions.

    Args:
        source: The whole rule file, which carries the definitions.
        text: The fragment to expand.

    Returns:
        The fragment with every known macro replaced by its contents.
    """
    macros = dict(re.findall(r"\$(\w+)\s*=\s*\[([^\]]*)\]", source))
    for name, contents in macros.items():
        text = text.replace(f"${name}", contents)
    return text


@pytest.mark.parametrize("lang", CYRILLIC_LANGUAGES)
def test_no_left_context_is_written_in_the_source_script(lang: str) -> None:
    """A left context is matched against output, so it cannot be Cyrillic.

    ICU rewrites left to right, so by the time a rule is tried the text to
    its left has already been transliterated. A left context naming
    Cyrillic therefore never matches, and the rule silently does nothing.
    Kyrgyz and Uzbek Cyrillic both had glide rules written that way.

    Args:
        lang: The Cyrillic-script language whose rule file is checked.
    """
    source = (_RULE_DIR / f"{lang}_ipa.rules").read_text(encoding="utf-8")
    offenders = []
    for line in source.splitlines():
        rule = line.split("#")[0]
        if "{" not in rule or rule.strip().startswith("$"):
            continue
        left = resolve_macros(source, rule.split("{")[0])
        if any(ud.name(c, "").startswith("CYRILLIC") for c in left):
            offenders.append(rule.strip())

    assert not offenders, f"{lang} has left contexts in Cyrillic: {offenders}"


def test_the_glide_and_iotation_rules_actually_fire() -> None:
    """The contextual rules do what their comments say they do.

    Every one of these was written, commented and shipped while matching
    nothing, because the right-context bracket was used for a left
    context. That replaced the preceding vowel instead of the letter the
    rule named, so <тау> came out as [twu] rather than [tɑw]. These are
    the behaviours, stated so the brackets cannot quietly swap again.
    """
    assert to_ipa("тау", "ky") == "tɑw", "Kyrgyz glide after a vowel"
    assert to_ipa("маек", "ky") == "mɑjek", "Kyrgyz iotation after a vowel"
    assert to_ipa("тау", "kk") == "tɑw", "Kazakh glide after a vowel"
    assert to_ipa("уақ", "kk") == "wɑq", "Kazakh glide before a vowel"
    assert to_ipa("су", "kk") == "su", "Kazakh <у> is the vowel after a consonant"
    assert to_ipa("маен", "uzc") == "majen", "Uzbek iotation after a vowel"
    assert to_ipa("ъе", "uzc") == "ʔje", "Uzbek iotation after the hard sign"


def test_no_word_loses_its_only_vowel() -> None:
    """A syllable needs a nucleus, whatever the source says about a letter.

    This needs no source to adjudicate and would have caught the Kazakh
    defect on its own: mapping <у> to the glide everywhere turned
    су 'water' into [sw], which is not a possible syllable in any of
    these languages.
    """
    vowels = set("ɑaeiouøyɯɨəɪʊʏæɵɔ")
    for lang, words in (
        ("kk", ("су", "ту", "оқу", "бару", "университет")),
        ("ky", ("суу", "тоо", "жол")),
        ("uzc", ("сув", "салом")),
    ):
        for word in words:
            assert set(to_ipa(word, lang)) & vowels != set(), f"{lang} {word} lost its vowel"


# Every language's source alphabet, for the seam sweep below. The
# Cyrillic languages share the letter set above; the Latin alphabets are
# the official ones the rule files implement; the Uyghur letters are
# derived from the rule file itself, so a newly mapped letter is swept
# without this dict being edited.
APOSTROPHES = ("'", "ʻ", "’", "ʼ")


def source_alphabets() -> dict[str, str]:
    """The input alphabet the seam sweep feeds each language.

    Returns:
        Language code to its letters, covering every rule file.
    """
    ug_source = (_RULE_DIR / "ug_ipa.rules").read_text(encoding="utf-8")
    ug_letters = "".join(
        sorted({c for c in ug_source if ud.name(c, "").startswith("ARABIC LETTER")})
    )
    return {
        "kk": CYRILLIC_LETTERS,
        "ky": CYRILLIC_LETTERS,
        "uzc": CYRILLIC_LETTERS,
        "az": "abcçdeəfgğhxıijkqlmnoöprsştuüvyz",
        "tr": "abcçdefgğhıijklmnoöprsştuüvyz",
        "uz": "abdefghijklmnopqrstuvxyz",
        "fi": "abcdefghijklmnopqrstuvwxyzåäö",
        "ug": ug_letters,
    }


def multi_char_rule_heads(lang: str) -> list[str]:
    """The multi-character left-hand sides a rule file declares.

    Read from the file itself, so a newly added digraph is swept without
    this test being edited. A ``$Apo`` reference expands to every
    apostrophe variant; heads using contexts or other macros are skipped,
    because a bare concatenation cannot exercise them faithfully.

    Args:
        lang: The language whose rule file is read.

    Returns:
        Every literal multi-character head, expanded.
    """
    heads: list[str] = []
    for line in (_RULE_DIR / f"{lang}_ipa.rules").read_text(encoding="utf-8").splitlines():
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


def seam_inputs(lang: str, letters: str) -> list[str]:
    """Letter pairs plus every letter set against every multi-char head.

    Pairs trigger digraph and context rules against each other. The
    head combinations put each multi-character rule directly before and
    after every letter, which is where a shorter rule can steal a
    character from the rule behind it — the Uzbek <ngʻ> defect was the
    letter n followed by the head gʻ.

    Args:
        lang: The language the inputs are for.
        letters: Its source alphabet.

    Returns:
        The input strings to sweep.
    """
    heads = multi_char_rule_heads(lang)
    pairs = [a + b for a in letters for b in letters]
    before = [a + head for a in letters for head in heads]
    after = [head + a for head in heads for a in letters]
    return pairs + before + after


def test_alphabet_sweep_covers_every_rule_file() -> None:
    """The sweep's alphabet dict covers exactly the rule files present."""
    assert set(source_alphabets()) == set(LANGUAGES)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_no_source_mark_or_letter_survives_any_letter_pair(lang: str) -> None:
    """No seam between two rules leaks source orthography into output.

    The Uzbek rules once parsed <ngʻ> as ng + a stray mark, stranding
    the apostrophe in the corpus, where the cleaner then split the word.
    A stranded apostrophe or a surviving source-script letter after any
    swept input is that class of defect, for any language.

    Args:
        lang: The language whose rule file is swept.
    """
    letters = source_alphabets()[lang]
    source_script = "ARABIC" if lang == "ug" else "CYRILLIC"
    offenders = {}
    for text in seam_inputs(lang, letters):
        out = to_ipa(text, lang)
        stranded = [c for c in out if c in APOSTROPHES or ud.name(c, "").startswith(source_script)]
        if stranded:
            offenders[text] = out

    assert offenders == {}, f"{lang} strands source characters: {offenders}"


def test_the_ligature_range_covers_the_symbols_it_claims_to() -> None:
    """The range really is the withdrawn affricate block.

    Stated so that the range cannot be silently narrowed to make a
    failing rule file pass, which is the failure this file exists for.
    """
    assert WITHDRAWN_LIGATURES == ("ʣ", "ʤ", "ʥ", "ʦ", "ʧ", "ʨ")
