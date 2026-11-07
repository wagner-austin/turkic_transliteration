"""The record of which rules wrote an IPA file, and the cleaner's refusals on it.

On 2026-11-07 a Kazakh file transliterated before 0.5.7 went through the
0.5.10 cleaner and lost about 54,700 Kazakh tokens as "foreign", because it
spelled ‹я› with an ASCII ``a`` the installed rules no longer emit and
nothing recorded which rules had written it. These cases drive the real
transliteration command, the real record writer and the real cleaner over
real files. Expected digests are computed here with ``hashlib`` over the
bytes on disk, never by the functions under test, and expected IPA is read
off the Kazakh rule file by hand: с > s, а > ɑ, я > jɑ, т > t.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.corpus_clean_helpers import LONG, write_corpus
from turkic_translit.cli.clean_corpus import cli as clean_cli
from turkic_translit.cli.translit import translit
from turkic_translit.core import _RULE_DIR, get_supported_languages
from turkic_translit.corpus.clean import clean_corpora
from turkic_translit.corpus.errors import (
    TransliteratedByOtherRulesError,
    TransliterationOutputChangedError,
    TransliterationRecordLanguageError,
    TransliterationRecordMissingError,
)
from turkic_translit.corpus.transliteration_record import (
    decode_transliteration_record,
    encode_transliteration_record,
    read_transliteration_record,
    require_written_by_installed_rules,
    write_transliteration_record,
)
from turkic_translit.validation import FieldError

KAZAKH_RULES = _RULE_DIR / "kk_ipa.rules"
# Thirty-odd characters of Kazakh, so a cleaned line clears the minimum
# length; саясат ("politics") carries я, the letter old rules spelled ja.
KAZAKH = "саясат туралы мақала және саясат"


def _sha256(path: Path) -> str:
    """Hash a file's bytes with hashlib, outside the code under test.

    Args:
        path: The file.

    Returns:
        Its SHA-256 as lowercase hexadecimal.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_of(path: Path) -> dict[str, str]:
    """Read the record beside a corpus as raw JSON, bypassing the decoder.

    Args:
        path: The corpus.

    Returns:
        The record's fields as written to disk.
    """
    document: dict[str, str] = json.loads(Path(f"{path}.manifest.json").read_text(encoding="utf-8"))
    return document


def _transliterate(tmp_path: Path, name: str) -> Path:
    """Run the real translit command over the Kazakh sample.

    Args:
        tmp_path: Directory to work in.
        name: File name for the IPA, under ``raw``.

    Returns:
        The IPA file written.
    """
    source = tmp_path / "kk.txt"
    source.write_text(KAZAKH + "\n", encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    out = raw / name
    result = CliRunner().invoke(
        translit, ["--lang", "kk", "--in", str(source), "--out-ipa", str(out)]
    )
    assert result.exit_code == 0, result.output
    return out


def test_translit_records_the_installed_rules_and_the_bytes_it_wrote(tmp_path: Path) -> None:
    """The record names kk_ipa.rules, its digest, and the IPA file's digest."""
    out = _transliterate(tmp_path, "kk_ipa.txt")

    assert "sɑjɑsɑt" in out.read_text(encoding="utf-8")
    assert _record_of(out) == {
        "language": "kk",
        "rule_file": "kk_ipa.rules",
        "rule_digest": _sha256(KAZAKH_RULES),
        "output_sha256": _sha256(out),
    }


def test_ipa_on_standard_output_and_latin_alone_write_no_record(tmp_path: Path) -> None:
    """Only an IPA file has bytes on disk for a record to describe."""
    source = tmp_path / "kk.txt"
    source.write_text(KAZAKH + "\n", encoding="utf-8")
    latin = tmp_path / "kk_lat.txt"

    piped = CliRunner().invoke(translit, ["--lang", "kk", "--in", str(source), "--out-ipa", "-"])
    latin_only = CliRunner().invoke(
        translit, ["--lang", "kk", "--in", str(source), "--out-latin", str(latin)]
    )

    assert piped.exit_code == 0, piped.output
    assert "sɑjɑsɑt" in piped.output
    assert latin_only.exit_code == 0, latin_only.output
    assert latin.read_text(encoding="utf-8").strip() != ""
    assert sorted(path.name for path in tmp_path.iterdir()) == ["kk.txt", "kk_lat.txt"]


def test_ipa_the_installed_rules_wrote_is_cleaned_with_no_word_dropped(tmp_path: Path) -> None:
    """translit then turkic-clean-corpus keeps every Kazakh word."""
    _transliterate(tmp_path, "oscar_kk_ipa.txt")
    out = tmp_path / "clean"

    result = CliRunner().invoke(
        clean_cli, ["--input-dir", str(tmp_path / "raw"), "--output-dir", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert "dropped 0 foreign tokens" in result.output
    assert (out / "oscar_kk_ipa.txt").read_text(encoding="utf-8").count("sɑjɑsɑt") == 2


def test_ipa_with_no_record_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    """The colleague's file of 2026-11-07: old spelling, no record."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "oscar_kk_ipa.txt").write_text("sɑjasɑt turɑlə mɑqɑlɑ ʒəne sɑjasɑt\n", "utf-8")
    out = tmp_path / "clean"

    result = CliRunner().invoke(clean_cli, ["--input-dir", str(raw), "--output-dir", str(out)])

    assert result.exit_code == 1
    assert "TURKIC_CORPUS_006_TRANSLITERATION_RECORD_MISSING" in result.output
    assert f"{raw / 'oscar_kk_ipa.txt'} has no transliteration record" in result.output
    assert f"{raw / 'oscar_kk_ipa.txt.manifest.json'};" in result.output
    assert not out.exists()


def test_ipa_written_by_other_rules_is_refused_naming_both_digests(tmp_path: Path) -> None:
    """A record of the pre-0.5.10 shcha rule is not the installed rules."""
    old_rules = tmp_path / "old_kk_ipa.rules"
    old_rules.write_text(
        KAZAKH_RULES.read_text(encoding="utf-8").replace("щ > ʃː", "щ > ɕː"), encoding="utf-8"
    )
    corpus = write_corpus(tmp_path / "oscar_kk_ipa.txt", "kk", [LONG])
    record = Path(f"{corpus}.manifest.json")
    document = _record_of(corpus)
    document["rule_digest"] = _sha256(old_rules)
    record.write_text(json.dumps(document), encoding="utf-8")
    assert _sha256(old_rules) != _sha256(KAZAKH_RULES)

    with pytest.raises(TransliteratedByOtherRulesError) as caught:
        clean_corpora({"kk": corpus}, tmp_path / "clean", ())

    assert caught.value.code == "TURKIC_CORPUS_008_TRANSLITERATED_BY_OTHER_RULES"
    assert caught.value.recorded == f"kk_ipa.rules {_sha256(old_rules)}"
    assert caught.value.installed == f"kk_ipa.rules {_sha256(KAZAKH_RULES)}"
    assert (
        f"{corpus} was written by kk_ipa.rules {_sha256(old_rules)}, and the installed "
        f"rules are kk_ipa.rules {_sha256(KAZAKH_RULES)};"
    ) in str(caught.value)
    assert not (tmp_path / "clean").exists()


@pytest.mark.parametrize(
    ("rule_file", "digest_of"),
    [
        # Another rule file and its own digest.
        ("ky_ipa.rules", "ky_ipa.rules"),
        # The installed IPA rules' digest under another file's name: the
        # name alone is wrong, so only the name comparison can catch it.
        ("kk_lat.rules", "kk_ipa.rules"),
    ],
)
def test_a_record_naming_another_rule_file_is_refused(
    tmp_path: Path, rule_file: str, digest_of: str
) -> None:
    """Only kk_ipa.rules with its installed digest vouches for Kazakh IPA."""
    corpus = write_corpus(tmp_path / "oscar_kk_ipa.txt", "kk", [LONG])
    document = _record_of(corpus)
    document["rule_file"] = rule_file
    document["rule_digest"] = _sha256(_RULE_DIR / digest_of)
    Path(f"{corpus}.manifest.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(TransliteratedByOtherRulesError) as caught:
        require_written_by_installed_rules(corpus, "kk")

    assert caught.value.recorded == f"{rule_file} {_sha256(_RULE_DIR / digest_of)}"
    assert caught.value.installed == f"kk_ipa.rules {_sha256(KAZAKH_RULES)}"
    assert (
        f"{corpus} was written by {rule_file} {_sha256(_RULE_DIR / digest_of)}, and the "
        f"installed rules are kk_ipa.rules {_sha256(KAZAKH_RULES)};"
    ) in str(caught.value)


def test_a_record_of_another_language_is_refused(tmp_path: Path) -> None:
    """A file named Kazakh whose record says Kyrgyz rules wrote it."""
    corpus = write_corpus(tmp_path / "oscar_kk_ipa.txt", "ky", [LONG])

    with pytest.raises(TransliterationRecordLanguageError) as caught:
        require_written_by_installed_rules(corpus, "kk")

    assert caught.value.code == "TURKIC_CORPUS_007_TRANSLITERATION_RECORD_LANGUAGE"
    assert (caught.value.named, caught.value.recorded) == ("kk", "ky")
    assert str(caught.value) == (
        f"TURKIC_CORPUS_007_TRANSLITERATION_RECORD_LANGUAGE: {corpus} is named as kk "
        f"but its record says the ky rules wrote it"
    )


def test_a_file_changed_after_its_record_is_refused(tmp_path: Path) -> None:
    """The record describes bytes; an edited file is not those bytes."""
    corpus = write_corpus(tmp_path / "oscar_kk_ipa.txt", "kk", [LONG])
    recorded = _sha256(corpus)
    with corpus.open("a", encoding="utf-8") as stream:
        stream.write("hello world this line is english text\n")

    with pytest.raises(TransliterationOutputChangedError) as caught:
        require_written_by_installed_rules(corpus, "kk")

    assert caught.value.code == "TURKIC_CORPUS_009_TRANSLITERATION_OUTPUT_CHANGED"
    assert caught.value.recorded == recorded
    assert caught.value.found == _sha256(corpus)
    assert str(caught.value) == (
        f"TURKIC_CORPUS_009_TRANSLITERATION_OUTPUT_CHANGED: {corpus} has SHA-256 "
        f"{_sha256(corpus)}, and its transliteration record describes {recorded}; "
        f"the file changed after the rules wrote it"
    )


def test_one_refused_corpus_stops_the_whole_run_before_any_write(tmp_path: Path) -> None:
    """Kyrgyz is fine and Kazakh has no record: neither is cleaned."""
    good = write_corpus(tmp_path / "oscar_ky_ipa.txt", "ky", [LONG])
    bare = tmp_path / "oscar_kk_ipa.txt"
    bare.write_text(LONG + "\n", encoding="utf-8")

    with pytest.raises(TransliterationRecordMissingError) as caught:
        clean_corpora({"ky": good, "kk": bare}, tmp_path / "clean", ())

    assert caught.value.corpus == str(bare)
    assert not (tmp_path / "clean").exists()


@pytest.mark.parametrize(
    "language",
    sorted(code for code, modes in get_supported_languages().items() if "ipa" in modes),
)
def test_every_language_s_record_round_trips(tmp_path: Path, language: str) -> None:
    """Write, read, encode and decode agree for every language with IPA rules."""
    corpus = write_corpus(tmp_path / f"oscar_{language}_ipa.txt", language, [LONG])
    written = write_transliteration_record(corpus, language)

    read = read_transliteration_record(Path(f"{corpus}.manifest.json"))

    assert read == written
    assert decode_transliteration_record(encode_transliteration_record(read)) == read
    assert read["rule_digest"] == _sha256(_RULE_DIR / f"{language}_ipa.rules")
    assert require_written_by_installed_rules(corpus, language) == read


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"rule_digest": None}, "TURKIC_FIELD_001_MISSING"),
        ({"rule_digest": "ABC"}, "TURKIC_FIELD_002_TYPE"),
        ({"output_sha256": "0" * 63}, "TURKIC_FIELD_002_TYPE"),
        ({"language": ""}, "TURKIC_FIELD_003_EMPTY"),
    ],
)
def test_decoding_refuses_what_encoding_never_writes(
    tmp_path: Path, change: dict[str, str | None], code: str
) -> None:
    """A missing field, a non-digest or an empty language is not a record."""
    corpus = write_corpus(tmp_path / "oscar_kk_ipa.txt", "kk", [LONG])
    document: dict[str, str | None] = dict(_record_of(corpus))
    for field, value in change.items():
        if value is None:
            del document[field]
        else:
            document[field] = value

    with pytest.raises(FieldError, match=code):
        decode_transliteration_record(document)


def test_a_record_that_is_not_a_mapping_is_refused(tmp_path: Path) -> None:
    """A JSON list is not a record, and says so with the field-type code."""
    path = tmp_path / "x.txt.manifest.json"
    path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(FieldError, match="TURKIC_FIELD_002_TYPE"):
        read_transliteration_record(path)
