"""Corpus cleaning, including the decisions it must refuse to make.

Every case writes real corpora to a real directory and runs the real
pipeline over them. The stages are ordered and the order is load-bearing,
so each stage is exercised both on its own and through the whole run.

The equalisation stage is what makes a seven-language comparison a
comparison rather than a measurement of who had more text, so its budget
and the language that set it are checked explicitly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from click.testing import CliRunner

from turkic_translit.cli.clean_corpus import (
    REPORT_NAME,
    cli,
    discover,
    harmonize,
    language_of,
    render,
)
from turkic_translit.core import _RULE_DIR
from turkic_translit.corpus.clean import (
    ALLOWED_CHARS,
    DEFAULT_MIN_IPA_RATIO,
    DEFAULT_MIN_LINE_CHARS,
    clean_corpora,
    clean_lines,
    decode_clean_report,
    decode_clean_stats,
    encode_clean_report,
    encode_clean_stats,
    harmonized_emitted,
    sanitize_line,
    transcription_ratio,
    truncate_to_budget,
)
from turkic_translit.corpus.errors import (
    ERR_NO_CORPORA,
    ERR_SYMBOL_MAP_MALFORMED,
    NoCorporaError,
    SymbolMapMalformedError,
)
from turkic_translit.corpus.inventory import emitted_characters
from turkic_translit.corpus.symbols import (
    PACKAGED_SYMBOL_MAP,
    apply_substitutions,
    decode_symbol_rule,
    encode_symbol_rule,
    parse_symbol_map,
    read_symbol_map,
    scopes_of,
    substitutions_for,
)
from turkic_translit.validation import FieldError

HEADER = "action,scope,from,to,verdict,rationale,citation"
LIGATURE_ROW = "merge,all,ʧ,t͡ʃ,NOTATION,ligature withdrawn from the IPA,IPA Handbook (1999)"
TURKISH_ROW = "merge,tr,a,ɑ,MOSTLY-NOTATION,written a by convention only,Zimmer & Orgun 1992"
KEPT_ROW = "keep,all,q,,CONTRAST,a uvular plosive is phonemic in Kazakh,McCollum & Chen 2021"

SMALL_MAP = "\n".join([HEADER, LIGATURE_ROW, TURKISH_ROW, KEPT_ROW]) + "\n"

# Long enough to clear the default minimum line length, and written
# entirely in characters both the Kazakh and the Kyrgyz rules can emit,
# so the token filter keeps every word for either language.
LONG = "sɑlɑmɑt bɑlɑ ʒoldɑ turɑt eki kun ʒɑj bolup"

# The emitted set the unit tests hand the sanitiser directly: the test
# line's own characters, plus the affricate the symbol-map test writes.
EMITTED = frozenset(LONG + "t͡ʃ")


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


def test_the_packaged_map_is_readable_and_records_both_actions() -> None:
    """The shipped map parses, and it keeps its non-merge rows.

    A map holding only merges would have lost the record of which
    contrasts were examined and deliberately left alone, which is half of
    what the file is for.
    """
    rules = read_symbol_map()

    assert rules
    assert {rule["action"] for rule in rules} == {"merge", "keep"}
    assert all(rule["citation"] for rule in rules)


def test_the_packaged_map_merges_the_withdrawn_ligatures() -> None:
    """The merge that the Kyrgyz corpus actually needed is present."""
    substitutions = substitutions_for(read_symbol_map(), "ky")

    assert substitutions["ʧ"] == "t͡ʃ"
    assert substitutions["ʤ"] == "d͡ʒ"


def test_a_scoped_merge_reaches_only_its_language() -> None:
    """The Turkish low vowel is rewritten for Turkish and nowhere else."""
    rules = parse_symbol_map(SMALL_MAP, "small.csv")

    assert substitutions_for(rules, "tr")["a"] == "ɑ"
    assert "a" not in substitutions_for(rules, "ky")


def test_a_kept_row_produces_no_substitution() -> None:
    """A recorded contrast changes nothing about the text."""
    rules = parse_symbol_map(SMALL_MAP, "small.csv")

    assert "q" not in substitutions_for(rules, "tr")


def test_a_row_scoped_to_another_language_is_skipped_not_refused() -> None:
    """Cleaning a subset of the language set is an ordinary thing to want.

    The packaged map has an opinion about all seven languages. A run over
    two of them must not fail because of a row about a third.
    """
    rules = parse_symbol_map(SMALL_MAP, "small.csv")

    assert substitutions_for(rules, "kk") == {"ʧ": "t͡ʃ"}


def test_every_scope_in_the_packaged_map_names_a_language_we_have_rules_for() -> None:
    """Where a typo in a scope is actually detectable.

    Per run the scope check has to be permissive, or a subset could not
    be cleaned. So the strict check lives here instead: a scope naming a
    language this project ships no rules for is a typo, and it would
    otherwise disable its merge in silence.
    """
    shipped = {path.stem.split("_")[0] for path in _RULE_DIR.glob("*_ipa.rules")}

    assert scopes_of(read_symbol_map()) <= shipped


def test_an_empty_map_is_refused() -> None:
    """A file with no header is not a table of decisions."""
    with pytest.raises(SymbolMapMalformedError) as raised:
        parse_symbol_map("", "empty.csv")

    assert raised.value.code == ERR_SYMBOL_MAP_MALFORMED


def test_a_row_missing_a_column_is_refused() -> None:
    """Every column carries part of the justification, so all are required."""
    with pytest.raises(SymbolMapMalformedError, match="citation"):
        parse_symbol_map("action,scope,from,to,verdict,rationale\nmerge,all,a,ɑ,N,why\n", "x.csv")


def test_a_merge_naming_no_symbol_is_refused() -> None:
    """A merge from nothing would rewrite every position in the text."""
    with pytest.raises(SymbolMapMalformedError, match="names no symbol"):
        parse_symbol_map(f"{HEADER}\nmerge,all,,ɑ,N,why,source\n", "x.csv")


def test_a_row_with_an_empty_action_is_refused() -> None:
    """An action is the one field with no sensible default."""
    with pytest.raises(FieldError):
        parse_symbol_map(f"{HEADER}\n,all,a,ɑ,N,why,source\n", "x.csv")


def test_symbol_rules_round_trip() -> None:
    """A rule encodes to the CSV's own column names and back."""
    rules = parse_symbol_map(SMALL_MAP, "small.csv")

    assert decode_symbol_rule(encode_symbol_rule(rules[0]), "small.csv") == rules[0]
    assert set(encode_symbol_rule(rules[0])) == {
        "action",
        "scope",
        "from",
        "to",
        "verdict",
        "rationale",
        "citation",
    }


def test_substitutions_apply_in_the_map_s_order() -> None:
    """Order matters when one row's target is another row's source."""
    assert apply_substitutions("ʧɑj", {"ʧ": "t͡ʃ"}) == "t͡ʃɑj"
    assert apply_substitutions("abc", {"a": "b", "b": "c"}) == "ccc"


def test_a_line_of_transcription_scores_one() -> None:
    """The ratio is over the allowed set, which IPA text sits inside."""
    assert transcription_ratio("sɑlɑm") == 1.0


def test_a_line_of_another_script_scores_low() -> None:
    """Foreign-script leakage is what the ratio test exists to catch."""
    assert transcription_ratio("これはテストです") == 0.0


def test_an_empty_line_has_no_ratio() -> None:
    """Refused rather than given a number that would be a guess.

    Callers drop empty lines on length before reaching here, so this
    states the boundary rather than papering over it.
    """
    with pytest.raises(ZeroDivisionError):
        transcription_ratio("")


def test_stray_characters_become_spaces_rather_than_vanishing() -> None:
    """Deleting a character would fuse its neighbours into a false sequence."""
    assert sanitize_line("sɑl☺ɑm", EMITTED) == ("sɑl ɑm", 0, 1)
    assert "☺" not in ALLOWED_CHARS


def test_sanitising_collapses_the_spaces_it_creates() -> None:
    """A run of junk leaves one space, not one per character."""
    assert sanitize_line("sɑl☺☺☺ɑm", EMITTED) == ("sɑl ɑm", 0, 3)


def test_a_token_with_a_letter_the_rules_cannot_emit_is_dropped_whole() -> None:
    """Foreign material goes as a word, not as fragments of one.

    Stripping the c out of a quoted brand name would leave a fragment
    that reads as a native word; dropping the token removes the quote
    and nothing else.
    """
    assert sanitize_line("sɑlɑm facebook eki", EMITTED) == ("sɑlɑm eki", 1, 0)


def test_punctuation_and_digits_are_stripped_to_spaces() -> None:
    """Corpus style is removed; the phoneme stream stays intact."""
    line, dropped, replaced = sanitize_line("sɑlɑm, eki 45.", EMITTED)

    assert line == "sɑlɑm eki"
    assert dropped == 0
    assert replaced == 4


def test_each_filter_is_counted_separately() -> None:
    """The report distinguishes why each line went, not just how many."""
    lines = [
        LONG,
        LONG,  # duplicate
        "short",  # under the minimum
        "これはとても長い日本語のテキストでありフィルタを通過しません",  # low ratio
    ]

    kept, stats = clean_lines(lines, {}, DEFAULT_MIN_LINE_CHARS, DEFAULT_MIN_IPA_RATIO, EMITTED)

    assert kept == [LONG]
    assert stats["lines_in"] == 4
    assert stats["dropped_duplicate"] == 1
    assert stats["dropped_short"] == 1
    assert stats["dropped_low_ipa"] == 1
    assert stats["lines_kept"] == 1


def test_a_line_shortened_past_the_minimum_by_sanitising_is_dropped() -> None:
    """The length test runs again afterwards, because sanitising shortens."""
    line = "sɑlɑm" + "☺" * 40

    kept, stats = clean_lines([line], {}, DEFAULT_MIN_LINE_CHARS, 0.0, EMITTED)

    assert kept == []
    assert stats["dropped_short"] == 1
    assert stats["chars_replaced"] == 40


def test_the_symbol_map_runs_before_the_filters() -> None:
    """A merge must be able to change whether a line survives.

    Ordering the stages the other way would filter on notation this
    project is in the middle of removing.
    """
    kept, _stats = clean_lines([LONG.replace("j", "ʧ")], {"ʧ": "t͡ʃ"}, 1, 1.0, EMITTED)

    assert kept == [LONG.replace("j", "t͡ʃ")]


def test_truncation_keeps_whole_lines() -> None:
    """No corpus ends in a fragment of a word."""
    assert truncate_to_budget(["abc", "de", "fghi"], 8) == ["abc", "de"]
    assert truncate_to_budget(["abcdefgh"], 3) == []


def test_every_corpus_is_cut_to_the_smallest(tmp_path: Path) -> None:
    """Equalisation is what makes the languages comparable."""
    raw = write_corpora(
        tmp_path,
        {
            "ky": [LONG, LONG + " ekint͡ʃi", LONG + " birdi"],
            "kk": [LONG],
        },
    )
    out = tmp_path / "clean"

    report = clean_corpora(
        {"ky": raw / "oscar_ky_ipa.txt", "kk": raw / "oscar_kk_ipa.txt"}, out, ()
    )

    written = {
        language: (out / f"oscar_{language}_ipa.txt").read_text(encoding="utf-8")
        for language in ("ky", "kk")
    }
    assert len(written["ky"]) <= report["equalized_char_budget"]
    assert len(written["kk"]) <= report["equalized_char_budget"]
    assert report["languages"]["ky"]["chars_written"] == len(written["ky"])


def test_the_report_names_the_language_that_set_the_budget(tmp_path: Path) -> None:
    """The budget is otherwise a number with no owner.

    Which language is the set's lowest-resource one is the fact the
    equalisation step establishes, and it is the fact a reader of the
    results most often wants.
    """
    raw = write_corpora(tmp_path, {"ky": [LONG, LONG + " more"], "kk": [LONG]})

    report = clean_corpora(
        {"ky": raw / "oscar_ky_ipa.txt", "kk": raw / "oscar_kk_ipa.txt"},
        tmp_path / "clean",
        (),
    )

    assert report["budget_language"] == "kk"
    assert report["equalized_char_budget"] == report["languages"]["kk"]["chars_kept"]


def test_cleaning_nothing_is_refused(tmp_path: Path) -> None:
    """Equalising one corpus against no others has no meaning."""
    with pytest.raises(NoCorporaError) as raised:
        clean_corpora({}, tmp_path / "clean", ())

    assert raised.value.code == ERR_NO_CORPORA


def test_a_corpus_that_loses_everything_still_writes_a_file(tmp_path: Path) -> None:
    """An empty output is a result, not a crash."""
    raw = write_corpora(tmp_path, {"ky": ["short"], "kk": ["short"]})
    out = tmp_path / "clean"

    report = clean_corpora(
        {"ky": raw / "oscar_ky_ipa.txt", "kk": raw / "oscar_kk_ipa.txt"}, out, ()
    )

    assert (out / "oscar_ky_ipa.txt").read_text(encoding="utf-8") == ""
    assert report["equalized_char_budget"] == 0


def test_statistics_round_trip() -> None:
    """A statistics record encodes and decodes to itself."""
    _kept, stats = clean_lines([LONG], {}, DEFAULT_MIN_LINE_CHARS, DEFAULT_MIN_IPA_RATIO, EMITTED)

    assert decode_clean_stats(encode_clean_stats(stats)) == stats


def test_a_report_round_trips_through_json(tmp_path: Path) -> None:
    """The written report reads back as the report that was written."""
    raw = write_corpora(tmp_path, {"ky": [LONG], "kk": [LONG, LONG + " more"]})
    report = clean_corpora(
        {"ky": raw / "oscar_ky_ipa.txt", "kk": raw / "oscar_kk_ipa.txt"},
        tmp_path / "clean",
        (),
    )

    written = json.dumps(encode_clean_report(report))

    assert decode_clean_report(json.loads(written)) == report


def test_decoding_a_report_short_of_a_field_fails() -> None:
    """The decoder validates rather than trusting its caller."""
    with pytest.raises(FieldError):
        decode_clean_report({"min_line_chars": 30, "min_ipa_ratio": 0.95, "languages": {}})


def test_decoding_a_report_with_a_bad_ratio_fails() -> None:
    """The ratio is a float, and the type is checked at the boundary."""
    with pytest.raises(TypeError, match="min_ipa_ratio"):
        decode_clean_report(
            {
                "min_line_chars": 30,
                "min_ipa_ratio": "0.95",
                "equalized_char_budget": 0,
                "budget_language": "ky",
                "rules_fingerprint": {},
                "languages": {},
            }
        )


def test_decoding_a_report_whose_languages_are_not_mappings_fails() -> None:
    """Per-language statistics must be records, not scalars."""
    base: dict[str, str | int | float | bool | None | Mapping[str, str | int | float | bool]] = {
        "min_line_chars": 30,
        "min_ipa_ratio": 0.95,
        "equalized_char_budget": 0,
        "budget_language": "ky",
        "rules_fingerprint": {},
    }

    with pytest.raises(TypeError, match="mapping"):
        decode_clean_report({**base, "languages": 3})

    with pytest.raises(TypeError, match="statistics"):
        decode_clean_report({**base, "languages": {"ky": 3}})


def test_decoding_a_report_whose_fingerprint_is_not_a_mapping_fails() -> None:
    """The fingerprint is a record of files to digests, not a scalar."""
    with pytest.raises(TypeError, match="rules_fingerprint"):
        decode_clean_report(
            {
                "min_line_chars": 30,
                "min_ipa_ratio": 0.95,
                "equalized_char_budget": 0,
                "budget_language": "ky",
                "rules_fingerprint": 3,
                "languages": {},
            }
        )


def test_the_report_fingerprints_the_rules_the_run_depended_on(tmp_path: Path) -> None:
    """Each rule file and the applied map are hashed into the report.

    The digests are recomputed here from the same bytes, so the report
    can be checked against the rules that exist now rather than trusted.
    """
    raw = write_corpora(tmp_path, {"ky": [LONG], "kk": [LONG, LONG + " more"]})

    report = clean_corpora(
        {"ky": raw / "oscar_ky_ipa.txt", "kk": raw / "oscar_kk_ipa.txt"},
        tmp_path / "clean",
        (),
    )

    fingerprint = report["rules_fingerprint"]
    assert set(fingerprint) == {"kk_ipa.rules", "ky_ipa.rules", "symbol_map"}
    expected = hashlib.sha256((_RULE_DIR / "ky_ipa.rules").read_bytes()).hexdigest()
    assert fingerprint["ky_ipa.rules"] == expected


def test_foreign_tokens_are_dropped_and_counted_end_to_end(tmp_path: Path) -> None:
    """A quoted foreign word leaves the corpus whole, and the report says so."""
    raw = write_corpora(tmp_path, {"ky": [LONG + " facebook"], "kk": [LONG]})
    out = tmp_path / "clean"

    report = clean_corpora(
        {"ky": raw / "oscar_ky_ipa.txt", "kk": raw / "oscar_kk_ipa.txt"}, out, ()
    )

    cleaned = (out / "oscar_ky_ipa.txt").read_text(encoding="utf-8")
    assert "facebook" not in cleaned
    assert report["languages"]["ky"]["dropped_foreign_tokens"] == 1
    assert report["languages"]["kk"]["dropped_foreign_tokens"] == 0


def test_a_cleaned_corpus_holds_only_what_the_rules_can_emit(tmp_path: Path) -> None:
    """The vocabulary property, stated over the file that was written."""
    raw = write_corpora(
        tmp_path,
        {"ky": [LONG + ", 45 % (facebook)"], "kk": [LONG + " eki 7."]},
    )
    out = tmp_path / "clean"

    clean_corpora({"ky": raw / "oscar_ky_ipa.txt", "kk": raw / "oscar_kk_ipa.txt"}, out, ())

    for language in ("ky", "kk"):
        text = (out / f"oscar_{language}_ipa.txt").read_text(encoding="utf-8")
        allowed = emitted_characters(language) | {" ", "\n"}
        assert set(text) <= allowed, f"{language} kept {set(text) - allowed}"


def test_the_token_filter_sees_the_symbol_map_s_own_rewrites(tmp_path: Path) -> None:
    """A token the map itself harmonised is native, not foreign.

    The Turkish rules emit a, and the map rewrites it to ɑ before the
    sanitiser runs. A filter built on the raw emitted set read every
    harmonised Turkish token as foreign and dropped half the corpus.
    """
    rules = parse_symbol_map(SMALL_MAP, "small.csv")
    assert "ɑ" not in emitted_characters("tr")
    assert "ɑ" in harmonized_emitted("tr", rules)

    line = "bana bakan kalan zaman aslan taɾafa katan alan bakan"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "oscar_tr_ipa.txt").write_text(line + "\n", encoding="utf-8")
    out = tmp_path / "clean"

    report = clean_corpora({"tr": raw / "oscar_tr_ipa.txt"}, out, rules)

    cleaned = (out / "oscar_tr_ipa.txt").read_text(encoding="utf-8")
    assert "bɑnɑ" in cleaned
    assert report["languages"]["tr"]["dropped_foreign_tokens"] == 0


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("oscar_ky_ipa", "ky"),
        ("ky", "ky"),
        ("corpus_fi", "fi"),
        ("oscar_ky_kk_ipa", None),
        ("corpus", None),
    ],
)
def test_the_language_is_read_from_the_file_name(name: str, expected: str | None) -> None:
    """One unambiguous two-letter field, or nothing."""
    assert language_of(Path(f"{name}.txt")) == expected


def test_two_files_claiming_one_language_are_refused(tmp_path: Path) -> None:
    """Picking one would silently drop the other's text."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "oscar_ky_ipa.txt").write_text(LONG, encoding="utf-8")
    (raw / "wiki_ky_ipa.txt").write_text(LONG, encoding="utf-8")

    with pytest.raises(Exception, match="claim to be ky"):
        discover(raw, "*.txt")


def test_a_file_with_no_language_in_its_name_is_refused(tmp_path: Path) -> None:
    """Guessing which language a file holds is not the tool's to do."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "corpus.txt").write_text(LONG, encoding="utf-8")

    with pytest.raises(Exception, match="cannot tell which language"):
        discover(raw, "*.txt")


def test_the_rendering_names_every_language_and_the_budget(tmp_path: Path) -> None:
    """The terminal output says which language constrained the rest."""
    raw = write_corpora(tmp_path, {"ky": [LONG], "kk": [LONG, LONG + " more"]})
    report = clean_corpora(
        {"ky": raw / "oscar_ky_ipa.txt", "kk": raw / "oscar_kk_ipa.txt"},
        tmp_path / "clean",
        (),
    )

    text = render(report)

    assert "ky" in text
    assert "kk" in text
    assert "lowest-resource" in text


def test_the_command_cleans_a_directory_and_writes_a_report(tmp_path: Path) -> None:
    """End to end, through the console entry point, on real files."""
    raw = write_corpora(
        tmp_path,
        {"ky": [LONG.replace("j", "ʧ"), LONG], "kk": [LONG, LONG + " ekint͡ʃi"]},
    )
    out = tmp_path / "clean"

    result = CliRunner().invoke(
        cli,
        [
            "--input-dir",
            str(raw),
            "--output-dir",
            str(out),
            "--symbol-map",
            str(PACKAGED_SYMBOL_MAP),
        ],
    )

    assert result.exit_code == 0, result.output
    report = decode_clean_report(json.loads((out / REPORT_NAME).read_text(encoding="utf-8")))
    assert set(report["languages"]) == {"ky", "kk"}
    assert "ʧ" not in (out / "oscar_ky_ipa.txt").read_text(encoding="utf-8")


def test_the_command_refuses_an_empty_directory(tmp_path: Path) -> None:
    """A run that cleaned nothing must not report success."""
    raw = tmp_path / "raw"
    raw.mkdir()

    result = CliRunner().invoke(
        cli, ["--input-dir", str(raw), "--output-dir", str(tmp_path / "clean")]
    )

    assert result.exit_code != 0
    assert "no file matching" in result.output


# ---------------------------------------------------------------------------
# Harmonisation: the symbol map with nothing else done
# ---------------------------------------------------------------------------


def test_harmonising_rewrites_symbols_and_touches_nothing_else(tmp_path: Path) -> None:
    """Evaluation texts keep every line; only the symbols change.

    The corpus pipeline drops short lines and deduplicates, which would
    destroy a perception file's section headers and numeric markers. This
    mode exists so those files can share the corpus's symbol space while
    keeping their structure line for line.
    """
    raw = tmp_path / "eval"
    raw.mkdir()
    original = "KYRGYZ\nTEXT 1: s\u0251l\u0251m\n\n1\n" + LONG.replace("j", "\u02a7") + "\n1\n"
    (raw / "perception_ky.txt").write_text(original, encoding="utf-8")
    out = tmp_path / "harmonised"

    written = harmonize(discover(raw, "*.txt"), out, read_symbol_map(PACKAGED_SYMBOL_MAP))

    result = (out / "perception_ky.txt").read_text(encoding="utf-8")
    assert written == ["ky"]
    assert "\u02a7" not in result
    assert "t\u0361\u0283" in result
    assert len(result.splitlines()) == len(original.splitlines())
    assert result.splitlines()[0] == "KYRGYZ"
    assert result.splitlines()[3] == "1"


def test_the_command_harmonises_without_cleaning(tmp_path: Path) -> None:
    """The harmonise pair alone is a complete, report-free run."""
    raw = tmp_path / "eval"
    raw.mkdir()
    (raw / "perception_kk.txt").write_text(LONG + "\n", encoding="utf-8")
    out = tmp_path / "harmonised"

    result = CliRunner().invoke(
        cli,
        ["--harmonize-dir", str(raw), "--harmonize-output-dir", str(out)],
    )

    assert result.exit_code == 0, result.output
    assert "harmonised: kk" in result.output
    assert (out / "perception_kk.txt").exists()
    assert not (out / REPORT_NAME).exists()


def test_the_command_runs_both_modes_in_one_invocation(tmp_path: Path) -> None:
    """Corpora are cleaned and evaluation texts harmonised together."""
    raw = write_corpora(tmp_path, {"ky": [LONG], "kk": [LONG + " ekint͡ʃi"]})
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "perception_ky.txt").write_text("short\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "--input-dir",
            str(raw),
            "--output-dir",
            str(tmp_path / "clean"),
            "--harmonize-dir",
            str(eval_dir),
            "--harmonize-output-dir",
            str(tmp_path / "harmonised"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "clean" / REPORT_NAME).exists()
    # The harmonised file keeps its too-short line; only cleaning filters.
    harmonised = tmp_path / "harmonised" / "perception_ky.txt"
    assert harmonised.read_text(encoding="utf-8") == "short\n"


def test_a_half_given_directory_pair_is_refused(tmp_path: Path) -> None:
    """One directory of a pair is a mistake, not a request."""
    raw = tmp_path / "raw"
    raw.mkdir()

    for arguments, complaint in (
        (["--input-dir", str(raw)], "needs both --input-dir and --output-dir"),
        (["--harmonize-dir", str(raw)], "needs both --harmonize-dir"),
    ):
        result = CliRunner().invoke(cli, arguments)
        assert result.exit_code != 0
        assert complaint in result.output


def test_the_command_with_no_directories_is_refused() -> None:
    """A run that would do nothing must say so rather than succeed."""
    result = CliRunner().invoke(cli, [])

    assert result.exit_code != 0
    assert "nothing to do" in result.output


def test_harmonising_an_empty_directory_is_refused(tmp_path: Path) -> None:
    """No matching files is a mistake, not a quiet no-op."""
    raw = tmp_path / "eval"
    raw.mkdir()

    result = CliRunner().invoke(
        cli,
        ["--harmonize-dir", str(raw), "--harmonize-output-dir", str(tmp_path / "out")],
    )

    assert result.exit_code != 0
    assert "no file matching" in result.output
