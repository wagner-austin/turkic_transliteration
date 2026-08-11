"""Tests for the rules that keep provenance readable and inherited.

Every case writes a real rule file or a real test module and runs the
real rule over it. The rules read headers and parse source, so giving
them headers and source is the only honest way to exercise them.

Each of the three rules exists because of a failure this project had.
The cases below reproduce those failures, so a change that stops
detecting them fails here.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.guards.provenance_rules import (
    RuleSourceRule,
    SelfReferentialExpectationRule,
    TestSourceInheritanceRule,
    calls_transliterator,
    is_ipa_rule_file,
    is_test_module,
    states_expectations,
)

COMPLETE_HEADER = """\
# Example → IPA transliteration rules (zz_ipa.rules)
#
# Source-Authors: Doe, J.
# Source-Year: 1999
# Source-Title: An Illustration of the IPA
# Source-Container: Journal of the International Phonetic Association 29(1): 1-4
# Source-Id: https://doi.org/10.1017/S0000000000000000

a > ɑ ;
"""


@pytest.fixture
def write_rule_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Return a helper writing a rule file into a rules directory.

    Args:
        tmp_path: Root to build the tree in.

    Returns:
        A function taking a file name and contents, returning its path.
    """
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()

    def write(name: str, text: str) -> Path:
        """Write one rule file.

        Args:
            name: File name within the rules directory.
            text: File contents.

        Returns:
            The written path.
        """
        path = rules_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    return write


@pytest.fixture
def write_test_module(tmp_path: Path) -> Callable[[str, str], Path]:
    """Return a helper writing source into a ``tests`` directory.

    The rules only consider files under a directory named ``tests``, so
    the fixture builds that layout rather than leaving each case to.

    Args:
        tmp_path: Root to build the tree in.

    Returns:
        A function taking a file name and source, returning its path.
    """
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    def write(name: str, source: str) -> Path:
        """Write one test module.

        Args:
            name: File name within the tests directory.
            source: File contents.

        Returns:
            The written path.
        """
        path = tests_dir / name
        path.write_text(source, encoding="utf-8")
        return path

    return write


def source_kinds(paths: list[Path]) -> list[str]:
    """Run the rule-provenance rule and return the violation kinds.

    Args:
        paths: Files to scan.

    Returns:
        The kind of each violation, in the order reported.
    """
    return [violation.kind for violation in RuleSourceRule().run(paths)]


def inheritance_kinds(paths: list[Path]) -> list[str]:
    """Run the test-provenance rule and return the violation kinds.

    Args:
        paths: Files to scan.

    Returns:
        The kind of each violation, in the order reported.
    """
    return [violation.kind for violation in TestSourceInheritanceRule().run(paths)]


def circular_kinds(paths: list[Path]) -> list[str]:
    """Run the self-referential rule and return the violation kinds.

    Args:
        paths: Files to scan.

    Returns:
        The kind of each violation, in the order reported.
    """
    return [violation.kind for violation in SelfReferentialExpectationRule().run(paths)]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("tr_ipa.rules", True),
        ("uzc_ipa.rules", True),
        ("tr_lat.rules", False),
        ("ar_lat.rules", False),
        ("tr_ipa.txt", False),
    ],
)
def test_only_ipa_rule_files_are_in_scope(name: str, expected: bool) -> None:
    """A Latin-output rule file states no phonological source to check."""
    assert is_ipa_rule_file(Path("rules") / name) is expected


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("tests/test_a.py", True),
        ("src/turkic_translit/core.py", False),
        ("tests/data/sample.txt", False),
    ],
)
def test_only_python_under_tests_is_a_test_module(relative: str, expected: bool) -> None:
    """The inheritance rules apply to test source and nothing else."""
    assert is_test_module(Path(relative)) is expected


def test_a_complete_header_passes(write_rule_file: Callable[[str, str], Path]) -> None:
    """All five fields present is the whole requirement."""
    path = write_rule_file("zz_ipa.rules", COMPLETE_HEADER)

    assert source_kinds([path]) == []


@pytest.mark.parametrize("field", ["Authors", "Year", "Title", "Container", "Id"])
def test_each_missing_field_is_reported(
    field: str, write_rule_file: Callable[[str, str], Path]
) -> None:
    """One violation per absent field, naming the field in the message."""
    text = "\n".join(
        line for line in COMPLETE_HEADER.splitlines() if not line.startswith(f"# Source-{field}:")
    )
    path = write_rule_file("zz_ipa.rules", text)

    violations = RuleSourceRule().run([path])

    assert [violation.kind for violation in violations] == ["rule-file-declares-no-source"]
    assert field in violations[0].line


def test_a_prose_citation_is_not_a_declaration(
    write_rule_file: Callable[[str, str], Path],
) -> None:
    """The failure this rule exists for: a citation nothing can parse.

    A header naming its paper in ordinary prose reads fine and
    constrains nothing, which is how the Turkish rules came to quote a
    rule about soft g that they did not implement.
    """
    path = write_rule_file(
        "zz_ipa.rules",
        "# Based on Doe (1999), An Illustration of the IPA, JIPA 29(1).\n\na > ɑ ;\n",
    )

    assert len(source_kinds([path])) == 5


def test_a_latin_rule_file_is_not_asked_for_a_source(
    write_rule_file: Callable[[str, str], Path],
) -> None:
    """Only the IPA rule files implement a phonological description."""
    path = write_rule_file("zz_lat.rules", "a > a ;\n")

    assert source_kinds([path]) == []


def test_a_test_pinning_output_without_a_source_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The second failure: expected values naming no source at all."""
    path = write_test_module(
        "test_zz.py",
        "def test_letters() -> None:\n    assert to_ipa('a', 'zz') == 'ɑ'\n",
    )

    assert inheritance_kinds([path]) == ["test-pins-output-without-source"]


def test_declaring_the_inherited_source_satisfies_the_rule(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A module that names what it inherits is in the clear."""
    path = write_test_module(
        "test_zz.py",
        "INHERITS_SOURCE = 'https://doi.org/10.1017/S0000000000000000'\n\n"
        "def test_letters() -> None:\n    assert to_ipa('a', 'zz') == 'ɑ'\n",
    )

    assert inheritance_kinds([path]) == []


def test_a_table_of_pairs_counts_as_stating_expectations(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A parametrised gold table is an expectation even with no literal.

    The assertion in such a module compares two names, so a rule looking
    only at comparisons would miss the whole family.
    """
    path = write_test_module(
        "test_zz.py",
        "GOLD = {'a': 'ɑ', 'b': 'b'}\n\n"
        "def test_letters() -> None:\n"
        "    for letter, expected in GOLD.items():\n"
        "        assert to_ipa(letter, 'zz') == expected\n",
    )

    assert inheritance_kinds([path]) == ["test-pins-output-without-source"]


def test_a_test_that_pins_nothing_needs_no_source(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """An inventory check states no expected value, so it inherits none.

    It compares output against a published segment set rather than
    against a transcription, which is what makes it non-circular; it
    should not be pushed into declaring a source it does not quote.
    """
    path = write_test_module(
        "test_zz.py",
        "INVENTORY = frozenset('ɑbc')\n\n"
        "def test_inventory() -> None:\n"
        "    assert set(to_ipa('ab', 'zz')) <= INVENTORY\n",
    )

    assert inheritance_kinds([path]) == []


def test_a_module_that_never_transliterates_is_out_of_scope(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """String expectations elsewhere in the suite are not provenance."""
    path = write_test_module(
        "test_zz.py",
        "def test_upper() -> None:\n    assert 'a'.upper() == 'A'\n",
    )

    assert inheritance_kinds([path]) == []


def test_an_expectation_computed_by_the_code_under_test_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The third failure: a passage test that reports determinism only."""
    path = write_test_module(
        "test_zz.py",
        "def test_passage() -> None:\n"
        "    gold = to_ipa('a b c', 'zz')\n"
        "    assert to_ipa('a b c', 'zz') == gold\n",
    )

    assert circular_kinds([path]) == ["expectation-computed-by-code-under-test"]


def test_a_generated_expectation_table_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A comprehension over the transliterator produces current answers.

    Not the right answers. Both sides of the eventual assertion are
    plain names, so the table itself has to be caught where it is built.
    """
    path = write_test_module(
        "test_zz.py",
        "LINES = ['a', 'b']\nGOLD = [to_ipa(line, 'zz') for line in LINES]\n",
    )

    assert circular_kinds([path]) == ["expectation-table-generated-by-code-under-test"]


def test_a_comprehension_that_does_not_transliterate_is_left_alone(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Only tables built by running the transliterator are the problem.

    An ordinary comprehension in a test module is not an expectation
    table, and reporting one would make the rule noise.
    """
    path = write_test_module(
        "test_zz.py",
        "SQUARES = [n * n for n in range(3)]\n\n"
        "def test_squares() -> None:\n    assert SQUARES == [0, 1, 4]\n",
    )

    assert circular_kinds([path]) == []


def test_an_unpacked_assignment_taints_only_its_named_targets(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A tuple target is not a name, so it binds nothing the rule tracks.

    The value still came from the transliterator, so the assertion below
    is circular through the element that does have a name.
    """
    path = write_test_module(
        "test_zz.py",
        "def test_passage() -> None:\n"
        "    (first, second) = gold = to_ipa('a b', 'zz').split()\n"
        "    assert to_ipa('a b', 'zz').split() == gold\n"
        "    assert first != second\n",
    )

    assert circular_kinds([path]) == ["expectation-computed-by-code-under-test"]


def test_a_literal_expectation_is_not_circular(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A value written down by a person is what the rule wants to see."""
    path = write_test_module(
        "test_zz.py",
        "def test_letters() -> None:\n    assert to_ipa('a', 'zz') == 'ɑ'\n",
    )

    assert circular_kinds([path]) == []


def test_comparing_two_calls_states_a_property(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Two spellings agreeing is a claim about the rules, not about them.

    Neither side is a stored result, so the assertion can fail — which
    is the difference from the determinism shape above.
    """
    path = write_test_module(
        "test_zz.py",
        "def test_alphabets_agree() -> None:\n"
        "    assert to_ipa('salom', 'uz') == to_ipa('салом', 'uzc')\n",
    )

    assert circular_kinds([path]) == []


def test_one_function_local_is_not_read_as_another_s(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Scope matters, or a reused name makes an honest test look circular."""
    path = write_test_module(
        "test_zz.py",
        "def test_first() -> None:\n"
        "    produced = to_ipa('a', 'zz')\n"
        "    assert produced == 'ɑ'\n"
        "\n"
        "def test_second() -> None:\n"
        "    produced = 'b'\n"
        "    assert to_ipa('b', 'zz') == produced\n",
    )

    assert circular_kinds([path]) == []


def test_a_loop_target_over_derived_output_is_derived(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Binding through a loop carries the taint the assignment carried.

    The table here is built without a comprehension, so the rule that
    catches generated tables stays out of the way and what is left is
    the loop-target path on its own.
    """
    path = write_test_module(
        "test_zz.py",
        "def test_passage() -> None:\n"
        "    gold = to_ipa('a b', 'zz').split()\n"
        "    for line, expected in zip(['a', 'b'], gold, strict=True):\n"
        "        assert to_ipa(line, 'zz') == expected\n",
    )

    assert circular_kinds([path]) == ["expectation-computed-by-code-under-test"]


def test_a_generated_table_read_in_a_loop_is_reported_twice(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Both rules fire on the shape the deleted passage tests had.

    The table is generated and then compared against, so it is caught
    where it is built and again where it is used. Reporting both is what
    tells a reader that removing the loop alone would not fix it.
    """
    path = write_test_module(
        "test_zz.py",
        "def test_passage() -> None:\n"
        "    gold = [to_ipa(line, 'zz') for line in ['a', 'b']]\n"
        "    for line, expected in zip(['a', 'b'], gold, strict=True):\n"
        "        assert to_ipa(line, 'zz') == expected\n",
    )

    assert sorted(circular_kinds([path])) == [
        "expectation-computed-by-code-under-test",
        "expectation-table-generated-by-code-under-test",
    ]


def test_a_module_level_derived_name_is_visible_to_every_test(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A gold table built at import time taints the tests that read it."""
    path = write_test_module(
        "test_zz.py",
        "GOLD = to_ipa('a', 'zz')\n\n"
        "def test_letters() -> None:\n"
        "    assert to_ipa('a', 'zz') == GOLD\n",
    )

    assert circular_kinds([path]) == ["expectation-computed-by-code-under-test"]


def test_a_non_test_file_is_skipped_by_every_rule(tmp_path: Path) -> None:
    """Source outside the tests directory is not scanned for either rule."""
    path = tmp_path / "helper.py"
    path.write_text(
        "GOLD = to_ipa('a', 'zz')\nassert to_ipa('a', 'zz') == GOLD\n",
        encoding="utf-8",
    )

    assert inheritance_kinds([path]) == []
    assert circular_kinds([path]) == []


def test_the_transliterator_detector_names_the_two_entry_points() -> None:
    """Only to_ipa and to_latin count; an unrelated call does not."""
    assert calls_transliterator(ast.parse("to_ipa('a', 'zz')"))
    assert calls_transliterator(ast.parse("to_latin('a', 'zz')"))
    assert not calls_transliterator(ast.parse("normalize('NFC', 'a')"))


def test_the_expectation_detector_needs_a_literal_or_a_pair_table() -> None:
    """A comparison against a name alone states no expectation."""
    assert states_expectations(ast.parse("assert to_ipa('a', 'zz') == 'ɑ'"))
    assert states_expectations(ast.parse("GOLD = {'a': 'ɑ'}"))
    assert not states_expectations(ast.parse("assert to_ipa('a', 'zz') == other"))
    assert not states_expectations(ast.parse("COUNTS = {'a': 1}"))
    assert not states_expectations(ast.parse("EMPTY = {}"))
