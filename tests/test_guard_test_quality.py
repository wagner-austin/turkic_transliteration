"""Tests for the rules that detect weak or fake tests.

Every case writes real Python to a real ``tests`` directory and runs the
real rule over it. The rules parse source, so giving them source is the
only honest way to exercise them: a stand-in AST would test the test, not
the rule.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.guards.test_quality_rules import (
    PatchingRule,
    TransliterationTestQualityRule,
    WeakAssertionRule,
    is_test_module,
)


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
        """Write source into the tests directory being scanned.

        Args:
            name: File name within that directory.
            source: File contents.

        Returns:
            The written path.
        """
        path = tests_dir / name
        path.write_text(source, encoding="utf-8")
        return path

    return write


def kinds(paths: list[Path]) -> list[str]:
    """Run the weak-assertion rule and return the violation kinds.

    Args:
        paths: Files to scan.

    Returns:
        The kind of each violation, in the order reported.
    """
    return [violation.kind for violation in WeakAssertionRule().run(paths)]


def patch_kinds(paths: list[Path]) -> list[str]:
    """Run the patching rule and return the violation kinds.

    Args:
        paths: Files to scan.

    Returns:
        The kind of each violation, in the order reported.
    """
    return [violation.kind for violation in PatchingRule().run(paths)]


def test_a_monkeypatch_fixture_argument_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Requesting the fixture is the violation, before any call."""
    path = write_test_module("test_a.py", "def test_x(monkeypatch) -> None:\n    assert True\n")

    assert patch_kinds([path]) == ["monkeypatch-usage"]


def test_a_monkeypatch_call_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Using the name anywhere is flagged, however it arrived."""
    path = write_test_module(
        "test_a.py", "def test_x() -> None:\n    monkeypatch.setenv('A', 'b')\n"
    )

    assert patch_kinds([path]) == ["monkeypatch-usage"]


def test_the_monkeypatch_type_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """An annotation naming the class is the same coupling."""
    path = write_test_module(
        "test_a.py",
        "import pytest\ndef test_x(mp: pytest.MonkeyPatch) -> None:\n    assert True\n",
    )

    assert patch_kinds([path]) == ["monkeypatch-usage"]


def test_one_line_reports_one_violation_per_kind(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The argument and its annotation on one line count once."""
    path = write_test_module(
        "test_a.py",
        "import pytest\ndef test_x(monkeypatch: pytest.MonkeyPatch) -> None:\n    assert True\n",
    )

    assert patch_kinds([path]) == ["monkeypatch-usage"]


@pytest.mark.parametrize(
    "source",
    [
        "import unittest.mock\n",
        "import mock\n",
        "from unittest.mock import Mock\n",
        "from mock import patch\n",
        "from unittest import mock\n",
    ],
)
def test_every_way_of_importing_a_mock_library_is_reported(
    write_test_module: Callable[[str, str], Path], source: str
) -> None:
    """The double answers every call, so the import itself is the defect.

    Args:
        write_test_module: Helper writing the module.
        source: One spelling of the import.
    """
    path = write_test_module("test_a.py", source)

    assert patch_kinds([path]) == ["mock-library-import"]


@pytest.mark.parametrize(
    "source",
    [
        "import unittest\n",
        "from unittest import TestCase\n",
        "from turkic_translit import _test_hooks\n",
        "def test_x() -> None:\n    assert True\n",
    ],
)
def test_ordinary_imports_and_tests_are_left_alone(
    write_test_module: Callable[[str, str], Path], source: str
) -> None:
    """Only the patching libraries are restricted.

    Args:
        write_test_module: Helper writing the module.
        source: Source that must not be flagged.
    """
    path = write_test_module("test_a.py", source)

    assert patch_kinds([path]) == []


def test_violations_are_reported_in_source_order(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Several offences in one file come back top to bottom."""
    path = write_test_module(
        "test_a.py",
        "from unittest.mock import Mock\n\n\ndef test_x(monkeypatch) -> None:\n    assert True\n",
    )

    violations = PatchingRule().run([path])

    assert [violation.kind for violation in violations] == [
        "mock-library-import",
        "monkeypatch-usage",
    ]
    assert [violation.line_no for violation in violations] == [1, 4]
    assert violations[0].line == "from unittest.mock import Mock"


def test_a_module_outside_tests_is_not_scanned(tmp_path: Path) -> None:
    """Production code may name anything it likes."""
    path = tmp_path / "src" / "monkeypatch_helper.py"
    path.parent.mkdir()
    path.write_text("monkeypatch = 1\n", encoding="utf-8")

    assert patch_kinds([path]) == []


def test_a_helper_module_inside_tests_is_not_scanned(tmp_path: Path) -> None:
    """The rule applies to test modules, not to their helpers."""
    path = tmp_path / "tests" / "helpers.py"
    path.parent.mkdir()
    path.write_text("from unittest.mock import Mock\n", encoding="utf-8")

    assert patch_kinds([path]) == []


def test_the_rule_is_named_for_its_report() -> None:
    """The summary line names this rule as ``patching``."""
    assert PatchingRule().name == "patching"


def test_only_test_modules_inside_a_tests_directory_are_scanned(tmp_path: Path) -> None:
    """The filter needs both the directory and the file-name prefix."""
    inside = tmp_path / "tests" / "test_a.py"
    inside.parent.mkdir()
    inside.touch()
    helper = tmp_path / "tests" / "helpers.py"
    helper.touch()
    outside = tmp_path / "src" / "test_b.py"
    outside.parent.mkdir()
    outside.touch()

    assert is_test_module(inside) is True
    assert is_test_module(helper) is False
    assert is_test_module(outside) is False


def test_a_source_module_is_skipped_entirely(tmp_path: Path) -> None:
    """A weak assertion outside tests is not this rule's business."""
    source = tmp_path / "src" / "thing.py"
    source.parent.mkdir()
    source.write_text("def test_x() -> None:\n    assert x is not None\n", encoding="utf-8")

    assert kinds([source]) == []


def test_a_helper_module_in_tests_is_skipped(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Only ``test_*`` modules are scanned, so helpers may do as they like."""
    helper = write_test_module("conftest.py", "def test_x():\n    assert x is not None\n")

    assert kinds([helper]) == []


def test_an_is_not_none_assertion_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Existence is not behaviour."""
    path = write_test_module("test_a.py", "def test_a():\n    assert value is not None\n")

    assert kinds([path]) == ["weak-assertion-is-not-none"]


def test_an_is_none_assertion_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Asserting a value *is* None pins it, so it is not weak."""
    path = write_test_module("test_a.py", "def test_a():\n    assert value is None\n")

    assert kinds([path]) == []


def test_is_not_against_a_non_constant_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """``x is not y`` compares two objects and is a real claim."""
    path = write_test_module("test_a.py", "def test_a():\n    assert first is not second\n")

    assert kinds([path]) == []


def test_is_not_against_another_constant_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Only ``None`` is singled out; other constants are left alone."""
    path = write_test_module("test_a.py", "def test_a():\n    assert value is not False\n")

    assert kinds([path]) == []


def test_a_chained_comparison_is_not_treated_as_identity(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A multi-operator compare is not the single ``is not`` form."""
    path = write_test_module("test_a.py", "def test_a():\n    assert a < b < c\n")

    assert kinds([path]) == []


def test_an_isinstance_assertion_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A type check says nothing about the value."""
    path = write_test_module("test_a.py", "def test_a():\n    assert isinstance(v, str)\n")

    assert kinds([path]) == ["weak-assertion-isinstance"]


def test_a_hasattr_assertion_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """An attribute existing says nothing about its value."""
    path = write_test_module("test_a.py", "def test_a():\n    assert hasattr(v, 'name')\n")

    assert kinds([path]) == ["weak-assertion-hasattr"]


def test_an_ordinary_call_in_an_assertion_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Only the two named builtins are weak; other calls are fine."""
    path = write_test_module("test_a.py", "def test_a():\n    assert callable(v)\n")

    assert kinds([path]) == []


def test_a_length_greater_than_zero_assertion_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Non-emptiness is existence, not content."""
    path = write_test_module("test_a.py", "def test_a():\n    assert len(items) > 0\n")

    assert kinds([path]) == ["weak-assertion-len-zero"]


def test_a_length_at_least_one_assertion_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The other spelling of the same weak claim."""
    path = write_test_module("test_a.py", "def test_a():\n    assert len(items) >= 1\n")

    assert kinds([path]) == ["weak-assertion-len-zero"]


def test_an_exact_length_assertion_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Pinning the count is a real claim."""
    path = write_test_module("test_a.py", "def test_a():\n    assert len(items) == 3\n")

    assert kinds([path]) == []


def test_a_length_above_a_real_bound_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Only the zero and one bounds are the existence check."""
    path = write_test_module("test_a.py", "def test_a():\n    assert len(items) > 5\n")

    assert kinds([path]) == []


def test_a_length_compared_to_a_variable_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A non-constant bound cannot be the existence check."""
    path = write_test_module("test_a.py", "def test_a():\n    assert len(items) > limit\n")

    assert kinds([path]) == []


def test_a_non_len_call_compared_to_zero_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The rule is about ``len``, not about comparisons to zero."""
    path = write_test_module("test_a.py", "def test_a():\n    assert count(items) > 0\n")

    assert kinds([path]) == []


def test_a_bare_name_compared_to_zero_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Without a call on the left there is no ``len`` to object to."""
    path = write_test_module("test_a.py", "def test_a():\n    assert total > 0\n")

    assert kinds([path]) == []


def test_a_chained_length_comparison_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A range check is more specific than the existence check."""
    path = write_test_module("test_a.py", "def test_a():\n    assert 0 < len(items) < 9\n")

    assert kinds([path]) == []


@pytest.mark.parametrize("stream", ["out", "err", "stdout", "stderr"])
def test_substring_matching_on_captured_output_is_reported(
    write_test_module: Callable[[str, str], Path], stream: str
) -> None:
    """Searching a captured stream for a phrase is fragile.

    Args:
        stream: The attribute name the assertion reads.
    """
    path = write_test_module("test_a.py", f"def test_a():\n    assert 'hi' in result.{stream}\n")

    assert kinds([path]) == ["weak-assertion-in-output"]


def test_substring_matching_on_another_attribute_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Only the captured-stream names are singled out."""
    path = write_test_module("test_a.py", "def test_a():\n    assert 'hi' in result.output\n")

    assert kinds([path]) == []


def test_membership_in_a_plain_name_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Without an attribute on the right there is no captured stream."""
    path = write_test_module("test_a.py", "def test_a():\n    assert 'hi' in greeting\n")

    assert kinds([path]) == []


def test_not_in_a_captured_stream_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The rule matches membership, not its negation."""
    path = write_test_module("test_a.py", "def test_a():\n    assert 'hi' not in result.out\n")

    assert kinds([path]) == []


def test_asserting_a_mock_was_called_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """That it was called says nothing about with what."""
    path = write_test_module("test_a.py", "def test_a():\n    assert sender.called\n")

    assert kinds([path]) == ["mock-without-assert-called-with"]


def test_asserting_another_attribute_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Only ``.called`` is the mock smell."""
    path = write_test_module("test_a.py", "def test_a():\n    assert sender.delivered\n")

    assert kinds([path]) == []


def test_four_patches_in_one_test_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Past three patches the test is describing a different program."""
    source = "def test_a():\n    patch('a')\n    patch('b')\n    patch('c')\n    patch('d')\n"
    path = write_test_module("test_a.py", source)

    assert kinds([path]) == ["excessive-mocking"]


def test_three_patches_is_allowed(write_test_module: Callable[[str, str], Path]) -> None:
    """Three is the documented limit, not the first violation."""
    source = "def test_a():\n    patch('a')\n    patch('b')\n    patch('c')\n"
    path = write_test_module("test_a.py", source)

    assert kinds([path]) == []


def test_patches_are_counted_through_an_attribute_too(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """``mock.patch`` counts the same as a bare ``patch``."""
    source = (
        "def test_a():\n"
        "    mock.patch('a')\n"
        "    mock.patch('b')\n"
        "    mock.patch('c')\n"
        "    mock.patch('d')\n"
    )
    path = write_test_module("test_a.py", source)

    assert kinds([path]) == ["excessive-mocking"]


def test_another_call_is_not_counted_as_a_patch(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Only ``patch`` is counted, however many other calls there are."""
    source = "def test_a():\n" + "".join(f"    build('{n}')\n" for n in "abcdef")
    path = write_test_module("test_a.py", source)

    assert kinds([path]) == []


def test_an_async_test_is_analysed_like_a_synchronous_one(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The visitor handles both function forms."""
    path = write_test_module("test_a.py", "async def test_a():\n    assert value is not None\n")

    assert kinds([path]) == ["weak-assertion-is-not-none"]


def test_a_non_test_function_is_not_analysed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Helpers inside a test module may assert however they need to."""
    path = write_test_module("test_a.py", "def helper():\n    assert value is not None\n")

    assert kinds([path]) == []


def test_a_non_test_async_function_is_not_analysed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The async visitor applies the same name filter as the sync one."""
    path = write_test_module("test_a.py", "async def helper():\n    assert value is not None\n")

    assert kinds([path]) == []


def test_a_length_check_against_a_non_call_left_side_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Without a call on the left there is nothing that could be ``len``."""
    path = write_test_module("test_a.py", "def test_a():\n    assert items.count > 0\n")

    assert kinds([path]) == []


def test_a_length_leading_a_chained_comparison_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """``len(x) > 0 > floor`` says more than plain non-emptiness.

    The left side is a ``len`` call, so the rule gets past its first two
    checks and is turned away by the operator count.
    """
    path = write_test_module("test_a.py", "def test_a():\n    assert len(items) > 0 > floor\n")

    assert kinds([path]) == []


def test_a_meaningful_comparison_is_recorded_without_complaint(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Comparing two variables is the pattern the rules want to see."""
    path = write_test_module("test_a.py", "def test_a():\n    assert actual == expected\n")

    assert kinds([path]) == []


def test_a_comparison_of_two_constants_is_not_meaningful(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Neither side is a variable, so nothing about the code is claimed."""
    path = write_test_module("test_a.py", "def test_a():\n    assert 1 == 1\n")

    assert kinds([path]) == []


def test_an_identity_comparison_is_not_counted_as_meaningful(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """``is`` is not one of the comparison operators tracked."""
    path = write_test_module("test_a.py", "def test_a():\n    assert actual is expected\n")

    assert kinds([path]) == []


def translit_kinds(paths: list[Path]) -> list[str]:
    """Run the transliteration rule and return the violation kinds.

    Args:
        paths: Files to scan.

    Returns:
        The kind of each violation, in the order reported.
    """
    return [violation.kind for violation in TransliterationTestQualityRule().run(paths)]


def test_a_domain_call_without_a_value_assertion_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Calling to_ipa and asserting nothing proves only that it ran."""
    path = write_test_module("test_a.py", "def test_a():\n    to_ipa('x', 'kk')\n")

    assert translit_kinds([path]) == ["translit-call-without-value-assertion"]


def test_a_domain_call_inside_pytest_raises_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Naming the exception pins the outcome as firmly as an equality.

    A call that is expected to raise returns no value to compare, so
    requiring an equality assertion would require a value that does not
    exist.
    """
    path = write_test_module(
        "test_a.py",
        "def test_a():\n"
        "    with pytest.raises(ValueError, match='no rules'):\n"
        "        to_ipa('x', 'zz')\n",
    )

    assert translit_kinds([path]) == []


def test_a_domain_call_inside_a_bare_raises_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """``raises`` imported from pytest is recognised unqualified."""
    path = write_test_module(
        "test_a.py",
        "def test_a():\n    with raises(ValueError):\n        to_latin('x', 'zz')\n",
    )

    assert translit_kinds([path]) == []


def test_a_with_item_that_is_not_a_call_is_still_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A context manager named rather than called verifies nothing."""
    path = write_test_module(
        "test_a.py",
        "def test_a():\n    with suppressor:\n        to_ipa('x', 'kk')\n",
    )

    assert translit_kinds([path]) == ["translit-call-without-value-assertion"]


def test_a_domain_call_inside_another_context_manager_is_still_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Only ``raises`` counts; an ordinary ``with`` verifies nothing."""
    path = write_test_module(
        "test_a.py",
        "def test_a():\n    with open('f') as handle:\n        to_ipa(handle.read(), 'kk')\n",
    )

    assert translit_kinds([path]) == ["translit-call-without-value-assertion"]


def test_a_domain_call_with_an_equality_assertion_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Pinning the result is what the rule is asking for."""
    path = write_test_module("test_a.py", "def test_a():\n    assert to_ipa('x', 'kk') == 'y'\n")

    assert translit_kinds([path]) == []


def test_a_domain_call_with_a_membership_assertion_is_allowed(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Membership pins the value too."""
    path = write_test_module("test_a.py", "def test_a():\n    assert 'a' in to_latin('x', 'kk')\n")

    assert translit_kinds([path]) == []


def test_a_domain_call_through_an_attribute_is_recognised(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """``core.transliterate(...)`` is the same domain call."""
    path = write_test_module("test_a.py", "def test_a():\n    core.transliterate('x')\n")

    assert translit_kinds([path]) == ["translit-call-without-value-assertion"]


def test_an_assert_equal_helper_counts_as_a_value_assertion(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The unittest-style helpers pin values just as well."""
    path = write_test_module(
        "test_a.py", "def test_a():\n    self.assertEqual(to_ipa('x', 'kk'), 'y')\n"
    )

    assert translit_kinds([path]) == []


def test_a_bare_truth_assertion_does_not_rescue_a_domain_call(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """``assert result`` is not a comparison and pins nothing."""
    path = write_test_module(
        "test_a.py", "def test_a():\n    result = to_ipa('x', 'kk')\n    assert result\n"
    )

    assert translit_kinds([path]) == ["translit-call-without-value-assertion"]


def test_a_test_without_a_domain_call_is_ignored(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The rule only speaks to tests that exercise the domain."""
    path = write_test_module("test_a.py", "def test_a():\n    helper()\n")

    assert translit_kinds([path]) == []


def test_the_transliteration_rule_skips_source_modules(tmp_path: Path) -> None:
    """Only test modules are considered, same as the other rule."""
    source = tmp_path / "src" / "thing.py"
    source.parent.mkdir()
    source.write_text("def test_a():\n    to_ipa('x', 'kk')\n", encoding="utf-8")

    assert translit_kinds([source]) == []


def test_the_transliteration_rule_skips_helper_modules(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """A conftest is not a test module."""
    path = write_test_module("conftest.py", "def test_a():\n    to_ipa('x', 'kk')\n")

    assert translit_kinds([path]) == []


def test_an_async_domain_test_is_reported(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Both function forms are walked."""
    path = write_test_module("test_a.py", "async def test_a():\n    to_ipa('x', 'kk')\n")

    assert translit_kinds([path]) == ["translit-call-without-value-assertion"]


def test_a_non_test_function_calling_the_domain_is_ignored(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """Helpers may call the domain without asserting."""
    path = write_test_module("test_a.py", "def build():\n    to_ipa('x', 'kk')\n")

    assert translit_kinds([path]) == []


def test_the_reported_line_quotes_the_offending_definition(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The violation points at the def so the report is actionable."""
    path = write_test_module("test_a.py", "\n\ndef test_late():\n    to_ipa('x', 'kk')\n")

    violations = TransliterationTestQualityRule().run([path])

    assert [(v.line_no, v.line) for v in violations] == [(3, "def test_late():")]


def test_the_weak_assertion_report_quotes_the_offending_line(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The violation carries the source line, stripped of indentation."""
    path = write_test_module("test_a.py", "def test_a():\n    assert value is not None\n")

    violations = WeakAssertionRule().run([path])

    assert [(v.line_no, v.line) for v in violations] == [(2, "assert value is not None")]
