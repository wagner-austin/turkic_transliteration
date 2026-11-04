"""Tests for the weak-assertion rule and the test-module filter it shares.

Every case writes real Python to a real ``tests`` directory and runs the
real rule over it. The rules parse source, so giving them source is the
only honest way to exercise them: a stand-in AST would test the test, not
the rule. The patching and transliteration rules are tested in
``test_guard_patching.py`` and ``test_guard_translit_quality.py``, and the
``write_test_module`` fixture all three use is in ``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.guards.test_quality_rules import WeakAssertionRule, is_test_module


def kinds(paths: list[Path]) -> list[str]:
    """Run the weak-assertion rule and return the violation kinds.

    Args:
        paths: Files to scan.

    Returns:
        The kind of each violation, in the order reported.
    """
    return [violation.kind for violation in WeakAssertionRule().run(paths)]


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


def test_the_weak_assertion_report_quotes_the_offending_line(
    write_test_module: Callable[[str, str], Path],
) -> None:
    """The violation carries the source line, stripped of indentation."""
    path = write_test_module("test_a.py", "def test_a():\n    assert value is not None\n")

    violations = WeakAssertionRule().run([path])

    assert [(v.line_no, v.line) for v in violations] == [(2, "assert value is not None")]
