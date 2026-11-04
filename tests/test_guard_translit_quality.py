"""Tests for the transliteration test-quality rule.

Every case writes real Python to a real ``tests`` directory and runs the
real rule over it. These cases were in ``test_guard_test_quality.py``
until 2026-11-04, when they moved here unchanged beside the rule, which
moved to ``scripts/guards/translit_test_rules.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from scripts.guards.translit_test_rules import TransliterationTestQualityRule


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
