"""Tests for the guard's typing, comment, suppression and logging rules.

Each case writes real Python into a real file or parses real source and
runs the real rule over it; nothing here substitutes an AST or a reader.
These cases were in ``tests/test_guard_rules.py`` until 2026-11-04, when
they moved here unchanged beside the rules they test, which moved to
``scripts/guards/typing_rules.py``, so both files stay under the
600-line limit.
"""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.guards.typing_rules import (
    _check_object_annotations,
    _is_suppress,
    _run_comments_rule,
    _run_logging_rule,
    _run_suppress_rule,
    _run_typing_rule,
)


def typing_kinds(source: str) -> list[str]:
    """Run the typing rule over source and return the violation kinds.

    Args:
        source: Python source to analyse.

    Returns:
        The kind of each violation, in the order reported.
    """
    tree = ast.parse(source)
    return [v.kind for v in _run_typing_rule(Path("m.py"), tree)]


def sole_expression(source: str) -> ast.expr:
    """Parse a one-expression module and return that expression.

    Args:
        source: Source holding exactly one expression statement.

    Returns:
        The expression node.

    Raises:
        TypeError: If the source's first statement is something else,
            which means the test case was written wrong.
    """
    statement = ast.parse(source).body[0]
    if not isinstance(statement, ast.Expr):
        raise TypeError(f"expected an expression, got {type(statement).__name__}")
    return statement.value


def test_importing_any_from_typing_is_reported() -> None:
    """The import is flagged even before the name is used."""
    assert typing_kinds("from typing import Any\n") == ["typing-import-any"]


def test_importing_cast_and_type_alias_are_reported() -> None:
    """All three forbidden names are recognised."""
    assert typing_kinds("from typing import cast, TypeAlias\n") == [
        "typing-import-cast",
        "typing-import-typealias",
    ]


def test_importing_type_checking_is_reported() -> None:
    """The import alone is the violation, before any block is opened."""
    assert typing_kinds("from typing import TYPE_CHECKING\n") == ["typing-import-type_checking"]


def test_a_type_checking_block_is_reported() -> None:
    """The guarded import exists for the checker and not at runtime."""
    assert typing_kinds("if TYPE_CHECKING:\n    from a import B\n") == ["type-checking-usage"]


def test_a_qualified_type_checking_attribute_is_reported() -> None:
    """``typing.TYPE_CHECKING`` is the same violation spelled differently."""
    assert typing_kinds("import typing\nif typing.TYPE_CHECKING:\n    pass\n") == [
        "typing-type_checking-usage"
    ]


def test_importing_a_permitted_name_from_typing_is_allowed() -> None:
    """Protocol and the rest of typing are not restricted."""
    assert typing_kinds("from typing import Protocol\n") == []


def test_importing_from_another_module_is_allowed() -> None:
    """A name called Any from elsewhere is not typing's Any."""
    assert typing_kinds("from mymod import Any\n") == []


def test_a_qualified_typing_attribute_is_reported() -> None:
    """``typing.Any`` is the same violation spelled differently."""
    assert typing_kinds("import typing\nx: typing.Any\n") == ["typing-any-usage"]


def test_a_qualified_attribute_on_another_module_is_allowed() -> None:
    """``other.Any`` is not typing's Any."""
    assert typing_kinds("import other\nx = other.Any\n") == []


def test_a_permitted_typing_attribute_is_allowed() -> None:
    """``typing.Protocol`` is not on the forbidden list."""
    assert typing_kinds("import typing\nx = typing.Protocol\n") == []


def test_a_cast_call_is_reported() -> None:
    """Calling cast is flagged wherever it came from."""
    assert typing_kinds("y = cast(int, x)\n") == ["cast-call"]


def test_a_call_through_an_attribute_is_not_a_cast_call() -> None:
    """``t.cast(...)`` is caught by the attribute rule, not this one."""
    assert typing_kinds("import other\ny = other.cast(int, x)\n") == []


def test_a_bare_any_name_is_reported() -> None:
    """Using the name at all is a violation."""
    assert typing_kinds("def f(x: Any) -> None: ...\n") == ["any-usage"]


def test_object_in_a_variable_annotation_is_reported() -> None:
    """An annotated assignment naming object is flagged."""
    assert typing_kinds("x: object = 1\n") == ["object-in-annotation"]


def test_object_in_an_argument_annotation_is_reported() -> None:
    """A parameter annotated object is flagged."""
    assert typing_kinds("def f(x: object) -> None: ...\n") == ["object-in-annotation"]


def test_object_in_a_return_annotation_is_reported() -> None:
    """A return annotated object is flagged."""
    assert typing_kinds("def f() -> object: ...\n") == ["object-in-annotation"]


def test_object_in_an_async_return_annotation_is_reported() -> None:
    """Async definitions are checked the same way."""
    assert typing_kinds("async def f() -> object: ...\n") == ["object-in-annotation"]


def test_object_nested_inside_an_annotation_is_reported() -> None:
    """``dict[str, object]`` hides the same escape hatch."""
    assert typing_kinds("x: dict[str, object] = {}\n") == ["object-in-annotation"]


def test_an_unannotated_assignment_is_allowed() -> None:
    """A plain assignment carries no annotation to inspect."""
    assert typing_kinds("x = object()\n") == []


def test_an_unannotated_parameter_is_allowed() -> None:
    """A bare parameter has no annotation to contain object."""
    assert typing_kinds("def f(x): ...\n") == []


def test_a_function_without_a_return_annotation_is_allowed() -> None:
    """Missing annotations are Ruff's business, not this rule's."""
    assert typing_kinds("def f(x: int): ...\n") == []


def test_the_object_check_ignores_unrelated_nodes() -> None:
    """A node that is neither an annotation nor a def yields nothing."""
    node = ast.parse("x = 1\n").body[0]

    assert _check_object_annotations(Path("m.py"), node) == []


def test_a_type_ignore_comment_is_reported(tmp_path: Path) -> None:
    """The suppression comment is flagged wherever it appears."""
    source = "x = 1  # type: ignore[arg-type]\n"

    kinds = [v.kind for v in _run_comments_rule(tmp_path / "m.py", source)]

    assert kinds == ["type-ignore"]


def test_a_pragma_comment_is_reported(tmp_path: Path) -> None:
    """Coverage pragmas hide untested paths."""
    source = "if x:  # pragma: no cover\n    pass\n"

    kinds = [v.kind for v in _run_comments_rule(tmp_path / "m.py", source)]

    assert kinds == ["pragma-comment"]


def test_one_comment_can_be_both_violations(tmp_path: Path) -> None:
    """A comment carrying both markers is reported twice."""
    source = "x = 1  # type: ignore and pragma too\n"

    kinds = [v.kind for v in _run_comments_rule(tmp_path / "m.py", source)]

    assert kinds == ["type-ignore", "pragma-comment"]


def test_an_ordinary_comment_is_allowed(tmp_path: Path) -> None:
    """Comments are not otherwise restricted."""
    source = "x = 1  # this explains why\n"

    assert _run_comments_rule(tmp_path / "m.py", source) == []


def test_the_phrase_in_a_string_is_not_a_comment(tmp_path: Path) -> None:
    """Only comment tokens are inspected."""
    source = 'x = "type: ignore"\n'

    assert _run_comments_rule(tmp_path / "m.py", source) == []


def test_contextlib_suppress_is_recognised() -> None:
    """The qualified call is the canonical form."""
    call = sole_expression("contextlib.suppress(OSError)\n")

    assert _is_suppress(call) is True


def test_a_bare_suppress_name_is_recognised() -> None:
    """An imported suppress used directly counts too."""
    call = sole_expression("suppress(OSError)\n")

    assert _is_suppress(call) is True


def test_another_module_s_suppress_is_not_recognised() -> None:
    """Only contextlib's suppress is the one being banned."""
    call = sole_expression("other.suppress(OSError)\n")

    assert _is_suppress(call) is False


def test_another_contextlib_helper_is_not_recognised() -> None:
    """``contextlib.closing`` is not suppression."""
    call = sole_expression("contextlib.closing(handle)\n")

    assert _is_suppress(call) is False


def test_an_unrelated_call_is_not_recognised() -> None:
    """A plain function call is not suppression."""
    call = sole_expression("build(OSError)\n")

    assert _is_suppress(call) is False


def suppress_kinds(source: str) -> list[str]:
    """Run the suppress rule over source and return violation kinds.

    Args:
        source: Python source to analyse.

    Returns:
        The kind of each violation, in the order reported.
    """
    lines = source.splitlines()
    return [v.kind for v in _run_suppress_rule(Path("m.py"), ast.parse(source), lines)]


def test_a_with_suppress_block_is_reported() -> None:
    """Suppressing an exception is the thing being banned."""
    assert suppress_kinds("import contextlib\nwith contextlib.suppress(OSError):\n    go()\n") == [
        "suppress-usage"
    ]


def test_an_async_with_suppress_block_is_reported() -> None:
    """Both statement forms are walked."""
    source = (
        "import contextlib\n"
        "async def f():\n"
        "    async with contextlib.suppress(OSError):\n"
        "        await go()\n"
    )

    assert suppress_kinds(source) == ["suppress-usage"]


def test_two_suppressions_on_one_line_are_reported_once() -> None:
    """The report is per line, so a repeat on the same line is skipped."""
    source = "import contextlib\nwith contextlib.suppress(OSError), contextlib.suppress(KeyError):\n    go()\n"

    assert suppress_kinds(source) == ["suppress-usage"]


def test_an_ordinary_with_block_is_allowed() -> None:
    """Context managers in general are fine."""
    assert suppress_kinds("with open('f') as handle:\n    handle.read()\n") == []


def test_the_reported_suppress_line_is_the_source_line() -> None:
    """The violation quotes the line so the report is actionable."""
    source = "import contextlib\nwith contextlib.suppress(OSError):\n    go()\n"

    violations = _run_suppress_rule(Path("m.py"), ast.parse(source), source.splitlines())

    assert [v.line for v in violations] == ["with contextlib.suppress(OSError):"]


def test_a_print_in_source_is_reported(tmp_path: Path) -> None:
    """Library code emits through logging so callers control the sink."""
    path = tmp_path / "src" / "pkg" / "m.py"
    path.parent.mkdir(parents=True)
    tree = ast.parse("print('hello')\n")

    assert [v.kind for v in _run_logging_rule(path, tree)] == ["print-usage"]


def test_a_print_in_tests_is_allowed(tmp_path: Path) -> None:
    """Only files under src are restricted."""
    path = tmp_path / "tests" / "test_m.py"
    path.parent.mkdir(parents=True)
    tree = ast.parse("print('hello')\n")

    assert _run_logging_rule(path, tree) == []


def test_a_print_through_an_attribute_is_allowed(tmp_path: Path) -> None:
    """``logger.print`` is not the builtin."""
    path = tmp_path / "src" / "m.py"
    path.parent.mkdir(parents=True)
    tree = ast.parse("logger.print('hello')\n")

    assert _run_logging_rule(path, tree) == []
