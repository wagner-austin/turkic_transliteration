"""Tests for the rule that keeps docstrings in Google style.

Every case writes real Python to a real file and runs the real rule over
it. The rule parses source, so source is what it is given.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.guards.docstring_rules import (
    DocstringRule,
    declared_parameters,
    documents_arguments,
    is_python_module,
    returns_a_value,
)


@pytest.fixture
def write_module(tmp_path: Path) -> Callable[[str, str], Path]:
    """Return a helper writing source anywhere under a project tree.

    Args:
        tmp_path: Root to build the tree in.

    Returns:
        A function taking a repo-relative path and source, returning the
        written path.
    """

    def write(relative: str, source: str) -> Path:
        """Write source at a path within the tree.

        Args:
            relative: Path within the tree, directories created as
                needed.
            source: File contents.

        Returns:
            The written path.
        """
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    return write


def kinds(paths: list[Path]) -> list[str]:
    """Run the docstring rule and return the violation kinds.

    Args:
        paths: Files to scan.

    Returns:
        The kind of each violation, in the order reported.
    """
    return [violation.kind for violation in DocstringRule().run(paths)]


def test_an_undocumented_function_is_reported(
    write_module: Callable[[str, str], Path],
) -> None:
    """A function with no docstring at all is the first thing flagged."""
    path = write_module("src/m.py", "def f(x: int) -> int:\n    return x\n")

    assert kinds([path]) == ["docstring-missing"]


def test_a_documented_function_naming_nothing_is_reported(
    write_module: Callable[[str, str], Path],
) -> None:
    """A summary alone leaves the signature undescribed."""
    path = write_module("src/m.py", 'def f(x: int) -> int:\n    """Do it."""\n    return x\n')

    assert kinds([path]) == ["docstring-missing-args", "docstring-missing-returns"]


def test_a_fully_documented_function_is_allowed(
    write_module: Callable[[str, str], Path],
) -> None:
    """Naming the parameters and the result satisfies the rule."""
    source = (
        "def f(x: int) -> int:\n"
        '    """Do it.\n\n'
        "    Args:\n"
        "        x: The number.\n\n"
        "    Returns:\n"
        "        The number.\n"
        '    """\n'
        "    return x\n"
    )
    path = write_module("src/m.py", source)

    assert kinds([path]) == []


def test_a_generator_may_document_yields_instead(
    write_module: Callable[[str, str], Path],
) -> None:
    """``Yields:`` describes a generator's result as well as ``Returns:``."""
    source = (
        "def f() -> Iterator[int]:\n"
        '    """Count.\n\n'
        "    Yields:\n"
        "        Each number.\n"
        '    """\n'
        "    yield 1\n"
    )
    path = write_module("src/m.py", source)

    assert kinds([path]) == []


def test_a_function_returning_nothing_needs_no_return_section(
    write_module: Callable[[str, str], Path],
) -> None:
    """An explicit ``None`` return is not a result to describe."""
    path = write_module("src/m.py", 'def f() -> None:\n    """Do it."""\n')

    assert kinds([path]) == []


def test_a_function_that_never_returns_needs_no_return_section(
    write_module: Callable[[str, str], Path],
) -> None:
    """``NoReturn`` describes a function that raises, not one that returns."""
    source = (
        "def f() -> NoReturn:\n"
        '    """Stop.\n\n'
        "    Raises:\n"
        "        SystemExit: Always.\n"
        '    """\n'
        "    raise SystemExit(1)\n"
    )
    path = write_module("src/m.py", source)

    assert kinds([path]) == []


def test_a_qualified_no_return_is_recognised(
    write_module: Callable[[str, str], Path],
) -> None:
    """``typing.NoReturn`` is the same annotation spelled differently."""
    path = write_module("src/m.py", 'def f() -> typing.NoReturn:\n    """Stop."""\n    raise E\n')

    assert kinds([path]) == []


def test_an_unannotated_return_needs_no_return_section(
    write_module: Callable[[str, str], Path],
) -> None:
    """With no annotation there is nothing for the rule to check."""
    path = write_module("src/m.py", 'def f():\n    """Do it."""\n    return 1\n')

    assert kinds([path]) == []


def test_a_constructor_may_take_its_args_from_the_class(
    write_module: Callable[[str, str], Path],
) -> None:
    """Google style documents constructor arguments on the class."""
    source = (
        "class C:\n"
        '    """A thing.\n\n'
        "    Args:\n"
        "        value: What it holds.\n"
        '    """\n\n'
        "    def __init__(self, value: int) -> None:\n"
        '        """Store the value."""\n'
        "        self.value = value\n"
    )
    path = write_module("src/m.py", source)

    assert kinds([path]) == []


def test_a_constructor_is_reported_when_the_class_names_nothing(
    write_module: Callable[[str, str], Path],
) -> None:
    """Without the class docstring the constructor must name them itself."""
    source = (
        "class C:\n"
        '    """A thing."""\n\n'
        "    def __init__(self, value: int) -> None:\n"
        '        """Store the value."""\n'
        "        self.value = value\n"
    )
    path = write_module("src/m.py", source)

    assert kinds([path]) == ["docstring-missing-args"]


def test_a_class_without_a_docstring_does_not_exempt_its_constructor(
    write_module: Callable[[str, str], Path],
) -> None:
    """A class with no docstring documents nothing for anyone."""
    source = 'class C:\n    def __init__(self, value: int) -> None:\n        """Store."""\n'
    path = write_module("src/m.py", source)

    assert kinds([path]) == ["docstring-missing-args"]


def test_self_and_cls_are_not_parameters_to_document(
    write_module: Callable[[str, str], Path],
) -> None:
    """The receiver describes the call's subject, not its arguments."""
    source = (
        "class C:\n"
        '    """A thing."""\n\n'
        "    def method(self) -> None:\n"
        '        """Do it."""\n\n'
        "    @classmethod\n"
        "    def build(cls) -> None:\n"
        '        """Build it."""\n'
    )
    path = write_module("src/m.py", source)

    assert kinds([path]) == []


def test_variadic_parameters_must_be_documented(
    write_module: Callable[[str, str], Path],
) -> None:
    """``*args`` and ``**kwargs`` carry the call's data like any other."""
    path = write_module("src/m.py", 'def f(*a: int, **k: str) -> None:\n    """Do it."""\n')

    assert kinds([path]) == ["docstring-missing-args"]


def test_a_test_module_needs_a_docstring_but_not_args(
    write_module: Callable[[str, str], Path],
) -> None:
    """A test's parameters are fixtures, which document themselves."""
    documented = write_module(
        "tests/test_a.py", 'def test_x(tmp_path) -> None:\n    """Do it."""\n'
    )
    bare = write_module("tests/test_b.py", "def test_y(tmp_path) -> None:\n    pass\n")

    assert kinds([documented]) == []
    assert kinds([bare]) == ["docstring-missing"]


def test_a_test_module_still_documents_what_it_returns(
    write_module: Callable[[str, str], Path],
) -> None:
    """The return exemption is for parameters only."""
    path = write_module("tests/test_a.py", 'def helper() -> int:\n    """Do it."""\n    return 1\n')

    assert kinds([path]) == ["docstring-missing-returns"]


def test_a_non_python_file_is_skipped(write_module: Callable[[str, str], Path]) -> None:
    """The rule reads Python, so it is given only Python."""
    path = write_module("src/notes.txt", "def f(): pass\n")

    assert kinds([path]) == []


def test_violations_are_reported_in_source_order(
    write_module: Callable[[str, str], Path],
) -> None:
    """Several offences in one file come back top to bottom."""
    source = 'def a() -> int:\n    return 1\n\n\ndef b(x: int) -> None:\n    """Do it."""\n'
    path = write_module("src/m.py", source)

    violations = DocstringRule().run([path])

    assert [violation.kind for violation in violations] == [
        "docstring-missing",
        "docstring-missing-args",
    ]
    assert [violation.line_no for violation in violations] == [1, 5]
    assert violations[0].line == "def a() -> int:"


def test_the_rule_is_named_for_its_report() -> None:
    """The summary line names this rule as ``docstrings``."""
    assert DocstringRule().name == "docstrings"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("m.py", True), ("notes.txt", False), ("data.json", False)],
)
def test_only_python_files_are_read(name: str, expected: bool) -> None:
    """The file filter is by suffix.

    Args:
        name: File name to classify.
        expected: Whether the rule should read it.
    """
    assert is_python_module(Path(name)) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [("src/m.py", True), ("scripts/m.py", True), ("tests/test_m.py", False)],
)
def test_only_non_test_files_must_name_their_arguments(path: str, expected: bool) -> None:
    """The ``Args:`` requirement stops at the tests directory.

    Args:
        path: Repo-relative path to classify.
        expected: Whether ``Args:`` is required there.
    """
    assert documents_arguments(Path(path)) is expected


def sole_function(source: str) -> ast.FunctionDef:
    """Parse a one-function module and return that function.

    Args:
        source: Source holding exactly one function definition.

    Returns:
        The function node.

    Raises:
        TypeError: If the source's first statement is something else,
            which means the test case was written wrong.
    """
    statement = ast.parse(source).body[0]
    if not isinstance(statement, ast.FunctionDef):
        raise TypeError(f"expected a function, got {type(statement).__name__}")
    return statement


def test_the_parameter_list_covers_every_calling_convention() -> None:
    """Positional-only, keyword-only and variadic parameters all count."""
    node = sole_function("def f(a, /, b, *args, c, **kwargs) -> None: ...\n")

    assert declared_parameters(node) == ["a", "b", "c", "args", "kwargs"]


def test_an_absent_return_annotation_is_not_a_return() -> None:
    """A function with no annotation promises nothing to describe."""
    assert returns_a_value(sole_function("def f(): ...\n")) is False
