"""Tests for the guard rules that enforce this project's own standards.

Each case writes real Python into a real project tree and runs the real
rule over it. The rules read files and parse source, so files and source
are what they are given; nothing here substitutes an AST or a reader.

The helpers below are exercised directly where a branch cannot be reached
through a whole-tree run — a malformed ``except`` header, for instance,
never survives ``ast.parse`` but the line scanner still has to handle it.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.guard import (
    PROJECT_ROOT,
    _check_object_annotations,
    _find_body_start,
    _first_body_is_trivial,
    _is_broad_exception,
    _is_suppress,
    _iter_py_files,
    _parse_except_header,
    _run_comments_rule,
    _run_exceptions_rule,
    _run_logging_rule,
    _run_suppress_rule,
    _run_typing_rule,
    _scan_except_body,
    main,
    parse_root_argument,
    run_guards,
)


@pytest.fixture
def project(tmp_path: Path) -> Callable[[str, str], Path]:
    """Return a helper writing a module into a project tree.

    Args:
        tmp_path: Root to build the tree in.

    Returns:
        A function taking a repo-relative path and source, returning the
        file's path. The project's ``pyproject.toml`` is created too, so
        the tree is one the entry point accepts.
    """
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    def write(relative: str, source: str) -> Path:
        """Write source into the project tree being scanned.

        Args:
            relative: Path within the project, with directories
                created as needed.
            source: File contents.

        Returns:
            The written path.
        """
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    return write


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


def test_only_the_scanned_directories_are_walked(tmp_path: Path) -> None:
    """src, tests and scripts are scanned; anything else is not."""
    for relative in ("src/a.py", "tests/b.py", "scripts/c.py", "docs/d.py", "e.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    found = sorted(p.name for p in _iter_py_files(tmp_path))

    assert found == ["a.py", "b.py", "c.py"]


def test_a_missing_scanned_directory_is_not_an_error(tmp_path: Path) -> None:
    """A project without a scripts directory still scans the others."""
    path = tmp_path / "src" / "only.py"
    path.parent.mkdir(parents=True)
    path.touch()

    assert [p.name for p in _iter_py_files(tmp_path)] == ["only.py"]


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


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("except:", (0, "")),
        ("    except ValueError:", (4, "ValueError")),
        ("        except (OSError, KeyError):", (8, "(OSError, KeyError)")),
        ("except ValueError as exc:", (0, "ValueError as exc")),
        ("x = 1", None),
        ("except ValueError:  # trailing", None),
    ],
)
def test_except_headers_are_parsed_for_indent_and_types(
    header: str, expected: tuple[int, str] | None
) -> None:
    """The scanner reads indentation and the caught types off the line.

    Args:
        header: The source line to parse.
        expected: The indent and types, or None when the line is not a
            header this scanner recognises.
    """
    assert _parse_except_header(header) == expected


@pytest.mark.parametrize(
    ("types", "broad"),
    [
        ("", True),
        ("Exception", True),
        ("BaseException", True),
        ("(ValueError, Exception)", True),
        ("ValueError", False),
        ("(OSError, KeyError)", False),
    ],
)
def test_broadness_is_decided_by_the_caught_types(types: str, broad: bool) -> None:
    """A bare except and the two base classes are the broad cases.

    Args:
        types: The types text from the header.
        broad: Whether that text should count as broad.
    """
    assert _is_broad_exception(types) is broad


@pytest.mark.parametrize(
    ("line", "trivial"),
    [
        ("    pass", True),
        ("    ...", True),
        ("    pass  # deliberate", True),
        ("    return False", False),
        ("    raise", False),
    ],
)
def test_a_trivial_body_is_recognised(line: str, trivial: bool) -> None:
    """A body that only passes is a silently swallowed exception.

    Args:
        line: The first body line.
        trivial: Whether it should count as trivial.
    """
    assert _first_body_is_trivial(line) is trivial


def test_the_body_scanner_reports_what_it_found() -> None:
    """Logging and re-raising are both detected, and the body ends."""
    lines = [
        "    except ValueError:",
        "",
        "        logger.warning('x')",
        "        raise",
        "    finally:",
        "        cleanup()",
    ]

    has_log, has_raise, end = _scan_except_body(lines, 1, 4)

    assert (has_log, has_raise) == (True, True)
    assert end == 4


def test_the_body_scanner_runs_to_the_end_of_the_file() -> None:
    """A body reaching the last line ends there."""
    lines = ["    except ValueError:", "        return None"]

    has_log, has_raise, end = _scan_except_body(lines, 1, 4)

    assert (has_log, has_raise) == (False, False)
    assert end == 2


def test_the_body_start_skips_blank_lines() -> None:
    """Blank lines between the header and the body are stepped over."""
    assert _find_body_start(["except X:", "", "   ", "    pass"], 1) == 3


def test_the_body_start_of_a_truncated_file_is_its_length() -> None:
    """A header with nothing after it reports the end of the file."""
    assert _find_body_start(["except X:", "", "  "], 1) == 3


def exception_kinds(source: str, path: Path) -> list[str]:
    """Run the exceptions rule over source and return violation kinds.

    Args:
        source: Python source to analyse.
        path: Path the source is attributed to.

    Returns:
        The kind of each violation, in the order reported.
    """
    return [v.kind for v in _run_exceptions_rule(path, source.splitlines())]


def test_a_broad_except_must_log_and_raise(tmp_path: Path) -> None:
    """Catching Exception without doing both is reported."""
    source = "try:\n    go()\nexcept Exception:\n    logger.warning('x')\n"

    assert exception_kinds(source, tmp_path / "src" / "m.py") == [
        "broad-except-requires-log-and-raise"
    ]


def test_a_broad_except_that_logs_and_raises_is_allowed(tmp_path: Path) -> None:
    """Doing both is the documented way to catch broadly."""
    source = "try:\n    go()\nexcept Exception:\n    logger.exception('x')\n    raise\n"

    assert exception_kinds(source, tmp_path / "src" / "m.py") == []


def test_a_specific_except_may_log_or_raise(tmp_path: Path) -> None:
    """Either is enough when the caught type is specific."""
    logs = "try:\n    go()\nexcept ValueError:\n    logger.warning('x')\n"
    raises = "try:\n    go()\nexcept ValueError:\n    raise Other() from None\n"

    assert exception_kinds(logs, tmp_path / "src" / "m.py") == []
    assert exception_kinds(raises, tmp_path / "src" / "m.py") == []


def test_a_specific_except_doing_neither_is_reported(tmp_path: Path) -> None:
    """Swallowing a specific exception still hides a failure."""
    source = "try:\n    go()\nexcept ValueError:\n    return None\n"

    assert exception_kinds(source, tmp_path / "src" / "m.py") == ["except-without-log-or-raise"]


def test_a_silent_body_is_reported_twice(tmp_path: Path) -> None:
    """A bare pass is both silent and missing its log or raise."""
    source = "try:\n    go()\nexcept ValueError:\n    pass\n"

    assert exception_kinds(source, tmp_path / "src" / "m.py") == [
        "silent-except-body",
        "except-without-log-or-raise",
    ]


def test_exceptions_in_tests_are_not_checked(tmp_path: Path) -> None:
    """Tests legitimately catch exceptions to assert on them."""
    source = "try:\n    go()\nexcept ValueError:\n    pass\n"

    assert exception_kinds(source, tmp_path / "tests" / "test_m.py") == []


def test_a_module_without_handlers_reports_nothing(tmp_path: Path) -> None:
    """The scanner steps over ordinary lines."""
    source = "x = 1\ny = 2\n"

    assert exception_kinds(source, tmp_path / "src" / "m.py") == []


def test_a_clean_tree_passes(project: Callable[[str, str], Path], tmp_path: Path) -> None:
    """A project with nothing to report exits zero.

    The module is written to satisfy every rule, the docstring one
    included, so that this asserts a clean run rather than the absence
    of whichever rules happen to be registered.
    """
    project(
        "src/clean.py",
        "def f(x: int) -> int:\n"
        '    """Add one.\n\n'
        "    Args:\n"
        "        x: The number.\n\n"
        "    Returns:\n"
        "        One more than the number.\n"
        '    """\n'
        "    return x + 1\n",
    )

    assert run_guards(tmp_path) == 0


def test_a_tree_with_a_violation_fails(project: Callable[[str, str], Path], tmp_path: Path) -> None:
    """Any violation exits two."""
    project("src/dirty.py", "from typing import Any\n")

    assert run_guards(tmp_path) == 2


def reported_violations(captured_err: str) -> list[str]:
    """Extract the violation lines from a guard run's error output.

    Args:
        captured_err: Everything the run wrote to standard error.

    Returns:
        Each violation line, stripped of its leading indentation.
    """
    return [
        line.strip()
        for line in captured_err.splitlines()
        if line.startswith("  ") and ".py:" in line
    ]


def test_a_long_violation_line_is_truncated_in_the_report(
    project: Callable[[str, str], Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A long source line is cut to eighty characters plus an ellipsis."""
    project("src/dirty.py", f"x = 1  # type: ignore {'y' * 120}\n")

    run_guards(tmp_path)
    reported = reported_violations(capsys.readouterr().err)

    quoted = reported[0].split("type-ignore ", 1)[1]
    assert quoted.endswith("...")
    assert len(quoted) == 83


def test_a_short_violation_line_is_quoted_whole(
    project: Callable[[str, str], Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under the limit the source line is reported verbatim."""
    project("src/dirty.py", "x = 1  # type: ignore\n")

    run_guards(tmp_path)
    reported = reported_violations(capsys.readouterr().err)

    assert reported == ["src\\dirty.py:1: type-ignore x = 1  # type: ignore".replace("\\", os.sep)]


def test_a_violation_is_reported_relative_to_the_scanned_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Paths are shortened against the root so reports stay readable."""
    scanned = tmp_path / "scanned"
    (scanned / "src").mkdir(parents=True)
    (scanned / "src" / "dirty.py").write_text("from typing import Any\n", encoding="utf-8")

    assert run_guards(scanned) == 2
    reported = reported_violations(capsys.readouterr().err)
    assert reported == [f"src{os.sep}dirty.py:1: typing-import-any"]


def test_unparseable_source_raises_naming_the_file(
    project: Callable[[str, str], Path], tmp_path: Path
) -> None:
    """A syntax error is surfaced, not counted as a clean file."""
    project("src/broken.py", "def f(:\n")

    with pytest.raises(SyntaxError):
        run_guards(tmp_path)


def test_the_root_argument_is_read_from_the_command_line(tmp_path: Path) -> None:
    """``--root <path>`` selects the tree to scan."""
    assert parse_root_argument(["--root", str(tmp_path)]) == tmp_path.resolve()


def test_a_root_argument_without_a_value_is_ignored() -> None:
    """A trailing ``--root`` names nothing."""
    assert parse_root_argument(["--root"]) is None


def test_other_arguments_are_stepped_over(tmp_path: Path) -> None:
    """The scan finds ``--root`` wherever it appears."""
    assert parse_root_argument(["-v", "--root", str(tmp_path)]) == tmp_path.resolve()


def test_no_arguments_selects_no_root() -> None:
    """Without the flag the caller falls back to its default."""
    assert parse_root_argument([]) is None


def test_the_entry_point_scans_the_requested_root(
    project: Callable[[str, str], Path], tmp_path: Path
) -> None:
    """``--root`` is honoured in preference to the default."""
    project("src/dirty.py", "from typing import cast\n")

    assert main(["--root", str(tmp_path)]) == 2


def test_the_entry_point_falls_back_to_the_project_root(
    project: Callable[[str, str], Path], tmp_path: Path
) -> None:
    """Without ``--root`` the given default tree is scanned."""
    project("src/clean.py", "x: int = 1\n")

    assert main([], default_root=tmp_path) == 0


def test_a_default_root_that_is_not_a_project_is_refused(tmp_path: Path) -> None:
    """Scanning a tree with no project file would report a false pass."""
    assert main([], default_root=tmp_path / "nowhere") == 1


@pytest.mark.parametrize(
    ("source", "status"),
    [("x: int = 1\n", 0), ("from typing import Any\n", 2)],
)
def test_running_the_module_as_a_script_exits_with_the_rule_status(
    project: Callable[[str, str], Path], tmp_path: Path, source: str, status: int
) -> None:
    """``python -m scripts.guard`` turns the result into an exit code.

    Spawned as a real process so the module-level entry point runs with
    a real ``sys.argv``. Coverage follows into the child, so this needs
    no substitution of process state to be measured.

    Args:
        source: Module written into the scanned tree.
        status: Exit code the run should report.
    """
    project("src/module.py", source)

    completed = subprocess.run(
        [sys.executable, "-m", "scripts.guard", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )

    assert completed.returncode == status
