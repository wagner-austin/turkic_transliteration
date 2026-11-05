"""Tests for the guard's exception-handling rule.

Every case runs the rule over real source, parsed as the guard parses
it, and asserts the violation kinds it reports. These cases were in
``tests/test_guard_rules.py`` until 2026-11-04, when they moved here
beside the rule they test, which moved to
``scripts/guards/exception_rules.py``, so both files stay under the
600-line limit. On 2026-11-05 the rule moved from scanning lines to
reading ``ast.ExceptHandler`` nodes; the cases for the four shapes the
line scanner missed are the ones named after the old ``sweep_once``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.guards import Violation
from scripts.guards.exception_rules import _run_exceptions_rule

#: The janitor's sweep as it stood before d3a4b42, byte for byte from
#: git show d3a4b42^:src/turkic_translit/web/web_utils.py, its docstring cut.
OLD_SWEEP_ONCE = (
    "def sweep_once(max_age_sec: int) -> int:\n"
    "    try:\n"
    "        return purge_expired_downloads(max_age_sec)\n"
    "    except OSError:\n"
    '        log.exception("janitor sweep failed")\n'
    "        return 0\n"
)


def exception_kinds(source: str, path: Path) -> list[str]:
    """Run the exceptions rule over source and return violation kinds.

    Args:
        source: Python source to analyse.
        path: Path the source is attributed to.

    Returns:
        The kind of each violation, in the order reported.
    """
    tree = ast.parse(source)
    return [v.kind for v in _run_exceptions_rule(path, tree, source.splitlines())]


def handler(body: str, header: str = "    except OSError as exc:\n") -> str:
    """Wrap a handler body in a function with a try statement.

    Args:
        body: The handler's lines, indented eight spaces.
        header: The except line.

    Returns:
        A module holding the function.
    """
    return "def f():\n    try:\n        go()\n" + header + body


@pytest.mark.parametrize(
    "header",
    [
        "    except Exception:\n",
        "    except BaseException:\n",
        "    except:\n",
        "    except (ValueError, Exception):\n",
        "    except builtins.Exception:\n",
    ],
)
def test_a_broad_except_must_log_and_raise(tmp_path: Path, header: str) -> None:
    """A bare except and both base classes, however named, need both.

    Args:
        tmp_path: Root the module is attributed under.
        header: The except line.
    """
    source = handler("        logger.warning('x')\n        return 1\n", header)

    assert exception_kinds(source, tmp_path / "src" / "m.py") == [
        "broad-except-requires-log-and-raise"
    ]


def test_a_broad_except_that_logs_and_raises_is_allowed(tmp_path: Path) -> None:
    """Doing both is the documented way to catch broadly."""
    source = handler("        logger.exception('x')\n        raise\n", "    except Exception:\n")

    assert exception_kinds(source, tmp_path / "src" / "m.py") == []


def test_a_specific_except_may_raise_or_log_and_return_the_error(tmp_path: Path) -> None:
    """Re-raising, or logging and answering with the caught error, is enough."""
    answers = handler("        logger.warning('x')\n        return f'Error: {exc!s}'\n")
    raises = handler("        raise Other() from None\n", "    except ValueError:\n")

    assert exception_kinds(answers, tmp_path / "src" / "m.py") == []
    assert exception_kinds(raises, tmp_path / "src" / "m.py") == []


def test_the_old_sweep_once_is_reported_as_returning_a_default(tmp_path: Path) -> None:
    """Logging a sweep's error and answering zero files swept hides the failure."""
    assert exception_kinds(OLD_SWEEP_ONCE, tmp_path / "src" / "m.py") == [
        "except-returns-a-default"
    ]


def test_the_old_sweep_once_with_a_comment_on_its_header_is_reported(tmp_path: Path) -> None:
    """A comment after the colon does not take the handler out of the rule."""
    source = OLD_SWEEP_ONCE.replace(
        "except OSError:\n", "except OSError:  # the janitor must stay alive\n"
    )

    assert exception_kinds(source, tmp_path / "src" / "m.py") == ["except-returns-a-default"]


def test_the_old_sweep_once_with_raise_in_a_body_comment_is_reported(tmp_path: Path) -> None:
    """The word raise in a comment is not a raise statement."""
    source = OLD_SWEEP_ONCE.replace(
        "        return 0\n",
        "        # do not raise: the janitor must stay alive\n        return 0\n",
    )

    assert exception_kinds(source, tmp_path / "src" / "m.py") == ["except-returns-a-default"]


def test_the_old_sweep_once_with_raise_in_its_log_text_is_reported(tmp_path: Path) -> None:
    """The word raise in the logged message is not a raise statement."""
    source = OLD_SWEEP_ONCE.replace(
        '"janitor sweep failed"', '"janitor sweep failed; will not raise"'
    )

    assert exception_kinds(source, tmp_path / "src" / "m.py") == ["except-returns-a-default"]


def test_the_old_sweep_once_with_its_types_across_lines_is_reported(tmp_path: Path) -> None:
    """A header whose types span lines is still one handler."""
    source = OLD_SWEEP_ONCE.replace(
        "    except OSError:\n", "    except (\n        OSError,\n        ValueError,\n    ):\n"
    )

    assert exception_kinds(source, tmp_path / "src" / "m.py") == ["except-returns-a-default"]


def test_a_commented_header_doing_neither_is_reported(tmp_path: Path) -> None:
    """A handler neither logging nor raising is read whatever follows its colon."""
    source = handler("        return None\n", "    except ValueError:  # nothing to do\n")

    assert exception_kinds(source, tmp_path / "src" / "m.py") == ["except-without-log-or-raise"]


@pytest.mark.parametrize(
    "body",
    [
        "        log.warning('failed: %s', exc)\n        return 0\n",
        "        log.warning('failed: %s', exc)\n        return []\n",
        "        log.warning('failed: %s', exc)\n        return False\n",
        "        log.warning('failed: %s', exc)\n        return 'try again later'\n",
        "        log.warning('failed: %s', exc)\n        swept = 0\n        return swept\n",
        "        _log.exception('failed: %s', exc)\n        return 0\n",
        "        log.warning('x')\n        if exc.errno:\n            return str(exc)\n"
        "        return 0\n",
    ],
)
def test_a_default_is_not_excused_by_naming_the_error_in_the_log(tmp_path: Path, body: str) -> None:
    """The log call's mention reaches the log, not the caller.

    The last case returns the error on one path and a default on the
    other, and the default is enough to report it.

    Args:
        tmp_path: Root the module is attributed under.
        body: A handler body that logs the error and returns something else.
    """
    assert exception_kinds(handler(body), tmp_path / "src" / "m.py") == ["except-returns-a-default"]


def test_a_value_without_a_bound_error_is_a_default(tmp_path: Path) -> None:
    """A handler with no as name has no error to return."""
    source = handler("        log.warning('x')\n        return 'failed'\n", "    except OSError:\n")

    assert exception_kinds(source, tmp_path / "src" / "m.py") == ["except-returns-a-default"]


def test_a_raise_in_a_nested_function_is_not_the_handlers(tmp_path: Path) -> None:
    """A raise that runs only when a nested function is called raises nothing here."""
    body = "        def later():\n            raise\n        log.warning('x')\n        return 0\n"

    assert exception_kinds(handler(body), tmp_path / "src" / "m.py") == ["except-returns-a-default"]


@pytest.mark.parametrize(
    "body",
    [
        "        logger.warning('x: %s', exc)\n        return f'Error reading file: {exc!s}'\n",
        "        logger.warning('x: %s', exc)\n        return '', f'**Error**: {exc!s}'\n",
        "        logger.info('x: %s', exc)\n        return str(exc)\n",
        "        logger.exception('x')\n        payload = error_response(str(exc), status=500)\n"
        "        return error_markdown(payload), None\n",
        "        log.warning('x: %s', exc)\n        return (\n            'unavailable\\n\\n'\n"
        "            f'Error: {exc!s}'\n        )\n",
        "        log.warning('x')\n        message: str = str(exc)\n        return message\n",
        "        log.warning('x')\n        count: int\n        first, rest = exc.args, 1\n"
        "        return first\n",
    ],
)
def test_a_value_built_from_the_error_is_the_callers_answer(tmp_path: Path, body: str) -> None:
    """The boundary shapes in src pass: directly, through a name, across lines.

    Args:
        tmp_path: Root the module is attributed under.
        body: A handler body returning a value built from the error.
    """
    assert exception_kinds(handler(body), tmp_path / "src" / "m.py") == []


def test_a_held_file_skipped_behind_a_filtered_reraise_is_reported(tmp_path: Path) -> None:
    """Re-raising only some errors leaves the held-file error swallowed as 0.

    This is the task's own failure, one held file absorbed by answering
    zero files swept, behind a raise that runs only for other errors.
    """
    body = (
        "        if exc.errno != errno.EBUSY:\n"
        "            raise\n"
        "        logger.warning('held file, skipping: %s', exc)\n"
        "        return 0\n"
    )

    assert exception_kinds(handler(body), tmp_path / "src" / "m.py") == ["except-returns-a-default"]


def test_a_broad_handler_raising_only_when_fatal_is_reported(tmp_path: Path) -> None:
    """A broad handler must re-raise on every path, not only a fatal one."""
    body = "        logger.exception('sweep failed')\n        if fatal:\n            raise\n"

    assert exception_kinds(handler(body, "    except Exception:\n"), tmp_path / "src" / "m.py") == [
        "broad-except-requires-log-and-raise"
    ]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "        if exc.errno:\n            raise\n        else:\n"
            "            raise Other() from exc\n",
            [],
        ),
        (
            "        if exc.errno:\n            raise\n        return 0\n",
            ["except-returns-a-default"],
        ),
        (
            "        log.warning('x')\n        for attempt in range(3):\n            raise\n",
            ["except-logs-and-continues"],
        ),
        (
            "        log.warning('x')\n        while True:\n            break\n        return 0\n",
            ["except-returns-a-default"],
        ),
        ("        log.warning('x')\n        with lock:\n            raise\n", []),
        (
            "        log.warning('x')\n        try:\n            return 0\n        finally:\n"
            "            raise\n",
            [],
        ),
        (
            "        log.warning('x')\n        try:\n            return 0\n        finally:\n"
            "            release()\n        raise\n",
            ["except-returns-a-default"],
        ),
        (
            "        log.warning('x')\n        match exc.errno:\n            case 1:\n"
            "                raise\n",
            ["except-logs-and-continues"],
        ),
        (
            "        log.warning('x')\n        if exc.errno:\n            return str(exc)\n"
            "        raise\n",
            [],
        ),
    ],
)
def test_every_ending_of_a_handler_is_judged(
    tmp_path: Path, body: str, expected: list[str]
) -> None:
    """Each way out of a handler must re-raise or return the error.

    The endings are read through if, loops, with, try and match. A loop
    may run no times, a try whose finally always raises ends there, and
    a match may match no case.

    Args:
        tmp_path: Root the module is attributed under.
        body: The handler body.
        expected: The kinds it should earn.
    """
    assert exception_kinds(handler(body), tmp_path / "src" / "m.py") == expected


@pytest.mark.parametrize(
    "ending",
    [
        "",
        "        return\n",
        "        return None\n",
        "        continue\n",
    ],
)
def test_a_specific_except_that_logs_and_continues_is_reported(tmp_path: Path, ending: str) -> None:
    """Logging and then falling through, or returning nothing, hides the failure.

    The loop is the shape the download janitor had before 2026-11-04: a
    sweep's filesystem error was logged and the next sweep tried anyway.

    Args:
        tmp_path: Root the module is attributed under.
        ending: What follows the log call in the handler.
    """
    source = (
        "def sweep_forever():\n"
        "    while True:\n"
        "        try:\n"
        "            purge()\n"
        "        except OSError as exc:\n"
        "            log.warning('sweep failed: %s', exc)\n"
        f"{'    ' + ending if ending else ''}"
        "        sleep(60)\n"
    )

    assert exception_kinds(source, tmp_path / "src" / "m.py") == ["except-logs-and-continues"]


def test_a_return_after_the_try_statement_is_not_the_handlers(tmp_path: Path) -> None:
    """A later return of the function's result does not excuse the handler.

    Args:
        tmp_path: Root the module is attributed under.
    """
    source = (
        "def run_janitor(running):\n"
        "    swept = 0\n"
        "    while running():\n"
        "        try:\n"
        "            swept += purge()\n"
        "        except OSError as exc:\n"
        "            log.warning('sweep failed: %s', exc)\n"
        "        sleep(60)\n"
        "    return swept\n"
    )

    assert exception_kinds(source, tmp_path / "src" / "m.py") == ["except-logs-and-continues"]


@pytest.mark.parametrize(
    "body",
    [
        "        print(exc)\n        return str(exc)\n",
        "        log.flush()\n        return str(exc)\n",
        "        self.log.warning('x')\n        return str(exc)\n",
        "        return None\n",
    ],
)
def test_a_specific_except_doing_neither_is_reported(tmp_path: Path, body: str) -> None:
    """Only a logging method on a logger name is a log call.

    Args:
        tmp_path: Root the module is attributed under.
        body: A handler body that neither logs nor raises.
    """
    assert exception_kinds(handler(body), tmp_path / "src" / "m.py") == [
        "except-without-log-or-raise"
    ]


@pytest.mark.parametrize(
    ("body", "header", "second"),
    [
        ("        pass\n", "    except ValueError:\n", "except-without-log-or-raise"),
        ("        ...\n", "    except ValueError:\n", "except-without-log-or-raise"),
        ("        pass\n", "    except Exception:\n", "broad-except-requires-log-and-raise"),
    ],
)
def test_a_silent_body_is_reported_twice(
    tmp_path: Path, body: str, header: str, second: str
) -> None:
    """An empty body is both silent and missing what its catch requires.

    Args:
        tmp_path: Root the module is attributed under.
        body: The empty statement.
        header: The except line.
        second: The kind the catch's breadth adds.
    """
    assert exception_kinds(handler(body, header), tmp_path / "src" / "m.py") == [
        "silent-except-body",
        second,
    ]


def test_violations_name_the_handler_line_in_source_order(tmp_path: Path) -> None:
    """Each violation quotes its own except line, earliest first."""
    path = tmp_path / "src" / "m.py"
    source = (
        "def f():\n"
        "    try:\n"
        "        go()\n"
        "    except OSError:\n"
        "        try:\n"
        "            undo()\n"
        "        except ValueError:\n"
        "            pass\n"
        "        raise\n"
    )

    violations = _run_exceptions_rule(path, ast.parse(source), source.splitlines())

    assert violations == [
        Violation(
            file=path, line_no=7, kind="silent-except-body", line="        except ValueError:"
        ),
        Violation(
            file=path,
            line_no=7,
            kind="except-without-log-or-raise",
            line="        except ValueError:",
        ),
    ]


def test_exceptions_in_tests_are_not_checked(tmp_path: Path) -> None:
    """Tests legitimately catch exceptions to assert on them."""
    source = "try:\n    go()\nexcept ValueError:\n    pass\n"

    assert exception_kinds(source, tmp_path / "tests" / "test_m.py") == []


def test_a_module_without_handlers_reports_nothing(tmp_path: Path) -> None:
    """A module with no try statement has nothing to judge."""
    assert exception_kinds("x = 1\ny = 2\n", tmp_path / "src" / "m.py") == []
