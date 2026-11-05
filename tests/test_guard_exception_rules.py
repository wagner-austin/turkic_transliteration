"""Tests for the guard's exception-handling rule.

The helpers are exercised directly where a branch cannot be reached
through a whole-tree run: a malformed ``except`` header, for instance,
never survives ``ast.parse``, but the line scanner still has to handle
it. These cases were in ``tests/test_guard_rules.py`` until 2026-11-04,
when they moved here unchanged beside the rule they test, which moved to
``scripts/guards/exception_rules.py``, so both files stay under the
600-line limit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.guards.exception_rules import (
    _find_body_start,
    _first_body_is_trivial,
    _is_broad_exception,
    _parse_except_header,
    _run_exceptions_rule,
    _scan_except_body,
)


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

    assert _scan_except_body(lines, 1, 4) == (True, True, False, 4)


def test_the_body_scanner_runs_to_the_end_of_the_file() -> None:
    """A body reaching the last line ends there, and None is no value."""
    lines = ["    except ValueError:", "        return None"]

    assert _scan_except_body(lines, 1, 4) == (False, False, False, 2)


@pytest.mark.parametrize(
    ("body", "returns_value"),
    [
        ("        return 'message'", True),
        ("        return (", True),
        ("        return False", True),
        ("        return", False),
        ("        return None", False),
        ("        return None  # nothing to say", False),
        ("        returned = 1", False),
    ],
)
def test_the_body_scanner_tells_a_returned_value_from_none(body: str, returns_value: bool) -> None:
    """Only a return carrying something other than None counts as a value.

    Args:
        body: The handler's one body line.
        returns_value: Whether it returns a value.
    """
    assert _scan_except_body(["    except ValueError:", body], 1, 4)[2] is returns_value


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


def test_a_specific_except_may_raise_or_log_and_return_a_value(tmp_path: Path) -> None:
    """Re-raising, or logging and answering with a value, is enough."""
    answers = "def f():\n    try:\n        go()\n    except ValueError:\n"
    answers += "        logger.warning('x')\n        return 'shown to the user'\n"
    raises = "try:\n    go()\nexcept ValueError:\n    raise Other() from None\n"

    assert exception_kinds(answers, tmp_path / "src" / "m.py") == []
    assert exception_kinds(raises, tmp_path / "src" / "m.py") == []


@pytest.mark.parametrize(
    "ending",
    [
        "",
        "        return\n",
        "        return None\n",
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

    The handler ends where the code dedents to its header, so the return
    after the loop is the function's.

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
