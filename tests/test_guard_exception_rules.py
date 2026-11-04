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
