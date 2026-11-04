"""Exception-handling rules of the guard.

Broad handlers must log and re-raise, specific ones must log or re-raise,
and a silent body is reported. These checks were defined in
``scripts/guard.py`` until 2026-11-04, when they moved here unchanged so
that the runner and each group of rules stay under the 600-line file
limit the file-length rule enforces.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from scripts.guards import Violation

_EXCEPT_HEADER = re.compile(r"^(\s*)except(\s+([^:]+))?:\s*$")
_BROAD_TYPES = re.compile(r"\b(Exception|BaseException)\b")
_LOG_CALL = re.compile(r"\b(logging|log|logger)\.(debug|info|warning|error|exception|critical)\(")
_RAISE_RE = re.compile(r"\braise\b")


def _parse_except_header(raw: str) -> tuple[int, str] | None:
    """Split an ``except`` header into its indent and its types.

    Args:
        raw: One source line, which may be anything.

    Returns:
        The indent width and the exception types as written, or
        ``None`` when the line is not an ``except`` header. A bare
        ``except:`` reports an empty type string.
    """
    match = _EXCEPT_HEADER.match(raw)
    if match is None:
        return None
    indent_group = match.group(1)
    group3 = match.group(3)
    indent_str = indent_group if indent_group is not None else ""
    types_str = group3 if group3 is not None else ""
    return len(indent_str), types_str.strip()


def _is_broad_exception(types: str) -> bool:
    """Report whether a caught type is a broad one.

    Args:
        types: Exception types as written, empty for a bare ``except``.

    Returns:
        True for a bare ``except`` and for anything naming
        ``Exception`` or ``BaseException``.
    """
    return types == "" or _BROAD_TYPES.search(types) is not None


def _first_body_is_trivial(line: str) -> bool:
    """Report whether a line is an empty statement.

    Args:
        line: One source line.

    Returns:
        True for ``pass`` and for ``...``, with or without a trailing
        comment. Either means the handler discards the exception.
    """
    return re.match(r"^\s+(pass|\.\.\.)\s*(#.*)?$", line) is not None


def _scan_except_body(
    lines: Sequence[str], start: int, header_indent: int
) -> tuple[bool, bool, int]:
    """Scan one ``except`` body for logging and re-raising.

    Args:
        lines: All source lines of the module.
        start: Index of the body's first line.
        header_indent: Indent width of the ``except`` header, which
            marks where the body ends.

    Returns:
        Whether the body logs, whether it raises, and the index of the
        first line after it.
    """
    total = len(lines)
    has_log = False
    has_raise = False
    i = start
    while i < total:
        body_line = lines[i]
        if body_line.strip() == "":
            i += 1
            continue
        body_indent = len(body_line) - len(body_line.lstrip(" \t"))
        if body_indent <= header_indent and re.match(
            r"^\s*(except\b|finally\b|else\b|$)", body_line
        ):
            break
        if _RAISE_RE.search(body_line):
            has_raise = True
        if _LOG_CALL.search(body_line):
            has_log = True
        i += 1
    return has_log, has_raise, i


def _find_body_start(lines: Sequence[str], start: int) -> int:
    """Find the first non-empty line after an ``except`` header.

    Args:
        lines: All source lines of the module.
        start: Index to begin looking from.

    Returns:
        The index of that line, or the number of lines when the file
        ends first. Valid Python always has a body, so the second case
        only arises on a truncated file.
    """
    total = len(lines)
    i = start
    while i < total:
        if lines[i].strip() != "":
            return i
        i += 1
    return total


def _run_exceptions_rule(path: Path, lines: list[str]) -> list[Violation]:
    """Report the exception handlers that neither log nor re-raise.

    A broad handler must do both; a specific one must do at least one.
    Test files are skipped, because a test catching an exception to
    assert on it is doing exactly what it should.

    Args:
        path: File being scanned, for the violation record.
        lines: The module's source lines. Scanned as text rather than
            parsed, so that a handler's formatting is visible.

    Returns:
        One violation per offending handler.
    """
    # Skip test files
    if "tests" in path.parts:
        return []

    violations: list[Violation] = []
    total = len(lines)
    idx = 0
    while idx < total:
        raw = lines[idx]
        parsed = _parse_except_header(raw)
        if parsed is None:
            idx += 1
            continue
        indent, types = parsed
        broad = _is_broad_exception(types)

        body_start = _find_body_start(lines, idx + 1)

        if body_start < total and _first_body_is_trivial(lines[body_start]):
            violations.append(
                Violation(
                    file=path,
                    line_no=idx + 1,
                    kind="silent-except-body",
                    line=raw.rstrip("\n"),
                )
            )

        has_log, has_raise, body_end = _scan_except_body(lines, body_start, indent)
        if broad:
            if not (has_log and has_raise):
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx + 1,
                        kind="broad-except-requires-log-and-raise",
                        line=raw.rstrip("\n"),
                    )
                )
        else:
            if not (has_log or has_raise):
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx + 1,
                        kind="except-without-log-or-raise",
                        line=raw.rstrip("\n"),
                    )
                )

        idx = body_end if body_end > idx else idx + 1
    return violations
