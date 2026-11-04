"""The file-length rule: no Python file longer than 600 lines.

The coding standard, as the operator gave it on 2026-09-19: one file, one
concern, 400 to 600 lines at most, source and tests alike. A file past the
limit is split into modules by concern; there is no allow-list, because an
exempted file is the one that keeps growing.
"""

from __future__ import annotations

from pathlib import Path

from scripts.guards import Violation
from scripts.guards.docstring_rules import is_python_module
from scripts.guards.util import read_lines

MAX_LINES = 600


class FileLengthRule:
    """Guard rule refusing any Python file longer than :data:`MAX_LINES`."""

    name = "file-length"

    def run(self, files: list[Path]) -> list[Violation]:
        """Report every Python file whose line count exceeds the limit.

        Args:
            files: Paths to consider; non-Python files are skipped.

        Returns:
            One violation per file over the limit, carrying its line count
            as the violation's line number so the report names both.
        """
        out: list[Violation] = []
        for path in files:
            if not is_python_module(path):
                continue
            count = len(read_lines(path))
            if count > MAX_LINES:
                out.append(
                    Violation(
                        file=path,
                        line_no=count,
                        kind="file-too-long",
                        line=f"{count} lines, over the {MAX_LINES}-line limit: split it by concern",
                    )
                )
        return out


__all__ = ["MAX_LINES", "FileLengthRule"]
