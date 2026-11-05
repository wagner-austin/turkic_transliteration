"""The Windows rule: no tracked file is, or tells a reader to use, Windows tooling.

The operator, 2026-11-04: no more Windows code. The repository's Windows
scripts, its Windows-only branches and its Windows install steps were
removed by hand, and this rule keeps them out: it fails the guard on any
tracked file carrying the Windows shell's script extension, and on any line
of a tracked file that names the Windows shell, that extension, or either
Windows package manager. The changelog is the one file it does not read,
because it is the record of what was removed and has to name it.

The patterns are spelled with bracket classes, so this module and its tests
never contain the words they refuse and the rule can read itself.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from scripts.guards import Violation

#: The repository-relative path of the one file the rule leaves unread.
HISTORY = Path("CHANGELOG.md")

#: A file name ending in the Windows shell's script extension.
SCRIPT_NAME = re.compile(r"[.]p[s]1\Z", re.IGNORECASE)

#: A line naming the Windows shell, its script extension (bare, as in an
#: editorconfig glob, or after a dot) or a Windows package manager.
WINDOWS_TEXT = re.compile(
    rb"power[s]hell"
    rb"|(?<![a-z0-9])p[s]1(?![a-z0-9])"
    rb"|(?<![a-z0-9])c[h]oco(?:latey)?(?![a-z0-9])"
    rb"|(?<![a-z0-9])w[i]nget(?![a-z0-9])",
    re.IGNORECASE,
)


def tracked_files(root: Path) -> list[Path]:
    """List the files git tracks under a project root.

    Untracked files are left out on purpose: a checkout also holds a
    virtual environment and downloaded model weights, which are not the
    repository's and which the rule has no business reading. The fleet
    stages every commit it checks as a one-commit repository, so the list
    there is exactly the commit's files.

    Args:
        root: The project root, which must be a git work tree.

    Returns:
        Each tracked file, as a path under ``root``, in git's order.

    Raises:
        subprocess.CalledProcessError: If git cannot list the tree, for
            example because ``root`` is not a git work tree.
    """
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    return [root / name for name in completed.stdout.decode("utf-8").split("\0") if name]


class WindowsRule:
    """Guard rule refusing tracked Windows scripts and Windows instructions.

    Args:
        root: The project root the files are tracked under, against which
            :data:`HISTORY` is resolved.
    """

    name = "windows"

    def __init__(self, root: Path) -> None:
        """Remember the root the changelog is resolved against."""
        self._history = root / HISTORY

    def run(self, files: list[Path]) -> list[Violation]:
        """Report every Windows script and every line naming Windows tooling.

        Args:
            files: The tracked files to read.

        Returns:
            One violation per Windows script, at line 1, and one per line
            that names Windows tooling, carrying that line.
        """
        out: list[Violation] = []
        for path in files:
            if path == self._history:
                continue
            if SCRIPT_NAME.search(path.name):
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="windows-script",
                        line="a Windows shell script: remove it",
                    )
                )
            for number, raw in enumerate(path.read_bytes().splitlines(), start=1):
                if WINDOWS_TEXT.search(raw):
                    text = raw.decode("utf-8", errors="backslashreplace").strip()
                    out.append(Violation(file=path, line_no=number, kind="windows-text", line=text))
        return out


__all__ = ["HISTORY", "SCRIPT_NAME", "WINDOWS_TEXT", "WindowsRule", "tracked_files"]
