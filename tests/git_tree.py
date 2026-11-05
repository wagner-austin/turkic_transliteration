"""Make a temporary project tree a git work tree, as the guard requires.

The guard's Windows rule reads the files git tracks, so every test that
runs the whole guard over a tree it built first makes that tree a
repository. ``git add`` is what marks a written file as tracked.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def init_git_tree(root: Path) -> Path:
    """Make a directory an empty git work tree.

    Args:
        root: The directory, created if it does not exist.

    Returns:
        The same directory.
    """
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def track(root: Path, relative: str, content: str) -> Path:
    """Write a file into a git work tree and mark it tracked.

    Args:
        root: A tree :func:`init_git_tree` made.
        relative: The file's path within it, directories created as needed.
        content: The file's text, written as UTF-8.

    Returns:
        The written path.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "--", relative], check=True)
    return path


__all__ = ["init_git_tree", "track"]
