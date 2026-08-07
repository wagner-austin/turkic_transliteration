"""Where language-identification weights are looked for and written.

Kept separate from the registry so that "which model" and "where its
bytes live" stay independent questions. A caller that already knows the
path passes it directly; a caller that does not asks here and gets the
same ordered list every time, which is what makes a run reproducible on a
second machine.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def default_search_dirs() -> tuple[Path, ...]:
    """List the directories consulted for weights, in priority order.

    The installed package directory comes first so that a checkout's own
    weights win over a stray copy in the user's home directory, which is
    where an interrupted manual download tends to land.

    Returns:
        Directories to search, most specific first.
    """
    return (_PACKAGE_ROOT, _PACKAGE_ROOT / "web", Path.home())


def default_destination_dir() -> Path:
    """Name the directory a downloaded model is written into.

    Returns:
        The installed package directory, which is also the first entry
        of :func:`default_search_dirs` so a fetched model is found again
        on the next run.
    """
    return _PACKAGE_ROOT


__all__ = ["default_destination_dir", "default_search_dirs"]
