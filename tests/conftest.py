"""Fixtures shared by more than one test module.

``write_test_module`` serves the three test-quality rule suites
(``test_guard_test_quality.py``, ``test_guard_patching.py`` and
``test_guard_translit_quality.py``), which were one file until
2026-11-04; it moved here unchanged when that file was split under the
600-line limit, so the suites share one definition of it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def write_test_module(tmp_path: Path) -> Callable[[str, str], Path]:
    """Return a helper writing source into a ``tests`` directory.

    The rules only consider files under a directory named ``tests``, so
    the fixture builds that layout rather than leaving each case to.

    Args:
        tmp_path: Root to build the tree in.

    Returns:
        A function taking a file name and source, returning its path.
    """
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    def write(name: str, source: str) -> Path:
        """Write source into the tests directory being scanned.

        Args:
            name: File name within that directory.
            source: File contents.

        Returns:
            The written path.
        """
        path = tests_dir / name
        path.write_text(source, encoding="utf-8")
        return path

    return write
