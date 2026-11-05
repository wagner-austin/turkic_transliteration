"""The run's per-model language-identification summary, driven through real pytest.

``tests/conftest.py`` ends every run with one line per model saying how
many of its marked tests ran against real weights and how many skipped.
No fleet node holds the weights, so the suite's own runs only ever show
zero ran; this case copies the real conftest into a fresh directory beside
marked tests that pass, skip and fail, runs real pytest over them with
two pytest-xdist workers, so every report crosses a process boundary as
it does under ``make test``, and reads the lines it prints. Removing or
breaking the hook fails it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

#: The conftest whose hook is under test, copied as it stands.
CONFTEST = Path(__file__).with_name("conftest.py")

#: Marked tests covering each outcome the summary distinguishes.
INNER_TESTS = """\
import pytest


@pytest.mark.lid176_weights
def test_lid176_with_weights():
    assert 1 + 1 == 2


@pytest.mark.lid176_weights
def test_lid176_without_weights():
    pytest.skip("lid.176 weights are not installed")


@pytest.mark.lid218e_weights
def test_lid218e_that_fails():
    assert 1 + 1 == 3


def test_unmarked():
    assert 2 * 2 == 4
"""


def test_the_summary_counts_ran_and_skipped_per_model_across_workers(tmp_path: Path) -> None:
    """A pass and a failure count as ran, a skip as skipped, and unmarked tests not at all.

    Args:
        tmp_path: Directory the inner run collects, holding nothing else.
    """
    (tmp_path / "conftest.py").write_text(CONFTEST.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "test_inner.py").write_text(INNER_TESTS, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-n", "2", "-p", "no:cacheprovider", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        check=False,
    )

    assert completed.returncode == 1
    assert [line for line in completed.stdout.splitlines() if line.startswith("LID ")] == [
        "LID lid.176: 1 ran against real weights, 1 skipped",
        "LID lid218e: 1 ran against real weights, 0 skipped",
    ]
