"""Fixtures shared by more than one test module, and the run's model summary.

``write_test_module`` serves the three test-quality rule suites
(``test_guard_test_quality.py``, ``test_guard_patching.py`` and
``test_guard_translit_quality.py``), which were one file until
2026-11-04; it moved here unchanged when that file was split under the
600-line limit, so the suites share one definition of it.

The language-identification tests in ``test_verification.py`` need real
fastText weights that are not distributed with the package, and skip when
the weights are absent. A green run would otherwise not say whether those
models were exercised, so each such test carries the marker of the model
it needs, and the end of the run states, per model, how many of its tests
ran against real weights and how many skipped. Markers travel with each
test's report, so the count holds when pytest-xdist runs the tests in
other processes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

#: Each marker a language-identification test carries, and the model whose
#: real weights it needs.
LID_MARKERS: dict[str, str] = {"lid176_weights": "lid.176", "lid218e_weights": "lid218e"}


def pytest_configure(config: pytest.Config) -> None:
    """Register the weight markers, so a misspelt one is an error.

    Args:
        config: The run's configuration.
    """
    for marker, model in LID_MARKERS.items():
        config.addinivalue_line("markers", f"{marker}: needs the real {model} weights")


def lid_counts(reports: list[pytest.TestReport]) -> list[str]:
    """Count, per model, the tests that ran against real weights and that skipped.

    A test ran when its call phase reported, passed or failed, and skipped
    when any phase reported a skip; each test is counted once by its id.

    Args:
        reports: Every test report of the run.

    Returns:
        One line per model, in :data:`LID_MARKERS` order.
    """
    lines: list[str] = []
    for marker, model in LID_MARKERS.items():
        ran: set[str] = set()
        skipped: set[str] = set()
        for report in reports:
            if marker not in report.keywords:
                continue
            if report.outcome == "skipped":
                skipped.add(report.nodeid)
            elif report.when == "call":
                ran.add(report.nodeid)
        lines.append(f"LID {model}: {len(ran)} ran against real weights, {len(skipped)} skipped")
    return lines


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """End the run with the per-model language-identification counts.

    Args:
        terminalreporter: The reporter holding every report of the run.
    """
    reports = [
        report
        for outcome in terminalreporter.stats.values()
        for report in outcome
        if isinstance(report, pytest.TestReport)
    ]
    for line in lid_counts(reports):
        terminalreporter.write_line(line)


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
