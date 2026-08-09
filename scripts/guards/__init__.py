"""Guard rules for enforcing code quality standards.

This module provides a modular guard system with reusable rule definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol


class Violation(NamedTuple):
    """A single guard violation."""

    file: Path
    line_no: int
    kind: str
    line: str


class RuleReport(NamedTuple):
    """Summary of violations for a rule."""

    name: str
    violations: int


class Rule(Protocol):
    """One guard rule, as the runner drives it."""

    @property
    def name(self) -> str:
        """Name this rule reports under.

        Returns:
            The name shown in the summary, e.g. ``patching``.
        """
        ...

    def run(self, files: list[Path]) -> list[Violation]:
        """Scan files and report what this rule forbids.

        Args:
            files: Every file the guards were pointed at. A rule that
                applies to some of them filters here rather than
                relying on the runner to pre-select.

        Returns:
            One violation per offence.
        """
        ...


__all__ = ["Rule", "RuleReport", "Violation"]
