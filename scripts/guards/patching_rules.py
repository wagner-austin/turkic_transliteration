"""The patching rule: no monkeypatching and no mock library in tests.

A test swaps a ``_test_hooks`` binding for a real implementation of its
Protocol instead of rebinding an attribute or importing ``unittest.mock``;
see :mod:`scripts.guards.test_quality_rules` for why. This rule was
defined there until 2026-11-04, when it moved here unchanged so that each
test-quality rule stays under the 600-line file limit.
"""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.guards import Violation
from scripts.guards.test_quality_rules import MOCK_MODULES, is_test_module
from scripts.guards.util import read_lines


class PatchingRule:
    """Guard rule banning attribute patching and the mock library.

    A test that rebinds an attribute on a module under test is coupled
    to that module's internals and silently stops substituting anything
    when the attribute is renamed. A test built on ``unittest.mock`` is
    worse: the double answers every call, so the assertions hold no
    matter what the real code does.

    Both have one replacement. Every boundary is a Protocol whose real
    implementation is bound at import time in a ``_test_hooks`` module,
    and a test rebinds that one name.
    """

    name = "patching"

    def run(self, files: list[Path]) -> list[Violation]:
        """Report every patch and every mock import in the test modules.

        Args:
            files: Paths to consider; non-test modules are skipped.

        Returns:
            One violation per offending name or import.
        """
        out: list[Violation] = []
        for path in files:
            if not is_test_module(path):
                continue
            lines = read_lines(path)
            tree = ast.parse("\n".join(lines), filename=str(path))
            out.extend(self._check_module(path, tree, lines))
        return out

    def _check_module(self, path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
        """Collect the violations in one parsed test module.

        Args:
            path: File being scanned, for the violation record.
            tree: Parsed module.
            lines: Source lines, used to quote the offending line.

        Returns:
            The violations found, in source order.
        """
        found: list[Violation] = []
        seen: set[tuple[int, str]] = set()
        for node in ast.walk(tree):
            kind = self._kind_of(node)
            # Every node kind this rule matches is an expression, a
            # statement, or an argument, and only those carry a position.
            # The check is what tells the type checker that, so the
            # position below is read without a suppression.
            if kind is None or not isinstance(node, ast.expr | ast.stmt | ast.arg):
                continue
            key = (node.lineno, kind)
            if key in seen:
                continue
            seen.add(key)
            found.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind=kind,
                    line=lines[node.lineno - 1].strip(),
                )
            )
        return sorted(found, key=lambda violation: violation.line_no)

    def _kind_of(self, node: ast.AST) -> str | None:
        """Classify one node as a patching violation, or not.

        Args:
            node: Node from the module's tree.

        Returns:
            The violation kind, or ``None`` when the node is fine.
        """
        if isinstance(node, ast.Name) and node.id == "monkeypatch":
            return "monkeypatch-usage"
        if isinstance(node, ast.arg) and node.arg == "monkeypatch":
            return "monkeypatch-usage"
        if isinstance(node, ast.Attribute) and node.attr == "MonkeyPatch":
            return "monkeypatch-usage"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in MOCK_MODULES:
                    return "mock-library-import"
        if isinstance(node, ast.ImportFrom):
            if node.module in MOCK_MODULES:
                return "mock-library-import"
            if node.module == "unittest" and any(a.name == "mock" for a in node.names):
                return "mock-library-import"
        return None


__all__ = ["PatchingRule"]
