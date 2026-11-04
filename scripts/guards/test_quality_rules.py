"""Guard rules for detecting weak or fake tests.

These rules identify test anti-patterns that achieve code coverage
without actually verifying behavior. Coverage shows lines executed,
not correctness proven.

Violations:
- weak-assertion-is-not-none: `assert x is not None` proves existence only
- weak-assertion-isinstance: Type check doesn't verify behavior
- weak-assertion-hasattr: Attribute exists, but what's its value?
- weak-assertion-len-zero: `assert len(x) > 0` checks existence not content
- weak-assertion-in-output: String matching in captured output is fragile
- mock-without-assert-called-with: Mock verified called but not with what args
- excessive-mocking: Test mocks more than 3 things, probably not integration
- translit-call-without-value-assertion: A test exercises a transliteration or
  language-identification entry point but never compares the returned value
  against an expected one, so it proves the call ran and nothing more
- monkeypatch-usage: Reaching in to rebind an attribute at test time, instead
  of injecting through the module's `_test_hooks` seam
- mock-library-import: unittest.mock produces objects that answer any call,
  so a test built on one passes whatever the real code does

The last two are the same mistake seen from different angles. Every
substitutable boundary in this project is a Protocol with a real
implementation bound at import time, so a test swaps the binding rather
than patching the code that reads it. Those replacements are real
classes: they implement the protocol, hold no assertion helpers, and
fail loudly when asked for something they were not given. That is what
separates them from a mock, which answers anything and therefore proves
nothing.

This module holds the weak-assertion rule and the names the test-quality
rules share. The patching rule is in :mod:`scripts.guards.patching_rules`
and the transliteration rule in :mod:`scripts.guards.translit_test_rules`.
"""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.guards import Violation
from scripts.guards.util import read_lines

MOCK_MODULES: frozenset[str] = frozenset({"unittest.mock", "mock"})


def _is_patch_call(func: ast.expr) -> bool:
    """Report whether a call expression is a ``patch`` call.

    Args:
        func: The callable part of a call expression.

    Returns:
        True for both ``patch(...)`` and ``something.patch(...)``.
    """
    if isinstance(func, ast.Attribute) and func.attr == "patch":
        return True
    return isinstance(func, ast.Name) and func.id == "patch"


class _AssertVisitor(ast.NodeVisitor):
    """Walks a test module and records its weak assertions.

    Args:
        path: File being scanned, for the violation record.
        lines: Source lines, used to quote the offending line.
    """

    def __init__(self, path: Path, lines: list[str]) -> None:
        """Start with no violations and no function in progress."""
        self.path = path
        self.lines = lines
        self.violations: list[Violation] = []
        self.current_function: str = ""
        self.function_has_comparison: bool = False
        self.function_mock_count: int = 0
        self.function_start_line: int = 0

    def _get_line(self, line_no: int) -> str:
        """Return the source line a violation sits on.

        The lines were read from the same file the AST was parsed from,
        so a node's line number always indexes into them. There is no
        out-of-range case to defend against.

        Args:
            line_no: One-based line number from an AST node.

        Returns:
            The line, stripped of indentation.
        """
        return self.lines[line_no - 1].strip()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyse a function definition if it is a test.

        Args:
            node: The function being visited.
        """
        if node.name.startswith("test_"):
            self._analyze_test_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Analyse an async function definition if it is a test.

        Args:
            node: The function being visited.
        """
        if node.name.startswith("test_"):
            self._analyze_test_function(node)
        self.generic_visit(node)

    def _analyze_test_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Examine one test for weak assertions and unverified doubles.

        Args:
            node: The test function to analyse.
        """
        self.current_function = node.name
        self.function_has_comparison = False
        self.function_mock_count = 0
        self.function_start_line = node.lineno

        for child in ast.walk(node):
            self._check_assert(child)
            self._check_mock_usage(child)
            self._check_comparison(child)

        self._check_function_level_issues()

    def _check_assert(self, node: ast.AST) -> None:
        """Report an assert whose test proves nothing about a value.

        Args:
            node: Node from the test function's tree.
        """
        if not isinstance(node, ast.Assert):
            return

        test = node.test

        if self._is_identity_check_negated(test, "None"):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-is-not-none",
                    line=self._get_line(node.lineno),
                )
            )

        if self._is_isinstance_check(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-isinstance",
                    line=self._get_line(node.lineno),
                )
            )

        if self._is_hasattr_check(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-hasattr",
                    line=self._get_line(node.lineno),
                )
            )

        if self._is_len_existence_check(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-len-zero",
                    line=self._get_line(node.lineno),
                )
            )

        if self._is_string_in_output(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-in-output",
                    line=self._get_line(node.lineno),
                )
            )

    def _check_mock_usage(self, node: ast.AST) -> None:
        """Count the doubles a test builds.

        Args:
            node: Node from the test function's tree.
        """
        if isinstance(node, ast.Call) and _is_patch_call(node.func):
            self.function_mock_count += 1

        if isinstance(node, ast.Assert) and self._is_mock_called_check(node.test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="mock-without-assert-called-with",
                    line=self._get_line(node.lineno),
                )
            )

    def _check_comparison(self, node: ast.AST) -> None:
        """Note that the test compares a value against an expected one.

        Args:
            node: Node from the test function's tree.
        """
        if not isinstance(node, ast.Compare):
            return

        comparison_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
        for op in node.ops:
            if isinstance(op, comparison_ops) and self._is_variable_comparison(node):
                self.function_has_comparison = True

    def _check_function_level_issues(self) -> None:
        """Check issues that require analyzing the whole function."""
        if self.function_mock_count > 3:
            src_line = self._get_line(self.function_start_line)
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=self.function_start_line,
                    kind="excessive-mocking",
                    line=f"{src_line} ({self.function_mock_count} patches)",
                )
            )

    def _is_identity_check_negated(self, node: ast.expr, const_name: str) -> bool:
        """Report whether an expression is ``x is not <const>``.

        Args:
            node: The expression being examined.
            const_name: Name of the constant compared against.

        Returns:
            True when it matches that pattern.
        """
        if not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.IsNot):
            return False

        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Constant):
            return False

        return const_name == "None" and comparator.value is None

    def _is_isinstance_check(self, node: ast.expr) -> bool:
        """Report whether an expression calls ``isinstance``.

        Args:
            node: The expression being examined.

        Returns:
            True when it matches that pattern.
        """
        if not isinstance(node, ast.Call):
            return False
        return isinstance(node.func, ast.Name) and node.func.id == "isinstance"

    def _is_hasattr_check(self, node: ast.expr) -> bool:
        """Report whether an expression calls ``hasattr``.

        Args:
            node: The expression being examined.

        Returns:
            True when it matches that pattern.
        """
        if not isinstance(node, ast.Call):
            return False
        return isinstance(node.func, ast.Name) and node.func.id == "hasattr"

    def _is_len_existence_check(self, node: ast.expr) -> bool:
        """Report whether an expression is ``len(x) > 0`` or ``>= 1``.

        Args:
            node: The expression being examined.

        Returns:
            True when it matches that pattern.
        """
        if not isinstance(node, ast.Compare):
            return False
        if not isinstance(node.left, ast.Call):
            return False

        func = node.left.func
        if not (isinstance(func, ast.Name) and func.id == "len"):
            return False
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return False

        op = node.ops[0]
        comp = node.comparators[0]
        if not isinstance(comp, ast.Constant):
            return False

        if isinstance(op, ast.Gt) and comp.value == 0:
            return True
        return isinstance(op, ast.GtE) and comp.value == 1

    def _is_string_in_output(self, node: ast.expr) -> bool:
        """Report whether an expression matches a string in captured output.

        Args:
            node: The expression being examined.

        Returns:
            True when it matches that pattern.
        """
        if not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.In):
            return False

        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Attribute):
            return False

        return comparator.attr in ("out", "err", "stdout", "stderr")

    def _is_mock_called_check(self, node: ast.expr) -> bool:
        """Report whether an expression reads a double's ``called`` flag.

        Args:
            node: The expression being examined.

        Returns:
            True when it matches that pattern.
        """
        return isinstance(node, ast.Attribute) and node.attr == "called"

    def _is_variable_comparison(self, node: ast.Compare) -> bool:
        """Report whether a comparison pins a value rather than a literal.

        Args:
            node: The comparison being examined.

        Returns:
            True when at least one side is a name, attribute or
            subscript, which is what makes the comparison say something
            about the code under test.
        """
        var_types = (ast.Name, ast.Attribute, ast.Subscript)
        left_is_var = isinstance(node.left, var_types)
        right_is_var = any(isinstance(c, var_types) for c in node.comparators)
        return left_is_var and right_is_var


def is_test_module(path: Path) -> bool:
    """Report whether a path is a test module these rules apply to.

    ``as_posix`` normalises the separator on every platform, so one
    check covers both. The previous version paired it with a literal
    that was meant to be a Windows path but was written ``"\tests\\"`` —
    a tab followed by ``ests`` — and so never matched anything.

    Args:
        path: File being considered.

    Returns:
        True for a ``test_*.py`` file inside a ``tests`` directory.
    """
    return "/tests/" in path.as_posix() and path.name.startswith("test_")


class WeakAssertionRule:
    """Guard rule for detecting weak or fake tests."""

    name = "test-quality"

    def run(self, files: list[Path]) -> list[Violation]:
        """Report every weak assertion in the given test modules.

        Args:
            files: Paths to consider; non-test modules are skipped.

        Returns:
            One violation per weak assertion found.
        """
        out: list[Violation] = []
        for path in files:
            if not is_test_module(path):
                continue
            lines = read_lines(path)
            visitor = _AssertVisitor(path, lines)
            visitor.visit(ast.parse("\n".join(lines), filename=str(path)))
            out.extend(visitor.violations)
        return out


__all__ = ["MOCK_MODULES", "WeakAssertionRule", "is_test_module"]
