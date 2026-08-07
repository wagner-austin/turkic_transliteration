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
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from scripts.guards import Violation
from scripts.guards.util import read_lines


def _is_patch_call(func: ast.expr) -> bool:
    """Check if func is a patch() call."""
    if isinstance(func, ast.Attribute) and func.attr == "patch":
        return True
    return isinstance(func, ast.Name) and func.id == "patch"


class _AssertVisitor(ast.NodeVisitor):
    """Visitor to analyze assert statements in test functions."""

    def __init__(self, path: Path, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.violations: list[Violation] = []
        self.current_function: str = ""
        self.function_has_comparison: bool = False
        self.function_mock_count: int = 0
        self.function_start_line: int = 0

    def _get_line(self, line_no: int) -> str:
        """Get source line content by line number (1-indexed)."""
        idx = line_no - 1
        if 0 <= idx < len(self.lines):
            return self.lines[idx].strip()
        return ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("test_"):
            self._analyze_test_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name.startswith("test_"):
            self._analyze_test_function(node)
        self.generic_visit(node)

    def _analyze_test_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
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
        """Check for weak assertion patterns."""
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
        """Check for mock-related issues."""
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
        """Track if test has meaningful comparisons."""
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
        """Check if node is `x is not <const>`."""
        if not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.IsNot):
            return False

        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Constant):
            return False

        return const_name == "None" and comparator.value is None

    def _is_isinstance_check(self, node: ast.expr) -> bool:
        """Check if node is isinstance(x, Y)."""
        if not isinstance(node, ast.Call):
            return False
        return isinstance(node.func, ast.Name) and node.func.id == "isinstance"

    def _is_hasattr_check(self, node: ast.expr) -> bool:
        """Check if node is hasattr(x, "y")."""
        if not isinstance(node, ast.Call):
            return False
        return isinstance(node.func, ast.Name) and node.func.id == "hasattr"

    def _is_len_existence_check(self, node: ast.expr) -> bool:
        """Check if node is len(x) > 0 or len(x) >= 1."""
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
        """Check if node is 'string' in x.out or x.err."""
        if not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.In):
            return False

        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Attribute):
            return False

        return comparator.attr in ("out", "err", "stdout", "stderr")

    def _is_mock_called_check(self, node: ast.expr) -> bool:
        """Check if node is mock.called without args check."""
        return isinstance(node, ast.Attribute) and node.attr == "called"

    def _is_variable_comparison(self, node: ast.Compare) -> bool:
        """Check if comparison involves variables (not just constants)."""
        var_types = (ast.Name, ast.Attribute, ast.Subscript)
        left_is_var = isinstance(node.left, var_types)
        right_is_var = any(isinstance(c, var_types) for c in node.comparators)
        return left_is_var and right_is_var


class WeakAssertionRule:
    """Guard rule for detecting weak or fake tests."""

    name = "test-quality"

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []

        for path in files:
            if "/tests/" not in path.as_posix() and "\\tests\\" not in str(path):
                continue
            if not path.name.startswith("test_"):
                continue

            lines = read_lines(path)
            source = "\n".join(lines)

            tree = ast.parse(source, filename=str(path))

            visitor = _AssertVisitor(path, lines)
            visitor.visit(tree)
            out.extend(visitor.violations)

        return out


class _TranslitPatternVisitor(ast.NodeVisitor):
    """Record whether a test body exercises and then verifies a domain call.

    A domain call is any invocation of a transliteration or
    language-identification entry point. A value assertion is any
    comparison that pins a result against an expected value, whether by
    equality, membership, or an ``assertEqual``-family method.
    """

    _DOMAIN_CALLS: ClassVar[frozenset[str]] = frozenset(
        {
            "to_ipa",
            "to_latin",
            "transliterate",
            "translit_line",
            "classify_line",
            "detect_language",
            "resolve_lid_model",
            "load_lid_model",
        }
    )

    _VALUE_ASSERT_METHODS: ClassVar[frozenset[str]] = frozenset(
        {"assertEqual", "assertNotEqual", "assertIn", "assertNotIn"}
    )

    _VALUE_COMPARE_OPS: ClassVar[tuple[type[ast.cmpop], ...]] = (
        ast.Eq,
        ast.NotEq,
        ast.In,
        ast.NotIn,
    )

    def __init__(self) -> None:
        """Initialise both detection flags to unseen."""
        self.has_domain_call: bool = False
        self.has_value_assertion: bool = False

    def visit_Call(self, node: ast.Call) -> None:
        """Flag a domain call or a value-asserting helper method.

        Args:
            node: The call expression being visited.
        """
        func = node.func
        if isinstance(func, ast.Name) and func.id in self._DOMAIN_CALLS:
            self.has_domain_call = True
        if isinstance(func, ast.Attribute):
            if func.attr in self._DOMAIN_CALLS:
                self.has_domain_call = True
            if func.attr in self._VALUE_ASSERT_METHODS:
                self.has_value_assertion = True
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        """Flag an ``assert`` whose test pins a value by comparison.

        Args:
            node: The assert statement being visited.
        """
        test = node.test
        if isinstance(test, ast.Compare) and any(
            isinstance(op, self._VALUE_COMPARE_OPS) for op in test.ops
        ):
            self.has_value_assertion = True
        self.generic_visit(node)


class TransliterationTestQualityRule:
    """Guard rule enforcing that domain tests verify output, not execution.

    A test that calls :func:`to_ipa` and asserts nothing about the string
    it returns proves only that the function did not raise. This rule
    fails such tests so that coverage cannot be bought with calls that
    check nothing.
    """

    name = "translit-test-quality"

    def run(self, files: list[Path]) -> list[Violation]:
        """Scan test files for domain calls lacking a value assertion.

        Args:
            files: Python files to consider; non-test files are skipped.

        Returns:
            One violation per offending test function.
        """
        out: list[Violation] = []

        for path in files:
            if "/tests/" not in path.as_posix() and "\tests\\" not in str(path):
                continue
            if not path.name.startswith("test_"):
                continue

            lines = read_lines(path)
            tree = ast.parse("\n".join(lines), filename=str(path))
            out.extend(self._check_translit_patterns(path, tree, lines))

        return out

    def _check_translit_patterns(
        self, path: Path, tree: ast.AST, lines: list[str]
    ) -> list[Violation]:
        """Report test functions that call domain code without verifying it.

        Args:
            path: File being scanned, used for violation reporting.
            tree: Parsed module AST.
            lines: Source lines, used to quote the offending definition.

        Returns:
            Violations for every unverified domain-calling test.
        """
        violations: list[Violation] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue

            visitor = _TranslitPatternVisitor()
            for child in node.body:
                visitor.visit(child)

            if visitor.has_domain_call and not visitor.has_value_assertion:
                idx = node.lineno - 1
                text = lines[idx].strip() if 0 <= idx < len(lines) else ""
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="translit-call-without-value-assertion",
                        line=text,
                    )
                )

        return violations


__all__ = ["TransliterationTestQualityRule", "WeakAssertionRule"]
