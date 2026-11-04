"""The transliteration test-quality rule.

A test that calls a transliteration or language-identification entry
point must compare the returned value against an expected one; a test
that only proves the call ran is reported. This rule was defined in
:mod:`scripts.guards.test_quality_rules` until 2026-11-04, when it moved
here unchanged so that each test-quality rule stays under the 600-line
file limit.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from scripts.guards import Violation
from scripts.guards.test_quality_rules import is_test_module
from scripts.guards.util import read_lines


class _TranslitPatternVisitor(ast.NodeVisitor):
    """Record whether a test body exercises and then verifies a domain call.

    A domain call is any invocation of a transliteration or
    language-identification entry point. A value assertion is any
    comparison that pins a result against an expected value, whether by
    equality, membership, containment, or an ``assertEqual``-family
    method.

    Containment is here because of the inventory tests. A published
    phonological description states which segments a language has, and
    the check that output stays inside that set needs no expected value
    for any particular word — which is exactly why it cannot be
    circular. It pins the result against the source no less than an
    equality does. ``assert len(x) > 0`` is not admitted by this: the
    weak-assertion rules reject it separately.
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
        ast.LtE,
        ast.GtE,
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

    def visit_With(self, node: ast.With) -> None:
        """Flag a ``pytest.raises`` block as pinning the call's outcome.

        A test that calls a domain function inside ``pytest.raises`` is
        asserting what the call does: it names the exception type, and
        usually the message too. Requiring an equality assertion as well
        would mean requiring a return value the call never produces.

        Args:
            node: The ``with`` statement being visited.
        """
        if any(self._is_raises(item.context_expr) for item in node.items):
            self.has_value_assertion = True
        self.generic_visit(node)

    def _is_raises(self, expression: ast.expr) -> bool:
        """Report whether an expression is a call to ``pytest.raises``.

        Args:
            expression: The context expression of a ``with`` item.

        Returns:
            True for ``pytest.raises(...)`` and for a bare ``raises(...)``
            imported from pytest.
        """
        if not isinstance(expression, ast.Call):
            return False
        func = expression.func
        if isinstance(func, ast.Attribute):
            return func.attr == "raises"
        return isinstance(func, ast.Name) and func.id == "raises"


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
            if not is_test_module(path):
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


__all__ = ["TransliterationTestQualityRule"]
