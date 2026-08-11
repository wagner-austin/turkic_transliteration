"""Rules keeping rule-file provenance readable and inherited by tests.

Three failures this project actually had, none of them visible to any
other check.

A rule file could carry a source citation that nothing parsed, so the
citation constrained nothing and drifted from the rules beneath it: the
Turkish header quoted a rule about soft g that the rules did not
implement, and no check could tell.

A gold-standard test could name a source in a comment while its expected
values came from somewhere else, or from nowhere. Five test modules
pinned expected IPA while naming no source at all.

And a test could compute its expected value by calling the very function
under test. Seven passage tests did exactly that, so they passed under
every possible implementation, including a wrong one.

The rules below close each in turn: provenance must be machine-readable,
a test that states expected output must say which rule file's source it
inherits, and an expectation may not be derived from the code it checks.
"""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.guards import Violation
from scripts.guards.util import read_lines

RULES_SUFFIX = ".rules"
IPA_MARKER = "_ipa"
SOURCE_FIELD_PREFIX = "# Source-"
REQUIRED_SOURCE_FIELDS = ("Authors", "Year", "Title", "Container", "Id")

TESTS_DIRECTORY = "tests"
TRANSLITERATION_CALLS = frozenset({"to_ipa", "to_latin"})
INHERITS_DECLARATION = "INHERITS_SOURCE"


def is_ipa_rule_file(path: Path) -> bool:
    """Report whether a path is an IPA rule file.

    Args:
        path: File being considered.

    Returns:
        True for a ``*_ipa*.rules`` file.
    """
    return path.suffix == RULES_SUFFIX and IPA_MARKER in path.stem


def is_test_module(path: Path) -> bool:
    """Report whether a path is a test module.

    Args:
        path: File being considered.

    Returns:
        True for a ``.py`` file under the tests directory.
    """
    return path.suffix == ".py" and TESTS_DIRECTORY in path.parts


def calls_transliterator(tree: ast.AST) -> bool:
    """Report whether a module calls a transliteration entry point.

    Args:
        tree: Parsed module.

    Returns:
        True when ``to_ipa`` or ``to_latin`` is called.
    """
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in TRANSLITERATION_CALLS
        for node in ast.walk(tree)
    )


def _subtree_calls_transliterator(node: ast.AST) -> bool:
    """Report whether one expression contains a transliteration call.

    Args:
        node: Expression subtree.

    Returns:
        True when the subtree calls ``to_ipa`` or ``to_latin``.
    """
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in TRANSLITERATION_CALLS
        for child in ast.walk(node)
    )


def states_expectations(tree: ast.AST) -> bool:
    """Report whether a module states expected transliteration output.

    Two shapes count: a comparison naming a string literal directly, and
    a table of string-to-string pairs feeding a parametrised test.

    Args:
        tree: Parsed module.

    Returns:
        True when the module states at least one expected string.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(_subtree_calls_transliterator(operand) for operand in operands) and any(
                isinstance(operand, ast.Constant) and isinstance(operand.value, str)
                for operand in operands
            ):
                return True
        if isinstance(node, ast.Dict) and node.keys:
            pairs = zip(node.keys, node.values, strict=True)
            if all(
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                for key, value in pairs
            ):
                return True
    return False


class RuleSourceRule:
    """Guard rule requiring readable provenance on every IPA rule file."""

    name = "rule-provenance"

    def run(self, files: list[Path]) -> list[Violation]:
        """Report every IPA rule file whose provenance cannot be read.

        Args:
            files: Paths to consider; non-rule files are skipped.

        Returns:
            One violation per missing field, in file order.
        """
        out: list[Violation] = []
        for path in sorted(files):
            if not is_ipa_rule_file(path):
                continue
            declared = {
                line[len(SOURCE_FIELD_PREFIX) :].partition(":")[0].strip()
                for line in read_lines(path)
                if line.startswith(SOURCE_FIELD_PREFIX)
            }
            for field in REQUIRED_SOURCE_FIELDS:
                if field not in declared:
                    out.append(
                        Violation(
                            file=path,
                            line_no=1,
                            kind="rule-file-declares-no-source",
                            line=f"missing '{SOURCE_FIELD_PREFIX}{field}:'",
                        )
                    )
        return out


class TestSourceInheritanceRule:
    """Guard rule requiring pinned transliteration tests to name their source."""

    name = "test-provenance"

    def run(self, files: list[Path]) -> list[Violation]:
        """Report every test module that pins output without naming a source.

        A module qualifies when it both calls a transliteration entry
        point and states an expected string. Such a module must bind
        :data:`INHERITS_DECLARATION` to the identifier the rule file
        declares, so the claim can be checked rather than believed.

        Args:
            files: Paths to consider; non-test files are skipped.

        Returns:
            One violation per offending module.
        """
        out: list[Violation] = []
        for path in sorted(files):
            if not is_test_module(path):
                continue
            tree = ast.parse("\n".join(read_lines(path)), filename=str(path))
            if not (calls_transliterator(tree) and states_expectations(tree)):
                continue
            if self._declares_source(tree):
                continue
            out.append(
                Violation(
                    file=path,
                    line_no=1,
                    kind="test-pins-output-without-source",
                    line=f"no {INHERITS_DECLARATION} binding",
                )
            )
        return out

    def _declares_source(self, tree: ast.AST) -> bool:
        """Report whether a module binds the source-inheritance name.

        Args:
            tree: Parsed test module.

        Returns:
            True when the module assigns :data:`INHERITS_DECLARATION`.
        """
        return any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == INHERITS_DECLARATION
                for target in node.targets
            )
            for node in ast.walk(tree)
        )


class SelfReferentialExpectationRule:
    """Guard rule rejecting expectations computed by the code under test."""

    name = "self-referential-expectation"

    def run(self, files: list[Path]) -> list[Violation]:
        """Report every expectation derived from the function it checks.

        A name bound to the result of a transliteration call, and then
        used as the expected side of an assertion about that same call,
        makes an assertion that holds under every implementation. Such a
        test reports determinism and nothing else.

        Args:
            files: Paths to consider; non-test files are skipped.

        Returns:
            One violation per offending assertion.
        """
        out: list[Violation] = []
        for path in sorted(files):
            if not is_test_module(path):
                continue
            lines = read_lines(path)
            tree = ast.parse("\n".join(lines), filename=str(path))
            out.extend(self._generated_tables(path, tree, lines))
            module_derived = self._names_derived_from_transliteration(
                tree, recurse=False, seed=frozenset()
            )
            for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
                derived = self._names_derived_from_transliteration(
                    function, recurse=True, seed=frozenset(module_derived)
                )
                if derived:
                    out.extend(self._offending_assertions(path, function, derived, lines))
        return out

    def _generated_tables(self, path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
        """Report expectation tables built by running the transliterator.

        A comprehension that maps a transliteration call over a list of
        inputs produces the answers the code currently gives, not the
        answers it should give. Assigning that to a name called a gold
        standard is the whole anti-pattern in one line, and it survives
        any comparison the rule might otherwise inspect, because both
        sides of the eventual assertion are plain names.

        Args:
            path: File being scanned, for the violation record.
            tree: Parsed test module.
            lines: Source lines, used to quote the assignment.

        Returns:
            One violation per generated table.
        """
        out: list[Violation] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.ListComp | ast.SetComp | ast.GeneratorExp):
                continue
            if not _subtree_calls_transliterator(node.value.elt):
                continue
            out.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="expectation-table-generated-by-code-under-test",
                    line=lines[node.lineno - 1].strip(),
                )
            )
        return out

    def _names_derived_from_transliteration(
        self, tree: ast.AST, recurse: bool, seed: frozenset[str]
    ) -> set[str]:
        """Collect names whose value comes from a transliteration call.

        Both direct assignment and binding through a comprehension or a
        loop target count, since a table built by mapping the function
        over inputs is the shape this rule exists to reject.

        Scope matters. Module-level bindings are visible to every test,
        but one function's local must not be read as another's, or a
        name reused across two unrelated tests makes the second look
        circular when it is not.

        Args:
            tree: Module or function to scan.
            recurse: When False, only bindings at this node's own top
                level are collected, so function locals stay local.
            seed: Names already known to carry transliteration output,
                normally the module-level bindings a function can see.

        Returns:
            The set of bound names carrying transliteration output.
        """
        nodes = list(ast.walk(tree)) if recurse else list(ast.iter_child_nodes(tree))
        derived: set[str] = set(seed)
        for node in nodes:
            if isinstance(node, ast.Assign) and _subtree_calls_transliterator(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        derived.add(target.id)
        for node in nodes:
            if (
                isinstance(node, ast.For)
                and isinstance(node.iter, ast.Call)
                and any(isinstance(arg, ast.Name) and arg.id in derived for arg in node.iter.args)
            ):
                for name in ast.walk(node.target):
                    if isinstance(name, ast.Name):
                        derived.add(name.id)
        return derived

    def _offending_assertions(
        self, path: Path, tree: ast.AST, derived: set[str], lines: list[str]
    ) -> list[Violation]:
        """Report assertions comparing a call against a derived name.

        Args:
            path: File being scanned, for the violation record.
            tree: Parsed test module.
            derived: Names carrying transliteration output.
            lines: Source lines, used to quote the assertion.

        Returns:
            One violation per offending assertion.
        """
        out: list[Violation] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
                continue
            operands = [node.test.left, *node.test.comparators]
            calls = any(_subtree_calls_transliterator(operand) for operand in operands)
            # Comparing one call against another states a property, such
            # as two spellings of the same input agreeing. Only a call
            # weighed against a stored result of that call is circular.
            expected_is_stored = any(
                isinstance(operand, ast.Name) and operand.id in derived for operand in operands
            )
            if calls and expected_is_stored:
                out.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="expectation-computed-by-code-under-test",
                        line=lines[node.lineno - 1].strip(),
                    )
                )
        return out


__all__ = [
    "INHERITS_DECLARATION",
    "REQUIRED_SOURCE_FIELDS",
    "RuleSourceRule",
    "SelfReferentialExpectationRule",
    "TestSourceInheritanceRule",
    "calls_transliterator",
    "is_ipa_rule_file",
    "is_test_module",
    "states_expectations",
]
