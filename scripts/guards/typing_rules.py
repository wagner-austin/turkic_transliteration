"""Typing, comment, suppression and logging rules of the guard.

These checks were defined in ``scripts/guard.py`` until 2026-11-04, when
they moved here unchanged so that the runner and each group of rules stay
under the 600-line file limit the file-length rule enforces.
"""

from __future__ import annotations

import ast
import tokenize
from collections.abc import Generator
from io import StringIO
from pathlib import Path

from scripts.guards import Violation


def _iter_tokens(text: str) -> Generator[tokenize.TokenInfo, None, None]:
    """Tokenise source text.

    Args:
        text: Python source.

    Yields:
        Each token, including comments, in source order.
    """
    reader = StringIO(text).readline
    yield from tokenize.generate_tokens(reader)


# =============================================================================
# Typing Rules
# =============================================================================


def _contains_object_in_annotation(node: ast.AST) -> bool:
    """Report whether an annotation names ``object`` anywhere within it.

    Args:
        node: An annotation subtree.

    Returns:
        True when ``object`` appears, including nested inside a
        subscript such as ``dict[str, object]``.
    """
    return any(isinstance(child, ast.Name) and child.id == "object" for child in ast.walk(node))


def _check_object_annotations(path: Path, node: ast.AST) -> list[Violation]:
    """Report every ``object`` annotation on one node.

    Args:
        path: File being scanned, for the violation record.
        node: Node from the module's tree.

    Returns:
        A violation for each of the variable, parameter and return
        annotations that names ``object``.
    """
    violations: list[Violation] = []
    kind = "object-in-annotation"

    if (
        isinstance(node, ast.AnnAssign)
        and node.annotation is not None
        and _contains_object_in_annotation(node.annotation)
    ):
        violations.append(Violation(file=path, line_no=node.lineno, kind=kind, line=""))

    if (
        isinstance(node, ast.arg)
        and node.annotation is not None
        and _contains_object_in_annotation(node.annotation)
    ):
        violations.append(Violation(file=path, line_no=node.lineno, kind=kind, line=""))

    if (
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.returns is not None
        and _contains_object_in_annotation(node.returns)
    ):
        violations.append(Violation(file=path, line_no=node.lineno, kind=kind, line=""))

    return violations


def _run_typing_rule(path: Path, tree: ast.AST) -> list[Violation]:
    """Report the forbidden typing constructs in one module.

    Args:
        path: File being scanned, for the violation record.
        tree: Parsed module.

    Returns:
        One violation per forbidden import, attribute, call or name.
    """
    violations: list[Violation] = []
    # TYPE_CHECKING is banned with the rest: a name imported under it
    # exists for the type checker and not at runtime, so the annotation
    # referring to it is checked against something the program never
    # loads. Where it is used to break an import cycle, the cycle is the
    # defect. Where it is used to defer a costly import, as it was in
    # web_utils for the transliteration pipeline, it duplicates an import
    # the function performs anyway.
    forbidden_imports = {"Any", "TYPE_CHECKING", "TypeAlias", "cast"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                if alias.name in forbidden_imports:
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind=f"typing-import-{alias.name.lower()}",
                            line="",
                        )
                    )

        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "typing"
            and node.attr in forbidden_imports
        ):
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind=f"typing-{node.attr.lower()}-usage",
                    line="",
                )
            )

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "cast"
        ):
            violations.append(Violation(file=path, line_no=node.lineno, kind="cast-call", line=""))

        if isinstance(node, ast.Name) and node.id == "Any":
            violations.append(Violation(file=path, line_no=node.lineno, kind="any-usage", line=""))

        if isinstance(node, ast.Name) and node.id == "TYPE_CHECKING":
            violations.append(
                Violation(file=path, line_no=node.lineno, kind="type-checking-usage", line="")
            )

        violations.extend(_check_object_annotations(path, node))

    return violations


def _run_comments_rule(path: Path, text: str) -> list[Violation]:
    """Report the suppression comments in one module.

    Args:
        path: File being scanned, for the violation record.
        text: The module's source.

    Returns:
        One violation per ``type: ignore`` and per ``pragma`` comment.
        A comment carrying both is reported twice.
    """
    violations: list[Violation] = []
    for tok in _iter_tokens(text):
        if tok.type == tokenize.COMMENT:
            if "type: ignore" in tok.string:
                violations.append(
                    Violation(
                        file=path,
                        line_no=tok.start[0],
                        kind="type-ignore",
                        line=tok.line.rstrip("\n"),
                    )
                )
            if "pragma" in tok.string:
                violations.append(
                    Violation(
                        file=path,
                        line_no=tok.start[0],
                        kind="pragma-comment",
                        line=tok.line.rstrip("\n"),
                    )
                )
    return violations


# =============================================================================
# Suppress Rules
# =============================================================================


def _is_suppress(expr: ast.AST) -> bool:
    """Report whether an expression is ``contextlib.suppress``.

    Args:
        expr: The context expression of a ``with`` item.

    Returns:
        True for both the qualified and the bare spelling, whether or
        not it is being called.
    """
    func = expr.func if isinstance(expr, ast.Call) else expr
    if isinstance(func, ast.Attribute):
        is_contextlib = isinstance(func.value, ast.Name) and func.value.id == "contextlib"
        return is_contextlib and func.attr == "suppress"
    return isinstance(func, ast.Name) and func.id == "suppress"


def _run_suppress_rule(path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
    """Report every ``contextlib.suppress`` block in one module.

    Args:
        path: File being scanned, for the violation record.
        tree: Parsed module.
        lines: Source lines, used to quote the offending line.

    Returns:
        One violation per suppressing ``with`` statement.
    """
    violations: list[Violation] = []
    seen: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                if _is_suppress(item.context_expr):
                    line_no = item.context_expr.lineno
                    if line_no in seen:
                        continue
                    seen.add(line_no)
                    idx = line_no - 1
                    text = lines[idx] if 0 <= idx < len(lines) else ""
                    violations.append(
                        Violation(
                            file=path,
                            line_no=line_no,
                            kind="suppress-usage",
                            line=text.rstrip("\n"),
                        )
                    )
    return violations


# =============================================================================
# Logging Rules
# =============================================================================


def _run_logging_rule(path: Path, tree: ast.AST) -> list[Violation]:
    """Report every ``print`` call in a published module.

    Library code emits through logging so the caller controls the sink.
    Tests and scripts may print freely, so only ``src`` is scanned.

    Args:
        path: File being scanned, for the violation record.
        tree: Parsed module.

    Returns:
        One violation per ``print`` call, or none outside ``src``.
    """
    # Only check src/ files
    if "src" not in path.parts:
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="print-usage",
                    line="Library code must log, not print",
                )
            )
    return violations
