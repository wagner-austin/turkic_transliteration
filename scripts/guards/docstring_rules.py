"""The rule that keeps docstrings in Google style.

A signature says what a function takes and returns; a docstring is where
its meaning lives, and a docstring that omits half the signature is the
one most likely to be wrong about the rest. This rule holds every
function to the same shape: it is documented, its parameters are named
under ``Args:``, and anything it hands back is described under
``Returns:`` or ``Yields:``.

Two scoping decisions are stated here rather than left implicit.

A constructor may take its ``Args:`` from its class's docstring, which is
where Google style puts them and where this project already documents
them.

Under ``tests``, a docstring is required but ``Args:`` is not. A test's
parameters are almost always fixtures, and naming each one in prose
repeats what the fixture's own docstring says. Return sections are still
required there, though a test that returns something is rare.
"""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.guards import Violation
from scripts.guards.util import read_lines

ARGS_SECTION = "Args:"
RETURN_SECTIONS = ("Returns:", "Yields:")
IMPLICIT_PARAMETERS = frozenset({"self", "cls"})
NO_RETURN = "NoReturn"
CONSTRUCTOR = "__init__"
TESTS_DIRECTORY = "tests"


def is_python_module(path: Path) -> bool:
    """Report whether a path is a Python file this rule reads.

    Args:
        path: File being considered.

    Returns:
        True for a ``.py`` file.
    """
    return path.suffix == ".py"


def documents_arguments(path: Path) -> bool:
    """Report whether a file's functions must name their parameters.

    Args:
        path: File being scanned.

    Returns:
        False under ``tests``, where a parameter is nearly always a
        fixture that documents itself, and True everywhere else. The
        directory is matched as a path component rather than as a
        substring, so the answer does not depend on whether the caller
        passed an absolute path.
    """
    return TESTS_DIRECTORY not in path.parts


def declared_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """List the parameters a docstring is expected to name.

    Args:
        node: The function being inspected.

    Returns:
        Every parameter except ``self`` and ``cls``, which describe the
        receiver rather than the call.
    """
    arguments = node.args
    named = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    if arguments.vararg is not None:
        named.append(arguments.vararg)
    if arguments.kwarg is not None:
        named.append(arguments.kwarg)
    return [argument.arg for argument in named if argument.arg not in IMPLICIT_PARAMETERS]


def returns_a_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Report whether a function is annotated as returning something.

    Args:
        node: The function being inspected.

    Returns:
        False when the annotation is absent, is literally ``None``, or
        is ``NoReturn``. A function annotated ``NoReturn`` does not
        return at all, so a ``Returns:`` section would describe
        something that never happens; what it does instead belongs
        under ``Raises:``.
    """
    annotation = node.returns
    if annotation is None:
        return False
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return False
    return not _names_no_return(annotation)


def _names_no_return(annotation: ast.expr) -> bool:
    """Report whether an annotation is ``NoReturn``.

    Args:
        annotation: The return annotation.

    Returns:
        True for the bare name and for ``typing.NoReturn``.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id == NO_RETURN
    return isinstance(annotation, ast.Attribute) and annotation.attr == NO_RETURN


class DocstringRule:
    """Guard rule requiring a Google-style docstring on every function."""

    name = "docstrings"

    def run(self, files: list[Path]) -> list[Violation]:
        """Report every function whose docstring is missing or partial.

        Args:
            files: Paths to consider; non-Python files are skipped.

        Returns:
            One violation per offending function, in source order.
        """
        out: list[Violation] = []
        for path in files:
            if not is_python_module(path):
                continue
            lines = read_lines(path)
            tree = ast.parse("\n".join(lines), filename=str(path))
            out.extend(self._check_module(path, tree, lines))
        return out

    def _check_module(self, path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
        """Collect the docstring violations in one parsed module.

        Args:
            path: File being scanned, for the violation record.
            tree: Parsed module.
            lines: Source lines, used to quote the offending definition.

        Returns:
            The violations found, in source order.
        """
        found: list[Violation] = []
        documented_by_class = self._constructors_documented_by_their_class(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            quoted = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
            for kind in self._kinds_for(node, path, documented_by_class):
                found.append(Violation(file=path, line_no=node.lineno, kind=kind, line=quoted))
        return sorted(found, key=lambda violation: (violation.line_no, violation.kind))

    def _constructors_documented_by_their_class(self, tree: ast.AST) -> set[int]:
        """Find constructors whose class docstring already names the args.

        Args:
            tree: Parsed module.

        Returns:
            The line numbers of those constructors.
        """
        exempt: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            class_doc = ast.get_docstring(node)
            if class_doc is None or ARGS_SECTION not in class_doc:
                continue
            for member in node.body:
                if (
                    isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
                    and member.name == CONSTRUCTOR
                ):
                    exempt.add(member.lineno)
        return exempt

    def _kinds_for(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        path: Path,
        documented_by_class: set[int],
    ) -> list[str]:
        """Classify one function's docstring shortcomings.

        Args:
            node: The function being inspected.
            path: File it lives in, which decides whether ``Args:`` is
                required.
            documented_by_class: Constructors whose class docstring
                already names their parameters.

        Returns:
            The violation kinds, which may be empty.
        """
        docstring = ast.get_docstring(node)
        if docstring is None:
            return ["docstring-missing"]

        kinds: list[str] = []
        needs_arguments = documents_arguments(path) and node.lineno not in documented_by_class
        if needs_arguments and declared_parameters(node) and ARGS_SECTION not in docstring:
            kinds.append("docstring-missing-args")
        if returns_a_value(node) and not any(section in docstring for section in RETURN_SECTIONS):
            kinds.append("docstring-missing-returns")
        return kinds


__all__ = [
    "ARGS_SECTION",
    "RETURN_SECTIONS",
    "DocstringRule",
    "declared_parameters",
    "documents_arguments",
    "is_python_module",
    "returns_a_value",
]
