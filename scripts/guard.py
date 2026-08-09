"""Guard script to enforce strict typing and exception handling rules.

Checks for violations:
- No `Any` usage
- No `cast` usage
- No `object` in type annotations
- No `type: ignore` comments
- No `TypeAlias` usage
- No `TYPE_CHECKING` blocks
- No `contextlib.suppress` usage
- No `# pragma` comments
- No silent exception handling (except: pass)
- Broad exceptions (Exception/BaseException) must log AND re-raise
- Specific exceptions must log OR re-raise
- No `print()` in src/ (use _console module instead)
- Weak test assertions (is not None, isinstance, hasattr, len > 0)
- Transliteration test quality (domain calls must assert on the returned value)
- Google-style docstrings (every function documented, parameters and
  return values named)

Run with: python -m scripts.guard
"""

from __future__ import annotations

import ast
import re
import sys
import tokenize
from collections.abc import Generator, Sequence
from io import StringIO
from pathlib import Path

from scripts.guards import RuleReport, Violation
from scripts.guards.docstring_rules import DocstringRule
from scripts.guards.test_quality_rules import (
    PatchingRule,
    TransliterationTestQualityRule,
    WeakAssertionRule,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCANNED_DIRECTORIES: tuple[str, ...] = ("src", "tests", "scripts")


def _iter_py_files(root: Path) -> Generator[Path, None, None]:
    """Yield every Python file the guards apply to.

    Args:
        root: Project root to scan beneath.

    Yields:
        Each ``.py`` file under the scanned directories that exist.
    """
    for subdir in SCANNED_DIRECTORIES:
        dir_path = root / subdir
        if dir_path.is_dir():
            yield from sorted(dir_path.rglob("*.py"))


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


# =============================================================================
# Exception Rules
# =============================================================================

_EXCEPT_HEADER = re.compile(r"^(\s*)except(\s+([^:]+))?:\s*$")
_BROAD_TYPES = re.compile(r"\b(Exception|BaseException)\b")
_LOG_CALL = re.compile(r"\b(logging|log|logger)\.(debug|info|warning|error|exception|critical)\(")
_RAISE_RE = re.compile(r"\braise\b")


def _parse_except_header(raw: str) -> tuple[int, str] | None:
    """Split an ``except`` header into its indent and its types.

    Args:
        raw: One source line, which may be anything.

    Returns:
        The indent width and the exception types as written, or
        ``None`` when the line is not an ``except`` header. A bare
        ``except:`` reports an empty type string.
    """
    match = _EXCEPT_HEADER.match(raw)
    if match is None:
        return None
    indent_group = match.group(1)
    group3 = match.group(3)
    indent_str = indent_group if indent_group is not None else ""
    types_str = group3 if group3 is not None else ""
    return len(indent_str), types_str.strip()


def _is_broad_exception(types: str) -> bool:
    """Report whether a caught type is a broad one.

    Args:
        types: Exception types as written, empty for a bare ``except``.

    Returns:
        True for a bare ``except`` and for anything naming
        ``Exception`` or ``BaseException``.
    """
    return types == "" or _BROAD_TYPES.search(types) is not None


def _first_body_is_trivial(line: str) -> bool:
    """Report whether a line is an empty statement.

    Args:
        line: One source line.

    Returns:
        True for ``pass`` and for ``...``, with or without a trailing
        comment. Either means the handler discards the exception.
    """
    return re.match(r"^\s+(pass|\.\.\.)\s*(#.*)?$", line) is not None


def _scan_except_body(
    lines: Sequence[str], start: int, header_indent: int
) -> tuple[bool, bool, int]:
    """Scan one ``except`` body for logging and re-raising.

    Args:
        lines: All source lines of the module.
        start: Index of the body's first line.
        header_indent: Indent width of the ``except`` header, which
            marks where the body ends.

    Returns:
        Whether the body logs, whether it raises, and the index of the
        first line after it.
    """
    total = len(lines)
    has_log = False
    has_raise = False
    i = start
    while i < total:
        body_line = lines[i]
        if body_line.strip() == "":
            i += 1
            continue
        body_indent = len(body_line) - len(body_line.lstrip(" \t"))
        if body_indent <= header_indent and re.match(
            r"^\s*(except\b|finally\b|else\b|$)", body_line
        ):
            break
        if _RAISE_RE.search(body_line):
            has_raise = True
        if _LOG_CALL.search(body_line):
            has_log = True
        i += 1
    return has_log, has_raise, i


def _find_body_start(lines: Sequence[str], start: int) -> int:
    """Find the first non-empty line after an ``except`` header.

    Args:
        lines: All source lines of the module.
        start: Index to begin looking from.

    Returns:
        The index of that line, or the number of lines when the file
        ends first. Valid Python always has a body, so the second case
        only arises on a truncated file.
    """
    total = len(lines)
    i = start
    while i < total:
        if lines[i].strip() != "":
            return i
        i += 1
    return total


def _run_exceptions_rule(path: Path, lines: list[str]) -> list[Violation]:
    """Report the exception handlers that neither log nor re-raise.

    A broad handler must do both; a specific one must do at least one.
    Test files are skipped, because a test catching an exception to
    assert on it is doing exactly what it should.

    Args:
        path: File being scanned, for the violation record.
        lines: The module's source lines. Scanned as text rather than
            parsed, so that a handler's formatting is visible.

    Returns:
        One violation per offending handler.
    """
    # Skip test files
    if "tests" in path.parts:
        return []

    violations: list[Violation] = []
    total = len(lines)
    idx = 0
    while idx < total:
        raw = lines[idx]
        parsed = _parse_except_header(raw)
        if parsed is None:
            idx += 1
            continue
        indent, types = parsed
        broad = _is_broad_exception(types)

        body_start = _find_body_start(lines, idx + 1)

        if body_start < total and _first_body_is_trivial(lines[body_start]):
            violations.append(
                Violation(
                    file=path,
                    line_no=idx + 1,
                    kind="silent-except-body",
                    line=raw.rstrip("\n"),
                )
            )

        has_log, has_raise, body_end = _scan_except_body(lines, body_start, indent)
        if broad:
            if not (has_log and has_raise):
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx + 1,
                        kind="broad-except-requires-log-and-raise",
                        line=raw.rstrip("\n"),
                    )
                )
        else:
            if not (has_log or has_raise):
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx + 1,
                        kind="except-without-log-or-raise",
                        line=raw.rstrip("\n"),
                    )
                )

        idx = body_end if body_end > idx else idx + 1
    return violations


# =============================================================================
# Main Runner
# =============================================================================


def run_guards(root: Path) -> int:
    """Run every guard rule over a project tree.

    Args:
        root: Project root; its ``src``, ``tests`` and ``scripts``
            directories are scanned.

    Returns:
        0 when nothing was found, 2 when any rule reported a violation.

    Raises:
        SyntaxError: If a scanned file does not parse. The exception
            already names the file, so it is not wrapped.
    """
    typing_violations: list[Violation] = []
    comments_violations: list[Violation] = []
    suppress_violations: list[Violation] = []
    exceptions_violations: list[Violation] = []
    logging_violations: list[Violation] = []

    all_files: list[Path] = list(_iter_py_files(root))

    for path in all_files:
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))

        typing_violations.extend(_run_typing_rule(path, tree))
        comments_violations.extend(_run_comments_rule(path, source))
        suppress_violations.extend(_run_suppress_rule(path, tree, lines))
        exceptions_violations.extend(_run_exceptions_rule(path, lines))
        logging_violations.extend(_run_logging_rule(path, tree))

    weak_assertion_rule = WeakAssertionRule()
    translit_test_quality_rule = TransliterationTestQualityRule()
    patching_rule = PatchingRule()
    docstring_rule = DocstringRule()
    weak_assertion_violations = weak_assertion_rule.run(all_files)
    translit_test_quality_violations = translit_test_quality_rule.run(all_files)
    patching_violations = patching_rule.run(all_files)
    docstring_violations = docstring_rule.run(all_files)

    reports = [
        RuleReport(name="typing", violations=len(typing_violations)),
        RuleReport(name="comments", violations=len(comments_violations)),
        RuleReport(name="suppress", violations=len(suppress_violations)),
        RuleReport(name="exceptions", violations=len(exceptions_violations)),
        RuleReport(name="logging", violations=len(logging_violations)),
        RuleReport(name="test-quality", violations=len(weak_assertion_violations)),
        RuleReport(name="translit-test-quality", violations=len(translit_test_quality_violations)),
        RuleReport(name="patching", violations=len(patching_violations)),
        RuleReport(name="docstrings", violations=len(docstring_violations)),
    ]

    all_violations = (
        typing_violations
        + comments_violations
        + suppress_violations
        + exceptions_violations
        + logging_violations
        + weak_assertion_violations
        + translit_test_quality_violations
        + patching_violations
        + docstring_violations
    )

    print("Guard rule summary:")
    for rep in reports:
        print(f"  {rep.name}: {rep.violations} violations")

    if all_violations:
        print("Guard checks failed:", file=sys.stderr)
        for v in all_violations:
            rel_path = v.file.relative_to(root) if v.file.is_relative_to(root) else v.file
            text = v.line[:80] + "..." if len(v.line) > 80 else v.line
            print(f"  {rel_path}:{v.line_no}: {v.kind} {text}", file=sys.stderr)
        return 2

    print("Guard checks passed: no violations found.")
    return 0


def parse_root_argument(args: Sequence[str]) -> Path | None:
    """Extract the ``--root`` value from a command line.

    Args:
        args: Arguments after the program name.

    Returns:
        The resolved root, or ``None`` when ``--root`` was not given with
        a value.
    """
    index = 0
    while index < len(args):
        if args[index] == "--root" and index + 1 < len(args):
            return Path(args[index + 1]).resolve()
        index += 1
    return None


def main(argv: Sequence[str], default_root: Path = PROJECT_ROOT) -> int:
    """Run the guards over ``--root`` or over this project.

    Args:
        argv: Arguments after the program name. Passed explicitly rather
            than read from ``sys.argv`` here, so the entry point is the
            only place that touches process state.
        default_root: Root used when ``--root`` is absent. Defaults to
            the directory holding this script's project file.

    Returns:
        0 when nothing was found, 2 when a rule reported a violation,
        and 1 when the default root is not a project tree.
    """
    root = parse_root_argument(argv)
    if root is not None:
        return run_guards(root)

    if not (default_root / "pyproject.toml").is_file():
        print(f"ERROR: pyproject.toml not found in {default_root}", file=sys.stderr)
        return 1

    return run_guards(default_root)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
