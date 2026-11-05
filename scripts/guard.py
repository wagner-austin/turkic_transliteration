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
- No Python file longer than 600 lines

This module is the runner. The rules live in :mod:`scripts.guards`, one
module per group, so that no file of the guard itself passes the limit
its file-length rule enforces.

Run with: python -m scripts.guard
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Generator, Sequence
from pathlib import Path

from scripts.guards import RuleReport, Violation
from scripts.guards.docstring_rules import DocstringRule
from scripts.guards.exception_rules import _run_exceptions_rule
from scripts.guards.file_length_rules import FileLengthRule
from scripts.guards.patching_rules import PatchingRule
from scripts.guards.provenance_rules import (
    RuleSourceRule,
    SelfReferentialExpectationRule,
    TestSourceInheritanceRule,
)
from scripts.guards.test_quality_rules import WeakAssertionRule
from scripts.guards.translit_test_rules import TransliterationTestQualityRule
from scripts.guards.typing_rules import (
    _run_comments_rule,
    _run_logging_rule,
    _run_suppress_rule,
    _run_typing_rule,
)
from scripts.guards.windows_rules import WindowsRule, tracked_files

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


def _iter_rule_files(root: Path) -> Generator[Path, None, None]:
    """Yield every transliteration rule file the guards apply to.

    Rule files are not Python and so are invisible to
    :func:`_iter_py_files`, but they carry the source citation the
    provenance rule checks, so they are collected separately.

    Args:
        root: Project root to scan beneath.

    Yields:
        Each ``.rules`` file under the rules directory.
    """
    rules_dir = root / "src" / "turkic_translit" / "rules"
    if rules_dir.is_dir():
        yield from sorted(rules_dir.rglob("*.rules"))


# =============================================================================
# Main Runner
# =============================================================================


def run_guards(root: Path) -> int:
    """Run every guard rule over a project tree.

    Args:
        root: Project root, a git work tree; its ``src``, ``tests`` and
            ``scripts`` directories are scanned, and every tracked file is
            read by the Windows rule.

    Returns:
        0 when nothing was found, 2 when any rule reported a violation.

    Raises:
        SyntaxError: If a scanned file does not parse. The exception
            already names the file, so it is not wrapped.
        subprocess.CalledProcessError: If git cannot list the tracked
            files, as when ``root`` is not a git work tree.
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
    rule_source_rule = RuleSourceRule()
    test_source_rule = TestSourceInheritanceRule()
    self_referential_rule = SelfReferentialExpectationRule()
    file_length_rule = FileLengthRule()
    windows_rule = WindowsRule(root)
    weak_assertion_violations = weak_assertion_rule.run(all_files)
    translit_test_quality_violations = translit_test_quality_rule.run(all_files)
    patching_violations = patching_rule.run(all_files)
    docstring_violations = docstring_rule.run(all_files)
    rule_source_violations = rule_source_rule.run(list(_iter_rule_files(root)))
    test_source_violations = test_source_rule.run(all_files)
    self_referential_violations = self_referential_rule.run(all_files)
    file_length_violations = file_length_rule.run(all_files)
    windows_violations = windows_rule.run(tracked_files(root))

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
        RuleReport(name="rule-provenance", violations=len(rule_source_violations)),
        RuleReport(name="test-provenance", violations=len(test_source_violations)),
        RuleReport(
            name="self-referential-expectation", violations=len(self_referential_violations)
        ),
        RuleReport(name=file_length_rule.name, violations=len(file_length_violations)),
        RuleReport(name=windows_rule.name, violations=len(windows_violations)),
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
        + rule_source_violations
        + test_source_violations
        + self_referential_violations
        + file_length_violations
        + windows_violations
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
