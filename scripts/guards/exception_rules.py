"""Exception-handling rules of the guard.

Broad handlers must log and re-raise, and a silent body is reported. A
specific handler must re-raise, or log and then return a value carrying
the error it caught: binding it with ``as`` and returning something built
from it, directly or through a name assigned from it in the handler. That
is how a failure becomes the caller's answer, as a Gradio handler returns
the message the page shows, built from the error, and a reachability
probe returns why the host could not be reached. Two shapes are the
swallowed error the coding standard bans (board task 3b18c6eb, after its
audits found both passing the guard): a handler that logs and carries on,
falling through or returning nothing, reported as
``except-logs-and-continues``; and one that logs and returns a default
the failure did not produce, as the old ``sweep_once`` returned 0,
reported as ``except-returns-a-default``. A mention of the error in the
log call is not carried to the caller, so it does not count.

Handlers are read from the parsed module, as ``ast.ExceptHandler`` nodes,
not from its lines. The line scanner this replaced on 2026-11-05 missed
the old ``sweep_once`` body in four ordinary shapes its third audit
found: a comment after the header's colon, a header whose types span
lines, and the word ``raise`` in a body comment or in the log message.
In the tree a comment is gone, a string is a string and a header is one
node however it is laid out, so only a ``raise`` statement counts as one.

Each way a handler can end is judged, not the handler as a whole. A
``raise`` somewhere in the body once excused all of it, so a handler that
re-raised only under a condition could log and return 0, or fall
through, on every other path, which is the held-file swallow this rule
exists for behind a filtered re-raise. Every ending of a handler must now
re-raise, or, for a specific handler, return a value built from the
error; the endings are read through ``if``, loops, ``with``, ``try`` and
``match``, and a ``break`` or ``continue`` that leaves the handler is the
handler carrying on.

These checks were defined in
``scripts/guard.py`` until 2026-11-04, when they moved here so that the
runner and each group of rules stay under the 600-line file limit the
file-length rule enforces.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

from scripts.guards import Violation

#: A logger's name: ``logging``, ``log`` or ``logger``, with any leading underscores.
_LOG_RECEIVER = re.compile(r"_*(logging|log|logger)")
_LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})
_BROAD_TYPES = frozenset({"Exception", "BaseException"})
_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _handler_nodes(handler: ast.ExceptHandler) -> Iterator[ast.AST]:
    """Yield every node of a handler's body, without entering nested scopes.

    A ``raise`` or ``return`` inside a function or class defined in the
    handler runs when that is called, not when the handler does, so it
    says nothing about what the handler does with its error.

    Args:
        handler: The handler to walk.

    Yields:
        Each node of the body, the nested definitions themselves included
        but not their contents.
    """
    pending: list[ast.AST] = list(handler.body)
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, _NESTED_SCOPES):
            pending.extend(ast.iter_child_nodes(node))


def _names(node: ast.AST) -> set[str]:
    """Collect the identifiers an expression or target uses.

    Args:
        node: Any node.

    Returns:
        The id of every ``ast.Name`` inside it.
    """
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _is_broad(caught: ast.expr | None) -> bool:
    """Report whether a handler catches broadly.

    Args:
        caught: The handler's type expression, ``None`` for a bare
            ``except``.

    Returns:
        True for a bare ``except`` and for a type, or a tuple holding
        one, whose last dotted part is ``Exception`` or ``BaseException``.
    """
    if caught is None:
        return True
    elements = caught.elts if isinstance(caught, ast.Tuple) else [caught]
    return any(ast.unparse(element).rsplit(".", 1)[-1] in _BROAD_TYPES for element in elements)


def _is_log_call(node: ast.AST) -> bool:
    """Report whether a node calls a logging method on a logger name.

    Args:
        node: Any node of a handler body.

    Returns:
        True for a call such as ``logger.warning(...)`` or
        ``_log.exception(...)``.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _LOG_METHODS
        and isinstance(node.func.value, ast.Name)
        and _LOG_RECEIVER.fullmatch(node.func.value.id) is not None
    )


def _is_trivial(statement: ast.stmt) -> bool:
    """Report whether a statement is empty.

    Args:
        statement: A handler's first statement.

    Returns:
        True for ``pass`` and for a bare ``...``, either of which
        discards the exception.
    """
    return isinstance(statement, ast.Pass) or (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    )


def _endings(
    statements: list[ast.stmt],
) -> list[ast.Raise | ast.Return | ast.Break | ast.Continue | None]:
    """List the ways a run of statements can end.

    Args:
        statements: A block, run in order.

    Returns:
        Each ``raise``, ``return``, ``break`` or ``continue`` that can end
        the block, and ``None`` when control can also run off its end.
        Statements after one that cannot fall through are unreachable and
        are not read.
    """
    endings: list[ast.Raise | ast.Return | ast.Break | ast.Continue | None] = []
    for statement in statements:
        reached = _statement_endings(statement)
        endings.extend(ending for ending in reached if ending is not None)
        if None not in reached:
            return endings
    endings.append(None)
    return endings


def _statement_endings(
    statement: ast.stmt,
) -> list[ast.Raise | ast.Return | ast.Break | ast.Continue | None]:
    """List the ways one statement can end, ``None`` for falling through.

    A loop may run its body no times, so it can always fall through, and
    a ``break`` or ``continue`` inside it stays inside it. A ``try`` is
    read conservatively: an exception its body raises may be caught by its
    own handlers, but counting the raise anyway adds only raising
    endings, which never hide a swallowing one. A ``finally`` that cannot
    fall through decides how the statement ends. A ``match`` may match no
    case. Anything else, a nested definition included, falls through.

    Args:
        statement: The statement.

    Returns:
        Its endings, as :func:`_endings` gives them.
    """
    if isinstance(statement, ast.Raise | ast.Return | ast.Break | ast.Continue):
        return [statement]
    if isinstance(statement, ast.If):
        return _endings(statement.body) + _endings(statement.orelse)
    if isinstance(statement, ast.For | ast.AsyncFor | ast.While):
        inside = [
            None if isinstance(ending, ast.Break | ast.Continue) else ending
            for ending in _endings(statement.body)
        ]
        return inside + _endings(statement.orelse)
    if isinstance(statement, ast.With | ast.AsyncWith):
        return _endings(statement.body)
    if isinstance(statement, ast.Try | ast.TryStar):
        paths = _endings(statement.body + statement.orelse)
        for inner in statement.handlers:
            paths += _endings(inner.body)
        final = _endings(statement.finalbody)
        if None not in final:
            return final
        return paths + [ending for ending in final if ending is not None]
    if isinstance(statement, ast.Match):
        return [ending for case in statement.cases for ending in _endings(case.body)] + [None]
    return [None]


def _is_none(value: ast.expr) -> bool:
    """Report whether a returned expression is the literal ``None``.

    Args:
        value: The returned expression.

    Returns:
        True for ``return None``, which, like a bare ``return``, gives
        the caller nothing.
    """
    return isinstance(value, ast.Constant) and value.value is None


def _carriers(handler: ast.ExceptHandler, nodes: list[ast.AST]) -> set[str]:
    """Name every variable holding the caught error or a value built from it.

    The ``as`` name carries it, and so does each name assigned, in source
    order, from an expression using a carrier. A handler without ``as``
    has no carrier, since nothing in it can refer to its error.

    Args:
        handler: The handler.
        nodes: Its body nodes.

    Returns:
        The carrying names.
    """
    carriers = set() if handler.name is None else {handler.name}
    assignments: list[tuple[ast.expr, list[ast.expr]]] = []
    for node in nodes:
        if isinstance(node, ast.Assign):
            assignments.append((node.value, node.targets))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assignments.append((node.value, [node.target]))
    for value, targets in sorted(
        assignments, key=lambda pair: (pair[0].lineno, pair[0].col_offset)
    ):
        if _names(value) & carriers:
            carriers |= {name for target in targets for name in _names(target)}
    return carriers


def _unraised_kind(
    handler: ast.ExceptHandler, nodes: list[ast.AST], unraised: list[ast.Return | None]
) -> list[str]:
    """Judge the endings of a specific handler that do not re-raise.

    Args:
        handler: The handler.
        nodes: Its body nodes.
        unraised: Its endings other than a raise: each ``return``, and
            ``None`` for carrying on past the handler.

    Returns:
        ``except-logs-and-continues`` when any of them carries on or
        returns nothing, ``except-returns-a-default`` when any value one
        returns is not built from the error, and nothing when every one
        returns a value that is.
    """
    values = [
        ending.value
        for ending in unraised
        if ending is not None and ending.value is not None and not _is_none(ending.value)
    ]
    if len(values) < len(unraised):
        return ["except-logs-and-continues"]
    carriers = _carriers(handler, nodes)
    if any(not _names(value) & carriers for value in values):
        return ["except-returns-a-default"]
    return []


def _handler_kinds(handler: ast.ExceptHandler) -> list[str]:
    """Report how one handler hides the failure it caught, if it does.

    Args:
        handler: The handler.

    Returns:
        The violation kinds it earns, in the order they are reported.
    """
    nodes = list(_handler_nodes(handler))
    kinds = ["silent-except-body"] if _is_trivial(handler.body[0]) else []
    has_log = any(_is_log_call(node) for node in nodes)
    endings = _endings(handler.body)
    unraised = [
        None if isinstance(ending, ast.Break | ast.Continue) else ending
        for ending in endings
        if not isinstance(ending, ast.Raise)
    ]
    if _is_broad(handler.type):
        if not has_log or unraised:
            kinds.append("broad-except-requires-log-and-raise")
    elif not has_log and len(unraised) == len(endings):
        kinds.append("except-without-log-or-raise")
    elif unraised:
        kinds.extend(_unraised_kind(handler, nodes, unraised))
    return kinds


def _run_exceptions_rule(path: Path, tree: ast.Module, lines: list[str]) -> list[Violation]:
    """Report the exception handlers that hide the failure they caught.

    A broad handler must log and re-raise. A specific one must re-raise,
    or log and return a value built from the caught error; one that does
    neither is reported, and so are one that logs and then continues and
    one that logs and returns a default. Test files are skipped, because
    a test catching an exception to assert on it is doing exactly what it
    should.

    Args:
        path: File being scanned, for the violation record.
        tree: The module, parsed.
        lines: Its source lines, for the line each violation quotes.

    Returns:
        One violation per offence, in source order.
    """
    if "tests" in path.parts:
        return []
    handlers = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)),
        key=lambda handler: (handler.lineno, handler.col_offset),
    )
    return [
        Violation(file=path, line_no=handler.lineno, kind=kind, line=lines[handler.lineno - 1])
        for handler in handlers
        for kind in _handler_kinds(handler)
    ]
