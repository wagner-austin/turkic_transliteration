"""Public test utilities for checking rule output against published sources.

Every source-fidelity test in this project compares rule output with a
transcription printed in a cited description. The two sides never differ
freely: each test declares its differences in two tables, NOTATION for
glyph choices that leave the phoneme identical, and
DECLARED_SIMPLIFICATIONS for real distinctions the rules deliberately
drop. :func:`as_project_notation` applies those tables to the published
side. Anything the tables do not cover must match exactly, so an
undeclared divergence fails the comparing test.
"""

from __future__ import annotations

Deviation = tuple[str, str, str]


def as_project_notation(published: str, *tables: tuple[Deviation, ...]) -> str:
    """Rewrite a published transcription in this project's notation.

    Entries are applied in table order, then entry order, by plain
    substring replacement. Callers order entries so that no replacement
    output feeds a later entry's input.

    Args:
        published: The transcription exactly as the source prints it.
        *tables: Deviation tables of ``(source_form, ours, reason)``.

    Returns:
        The transcription in the notation the rules emit.

    Raises:
        ValueError: If an entry has an empty reason; a deviation without
            a stated justification is not declared, only smuggled.
    """
    for table in tables:
        for source_form, ours, reason in table:
            if not reason:
                msg = f"Deviation {source_form!r} -> {ours!r} has no reason"
                raise ValueError(msg)
            published = published.replace(source_form, ours)
    return published


def unexercised_entries(
    data: tuple[str, ...],
    *tables: tuple[Deviation, ...],
) -> tuple[Deviation, ...]:
    """Deviation entries whose source form appears in none of the data.

    A deviation that no printed datum exercises is untested and therefore
    unjustified; the standard test asserts this returns an empty tuple.

    Args:
        data: Every published transcription the test file compares.
        *tables: The file's deviation tables.

    Returns:
        The entries never applied to any datum.
    """
    return tuple(
        entry
        for table in tables
        for entry in table
        if not any(entry[0] in datum for datum in data)
    )
