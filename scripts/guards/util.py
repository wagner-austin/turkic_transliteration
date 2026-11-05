"""Utility functions for guard rules."""

from __future__ import annotations

from pathlib import Path


def read_lines(path: Path) -> list[str]:
    """Read a source file as a list of lines.

    Decoded as plain UTF-8. A file that opens with a byte-order mark is
    refused rather than read: the mark would become a zero-width no-break
    space at the start of the first line, so every rule matching on
    column one would miss that line and the file would pass unexamined.
    A read failure propagates as the ``OSError`` it is: the path is
    already in that exception's message, so wrapping it would only hide
    its type.

    Args:
        path: File to read.

    Returns:
        The file's lines, without their terminators.

    Raises:
        OSError: If the file cannot be read.
        UnicodeDecodeError: If the file is not valid UTF-8.
        ValueError: If the file opens with a byte-order mark.
    """
    text = path.read_text(encoding="utf-8", errors="strict")
    if text.startswith("﻿"):
        raise ValueError(f"{path} opens with a byte-order mark: save it as UTF-8 without one")
    return text.splitlines()


__all__ = ["read_lines"]
