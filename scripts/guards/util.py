"""Utility functions for guard rules."""

from __future__ import annotations

from pathlib import Path


def read_lines(path: Path) -> list[str]:
    """Read a source file as a list of lines.

    Decoded as utf-8-sig so a byte-order mark, which Windows editors add,
    is consumed rather than becoming part of the first line. A read
    failure propagates as the ``OSError`` it is: the path is already in
    that exception's message, so wrapping it would only hide its type.

    Args:
        path: File to read.

    Returns:
        The file's lines, without their terminators.

    Raises:
        OSError: If the file cannot be read.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    return path.read_text(encoding="utf-8-sig", errors="strict").splitlines()


__all__ = ["read_lines"]
