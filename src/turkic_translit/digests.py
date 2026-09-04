"""One definition of "the SHA-256 of this file", and one of what one looks like.

Lifted here on 2026-09-03 from ``cli.train_spm``, which owned the only
implementation and could not be imported by anything below the interface
layer without inverting the dependency. Two callers now need it: that CLI,
which pins the sentencepiece model it trained, and the corpus manifest,
which pins the corpus it downloaded.

The validator is here rather than in :mod:`turkic_translit.validation`
because it encodes a fact about digests rather than a general shape rule:
a SHA-256 is exactly 64 lowercase hexadecimal characters, and a manifest
carrying anything else is not carrying a digest.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from turkic_translit.validation import (
    ERR_FIELD_TYPE,
    FieldError,
    require_non_empty_str,
)

_HASH_CHUNK_BYTES: Final[int] = 1 << 20

SHA256_HEX: Final = re.compile(r"^[0-9a-f]{64}$")
"""A SHA-256 as ``hexdigest`` writes one: 64 characters, lowercase."""


def sha256_of_file(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in chunks.

    Args:
        path: File to digest.

    Returns:
        The digest as lowercase hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256_hex(
    field: str,
    value: str | int | float | bool | None | Mapping[str, str | int | float | bool],
) -> str:
    """Return ``value`` as a SHA-256 hex digest, or raise.

    Args:
        field: Field name, used in the error message.
        value: Candidate value taken from a loosely-typed mapping.

    Returns:
        The validated digest.

    Raises:
        FieldError: If the value is not a string, is empty, or is not 64
            lowercase hexadecimal characters. Accepting a shorter string
            would let a truncated digest through, and a truncated digest
            compares equal to nothing and so silently never matches.
    """
    text = require_non_empty_str(field, value)
    if SHA256_HEX.match(text) is None:
        raise FieldError(
            ERR_FIELD_TYPE,
            field,
            "must be 64 lowercase hexadecimal characters",
        )
    return text


__all__ = ["SHA256_HEX", "require_sha256_hex", "sha256_of_file"]
