"""Where OSCAR's shards live on the Hub, and what their names mean.

Everything here is a pure function over a repository's file listing. The
reads themselves belong to ``_test_hooks``, which is where this package
keeps its boundaries; the naming rules belong here, because they are
decisions rather than I/O, and a decision reachable only through a
network call is a decision nothing exercises.

OSCAR 23.01 publishes one directory per language, ``<code>_meta/``,
holding Zstandard-compressed JSON lines — a single
``<code>_meta.jsonl.zst`` for a small language, and
``<code>_meta_part_<n>.jsonl.zst`` shards for a large one. Each line is
a document object whose ``content`` field carries the text; the
remaining fields are the WARC headers and OSCAR's own annotations.

Reading these files directly is what replaced ``datasets``. That package
learned the same layout by downloading and executing the dataset's
loading script, which meant the corpus tab ran third-party code to
answer "which languages are there", took its dependencies along with it
— OSCAR 23.01 moved to Zstandard, so the script imports ``zstandard``,
and the demo crashed at startup when nothing here declared it — and
could not work at all under ``datasets`` 4, which removed loading
scripts entirely.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

DATASET_API: Final = "https://huggingface.co/api/datasets"

RESOLVE_TEMPLATE: Final = "https://huggingface.co/datasets/{name}/resolve/main/{path}"

SHARD_SUFFIX: Final = ".jsonl.zst"

CONTENT_FIELD: Final = "content"

_LANGUAGE_DIRECTORY: Final = re.compile(r"^(?P<code>[^/]+)_meta/")

_PART_NUMBER: Final = re.compile(r"_part_(?P<number>\d+)\.jsonl\.zst$")


def dataset_api_url(dataset_name: str) -> str:
    """Name the endpoint describing a dataset repository.

    Args:
        dataset_name: Dataset identifier, e.g.
            ``oscar-corpus/OSCAR-2301``.

    Returns:
        The URL whose response lists the repository's files. It answers
        without a token even for a gated dataset, which is why the
        language list costs no credential while the shards do.
    """
    return f"{DATASET_API}/{dataset_name}"


def languages(files: Iterable[str]) -> tuple[str, ...]:
    """List the language codes a repository's files provide.

    Args:
        files: Every path in the repository.

    Returns:
        One code per ``<code>_meta/`` directory holding shards, sorted.
        Directories are read from the shard paths themselves, so a
        language whose data has not been uploaded is not offered.
    """
    found = {
        match.group("code")
        for name in files
        if name.endswith(SHARD_SUFFIX) and (match := _LANGUAGE_DIRECTORY.match(name))
    }
    return tuple(sorted(found))


def shard_paths(files: Iterable[str], language: str) -> tuple[str, ...]:
    """List one language's shards, in the order they should be read.

    Args:
        files: Every path in the repository.
        language: Language code to select.

    Returns:
        The shard paths, ordered by part number. The numbers run past
        nine, so they are compared as numbers: sorted as text,
        ``part_10`` precedes ``part_2`` and the corpus arrives out of
        order. A language published as one unpartitioned file sorts
        ahead of any parts, which is where it belongs.
    """
    prefix = f"{language}_meta/"
    selected = [name for name in files if name.startswith(prefix) and name.endswith(SHARD_SUFFIX)]
    return tuple(sorted(selected, key=_part_number))


def _part_number(path: str) -> int:
    """Read a shard's part number.

    Args:
        path: Shard path within the repository.

    Returns:
        The number in ``_part_<n>``, or 0 for a language published as a
        single file, which carries no part number at all.
    """
    match = _PART_NUMBER.search(path)
    return int(match.group("number")) if match is not None else 0


__all__ = [
    "CONTENT_FIELD",
    "DATASET_API",
    "RESOLVE_TEMPLATE",
    "SHARD_SUFFIX",
    "dataset_api_url",
    "languages",
    "shard_paths",
]
