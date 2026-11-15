"""Tell the ``mirror-lid-models`` workflow what to copy and where.

The language-identification weights are downloaded from this project's
Hugging Face model repo, which holds Meta's files byte for byte. The
workflow that fills it reads every fact it needs from the registry
through this script: the repo, and for each model its filename, the
SHA-256 Meta's file has and the URL Meta released it at. Nothing about
the models is written in the workflow itself, so the repo the package
downloads from and the repo the workflow fills cannot be two different
names, and the digest the workflow checks before uploading is the one
every download is checked against.

Run with:

    python -m scripts.lid_mirror --print-repo
    python -m scripts.lid_mirror --print-manifest
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from turkic_translit.lid.registry import MIRROR_REPO, REGISTRY

USAGE = "usage: python -m scripts.lid_mirror (--print-repo | --print-manifest)"


def manifest_lines() -> list[str]:
    """List what the workflow copies, one model per line.

    Returns:
        ``<filename> <sha256> <origin_url>`` for every registered model,
        in registry order. No field can hold a space: a filename and a
        URL from the registry never do, and a digest is hexadecimal.
    """
    return [
        f"{spec['filename']} {spec['sha256']} {spec['origin_url']}" for spec in REGISTRY.values()
    ]


def main(argv: Sequence[str]) -> int:
    """Print the mirror's repo or its manifest.

    Args:
        argv: Arguments after the program name.

    Returns:
        0 when one of the two was printed, and 2 when the command line
        named neither.
    """
    if "--print-repo" in argv:
        print(MIRROR_REPO)
        return 0

    if "--print-manifest" in argv:
        for line in manifest_lines():
            print(line)
        return 0

    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
