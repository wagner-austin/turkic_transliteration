"""Root *turkic_translit.cli* **package**.

This package fully replaces the former single-file module.  We expose a
click.Group called **main** so the original entry-point string in
*pyproject.toml* (``"turkic-translit" = "turkic_translit.cli:main"``) keeps
working without modification.
"""

from __future__ import annotations

import click

# --------------------------------------------------------------------------- #
# Import sub-command entry-points **before** any executable code to satisfy
# Ruff E402 (imports must precede definitions). We only *register* them after
# ``main`` is defined below.                                                  #
# --------------------------------------------------------------------------- #
# Lightweight commands – always available
from .build_spm import main as _spm  # noqa: E402
from .download_corpus import cli as _dl  # noqa: E402
from .filter_russian import main as _ru  # noqa: E402

# Heavyweight commands – may fail if transformers/evaluate unavailable
try:  # noqa: E402
    from .eval_lm import cli as _eval  # noqa: F401
    from .train_lm import cli as _train  # noqa: F401
except Exception:  # pragma: no cover – optional deps missing
    _train = _eval = None  # type: ignore


# --------------------------------------------------------------------------- #
# Top-level Click group                                                       #
# --------------------------------------------------------------------------- #


@click.group()
def main() -> None:  # noqa: D401 – CLI root
    """Turkic-Transliterate command-line tools."""


# --------------------------------------------------------------------------- #
# Register the commands now that *main* exists                                #
# --------------------------------------------------------------------------- #

main.add_command(_spm, "build-spm")
main.add_command(_dl, "download-corpus")
main.add_command(_ru, "filter-russian")

if _train is not None and _eval is not None:  # optional heavy deps present
    main.add_command(_train, "train-lm")
    main.add_command(_eval, "eval-lm")


__all__: list[str] = ["main"]
