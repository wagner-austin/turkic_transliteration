"""Root *turkic_translit.cli* **package**.

This package fully replaces the former single-file module.  We expose a
click.Group called **main** so the original entry-point string in
*pyproject.toml* (``"turkic-translit" = "turkic_translit.cli:main"``) keeps
working without modification. Additionally, the individual sub-command functions (e.g. ``train_spm``) are re-exported for programmatic use.
"""

from __future__ import annotations

import os

import click

from ..error_service import init_error_service, set_correlation_id
from ..logging_config import setup as _log_setup

# --------------------------------------------------------------------------- #
# Import sub-command entry-points **before** any executable code to satisfy
# Ruff E402 (imports must precede definitions). We only *register* them after
# ``main`` is defined below.                                                  #
# --------------------------------------------------------------------------- #
# Lightweight commands – always available
from .build_spm import main as _spm  # noqa: E402
from .download_corpus import cli as _dl  # noqa: E402
from .filter_russian import main as _ru  # noqa: E402
from .train_spm import main as _train_spm  # noqa: E402

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
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
    show_default=True,
    help="Set logging level for all commands",
)
def main(log_level: str) -> None:  # noqa: D401 – CLI root
    """Turkic-Transliterate command-line tools."""
    os.environ["TURKIC_LOG_LEVEL"] = log_level.upper()
    _log_setup()
    init_error_service()
    set_correlation_id(os.getenv("TURKIC_CORRELATION_ID"))


# --------------------------------------------------------------------------- #
# Register the commands now that *main* exists                                #
# --------------------------------------------------------------------------- #

main.add_command(_spm, "build-spm")
main.add_command(_dl, "download-corpus")
main.add_command(_ru, "filter-russian")
main.add_command(_train_spm, "train-spm")

if _train is not None and _eval is not None:  # optional heavy deps present
    main.add_command(_train, "train-lm")
    main.add_command(_eval, "eval-lm")


# Re-export for "import turkic_translit.cli as cli; cli.train_spm(...)"
train_spm = _train_spm

__all__: list[str] = ["main", "train_spm"]
