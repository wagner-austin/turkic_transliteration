import platform
import sys
from importlib.metadata import version

# Apply patches first, before any other imports
from . import patches as _patches  # noqa: F401

# Public sub-packages (optional heavy deps)
try:
    from . import lm as lm  # noqa: F401
except Exception:  # pragma: no cover – missing optional deps
    lm = None  # type: ignore
from .core import to_ipa, to_latin
from .web.web_utils import (
    direct_transliterate,
    mask_russian,
    median_levenshtein,
    pipeline_transliterate,
    token_table_markdown,
)

# Logging is configured by entrypoints (CLI/web). Package import does not
# alter global logging configuration.

__all__ = [
    "to_latin",
    "to_ipa",
    "direct_transliterate",
    "pipeline_transliterate",
    "token_table_markdown",
    "mask_russian",
    "median_levenshtein",
    "lang_filter",
    *(["lm"] if lm is not None else []),
]
# Retrieve version from package metadata. Package is published as "turkic-translit" on PyPI.
__version__ = version("turkic-translit")

# Warn Windows users on Python >=3.12 if PyICU is not importable
if platform.system() == "Windows" and sys.version_info >= (3, 12):
    try:
        import PyICU  # noqa: F401
    except ImportError:
        import logging as _logging

        _logging.getLogger("turkic_translit").error(
            "PyICU is not installed and cannot be built automatically on Windows for Python 3.12+. "
            "Use Python 3.11 and install via 'turkic-pyicu-install'. See README for details."
        )
