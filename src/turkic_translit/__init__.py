import platform
import sys
from importlib.metadata import version

from . import patches as _patches  # noqa: F401
from .core import to_ipa, to_latin
from .logging_config import setup as _log_setup
from .web.web_utils import (
    direct_transliterate,
    mask_russian,
    median_levenshtein,
    pipeline_transliterate,
    token_table_markdown,
)

_log_setup()

__all__ = [
    "to_latin",
    "to_ipa",
    "direct_transliterate",
    "pipeline_transliterate",
    "token_table_markdown",
    "mask_russian",
    "median_levenshtein",
]
__version__ = version("turkic_transliterate")

# Warn Windows users on Python >=3.12 if PyICU is not importable
if platform.system() == "Windows" and sys.version_info >= (3, 12):
    try:
        import PyICU  # noqa: F401
    except ImportError:
        import sys as _sys

        _sys.stderr.write(
            "[turkic-transliterate] ERROR: PyICU is not installed and cannot be built automatically on Windows for Python 3.12+!\n"
            "To use this package, please create a virtual environment with Python 3.11 and install as follows:\n\n"
            "    py -3.11 -m venv turkic311\n"
            "    turkic311\\Scripts\\activate\n"
            "    pip install turkic-transliterate\n"
            "    turkic-pyicu-install\n\n"
            "See the README for more details.\n"
        )
