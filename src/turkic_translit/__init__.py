from importlib.metadata import version
import sys
import platform

# Set up logging first before any other operations
from .logging_config import setup as _log_setup; _log_setup()
# Import patches next to ensure they're applied before other imports
from . import patches

# Warn Windows users on Python >=3.12 if PyICU is not importable
if platform.system() == "Windows" and sys.version_info >= (3,12):
    try:
        import PyICU
    except ImportError:
        import sys as _sys
        _sys.stderr.write(
            "[turkic-transliterate] ERROR: PyICU is not installed and cannot be built automatically on Windows for Python 3.12+!\n"
            "To use this package, please create a virtual environment with Python 3.11 and install as follows:\n\n"
            "    py -3.11 -m venv turkic311\n"
            "    turkic311\\Scripts\\activate\n"
            "    pip install turkic-transliterate\n"
            "    python scripts/get_pyicu_wheel.py\n\n"
            "See the README for more details.\n"
        )

from .core import to_latin, to_ipa

__all__ = ["to_latin", "to_ipa"]
__version__ = version("turkic_transliterate")
