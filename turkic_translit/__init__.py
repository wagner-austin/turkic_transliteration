from importlib.metadata import version
# Set up logging first before any other operations
from .logging_config import setup as _log_setup; _log_setup()
# Import patches next to ensure they're applied before other imports
from . import patches
from .core import to_latin, to_ipa

__all__ = ["to_latin", "to_ipa"]
__version__ = version("turkic_transliterate")
