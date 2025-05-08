from importlib.metadata import version
# Import patches first to ensure they're applied before any other imports
from . import patches
from .core import to_latin, to_ipa

__all__ = ["to_latin", "to_ipa"]
__version__ = version("turkic_transliterate")
