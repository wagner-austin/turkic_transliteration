from importlib.metadata import version
from .core import to_latin, to_ipa

__all__ = ["to_latin", "to_ipa"]
__version__ = version(__name__)
