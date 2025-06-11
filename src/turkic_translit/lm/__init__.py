"""Public re-exports so callers can do:

from turkic_translit.lm import DatasetStream, LMModel, cross_perplexity, centred_cosine_matrix
"""

from .data import DatasetStream  # noqa: F401
from .eval import cross_perplexity  # noqa: F401
from .similarity import centred_cosine_matrix  # noqa: F401
from .train import LMModel  # noqa: F401

# Explicit re-export list for static type checkers (mypy attr-defined)
__all__ = [
    "DatasetStream",
    "LMModel",
    "cross_perplexity",
    "centred_cosine_matrix",
]
