"""Public re-exports so callers can do:

from turkic_translit.lm import DatasetStream, LMModel, cross_perplexity, centred_cosine_matrix
"""

from .data import DatasetStream
from .eval import cross_perplexity
from .similarity import centred_cosine_matrix
from .train import LMModel

# Explicit re-export list for static type checkers (mypy attr-defined)
__all__ = [
    "DatasetStream",
    "LMModel",
    "centred_cosine_matrix",
    "cross_perplexity",
]
