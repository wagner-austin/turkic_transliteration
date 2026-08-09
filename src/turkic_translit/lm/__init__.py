"""Public re-exports so callers can do:

from turkic_translit.lm import DatasetStream, LMModel, cross_perplexity
"""

from .data import DatasetStream
from .eval import cross_perplexity
from .similarity import linear_cka, rbf_cka, representation_similarity
from .train import LMModel

# Explicit re-export list for static type checkers (mypy attr-defined)
__all__ = [
    "DatasetStream",
    "LMModel",
    "cross_perplexity",
    "linear_cka",
    "rbf_cka",
    "representation_similarity",
]
