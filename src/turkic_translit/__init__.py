"""Turkic transliteration package.

Sub-packages have heavyweight or optional dependencies (torch,
gradio, transformers) that this ``__init__`` deliberately does not
import at package-load time. Consumers reach into the sub-packages
explicitly:

* :mod:`turkic_translit.core` — the deterministic rule engine (PyICU-backed).
* :mod:`turkic_translit.web` — Gradio UI helpers.
* :mod:`turkic_translit.lm` — Hugging Face LM utilities.
* :mod:`turkic_translit.cli` — Click subcommand group (``turkic-translit``).
* :mod:`turkic_translit.corpus` — corpus streaming, cleaning and inventory.

Keeping ``import turkic_translit`` cheap is what lets a caller import
the package without paying for torch, and what lets the rule engine
report a missing ICU at the first call that needs one rather than at
an import nobody wrote.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("turkic-translit")

__all__ = ["__version__"]
