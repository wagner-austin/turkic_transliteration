"""
Core heuristics for deciding whether a single token is Russian.
Independent of any CLI / UI layer.  100 % unit-testable.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

import numpy as np  # For typing

__all__ = ["RU_ONLY", "KZ_EXTRA", "is_russian_token"]

# --- public, reusable constants -------------------------------------------------
RU_ONLY: re.Pattern[str] = re.compile(r"^[А-ЯЁа-яё]+$")  # pure Cyrillic, no Latin
KZ_EXTRA: set[str] = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")  # absent from Russian


# --- type protocol for fastText model -------------------------------------------
class FastTextLike(Protocol):
    def predict(
        self, text: str, k: int
    ) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]: ...


# --- the one public function ----------------------------------------------------
def is_russian_token(
    token: str,
    *,
    thr: float,
    min_len: int,
    lid: FastTextLike,  # fastText model, already loaded
    stoplist: set[str] | None = None,
    margin: float = 0.10,
) -> bool:
    """
    Return True iff *token* should be treated as Russian, under `thr`/`margin`.

    • `thr` – minimum confidence required when RU is best label.
    • `margin` – max distance RU may be behind the winner (0.10 ⇒ within 10 %).
    • Orthography fallback is applied only when `thr == 0.0`.
    """
    if len(token) < min_len:
        return False

    t = token.lower()
    if stoplist and t in stoplist:
        return False

    if any(ch in KZ_EXTRA for ch in t):
        return False  # Kazakh-specific letter → not RU

    # ── fastText inference (single call) ────────────────────────────────────────
    # FastText can return NumPy arrays or lists, depending on build.
    labels, confs = lid.predict(t, k=3)

    # Normalise to simple Python lists so the rest of the logic is type-safe
    labels = list(labels)
    if not isinstance(confs, list):
        confs = np.atleast_1d(confs).tolist()

    # If for some reason we got no scores, bail out early
    if not labels or not confs:
        return False

    if labels[0] == "__label__ru" and confs[0] >= thr:
        return True  # RU is winner

    if "__label__ru" in labels[1:]:
        idx = labels.index("__label__ru")
        # Extra bounds-check: some very small models may return fewer confs than labels
        if idx < len(confs) and confs[idx] >= thr and confs[idx] >= confs[0] - margin:
            return True  # RU close second/third

    # orthography fallback only when slider is at the very bottom
    return thr == 0.0 and RU_ONLY.fullmatch(t) is not None
