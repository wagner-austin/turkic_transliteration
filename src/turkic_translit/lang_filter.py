"""Deciding whether a single token is Russian.

Independent of any CLI or UI layer, and independent of any array
library. The classifier this takes is a
:class:`~turkic_translit.lid.classifier.LidClassifier`, so labels arrive
already stripped of their model prefix and confidences arrive as plain
floats. The previous version accepted whatever fastText's Python wrapper
returned and normalised it with ``numpy.atleast_1d``, which is what tied
this module — and through it the whole project — to NumPy 1.
"""

from __future__ import annotations

import re

from turkic_translit.lid.classifier import LidClassifier

__all__ = ["KZ_EXTRA", "PREDICTIONS_PER_TOKEN", "RU_ONLY", "is_russian_token"]

RU_ONLY: re.Pattern[str] = re.compile(r"^[А-ЯЁа-яё]+$")
KZ_EXTRA: set[str] = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")

PREDICTIONS_PER_TOKEN: int = 3
_RUSSIAN = "ru"


def is_russian_token(
    token: str,
    *,
    thr: float,
    min_len: int,
    lid: LidClassifier,
    stoplist: set[str] | None = None,
    margin: float = 0.10,
) -> bool:
    """Report whether ``token`` should be treated as Russian.

    Args:
        token: The token to judge.
        thr: Minimum confidence required when Russian is the top label.
        min_len: Tokens shorter than this are never Russian.
        lid: Loaded classifier, consulted at most once.
        stoplist: Lowercased tokens to exempt, or ``None``.
        margin: How far behind the winner Russian may be and still count,
            so 0.10 accepts Russian within ten points of the top label.

    Returns:
        True when the token should be treated as Russian. When ``thr`` is
        zero the pure-Cyrillic orthography test also counts, which is the
        documented meaning of the lowest threshold rather than a fallback
        for a failed classification.
    """
    if len(token) < min_len:
        return False

    lowered = token.lower()
    if stoplist is not None and lowered in stoplist:
        return False

    if any(character in KZ_EXTRA for character in lowered):
        return False

    predictions = lid.classify_many(lowered, PREDICTIONS_PER_TOKEN)
    labels = [prediction["label"] for prediction in predictions]
    confidences = [prediction["probability"] for prediction in predictions]

    if labels[0] == _RUSSIAN and confidences[0] >= thr:
        return True

    if _RUSSIAN in labels[1:]:
        index = labels.index(_RUSSIAN)
        if confidences[index] >= thr and confidences[index] >= confidences[0] - margin:
            return True

    return thr == 0.0 and RU_ONLY.fullmatch(lowered) is not None
