"""Integration test for FastText language identification via FastTextLangID.

This test loads the *real* FastText model (lid.176.bin) using the project
utility ``ensure_fasttext_model``.  It then checks a handful of medium-length
sentences in different languages to make sure the top prediction matches the
expected ISO code and the confidence is reasonably high (>0.5).

Rationale
---------
A corrupted/partial model often still loads but returns the fallback result
``en / 0.25`` for nearly everything – the exact failure that recently slipped
through.  This test fails loudly in that scenario, giving us an early warning
that the on-disk model is broken.
"""

from __future__ import annotations

import pytest

from turkic_translit.langid import FastTextLangID
from turkic_translit.model_utils import ensure_fasttext_model

fasttext = pytest.importorskip("fasttext")  # runtime dependency


@pytest.fixture(scope="session")
def ft_model() -> FastTextLangID:
    """Session-scoped fixture that returns a loaded FastTextLangID model.

    If the binary model cannot be loaded (e.g. download blocked), the whole
    test module is skipped so the rest of the suite can still run.
    """
    try:
        # Ensure the model file exists/downloaded; raises on failure
        ensure_fasttext_model()
        return FastTextLangID()
    except Exception as exc:  # pragma: no cover – environment without network
        pytest.skip(f"FastText model unavailable: {exc}")


SAMPLES: dict[str, str] = {
    "tr": "Türkiye'nin başkenti Ankara'dır ve ülkenin siyasi merkezidir.",
    "en": "FastText is an efficient library for text classification and word representation.",
    "ru": "Москва является столицей России и крупнейшим городом страны.",
}


@pytest.mark.parametrize(("iso", "text"), list(SAMPLES.items()))
def test_fasttext_langid(ft_model: FastTextLangID, iso: str, text: str) -> None:
    """FastText must identify each *text* as language *iso* with prob > 0.5."""
    pred, prob = ft_model.predict_with_prob(text)
    assert pred == iso, f"{text!r} → predicted {pred!r} (expected {iso!r})"
    assert prob > 0.5, f"Low confidence {prob:.3f} for {iso!r} sentence"
