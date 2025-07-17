"""Integration test for FastTextLangID

Ensures the real FastText model can be loaded via `ensure_fasttext_model()` and
that a few simple sentences are labelled with the expected language codes.

We deliberately keep the sample short to avoid large downloads / long runtime.
"""

# NumPy 2.x still breaks the C++ fastText code – guard on major only
import numpy as np
import pytest
from packaging.version import parse as vparse

if vparse(np.__version__).major >= 2:
    pytest.skip("fastText requires NumPy<2", allow_module_level=True)

# Absent fasttext on non-Windows is fine – mark xfail
from turkic_translit.langid import FastTextLangID
from turkic_translit.model_utils import ensure_fasttext_model

fasttext = pytest.importorskip("fasttext")  # runtime dependency

# Load (or auto-download) model once per session
model_path = ensure_fasttext_model()
ft = FastTextLangID(str(model_path))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Merhaba dünya", "tr"),
        ("Здравствуйте", "ru"),
        ("Hello world", "en"),
    ],
)
def test_fasttext_predict(text: str, expected: str) -> None:
    lang, prob = ft.predict_with_prob(text)
    assert lang == expected
    # Lower threshold for "Hello world" as it's a short phrase
    min_prob = 0.15 if text == "Hello world" else 0.3
    assert prob > min_prob
