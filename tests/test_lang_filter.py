import json
from typing import Any

import numpy as np
import pytest

from turkic_translit.lang_filter import is_russian_token
from turkic_translit.web.web_utils import mask_russian


class MockFastText:
    """Simple mock for FastText that returns predefined results"""

    def __init__(self) -> None:
        self.responses: dict[str, tuple[list[str], list[float]]] = {}

    def set_response(
        self, text: str, labels: list[str], confidences: list[float]
    ) -> None:
        """Set the response for a specific input text"""
        self.responses[text] = (labels, confidences)

    def predict(
        self, text: str, k: int = 3
    ) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:
        """Predict method that returns our predefined responses"""
        if text in self.responses:
            labels, confs = self.responses[text]
            return labels[:k], np.array(confs[:k], dtype=np.float64)
        return ["__label__en"], np.array([0.25], dtype=np.float64)  # Default response


@pytest.fixture
def fasttext_mock() -> MockFastText:
    """Fixture providing a fresh MockFastText instance for each test"""
    return MockFastText()


def test_short_token_rejection(fasttext_mock: MockFastText) -> None:
    """Test that tokens shorter than min_len are rejected"""
    # Short token should be rejected regardless of language
    assert not is_russian_token(
        "пр",  # 2-char Russian word
        thr=0.5,
        min_len=3,
        lid=fasttext_mock,
    )

    # When min_len is lower and token is Russian, it should be accepted
    fasttext_mock.set_response("пр", ["__label__ru"], [0.9])
    assert is_russian_token(
        "пр",
        thr=0.5,
        min_len=2,
        lid=fasttext_mock,
    )


def test_stoplist_rejection(fasttext_mock: MockFastText) -> None:
    """Test that tokens in stoplist are rejected"""
    stoplist = {"привет", "мир"}

    # Set up Russian responses for both tokens
    fasttext_mock.set_response("привет", ["__label__ru"], [0.9])
    fasttext_mock.set_response("здравствуйте", ["__label__ru"], [0.9])

    # Word in stoplist should be rejected even if detected as Russian
    assert not is_russian_token(
        "привет",
        thr=0.5,
        min_len=3,
        lid=fasttext_mock,
        stoplist=stoplist,
    )

    # Word not in stoplist should pass
    assert is_russian_token(
        "здравствуйте",
        thr=0.5,
        min_len=3,
        lid=fasttext_mock,
        stoplist=stoplist,
    )


def test_kazakh_letter_rejection(fasttext_mock: MockFastText) -> None:
    """Test that tokens with Kazakh-specific letters are rejected"""
    # Even though we'll say it's Russian, the Kazakh letter should trigger rejection
    fasttext_mock.set_response("сәлем", ["__label__ru"], [0.9])

    assert not is_russian_token(
        "сәлем",  # Contains ә which is Kazakh-specific
        thr=0.5,
        min_len=3,
        lid=fasttext_mock,
    )


def test_russian_rank1_acceptance(fasttext_mock: MockFastText) -> None:
    """Test that Russian is accepted when it's the top label"""
    # Russian as top label with high confidence
    fasttext_mock.set_response(
        "привет", ["__label__ru", "__label__uk", "__label__bg"], [0.8, 0.1, 0.05]
    )

    assert is_russian_token(
        "привет",
        thr=0.5,
        min_len=3,
        lid=fasttext_mock,
    )

    # Russian as top label but low confidence
    fasttext_mock.set_response(
        "привет", ["__label__ru", "__label__uk", "__label__bg"], [0.4, 0.3, 0.2]
    )

    assert not is_russian_token(
        "привет",
        thr=0.5,
        min_len=3,
        lid=fasttext_mock,
    )


def test_russian_close_second(fasttext_mock: MockFastText) -> None:
    """Test Russian is accepted when it's a close second place"""
    # Russian as second place but within margin
    fasttext_mock.set_response(
        "привет", ["__label__uk", "__label__ru", "__label__bg"], [0.55, 0.5, 0.2]
    )

    # Should pass with 0.1 margin (0.55 - 0.5 = 0.05 < 0.1)
    assert is_russian_token(
        "привет",
        thr=0.5,
        min_len=3,
        lid=fasttext_mock,
        margin=0.1,
    )

    # Should fail with tight margin
    assert not is_russian_token(
        "привет",
        thr=0.5,
        min_len=3,
        lid=fasttext_mock,
        margin=0.01,
    )

    # Russian with low confidence should fail
    fasttext_mock.set_response(
        "привет", ["__label__uk", "__label__ru", "__label__bg"], [0.6, 0.4, 0.2]
    )

    assert not is_russian_token(
        "привет",
        thr=0.5,
        min_len=3,
        lid=fasttext_mock,
    )


def test_orthography_fallback(fasttext_mock: MockFastText) -> None:
    """Test orthography fallback with thr=0.0"""
    # Set up a case where Russian is not the top prediction
    fasttext_mock.set_response(
        "привет", ["__label__uk", "__label__bg", "__label__ru"], [0.5, 0.3, 0.2]
    )

    # Should pass when pure Cyrillic and thr=0.0
    assert is_russian_token(
        "привет",
        thr=0.0,
        min_len=3,
        lid=fasttext_mock,
    )

    # Should fail when thr>0.0
    assert not is_russian_token(
        "привет",
        thr=0.1,
        min_len=3,
        lid=fasttext_mock,
    )

    # Set up for mixed Cyrillic/Latin
    fasttext_mock.set_response(
        "приветworld", ["__label__uk", "__label__bg", "__label__ru"], [0.5, 0.3, 0.2]
    )

    # Should fail with mixed script even with thr=0.0
    assert not is_russian_token(
        "приветworld",
        thr=0.0,
        min_len=3,
        lid=fasttext_mock,
    )


def test_slider_equivalence(fasttext_mock: MockFastText) -> None:
    """Test slider sensitivity by checking threshold effect"""
    # Set up a token with exactly 0.4 confidence
    fasttext_mock.set_response("тест", ["__label__ru"], [0.4])

    # Test at various thresholds from 0.0 to 1.0
    results = [
        is_russian_token(
            "тест",
            thr=thr,
            min_len=3,
            lid=fasttext_mock,
        )
        for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ]

    # Results should start True and switch to False exactly once
    changes = sum(1 for i in range(1, len(results)) if results[i] != results[i - 1])
    assert changes == 1, (
        f"Expected exactly one change in results but got {changes}: {results}"
    )

    # The change should happen after threshold 0.4
    assert results[4]  # At threshold 0.4
    assert not results[5]  # At threshold 0.5


def test_web_integration(
    fasttext_mock: MockFastText, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test integration with mask_russian function"""
    # Mock the language ID singleton in web_utils
    from turkic_translit.web import web_utils

    monkeypatch.setattr(
        web_utils,
        "_langid_singleton",
        lambda: type("obj", (object,), {"model": fasttext_mock}),
    )

    # Set up responses for tokens
    fasttext_mock.set_response("привет", ["__label__ru"], [0.9])
    fasttext_mock.set_response("мир", ["__label__ru"], [0.8])
    fasttext_mock.set_response("hello", ["__label__en"], [0.95])

    # Test with debug=True to check JSON output
    result = mask_russian(text="привет мир hello", thr=0.5, min_len=3, debug=True)

    # Basic output check
    assert "<RU> <RU> hello" in result

    # Debug output parsing
    debug_start = result.find("<!--debug ") + 10
    debug_end = result.find(" -->", debug_start)
    debug_json = result[debug_start:debug_end]
    debug_data = json.loads(debug_json)

    # Check debug structure
    assert len(debug_data) == 3
    assert debug_data[0]["tok"] == "привет"
    assert debug_data[0]["ru"] is True
    assert debug_data[2]["tok"] == "hello"
    assert debug_data[2]["ru"] is False
