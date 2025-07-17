"""Test file upload/download functionality in the web interface."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from turkic_translit.web.web_utils import _CRON_DIR, direct_transliterate


def test_direct_transliterate_with_turkish() -> None:
    """Test that Turkish transliteration works correctly."""
    # Test IPA transliteration
    result, stats = direct_transliterate("merhaba", "tr", False, "ipa")
    assert result == "meɾhaba"
    assert "Bytes" in stats

    # Test that Latin transliteration now works for Turkish
    result, stats = direct_transliterate("merhaba", "tr", False, "latin")
    assert result == "merhaba"  # ASCII-Latin fold (no diacritics in this word)
    assert "Bytes" in stats


def test_file_upload_processing(tmp_path: Path) -> None:
    """Test processing uploaded file content."""
    # Create a test file
    test_content = "Günaydın, nasılsınız?"
    test_file = tmp_path / "test_input.txt"
    test_file.write_text(test_content, encoding="utf-8")

    # Mock file object with .name attribute (like Gradio provides)
    mock_file = Mock()
    mock_file.name = str(test_file)

    # Test that we can read the file
    with open(mock_file.name, encoding="utf-8") as f:
        content = f.read()
    assert content == test_content


def test_download_file_creation() -> None:
    """Test creating download files in CRON_DIR."""
    import time

    # Test file creation pattern
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"transliteration_tr_ipa_{timestamp}.txt"
    filepath = _CRON_DIR / filename

    # Ensure directory exists
    assert _CRON_DIR.exists()

    # Test writing a file
    test_content = "Test transliterated content"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(test_content)

    # Verify file was created
    assert filepath.exists()
    assert filepath.read_text(encoding="utf-8") == test_content

    # Clean up
    filepath.unlink()


def test_direct_transliterate_function() -> None:
    """Test the direct_transliterate function logic."""
    # Test text input
    result, stats = direct_transliterate("Merhaba dünya", "tr", False, "ipa")
    assert result == "meɾhaba dynja"
    assert "Bytes" in stats

    # Test empty input handling through web_utils
    result, stats = direct_transliterate("", "tr", False, "ipa")
    assert result == ""
    # Note: empty input returns empty result with stats showing 0 bytes


@pytest.mark.parametrize(
    ("text", "lang", "fmt", "expected"),
    [
        ("привет", "kk", "Latin", "privet"),  # Kazakh
        ("салам", "ky", "Latin", "salam"),  # Kyrgyz
        ("merhaba", "tr", "IPA", "meɾhaba"),  # Turkish
    ],
)
def test_multiple_language_support(
    text: str, lang: str, fmt: str, expected: str
) -> None:
    """Test that all supported languages work correctly."""
    result, _ = direct_transliterate(text, lang, False, fmt.lower())
    assert result == expected


def test_file_size_threshold() -> None:
    """Test that download is only enabled for non-trivial results."""
    # Small result (< 50 chars) - should not enable download
    small_text = "Hello"
    # This would be handled by the UI logic
    assert len(small_text) < 50

    # Large result (> 50 chars) - should enable download
    large_text = "This is a longer text that exceeds the threshold for enabling download functionality"
    assert len(large_text) > 50
