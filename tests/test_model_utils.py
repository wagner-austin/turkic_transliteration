"""
Tests for the automatic model download functionality.
"""

import os
import pathlib
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from turkic_translit.langid import FastTextLangID
from turkic_translit.model_utils import download_fasttext_model, ensure_fasttext_model


class TestModelUtils:
    """Tests for the model_utils module."""

    def setup_method(self) -> None:
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_model_path = pathlib.Path(self.temp_dir) / "lid.176.ftz"

    def teardown_method(self) -> None:
        """Clean up temporary directory after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.skip(reason="Only run manually - downloads real model")
    def test_download_fasttext_model_real(self) -> None:
        """Test actual downloading of the model (skip by default to avoid network)."""
        result = download_fasttext_model(self.temp_model_path)
        assert result.exists()
        assert result.stat().st_size > 0
        assert str(result) == str(self.temp_model_path)

    @patch("turkic_translit.model_utils.urllib.request.urlretrieve")
    def test_download_fasttext_model(self, mock_urlretrieve: MagicMock) -> None:
        """Test downloading function with mocked urllib."""
        # Mock the download
        mock_urlretrieve.return_value = None

        # Create an empty file to simulate successful download
        self.temp_model_path.touch()

        result = download_fasttext_model(self.temp_model_path)

        # Verify the function called urlretrieve with correct arguments
        mock_urlretrieve.assert_called_once()
        assert (
            mock_urlretrieve.call_args[0][0]
            == "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        )
        assert mock_urlretrieve.call_args[0][1] == self.temp_model_path

        # Verify returned path
        assert result == self.temp_model_path

    @patch("turkic_translit.model_utils.download_fasttext_model")
    def test_ensure_fasttext_model_download(self, mock_download: MagicMock) -> None:
        """Test ensure_fasttext_model when model doesn't exist."""
        # Mock the download function to return our temp path
        mock_download.return_value = self.temp_model_path

        # Patch Path.exists to return False for all known model locations
        with patch("pathlib.Path.exists", return_value=False):
            result = ensure_fasttext_model()

            # Verify download was called
            mock_download.assert_called_once()
            assert result == self.temp_model_path

    def test_avoid_download_when_model_exists(self) -> None:
        """Test that download is not performed when the model exists."""
        # Create a temporary model file
        self.temp_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_model_path.touch()

        # Mock functions to make Path.home() return our temp directory
        # and ensure our path is found
        with (
            patch("pathlib.Path.home", return_value=pathlib.Path(self.temp_dir)),
            patch(
                "turkic_translit.model_utils.download_fasttext_model"
            ) as mock_download,
        ):
            # Call the function - should find our model and not try to download
            result = ensure_fasttext_model()
            mock_download.assert_not_called()

            # The function should return our temp model path
            assert result == self.temp_model_path

    @patch("turkic_translit.langid.ensure_fasttext_model")
    @patch("fasttext.load_model")
    def test_fasttext_langid_auto_download(
        self, mock_load_model: MagicMock, mock_ensure: MagicMock
    ) -> None:
        """Test FastTextLangID uses automatic download."""
        # Setup mocks
        test_model_path = str(self.temp_model_path)
        mock_ensure.return_value = pathlib.Path(test_model_path)
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Test initialization with no path (should trigger auto-download)
        langid = FastTextLangID()

        # Verify ensure_fasttext_model was called
        mock_ensure.assert_called_once()

        # Verify load_model was called with the right path
        mock_load_model.assert_called_once_with(test_model_path)

        # Verify model was set
        assert langid.model == mock_model

    @patch("turkic_translit.langid.ensure_fasttext_model")
    @patch("os.path.exists")
    @patch("fasttext.load_model")
    def test_fasttext_langid_download_fallback(
        self, mock_load_model: MagicMock, mock_exists: MagicMock, mock_ensure: MagicMock
    ) -> None:
        """Test FastTextLangID falls back to bin file when download fails."""
        # Setup mocks to simulate download failure and bin file existence
        mock_ensure.side_effect = Exception("Download failed")
        mock_exists.return_value = True  # Simulate bin file exists
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Test initialization (should try download, fail, then use bin)
        langid = FastTextLangID()

        # Verify ensure_fasttext_model was called
        mock_ensure.assert_called_once()

        # Verify load_model was called with a path ending in lid.176.bin
        assert mock_load_model.call_args[0][0].endswith("lid.176.bin")

        # Verify model was set
        assert langid.model == mock_model


# Integration tests that require real model - these are skipped by default
class TestModelIntegration:
    """Integration tests that use real model download functionality."""

    @pytest.mark.skip(reason="Only run manually - requires network")
    def test_langid_integration(self) -> None:
        """
        Test end-to-end FastTextLangID functionality with real download.

        This test downloads the actual model and tests language prediction.
        Skip by default to avoid network dependency during CI.
        """
        # Remove models from standard locations first to ensure test downloads
        try:
            import tempfile

            temp_dir = tempfile.mkdtemp()
            home_path = pathlib.Path.home() / "lid.176.ftz"
            pkg_path = (
                pathlib.Path(__file__).parent.parent
                / "src"
                / "turkic_translit"
                / "lid.176.ftz"
            )
            web_path = (
                pathlib.Path(__file__).parent.parent
                / "src"
                / "turkic_translit"
                / "web"
                / "lid.176.ftz"
            )

            # Temporarily move existing models if they exist
            for path in [home_path, pkg_path, web_path]:
                if path.exists():
                    backup_path = pathlib.Path(temp_dir) / path.name
                    shutil.move(str(path), str(backup_path))

            # Create and test the language ID model
            langid = FastTextLangID()

            # Test English identification - should be consistent
            assert langid.predict("Hello world") == "en"
            assert langid.predict("This is a test") == "en"

            # Test Russian identification - critical for the app's filtering functionality
            # These MUST be identified as Russian consistently
            russian_examples = [
                "Привет мир",  # Привет мир
                "здравствуйте",  # здравствуйте
                "добрый день",  # добрый день
                "как дела",  # как дела
                "спасибо",  # спасибо
            ]
            for russian_text in russian_examples:
                assert (
                    langid.predict(russian_text) == "ru"
                ), f"Failed to identify '{russian_text}' as Russian"

            # For Turkic languages, the compressed model may identify some phrases differently
            # This is expected due to the similarity between Turkic languages
            turkic_text = "Сәлем әлем"  # Kazakh greeting
            lang_code = langid.predict(turkic_text)

            # Rather than requiring a specific code, verify it's identified as a Turkic language
            turkic_codes = {"kk", "tt", "ky", "uz", "ba", "sah", "crh", "azb", "az"}
            assert (
                lang_code in turkic_codes
            ), f"'{turkic_text}' identified as '{lang_code}' which is not in Turkic language codes: {turkic_codes}"

            # Verify prediction of tokens
            tokens = ["Hello", "Привет", "Сәлем"]
            langs = langid.predict_tokens(tokens)

            # First two should be consistent
            assert langs[0] == "en"
            assert langs[1] == "ru"

            # For the Turkic token, verify it's identified as a Turkic language
            turkic_codes = {"kk", "tt", "ky", "uz", "ba", "sah", "crh", "azb", "az"}
            assert (
                langs[2] in turkic_codes
            ), f"'Сәлем' identified as '{langs[2]}' which is not in Turkic language codes: {turkic_codes}"

        finally:
            # Restore any models we moved
            for path in [home_path, pkg_path, web_path]:
                backup_path = pathlib.Path(temp_dir) / path.name
                if backup_path.exists():
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    shutil.move(str(backup_path), str(path))
            shutil.rmtree(temp_dir, ignore_errors=True)
