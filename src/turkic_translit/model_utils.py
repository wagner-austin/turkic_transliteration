"""
Utilities for downloading and managing model files required by the Turkic Transliteration Suite.
"""

import logging
import pathlib
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# Constants
FASTTEXT_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
)


def download_fasttext_model(target_path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    Download the FastText language identification model (lid.176.ftz).

    Args:
        target_path: Optional target path to save the model. If None, saves to package directory.

    Returns:
        Path object to the downloaded model file

    Raises:
        OSError: If download fails
    """
    if target_path is None:
        # Default to the package directory
        target_path = pathlib.Path(__file__).parent / "lid.176.ftz"

    logger.info(f"Downloading FastText model to {target_path}")

    # Create directory if it doesn't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Download with a simple progress callback
        def report_progress(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            if block_num % 100 == 0 or percent >= 100:
                logger.info(
                    f"Download progress: {percent:.1f}% ({downloaded}/{total_size} bytes)"
                )

        urllib.request.urlretrieve(
            FASTTEXT_MODEL_URL, target_path, reporthook=report_progress
        )

        logger.info(f"Successfully downloaded FastText model to {target_path}")
        return target_path

    except Exception as e:
        logger.error(f"Failed to download FastText model: {e}")
        raise OSError(f"Failed to download FastText model: {e}") from e


def ensure_fasttext_model() -> pathlib.Path:
    """
    Ensure the FastText language identification model is available.
    If not found in standard locations, will download it automatically.

    Returns:
        Path object to the model file

    Raises:
        OSError: If download fails and model cannot be found
    """
    # Check standard locations first
    home_lid = pathlib.Path.home() / "lid.176.ftz"
    pkg_dir = pathlib.Path(__file__).parent
    pkg_lid = pkg_dir / "lid.176.ftz"
    web_lid = pkg_dir / "web" / "lid.176.ftz"

    for path in [home_lid, pkg_lid, web_lid]:
        if path.exists():
            logger.info(f"Found existing FastText model at {path}")
            return path

    # If not found, download to package directory
    logger.info("FastText model not found in standard locations, downloading...")
    return download_fasttext_model(pkg_lid)
