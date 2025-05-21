"""
Utilities for downloading and managing model files required by the Turkic Transliteration Suite.

FastText Language Identification Model Information:
-------------------------------------------------
We use two models for language identification, which can recognize 176 languages:
- lid.176.bin: faster and slightly more accurate, file size of 126MB
- lid.176.ftz: compressed version of the model, file size of 917kB

These models were trained on data from Wikipedia, Tatoeba and SETimes, used under CC-BY-SA.
The models expect UTF-8 as input.

License: Creative Commons Attribution-Share-Alike License 3.0.

Supported languages (ISO codes):
af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de
diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie
ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr
mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah
sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu
xal xmf yi yo yue zh

References:
[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification
    @article{joulin2016bag,
      title={Bag of Tricks for Efficient Text Classification},
      author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
      journal={arXiv preprint arXiv:1607.01759},
      year={2016}
    }

[2] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, FastText.zip: Compressing text classification models
    @article{joulin2016fasttext,
      title={FastText.zip: Compressing text classification models},
      author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
      journal={arXiv preprint arXiv:1612.03651},
      year={2016}
    }
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
