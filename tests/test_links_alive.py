from typing import Any

import pytest

from turkic_translit.cli import download_corpus as dl
from turkic_translit.cli._net_utils import url_ok


@pytest.mark.network
@pytest.mark.parametrize(("source", "cfg"), dl._REG.items())
def test_head_ok(source: str, cfg: dict[str, Any]) -> None:
    if cfg["driver"] == "oscar":
        url = f"https://huggingface.co/api/datasets/{cfg['hf_name']}"
    elif cfg["driver"] == "wikipedia":
        url = "https://dumps.wikimedia.org/"
    elif cfg["driver"] == "leipzig":
        url = f"{cfg['base_url']}/deu_news_2012_1M.tar.gz"
    else:
        url = cfg["base_url"]
    assert url_ok(url), f"{source} site unreachable"
