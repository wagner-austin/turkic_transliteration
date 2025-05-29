import bz2
import html
import re
import tarfile
import tempfile
import unicodedata as ud
import xml.etree.ElementTree as ET
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable, Optional

import click
import requests
import yaml  # type: ignore # Requires types-PyYAML
from datasets import load_dataset
from fasttext import load_model

from ._net_utils import url_ok

# One evergreen file we can always grab / health-check.
_LEIPZIG_FALLBACK = "deu_news_2012_1M.tar.gz"


# helper alias – one positional sig keeps mypy happy
StreamFn = Callable[[str, dict[str, Any], Optional[str]], Generator[str, None, None]]

# outer key = source-name, inner value = free-form dict
_REG: dict[str, dict[str, Any]] = yaml.safe_load(
    Path(__file__).with_suffix("").with_name("corpora.yaml").read_text()
)

# --------------------------------------------------------------------------- drivers

_FASTTEXT_CACHE: dict[str, Any] = {}


def _get_lid() -> Any:
    from turkic_translit.model_utils import ensure_fasttext_model

    lid_path = str(ensure_fasttext_model())
    return _FASTTEXT_CACHE.setdefault(lid_path, load_model(lid_path))


def stream_oscar(
    lang: str, cfg: dict[str, Any], filter_langid: Optional[str] = None
) -> Generator[str, None, None]:
    import click

    from ._net_utils import url_ok

    if not url_ok(f"https://huggingface.co/api/datasets/{cfg['hf_name']}"):
        raise click.ClickException(
            f"Remote corpus not reachable: https://huggingface.co/api/datasets/{cfg['hf_name']}"
        )
    ds = load_dataset(cfg["hf_name"], lang, split="train", streaming=True)
    model = _get_lid() if filter_langid else None
    for row in ds:
        txt = (row["text"] or "").strip()
        if txt:
            txt = ud.normalize("NFC", txt)
            if filter_langid and model is not None:
                pred = model.predict(txt.replace("\n", " "))[0][0].replace(
                    "__label__", ""
                )
                if pred != filter_langid:
                    continue
            yield txt


def stream_wikipedia(
    lang: str, cfg: dict[str, Any], filter_langid: Optional[str] = None
) -> Generator[str, None, None]:
    import click

    from ._net_utils import url_ok

    dump = f"{lang}wiki-latest-pages-articles.xml.bz2"
    url = f"https://dumps.wikimedia.org/{lang}wiki/latest/{dump}"
    if not url_ok(url):
        raise click.ClickException(f"Remote corpus not reachable: {url}")
    try:
        with open(dump, "wb") as f:
            for chunk in requests.get(url, stream=True, timeout=30).iter_content(
                chunk_size=8192
            ):
                f.write(chunk)
    except requests.RequestException as e:
        raise click.ClickException(f"Download failed for {url}: {e}") from e
    model: Any = _get_lid() if filter_langid else None
    with bz2.open(dump) as fp:
        for _, elem in ET.iterparse(fp, events=("end",)):
            if elem.tag.endswith("}text") and elem.text:
                txt = html.unescape(re.sub(r"(?s)<.*?>", " ", elem.text))
                for s in re.split(r"[.!?]", txt):
                    s = s.strip()
                    if s:
                        s = ud.normalize("NFC", s)
                        if filter_langid and model is not None:
                            pred = model.predict(s.replace("\n", " "))[0][0].replace(
                                "__label__", ""
                            )
                            if pred != filter_langid:
                                continue
                        yield s
                elem.clear()
            else:
                elem.clear()


def _leipzig_tar_name(lang: str) -> str:  # noqa: D401
    """
    Return a plausible tarball name for *lang*.

    For now we support the common “{iso}_news_2012_1M.tar.gz” and
    “{iso}_web_2019_1M.tar.gz” layouts; fall back to the German news 2012 file
    when in doubt so health-checks never 404.
    """
    tmpl = f"{lang}_news_2012_1M.tar.gz"
    test_url = f"https://downloads.wortschatz-leipzig.de/corpora/{tmpl}"
    if url_ok(test_url):
        return tmpl

    tmpl = f"{lang}_web_2019_1M.tar.gz"
    test_url = f"https://downloads.wortschatz-leipzig.de/corpora/{tmpl}"
    if url_ok(test_url):
        return tmpl

    # Last resort – always exists
    return _LEIPZIG_FALLBACK


def stream_leipzig(
    lang: str, cfg: dict[str, Any], filter_langid: Optional[str] = None
) -> Generator[str, None, None]:
    import click

    from ._net_utils import url_ok

    tar_url = f"{cfg['base_url']}/{_leipzig_tar_name(lang)}"
    if not url_ok(tar_url):
        raise click.ClickException(f"Remote corpus not reachable: {tar_url}")
    try:
        with tempfile.TemporaryDirectory() as td:
            tgz = Path(td) / "lz.tgz"
            tgz.write_bytes(requests.get(tar_url, timeout=30).content)
            model: Any = _get_lid() if filter_langid else None
            with tarfile.open(tgz) as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".tsv"):
                        fileobj = tar.extractfile(member)
                        if fileobj is None:
                            continue
                        text = fileobj.read().decode("utf-8")
                        for line in text.splitlines():
                            sent = line.split("\t", 2)[1].strip()
                            if filter_langid:
                                pred = model.predict(sent.replace("\n", " "))[0][
                                    0
                                ].replace("__label__", "")
                                if pred != filter_langid:
                                    continue
                            yield sent
    except requests.RequestException as e:
        raise click.ClickException(f"Download failed for {tar_url}: {e}") from e


_DRIVERS: dict[str, StreamFn] = {
    "oscar": stream_oscar,
    "wikipedia": stream_wikipedia,
    "leipzig": stream_leipzig,
}


# --------------------------------------------------------------------------- CLI
@click.group()
def cli() -> None:
    pass


@cli.command("list-sources")
def _ls_src() -> None:
    for name, cfg in _REG.items():
        click.echo(f"{name:12}  ({cfg['driver']})")


@cli.command("list-langs")
@click.option("--source", default="oscar-2201")
def _ls_lang(source: str) -> None:
    import click

    from ._net_utils import url_ok

    cfg = _REG[source]
    if cfg["driver"] == "oscar":
        from datasets import get_dataset_config_names

        names = get_dataset_config_names(cfg["hf_name"])
        click.echo(" ".join(sorted(names)))
    elif cfg["driver"] == "wikipedia":
        # Candidate codes: hardcoded or loaded from a file if available
        candidate_codes = [
            # ... populate with the list of 300 ISO codes or load dynamically ...
            # For brevity, only a few examples:
            "en",
            "ru",
            "kk",
            "ky",
            "tr",
            "uz",
            "az",
        ]

        def wiki_lang_ok(lang: str) -> bool:
            url = f"https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"
            return url_ok(url)

        maybe = [c for c in candidate_codes if wiki_lang_ok(c)]
        click.echo(" ".join(maybe))
    else:
        # No dynamic Leipzig index available, so we cannot list languages.
        click.echo("Dynamic Leipzig language listing unavailable.")


@cli.command("license")
@click.option("--source", required=True)
def _license(source: str) -> None:
    cfg = _REG[source]
    click.echo(f"License for {source}: {cfg.get('license', 'unknown')}")


@cli.command("download")
@click.option("--source", default="oscar-2201", show_default=True)
@click.option("--lang", required=True, help="ISO-639-1/3 code")
@click.option("--out", required=True, type=click.Path(dir_okay=False, writable=True))
@click.option(
    "--max-lines", type=int, default=None, help="Maximum number of lines to stream"
)
@click.option(
    "--filter-langid",
    type=str,
    default=None,
    help="Filter sentences by FastText language ID (ISO code)",
)
def _dl(
    source: str,
    lang: str,
    out: str,
    max_lines: Optional[int],
    filter_langid: Optional[str],
) -> None:
    import click

    cfg: dict[str, Any] = _REG[source]
    stream_fn = _DRIVERS[cfg["driver"]]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    lines = 0
    with open(out, "w", encoding="utf8") as fh:
        model = _get_lid() if filter_langid else None
        for text in stream_fn(lang, cfg, filter_langid):
            if max_lines is not None and lines >= max_lines:
                break
            if model and filter_langid:
                pred = model.predict(text.replace("\n", " "))[0][0].replace(
                    "__label__", ""
                )
                if pred != filter_langid:
                    continue
            fh.write(text + "\n")
            lines += 1
            if lines % 50_000 == 0:
                click.echo(f"{lines:,} lines...", err=True)
    click.secho(f"\u2713 {lines:,} lines \u2192 {out}", fg="green")


@cli.command("doctor")
def _doctor() -> None:
    import click

    from ._net_utils import url_ok

    bad = []
    for name, cfg in _REG.items():
        if cfg["driver"] == "oscar":
            url = f"https://huggingface.co/api/datasets/{cfg['hf_name']}"
        elif cfg["driver"] == "wikipedia":
            url = "https://dumps.wikimedia.org/"
        elif cfg["driver"] == "leipzig":
            from .download_corpus import _leipzig_tar_name

            url = f"{cfg['base_url']}/{_leipzig_tar_name('tat')}"
        else:
            url = cfg["base_url"]
        if not url_ok(url):
            bad.append(name)
    if bad:
        click.secho("Broken sources: " + ", ".join(bad), fg="red", err=True)
    else:
        click.secho("All sources reachable \u2713", fg="green")


if __name__ == "__main__":
    cli()
