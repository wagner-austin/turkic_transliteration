import bz2
import html
import json
import logging
import os
import re
import sys
import tarfile
import tempfile
import time
import unicodedata as ud
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import click
import requests
import yaml  # type: ignore # Requires types-PyYAML
from typing_extensions import Never

# 3rd-party heavy deps are optional in constrained CI environments.
# We fallback to lightweight stubs so the module can still import and our
# tests can monkey-patch drivers without pulling gigabytes of data.
try:
    from datasets import get_dataset_config_names, load_dataset
except ModuleNotFoundError:  # pragma: no cover – stubbed in minimal envs

    def _missing(*_a: object, **_kw: object) -> Never:  # noqa: D401 — internal helper
        raise ModuleNotFoundError(
            "Package 'datasets' is required for this command; install with"
            " 'pip install datasets' or use turkic-transliterate[cli] extra."
        )

    load_dataset = _missing
    get_dataset_config_names = _missing

try:
    from fasttext import load_model
except ModuleNotFoundError:  # pragma: no cover – use stub in tests

    def load_model(_p: str) -> Any:  # noqa: D401
        raise ModuleNotFoundError(
            "fasttext not installed; install turkic-transliterate[cli] to use"
            " language-ID filtering."
        )


from ._net_utils import url_ok

# ---------------------------------------------------------------------------


T = TypeVar("T")


def _timeout_wrapper(
    func: Callable[..., T], timeout_seconds: int = 300
) -> Callable[..., T]:
    """Wrapper to add timeout to functions that might hang."""
    import functools
    import threading

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        result: list[Optional[T]] = [None]
        exception: list[Optional[Exception]] = [None]

        def target() -> None:
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

        if exception[0]:
            raise exception[0]

        if result[0] is None:
            raise RuntimeError("Function execution failed without raising an exception")

        return result[0]

    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wikipedia_lang_codes_from_sitematrix() -> list[str]:
    """Return ISO codes for open Wikipedia editions via SiteMatrix API."""
    url = (
        "https://meta.wikimedia.org/w/api.php?"
        "action=sitematrix&format=json&smtype=language&smsiteprop=code|closed"
    )
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.load(r)

    langs: set[str] = set()
    for key, block in data.get("sitematrix", {}).items():
        if not str(key).isdigit():
            continue  # skip special keys
        for site in block.get("site", []):
            if site.get("code") == "wiki" and not site.get("closed"):
                langs.add(block["code"])
                break
    return sorted(langs)


# One evergreen file we can always grab / health-check.
_LEIPZIG_FALLBACK = "deu_news_2012_1M.tar.gz"


def _leipzig_tar_name(lang: str) -> str:
    """Return a plausible Leipzig tarball name for *lang* (news 2012 or web 2019)."""
    tmpl = f"{lang}_news_2012_1M.tar.gz"
    if url_ok(f"https://downloads.wortschatz-leipzig.de/corpora/{tmpl}"):
        return tmpl
    tmpl = f"{lang}_web_2019_1M.tar.gz"
    if url_ok(f"https://downloads.wortschatz-leipzig.de/corpora/{tmpl}"):
        return tmpl
    return _LEIPZIG_FALLBACK


# helper alias – one positional sig keeps mypy happy
StreamFn = Callable[[str, dict[str, Any], Optional[str]], Generator[str, None, None]]

# outer key = source-name, inner value = free-form dict
_REG: dict[str, dict[str, Any]] = yaml.safe_load(
    Path(__file__).with_suffix("").with_name("corpora.yaml").read_text()
)


# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
# Keep large raw Wikipedia XML dumps out of the project root so they don't clutter
# the working tree.  The directory can be overridden via $TURKIC_DUMP_DIR.
DUMP_DIR: Path = Path(os.getenv("TURKIC_DUMP_DIR", "data/raw_wiki"))

# Ensure the directory exists early so concurrent downloads don't race.
DUMP_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- drivers

_FASTTEXT_CACHE: dict[str, Any] = {}


def _get_lid() -> Any:
    from turkic_translit.langid import FastTextLangID
    from turkic_translit.model_utils import ensure_fasttext_model

    lid_path = str(ensure_fasttext_model())
    if lid_path not in _FASTTEXT_CACHE:
        _FASTTEXT_CACHE[lid_path] = FastTextLangID(lid_path)
    return _FASTTEXT_CACHE[lid_path]


def stream_oscar(
    lang: str, cfg: dict[str, Any], filter_langid: Optional[str] = None
) -> Generator[str, None, None]:
    logger = logging.getLogger(__name__)

    # Log environment info for debugging
    logger.info(f"Starting OSCAR download for language: {lang}")
    logger.info(f"Dataset: {cfg['hf_name']}")
    logger.info(f"HF_TOKEN present: {'Yes' if os.getenv('HF_TOKEN') else 'No'}")
    logger.debug(f"HTTP_PROXY: {os.getenv('HTTP_PROXY', 'Not set')}")
    logger.debug(f"HTTPS_PROXY: {os.getenv('HTTPS_PROXY', 'Not set')}")

    # Allow gated OSCAR datasets that rely on custom loading scripts.
    # `trust_remote_code=True` is required from datasets>=2.19 to execute the
    # repository's loading script.  We inherit the user-scoped HF token so no
    # extra `token=`/`use_auth_token=` arg is necessary.
    try:
        logger.info("Initializing dataset stream...")
        start_time = time.time()

        ds = load_dataset(
            cfg["hf_name"],
            lang,
            split="train",
            streaming=True,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )

        init_time = time.time() - start_time
        logger.info(f"Dataset initialized in {init_time:.2f}s")

    except Exception as e:
        logger.error(f"Failed to initialize dataset: {type(e).__name__}: {e}")
        logger.debug("Full error:", exc_info=True)
        raise

    model = _get_lid() if filter_langid else None
    if filter_langid:
        logger.info(f"Language ID filtering enabled for: {filter_langid}")

    row_count = 0
    last_log_time = time.time()

    try:
        logger.info("Starting to stream data...")
        for row in ds:
            row_count += 1

            # Only log in debug mode to avoid confusion when called from web UI
            current_time = time.time()
            if current_time - last_log_time > 10:
                logger.debug(f"stream_oscar internal: processed {row_count} rows...")
                last_log_time = current_time

            txt = (row["text"] or "").strip()
            if txt:
                txt = ud.normalize("NFC", txt)
                if filter_langid and model is not None:
                    pred = model.predict(txt.replace("\n", " "))
                    if pred != filter_langid:
                        continue
                yield txt

    except Exception as e:
        logger.error(
            f"Error during streaming after {row_count} rows: {type(e).__name__}: {e}"
        )
        logger.debug("Full error:", exc_info=True)
        raise

    logger.info(f"Streaming completed. Total rows processed: {row_count}")


def _stream_wikipedia_xml(
    lang: str, cfg: dict[str, Any], filter_langid: Optional[str] = None
) -> Generator[str, None, None]:
    import click

    dump_version = "latest"
    dump_name = f"{lang}wiki-{dump_version}-pages-articles.xml.bz2"
    url = f"https://dumps.wikimedia.org/{lang}wiki/{dump_version}/{dump_name}"

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        # Decompress on the fly from HTTP stream.
        bz_stream = bz2.BZ2File(resp.raw)
        model: Any = _get_lid() if filter_langid else None
        for _, elem in ET.iterparse(bz_stream, events=("end",)):
            if elem.tag.endswith("}text") and elem.text:
                txt = html.unescape(re.sub(r"(?s)<.*?>", " ", elem.text))
                for s in re.split(r"[.!?]", txt):
                    s = s.strip()
                    if s:
                        s = ud.normalize("NFC", s)
                        if filter_langid and model is not None:
                            pred = model.predict(s.replace("\n", " "))
                            if pred != filter_langid:
                                continue
                        yield s
                elem.clear()
            else:
                elem.clear()
    except requests.RequestException as e:
        raise click.ClickException(f"Download failed for {url}: {e}") from e


# ---------------------------------------------------------------------------
# New fast Hugging Face streaming driver for Wikipedia
# ---------------------------------------------------------------------------


def stream_wikipedia(
    lang: str,
    cfg: dict[str, Any],
    filter_langid: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream sentences from *lang* Wikipedia via 🤗 *datasets*.

    Falls back transparently to the original XML-dump path when the
    `datasets` package or the requested language variant is unavailable.
    """
    # Bypass the optional Hugging Face Wikipedia dataset and stream
    # directly from the official XML dump. This avoids noisy 404 logs
    # from HEAD probes when the dataset or language variant is missing.
    yield from _stream_wikipedia_xml(lang, cfg, filter_langid)


def stream_leipzig(
    lang: str, cfg: dict[str, Any], filter_langid: Optional[str] = None
) -> Generator[str, None, None]:
    import click

    tar_url = f"{cfg['base_url']}/{_leipzig_tar_name(lang)}"
    try:
        with tempfile.TemporaryDirectory() as td:
            tgz = Path(td) / "lz.tgz"
            tgz.write_bytes(requests.get(tar_url, timeout=30).content)
            model: Any = _get_lid() if filter_langid else None
            with tarfile.open(tgz) as tar:
                for member in tar.getmembers():
                    if member.name.endswith((".tsv", "-sentences.txt")):
                        fileobj = tar.extractfile(member)
                        if fileobj is None:
                            continue
                        text = fileobj.read().decode("utf-8")
                        for line in text.splitlines():
                            cols = line.split("\t", 1)
                            sent = cols[1] if len(cols) > 1 else line.partition(" ")[2]
                            sent = sent.strip()
                            if filter_langid:
                                pred = model.predict(sent.replace("\n", " "))
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
@click.option("--source", default="oscar-2301")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose debug output.")
def _ls_lang(source: str, verbose: bool) -> None:
    import click

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    cfg = _REG[source]
    if cfg["driver"] == "oscar":
        from datasets import get_dataset_config_names

        logger = logging.getLogger(__name__)
        logger.info(f"Fetching available languages for {cfg['hf_name']}...")

        try:
            # Wrap the call with timeout
            get_configs_with_timeout = _timeout_wrapper(
                get_dataset_config_names, timeout_seconds=60
            )
            names = get_configs_with_timeout(cfg["hf_name"], trust_remote_code=True)
            click.echo(" ".join(sorted(names)))
        except TimeoutError as e:
            logger.error("Timeout: Failed to fetch dataset configs after 60 seconds")
            logger.error("This might be due to network issues or proxy configuration")
            raise click.ClickException(
                "Failed to fetch dataset languages (timeout). "
                "Check your internet connection and proxy settings."
            ) from e
        except Exception as e:
            logger.error(f"Failed to fetch dataset configs: {type(e).__name__}: {e}")
            raise
    elif cfg["driver"] == "wikipedia":
        names = _wikipedia_lang_codes_from_sitematrix()
        if verbose:
            logging.getLogger(__name__).debug(
                "Wikipedia languages detected: %d", len(names)
            )
        click.echo(" ".join(names))
    else:
        # No dynamic Leipzig index available, so we cannot list languages.
        click.echo("Dynamic Leipzig language listing unavailable.")


@cli.command("license")
@click.option("--source", required=True)
def _license(source: str) -> None:
    cfg = _REG[source]
    click.echo(f"License for {source}: {cfg.get('license', 'unknown')}")


@cli.command("download")
@click.option("--source", default="oscar-2301", show_default=True)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose debug output.")
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
    verbose: bool,
) -> None:
    import click

    # Configure logging based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Also set HuggingFace logging if available
    try:
        import datasets

        datasets.logging.set_verbosity(
            datasets.logging.DEBUG if verbose else datasets.logging.WARNING
        )
    except ImportError:
        pass

    logger = logging.getLogger(__name__)
    logger.info("Starting corpus download")
    logger.info(f"Source: {source}, Language: {lang}, Output: {out}")

    cfg: dict[str, Any] = _REG[source]
    stream_fn = _DRIVERS[cfg["driver"]]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    lines = 0
    with open(out, "w", encoding="utf8") as fh:
        start_time = time.time()
        last_progress_time = start_time

        logger.info("Starting to download and process corpus...")

        try:
            for text in stream_fn(lang, cfg, filter_langid):
                if max_lines is not None and lines >= max_lines:
                    logger.info(f"Reached requested limit of {max_lines} lines")
                    break
                # Optional post-filtering if driver did not honour --filter-langid
                if filter_langid:
                    # Lazily create the model once
                    nonlocal_model: Any
                    if "post_lid" not in locals():
                        nonlocal_model = _get_lid()
                        post_lid = nonlocal_model
                    else:
                        post_lid = locals().get("post_lid")
                    if post_lid.predict(text.replace("\n", " ")) != filter_langid:
                        continue
                fh.write(text + "\n")
                lines += 1

                # Progress reporting
                current_time = time.time()
                if lines % 100 == 0 or (current_time - last_progress_time > 10):
                    elapsed = current_time - start_time
                    rate = lines / elapsed if elapsed > 0 else 0
                    if max_lines:
                        logger.info(
                            f"Progress: {lines}/{max_lines} lines saved ({rate:.0f} lines/sec)"
                        )
                    else:
                        logger.info(
                            f"Progress: {lines:,} lines saved ({rate:.0f} lines/sec)"
                        )
                    last_progress_time = current_time

        except KeyboardInterrupt:
            logger.warning("Download interrupted by user")
            raise
        except Exception as e:
            logger.error(
                f"Download failed after {lines:,} lines: {type(e).__name__}: {e}"
            )
            raise

    elapsed_total = time.time() - start_time
    logger.info(f"Download completed in {elapsed_total:.1f}s")
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
            url = "https://meta.wikimedia.org/w/api.php?action=sitematrix&format=json"
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
