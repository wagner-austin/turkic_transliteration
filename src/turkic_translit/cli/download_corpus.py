"""Command-line interface for streaming public corpora to disk.

The interface is deliberately thin: it turns command-line strings into
validated requests and renders results, and every decision about what to
download and what to keep is made in :mod:`turkic_translit.corpus`.

``--filter-langid`` and ``--lid-model`` are required together and neither
has a default. Naming a language to keep without naming the classifier
that decides the language is exactly how this project's own corpora
became irreproducible: they were filtered with NLLB's ``lid218e`` while
the released tool silently resolved fastText's ``lid.176``, which keeps a
different set of lines at the same threshold. The two options are
therefore either both given or both absent, and whichever model is named
is written into the manifest beside the corpus.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click

from turkic_translit.corpus.catalogue import available_languages, source_reachable
from turkic_translit.corpus.errors import CorpusError
from turkic_translit.corpus.filtering import (
    LidFilterRequest,
    decode_lid_filter_request,
)
from turkic_translit.corpus.run import download_corpus
from turkic_translit.corpus.sources import (
    SOURCE_REGISTRY,
    get_source_spec,
    known_source_ids,
)
from turkic_translit.lid.errors import LidError
from turkic_translit.lid.registry import known_model_ids
from turkic_translit.logging_config import setup as configure_logging
from turkic_translit.validation import FieldError

logger = logging.getLogger(__name__)

DEFAULT_SOURCE: str = "oscar-2301"
DEFAULT_THRESHOLD: float = 0.95


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
    show_default=True,
    help="Set logging level for corpus commands",
)
def cli(log_level: str) -> None:
    """Stream text corpora from public sources.

    Args:
        log_level: Verbosity applied to every command in this group.
    """
    os.environ["TURKIC_LOG_LEVEL"] = log_level.upper()
    configure_logging()


@cli.command("list-sources")
def list_sources() -> None:
    """Print every registered source with its driver and licence."""
    for source_id, spec in SOURCE_REGISTRY.items():
        click.echo(f"{source_id:12}  {spec['driver']:10}  {spec['license']}")


@cli.command("list-langs")
@click.option(
    "--source",
    type=click.Choice(known_source_ids()),
    default=DEFAULT_SOURCE,
    show_default=True,
)
def list_langs(source: str) -> None:
    """Print the language codes a source can be asked for.

    Args:
        source: Registry key of the source to enumerate.
    """
    click.echo(" ".join(available_languages(get_source_spec(source))))


@cli.command("license")
@click.option(
    "--source",
    type=click.Choice(known_source_ids()),
    default=DEFAULT_SOURCE,
    show_default=True,
)
def show_license(source: str) -> None:
    """Print the licence a source is published under.

    Args:
        source: Registry key of the source to report on.
    """
    click.echo(f"{source}: {get_source_spec(source)['license']}")


@cli.command("doctor")
def doctor() -> None:
    """Probe every registered source and report which are unreachable."""
    unreachable = [
        source_id for source_id, spec in SOURCE_REGISTRY.items() if not source_reachable(spec)
    ]
    if unreachable:
        click.secho("Unreachable sources: " + ", ".join(unreachable), fg="red", err=True)
        return
    click.secho("All sources reachable", fg="green")


def build_filter_request(
    filter_langid: str | None, lid_model: str | None, lid_threshold: float
) -> LidFilterRequest | None:
    """Turn the filtering options into a validated request.

    Args:
        filter_langid: Language code to keep, or ``None``.
        lid_model: Registry key of the classifier to use, or ``None``.
        lid_threshold: Minimum probability required to keep a line.

    Returns:
        The validated request, or ``None`` when no filter was asked for.

    Raises:
        click.UsageError: If exactly one of ``--filter-langid`` and
            ``--lid-model`` was given, since neither is meaningful alone.
    """
    if filter_langid is None and lid_model is None:
        return None
    if filter_langid is None or lid_model is None:
        raise click.UsageError(
            "--filter-langid and --lid-model must be given together: a corpus "
            "that records which language it kept must also record which "
            "classifier decided it."
        )
    return decode_lid_filter_request(
        {
            "language": filter_langid,
            "model_id": lid_model,
            "threshold": lid_threshold,
        }
    )


@cli.command("download")
@click.option(
    "--source",
    type=click.Choice(known_source_ids()),
    default=DEFAULT_SOURCE,
    show_default=True,
)
@click.option("--lang", required=True, help="Language code within the source")
@click.option("--out", required=True, type=click.Path(dir_okay=False, writable=True))
@click.option("--max-lines", type=int, default=None, help="Stop after this many lines are kept")
@click.option(
    "--filter-langid",
    type=str,
    default=None,
    help="Keep only lines the classifier assigns to this language code",
)
@click.option(
    "--lid-model",
    type=click.Choice(known_model_ids()),
    default=None,
    help="Classifier that decides the language; required with --filter-langid",
)
@click.option(
    "--lid-threshold",
    type=float,
    default=DEFAULT_THRESHOLD,
    show_default=True,
    help="Minimum probability a line must reach to be kept",
)
def download(
    source: str,
    lang: str,
    out: str,
    max_lines: int | None,
    filter_langid: str | None,
    lid_model: str | None,
    lid_threshold: float,
) -> None:
    """Stream a corpus to a file and write its manifest beside it.

    Args:
        source: Registry key of the corpus to stream.
        lang: Language code within that source.
        out: File to write the corpus to.
        max_lines: Stop after this many lines are kept, or ``None``.
        filter_langid: Language code to keep, or ``None`` for no filter.
        lid_model: Classifier deciding the language, or ``None``.
        lid_threshold: Minimum probability required to keep a line.

    Raises:
        click.ClickException: If the source, the classifier, or the
            filter request is rejected, or the source cannot be read. The
            message carries the originating error code.
    """
    try:
        lid_filter = build_filter_request(filter_langid, lid_model, lid_threshold)
        manifest, manifest_path = download_corpus(
            source_id=source,
            language=lang,
            output_path=Path(out),
            max_lines=max_lines,
            access_token=os.getenv("HF_TOKEN"),
            lid_filter=lid_filter,
        )
    except (CorpusError, LidError, FieldError) as exc:
        logger.error("corpus download failed: %s", exc)
        raise click.ClickException(str(exc)) from exc

    click.secho(
        f"{manifest['lines_written']:,} lines of {manifest['lines_seen']:,} seen -> {out}",
        fg="green",
    )
    click.echo(f"manifest: {manifest_path}")

