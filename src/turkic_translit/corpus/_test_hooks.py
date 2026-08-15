"""Injection seam for the network and dataset boundaries.

Production binds each hook to its real adapter at import time and never
rebinds it. Tests bind them to in-memory implementations that satisfy the
same protocols. The drivers call the hooks unconditionally, so no
production branch exists purely to support testing.

Every real adapter here reaches the outside world through ``urllib``
rather than through ``requests``, and that is deliberate: ``urllib``
handles ``file://``, so each adapter can be exercised for real — not
stubbed — against files on disk, endpoints included.

The module is private because the seam is internal to this package and is
not part of the published API.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import IO, Final, Protocol
from urllib.error import URLError
from urllib.request import urlopen

from zstandard import ZstdDecompressor

from turkic_translit.corpus import hub
from turkic_translit.corpus.errors import CorpusStreamError
from turkic_translit.net import build_request

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS: Final = 30.0

SITEMATRIX_URL: Final = (
    "https://meta.wikimedia.org/w/api.php?"
    "action=sitematrix&format=json&smtype=language&smsiteprop=code|closed"
)


class DatasetTextStreamer(Protocol):
    """Reader yielding one language's documents from a corpus."""

    def texts(self, dataset_name: str, configuration: str, token: str | None) -> Iterator[str]:
        """Stream the text of every document that carries some.

        Args:
            dataset_name: Dataset identifier, e.g.
                ``oscar-corpus/OSCAR-2301``.
            configuration: Language code within that dataset.
            token: Access token for gated datasets, or ``None``.

        Yields:
            Each document's text, exactly as stored.
        """
        ...


def repository_files(api_url: str) -> tuple[str, ...]:
    """List every file a dataset repository holds.

    The listing is public even for a gated dataset, so this is what the
    language list is built from: naming a corpus's languages costs no
    credential, while reading its text does.

    Args:
        api_url: The repository's endpoint on the Hub.

    Returns:
        Every path in the repository, in the order the Hub reports them.

    Raises:
        CorpusStreamError: If the response carries no file listing,
            which is what a moved or renamed repository looks like.
    """
    with urlopen(build_request(api_url, "GET"), timeout=TIMEOUT_SECONDS) as response:
        document = json.load(response)

    siblings = document.get("siblings") if isinstance(document, Mapping) else None
    if not isinstance(siblings, list):
        raise CorpusStreamError(api_url, "response carries no file listing")

    return tuple(
        sibling["rfilename"]
        for sibling in siblings
        if isinstance(sibling, Mapping) and isinstance(sibling.get("rfilename"), str)
    )


class HubShardTextStreamer:
    """Streamer reading a language's shards straight from the Hub.

    Each shard is Zstandard-compressed JSON lines, decompressed as it
    arrives rather than downloaded whole: a language's shards run to
    gigabytes, and a caller asking for a hundred sentences should pay
    for a hundred sentences.

    A line whose ``content`` is not a string is skipped rather than
    turned into an empty document, and the shards are read in part
    order so the corpus arrives as it was published.

    Args:
        dataset_api: Endpoint listing a repository's files. Overridable
            so the whole read runs against files on disk.
        resolve_template: Template naming a file's download URL, with
            ``name`` and ``path`` fields.
    """

    def __init__(
        self,
        dataset_api: str = hub.DATASET_API,
        resolve_template: str = hub.RESOLVE_TEMPLATE,
    ) -> None:
        """Store the two endpoints this streamer reads through."""
        self._dataset_api = dataset_api
        self._resolve_template = resolve_template

    def texts(self, dataset_name: str, configuration: str, token: str | None) -> Iterator[str]:
        """Stream one language's documents, shard by shard.

        Args:
            dataset_name: Dataset identifier, e.g.
                ``oscar-corpus/OSCAR-2301``.
            configuration: Language code selecting the shards to read.
            token: Access token for the gated data, or ``None``.

        Yields:
            Each document's text, skipping lines that carry none.

        Raises:
            CorpusStreamError: If the language has no shards, which
                means it was offered by something other than this
                repository's contents.
        """
        api_url = f"{self._dataset_api}/{dataset_name}"
        paths = hub.shard_paths(repository_files(api_url), configuration)
        if not paths:
            raise CorpusStreamError(api_url, f"no shards for language {configuration}")

        for path in paths:
            url = self._resolve_template.format(name=dataset_name, path=path)
            with urlopen(build_request(url, "GET", token), timeout=TIMEOUT_SECONDS) as response:
                reader = ZstdDecompressor().stream_reader(response)
                for line in io.TextIOWrapper(reader, encoding="utf-8"):
                    document = json.loads(line)
                    value = (
                        document.get(hub.CONTENT_FIELD) if isinstance(document, Mapping) else None
                    )
                    if isinstance(value, str):
                        yield value


class MappingDatasetTextStreamer:
    """Streamer answering from an in-memory table of configurations.

    Args:
        texts_by_configuration: Mapping of configuration name to the
            lines that configuration holds.
    """

    def __init__(self, texts_by_configuration: Mapping[str, Sequence[str]]) -> None:
        """Store the table and start an empty request log."""
        self._texts = dict(texts_by_configuration)
        self.requests: list[tuple[str, str, str | None]] = []

    def texts(self, dataset_name: str, configuration: str, token: str | None) -> Iterator[str]:
        """Record the request and yield the configuration's lines.

        Args:
            dataset_name: Dataset identifier, recorded not consulted.
            configuration: Configuration to look up.
            token: Access token, recorded not consulted.

        Yields:
            Each stored line for the configuration.

        Raises:
            CorpusStreamError: If the configuration is not in the table.
        """
        self.requests.append((dataset_name, configuration, token))
        lines = self._texts.get(configuration)
        if lines is None:
            raise CorpusStreamError(
                f"{dataset_name}/{configuration}", "no configuration in the table"
            )
        yield from lines


class ByteStreamOpener(Protocol):
    """Opener returning a readable binary stream for a URL."""

    def open(self, url: str) -> IO[bytes]:
        """Open ``url`` for streaming reads.

        Args:
            url: Fully-qualified URL to read.

        Returns:
            A readable binary stream positioned at the start.
        """
        ...


class UrlByteStreamOpener:
    """Opener streaming over any scheme ``urllib`` handles.

    Carries the project User-Agent, which Wikimedia's dump hosts require.
    """

    def open(self, url: str) -> IO[bytes]:
        """Open ``url`` and return the response body as a stream.

        Args:
            url: Fully-qualified URL to read.

        Returns:
            The response body, readable in chunks.

        Raises:
            CorpusStreamError: If the host refused, failed, or could not
                be resolved.
        """
        try:
            stream: IO[bytes] = urlopen(build_request(url, "GET"), timeout=TIMEOUT_SECONDS)
        except URLError as exc:
            logger.error("could not open %s: %s", url, exc)
            raise CorpusStreamError(url, str(exc)) from exc
        return stream


class MappingByteStreamOpener:
    """Opener answering from an in-memory mapping of URL to bytes.

    Args:
        payloads: Mapping of exact URL to the bytes that URL serves.
    """

    def __init__(self, payloads: Mapping[str, bytes]) -> None:
        """Store the payloads and start an empty request log."""
        self._payloads = dict(payloads)
        self.requests: list[str] = []

    def open(self, url: str) -> IO[bytes]:
        """Record the request and return the stored bytes as a stream.

        Args:
            url: URL to look up.

        Returns:
            A stream over the stored bytes.

        Raises:
            CorpusStreamError: If no payload is registered for the URL,
                which is how a wrong URL is surfaced rather than being
                answered with empty content.
        """
        self.requests.append(url)
        payload = self._payloads.get(url)
        if payload is None:
            raise CorpusStreamError(url, "no payload registered for this URL")
        return io.BytesIO(payload)


class ReachabilityProbe(Protocol):
    """Probe reporting whether a host will serve a URL."""

    def reachable(self, url: str) -> bool:
        """Report whether ``url`` can be retrieved.

        Args:
            url: Fully-qualified URL to probe.

        Returns:
            True when the host answered successfully.
        """
        ...


class UrlReachabilityProbe:
    """Probe issuing a HEAD request and reporting whether it succeeded."""

    def reachable(self, url: str) -> bool:
        """Probe ``url`` without downloading its body.

        Args:
            url: Fully-qualified URL to probe.

        Returns:
            True when the request completed; False when the host
            answered with an error status or could not be reached.
        """
        try:
            with urlopen(build_request(url, "HEAD"), timeout=TIMEOUT_SECONDS):
                return True
        except URLError as exc:
            logger.info("%s is unreachable: %s", url, exc)
            return False


class MappingReachabilityProbe:
    """Probe answering from a set of URLs known to be reachable.

    Args:
        reachable_urls: Every URL this probe reports as reachable.
    """

    def __init__(self, reachable_urls: Sequence[str]) -> None:
        """Store the reachable set and start an empty request log."""
        self._reachable = set(reachable_urls)
        self.requests: list[str] = []

    def reachable(self, url: str) -> bool:
        """Record the probe and answer from the stored set.

        Args:
            url: URL to look up.

        Returns:
            True when the URL is in the reachable set.
        """
        self.requests.append(url)
        return url in self._reachable


class LanguageCatalogue(Protocol):
    """Catalogue of the language codes each driver can be asked for."""

    def oscar_configurations(self, dataset_name: str) -> tuple[str, ...]:
        """List the configurations a Hugging Face dataset publishes.

        Args:
            dataset_name: Dataset identifier.

        Returns:
            Configuration names, sorted.
        """
        ...

    def wikipedia_editions(self) -> tuple[str, ...]:
        """List the language codes with an open Wikipedia edition.

        Returns:
            Language codes, sorted.
        """
        ...


class RemoteLanguageCatalogue:
    """Catalogue reading from Hugging Face and from Wikimedia's SiteMatrix.

    Args:
        sitematrix_url: SiteMatrix endpoint to query. Overridable so the
            JSON walk below can be exercised against a captured response
            rather than against the live API.
    """

    def __init__(
        self,
        sitematrix_url: str = SITEMATRIX_URL,
        dataset_api: str = hub.DATASET_API,
    ) -> None:
        """Store the two endpoints this catalogue queries."""
        self._sitematrix_url = sitematrix_url
        self._dataset_api = dataset_api

    def oscar_configurations(self, dataset_name: str) -> tuple[str, ...]:
        """List the languages a dataset repository holds shards for.

        Read from the repository's file listing, which is public: the
        previous version learned the same list by downloading and
        executing the dataset's loading script, so naming the languages
        ran third-party code and needed that script's own dependencies
        installed.

        Args:
            dataset_name: Dataset identifier.

        Returns:
            Language codes, sorted.
        """
        files = repository_files(f"{self._dataset_api}/{dataset_name}")
        return hub.languages(files)

    def wikipedia_editions(self) -> tuple[str, ...]:
        """List language codes whose Wikipedia edition is open.

        A block's ``site`` entry carries a ``closed`` key only when that
        edition has been closed, and the key's value is the empty string.
        Membership is therefore the test; truthiness is not, which is
        what previously let closed editions such as ``aa`` through.

        Returns:
            Language codes with an open Wikipedia, sorted.

        Raises:
            CorpusStreamError: If the response carries no SiteMatrix.
        """
        with urlopen(
            build_request(self._sitematrix_url, "GET"), timeout=TIMEOUT_SECONDS
        ) as response:
            document = json.load(response)

        matrix = document.get("sitematrix")
        if not isinstance(matrix, Mapping):
            raise CorpusStreamError(self._sitematrix_url, "response carries no sitematrix mapping")

        codes: list[str] = []
        for key, block in matrix.items():
            if not key.isdigit() or not isinstance(block, Mapping):
                continue
            code = block.get("code")
            sites = block.get("site")
            if not isinstance(code, str) or not isinstance(sites, list):
                continue
            for site in sites:
                if (
                    isinstance(site, Mapping)
                    and site.get("code") == "wiki"
                    and "closed" not in site
                ):
                    codes.append(code)
                    break
        return tuple(sorted(codes))


class MappingLanguageCatalogue:
    """Catalogue answering from fixed lists.

    Args:
        oscar_by_dataset: Mapping of dataset name to its configurations.
        wikipedia: Language codes with an open Wikipedia edition.
    """

    def __init__(
        self,
        oscar_by_dataset: Mapping[str, Sequence[str]],
        wikipedia: Sequence[str],
    ) -> None:
        """Store both catalogues."""
        self._oscar = dict(oscar_by_dataset)
        self._wikipedia = tuple(wikipedia)

    def oscar_configurations(self, dataset_name: str) -> tuple[str, ...]:
        """Return the stored configurations for a dataset.

        Args:
            dataset_name: Dataset to look up.

        Returns:
            Configuration names, sorted.

        Raises:
            CorpusStreamError: If the dataset is not in the table.
        """
        configurations = self._oscar.get(dataset_name)
        if configurations is None:
            raise CorpusStreamError(dataset_name, "no dataset in the table")
        return tuple(sorted(configurations))

    def wikipedia_editions(self) -> tuple[str, ...]:
        """Return the stored Wikipedia language codes.

        Returns:
            Language codes, sorted.
        """
        return tuple(sorted(self._wikipedia))


dataset_texts: DatasetTextStreamer = HubShardTextStreamer()
byte_streams: ByteStreamOpener = UrlByteStreamOpener()
reachability: ReachabilityProbe = UrlReachabilityProbe()
languages: LanguageCatalogue = RemoteLanguageCatalogue()

__all__ = [
    "SITEMATRIX_URL",
    "TIMEOUT_SECONDS",
    "ByteStreamOpener",
    "DatasetTextStreamer",
    "HubShardTextStreamer",
    "LanguageCatalogue",
    "MappingByteStreamOpener",
    "MappingDatasetTextStreamer",
    "MappingLanguageCatalogue",
    "MappingReachabilityProbe",
    "ReachabilityProbe",
    "RemoteLanguageCatalogue",
    "UrlByteStreamOpener",
    "UrlReachabilityProbe",
    "byte_streams",
    "dataset_texts",
    "languages",
    "reachability",
]
