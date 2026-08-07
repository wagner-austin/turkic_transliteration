"""Injection seam for the network and dataset boundaries.

Production binds each hook to its real adapter at import time and never
rebinds it. Tests bind them to in-memory implementations that satisfy the
same protocols. The drivers call the hooks unconditionally, so no
production branch exists purely to support testing.

Every real adapter here reaches the outside world through ``urllib`` or
through the ``datasets`` package rather than through ``requests``. That is
deliberate: ``urllib`` handles ``file://``, and ``datasets`` reads a local
directory, so the adapters can be exercised for real — not stubbed —
without a network.

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

from turkic_translit.corpus.errors import CorpusStreamError
from turkic_translit.net import build_request

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS: Final = 30.0

SITEMATRIX_URL: Final = (
    "https://meta.wikimedia.org/w/api.php?"
    "action=sitematrix&format=json&smtype=language&smsiteprop=code|closed"
)


class DatasetTextStreamer(Protocol):
    """Reader yielding the text column of a streamed dataset."""

    def texts(self, dataset_name: str, configuration: str, token: str | None) -> Iterator[str]:
        """Stream the ``text`` field of every row that has one.

        Args:
            dataset_name: Dataset identifier, e.g.
                ``oscar-corpus/OSCAR-2301``.
            configuration: Configuration within that dataset, which for
                OSCAR is the language code.
            token: Access token for gated datasets, or ``None``.

        Yields:
            Each row's text, exactly as stored.
        """
        ...


class HuggingFaceDatasetTextStreamer:
    """Streamer backed by the ``datasets`` package.

    The import is deferred to call time so that installing this project
    without the corpus extra still imports. Rows arrive untyped, so the
    text field is checked before it is yielded: a row whose ``text`` is
    null is skipped rather than being turned into an empty line that
    would later look like a real but blank document.
    """

    def texts(self, dataset_name: str, configuration: str, token: str | None) -> Iterator[str]:
        """Stream one configuration of a dataset in streaming mode.

        Args:
            dataset_name: Dataset identifier or a local directory path.
            configuration: Configuration within that dataset.
            token: Access token for gated datasets, or ``None``.

        Yields:
            Each row's text field, skipping rows that carry no text.
        """
        module = __import__("datasets")
        dataset = module.load_dataset(
            dataset_name,
            configuration,
            split="train",
            streaming=True,
            trust_remote_code=True,
            token=token,
        )
        for row in dataset:
            value = row["text"]
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

    def __init__(self, sitematrix_url: str = SITEMATRIX_URL) -> None:
        """Store the SiteMatrix endpoint this catalogue queries."""
        self._sitematrix_url = sitematrix_url

    def oscar_configurations(self, dataset_name: str) -> tuple[str, ...]:
        """List a dataset's configurations via the ``datasets`` package.

        Args:
            dataset_name: Dataset identifier or a local directory path.

        Returns:
            Configuration names, sorted.
        """
        module = __import__("datasets")
        configurations: Sequence[str] = module.get_dataset_config_names(
            dataset_name, trust_remote_code=True
        )
        return tuple(sorted(configurations))

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


dataset_texts: DatasetTextStreamer = HuggingFaceDatasetTextStreamer()
byte_streams: ByteStreamOpener = UrlByteStreamOpener()
reachability: ReachabilityProbe = UrlReachabilityProbe()
languages: LanguageCatalogue = RemoteLanguageCatalogue()

__all__ = [
    "SITEMATRIX_URL",
    "TIMEOUT_SECONDS",
    "ByteStreamOpener",
    "DatasetTextStreamer",
    "HuggingFaceDatasetTextStreamer",
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
