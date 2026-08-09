"""Streaming sentence source for language-model training and evaluation.

Sentences come from the same drivers the corpus commands use, and are
normalised by the same function, so a language model and a downloaded
corpus see identical text for identical inputs.

No language filtering happens here. A stream that needs filtered text
should be built from a corpus that was downloaded with a named
classifier, whose manifest records which one; filtering again at read
time would produce training data whose filter is written down nowhere.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

from tqdm import tqdm

from turkic_translit import _test_hooks
from turkic_translit.corpus.drivers import stream_source
from turkic_translit.corpus.run import normalize_line
from turkic_translit.corpus.sources import get_source_spec

__all__ = ["DatasetStream"]


class DatasetStream(Iterable[str]):
    """Memory-frugal sentence iterator over a registered corpus source.

    Args:
        source: Registry key of the corpus to stream, e.g. ``oscar-2301``.
        lang: Language code within that source.
        max_sentences: Hard cap on sentences yielded, or ``None`` to read
            the source to exhaustion.
    """

    def __init__(
        self,
        source: str,
        lang: str,
        max_sentences: int | None = None,
    ) -> None:
        """Store the source, language and cap this stream will read with."""
        self.source = source
        self.lang = lang
        self.max_sent = max_sentences

    def __iter__(self) -> Iterator[str]:
        """Yield one normalised, non-empty sentence per iteration.

        Yields:
            Each sentence, whitespace-collapsed and NFC-normalised.

        Raises:
            UnknownCorpusSourceError: If the source is not registered.
            CorpusStreamError: If the source's host cannot be read.
        """
        spec = get_source_spec(self.source)
        fragments = stream_source(spec, self.lang, _test_hooks.environment.get("HF_TOKEN"))
        lines = (line for line in map(normalize_line, fragments) if line != "")
        capped = lines if self.max_sent is None else itertools.islice(lines, self.max_sent)
        yield from tqdm(
            capped,
            total=self.max_sent,
            desc=f"[data] {self.lang}",
            unit="sent",
        )

    def to_list(self) -> list[str]:
        """Read the stream into a list, honouring the sentence cap.

        Returns:
            Every sentence the stream yields, up to the cap.
        """
        return list(iter(self))
