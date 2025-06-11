"""Streaming dataset helpers for language-model training & evaluation."""

from __future__ import annotations

import itertools
import unicodedata as ud
from collections.abc import Iterable, Iterator

__all__ = ["DatasetStream"]


class DatasetStream(Iterable[str]):
    """Memory-frugal sentence iterator backed by *datasets* streaming mode.

    Parameters
    ----------
    source:
        Name of the corpus source as registered in ``turkic_translit.cli.download_corpus``.
    lang:
        ISO-639-1/3 language identifier.
    max_sentences:
        Optional hard cap on yielded sentences – handy for tests.
    """

    def __init__(
        self,
        source: str,
        lang: str,
        max_sentences: int | None = None,
    ) -> None:
        self.source = source
        self.lang = lang
        self.max_sent = max_sentences

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def __iter__(self) -> Iterator[str]:
        """Yield one NFC-normalised sentence per iteration."""
        # Import inside to avoid heavy deps for users that never touch LMs.
        from ..cli import download_corpus as dl  # local import to dodge cycles

        if self.source not in dl._REG:
            raise KeyError(
                f"Unknown source '{self.source}'. Registered: {list(dl._REG)}"
            )

        cfg = dl._REG[self.source]
        driver = dl._DRIVERS[cfg["driver"]]

        for i, line in enumerate(driver(self.lang, cfg, None)):
            if self.max_sent is not None and i >= self.max_sent:
                break
            yield ud.normalize("NFC", line)

    # Small helper used by *evaluate* metrics which expect list[str]
    def to_list(self) -> list[str]:
        return list(itertools.islice(iter(self), self.max_sent))
