"""Reading raw text out of each kind of corpus source.

A driver's only job is extraction: it yields whatever text fragments the
source contains, in source order, and does nothing else. It does not
normalise, does not drop blanks, and above all does not filter by
language. Filtering lives in :mod:`turkic_translit.corpus.run`, in one
place, so that the classifier that decided a corpus's contents is the one
the manifest records. The previous implementation filtered inside each
driver *and* again in the CLI loop, which meant a line could be judged
twice by two separately constructed models.
"""

from __future__ import annotations

import bz2
import html
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import Final

from turkic_translit.corpus import _test_hooks
from turkic_translit.corpus.sources import OscarSourceSpec, WikipediaSourceSpec

_MARKUP = re.compile(r"(?s)<.*?>")
_SENTENCE_BREAK = re.compile(r"[.!?]")

WIKIPEDIA_DUMP_HOST: Final = "https://dumps.wikimedia.org"


def wikipedia_dump_url(language: str) -> str:
    """Build the URL of a language's latest articles dump.

    Args:
        language: Wikipedia language code, e.g. ``kk``.

    Returns:
        The canonical location of that edition's newest
        ``pages-articles`` dump.
    """
    return (
        f"{WIKIPEDIA_DUMP_HOST}/{language}wiki/latest/{language}wiki-latest-pages-articles.xml.bz2"
    )


def stream_wikipedia(language: str) -> Iterator[str]:
    """Stream sentence-like fragments from a Wikipedia XML dump.

    The dump is decompressed as it arrives and parsed incrementally, so
    memory stays flat across a multi-gigabyte file. Each article's
    wikitext has its markup removed, its entities unescaped, and is split
    on sentence-final punctuation.

    Args:
        language: Wikipedia language code, e.g. ``kk``.

    Yields:
        Text fragments, unnormalised and possibly empty.

    Raises:
        CorpusStreamError: If the dump host cannot be read.
    """
    with _test_hooks.byte_streams.open(wikipedia_dump_url(language)) as raw:
        for _event, element in ET.iterparse(bz2.BZ2File(raw), events=("end",)):
            if element.tag.endswith("}text") and element.text is not None:
                yield from _SENTENCE_BREAK.split(html.unescape(_MARKUP.sub(" ", element.text)))
            element.clear()


def stream_source(
    spec: OscarSourceSpec | WikipediaSourceSpec,
    language: str,
    token: str | None,
) -> Iterator[str]:
    """Stream raw text from whichever source ``spec`` describes.

    Args:
        spec: The source to read, which fixes the driver used.
        language: Language code within that source. For a Hugging Face
            dataset this is the configuration name; for Wikipedia it
            selects the edition.
        token: Access token for gated datasets, or ``None``. Ignored by
            the Wikipedia driver, whose dumps need no credentials.

    Returns:
        An iterator of raw text fragments.
    """
    if spec["driver"] == "oscar":
        return _test_hooks.dataset_texts.texts(spec["hf_name"], language, token)
    return stream_wikipedia(language)


__all__ = [
    "WIKIPEDIA_DUMP_HOST",
    "stream_source",
    "stream_wikipedia",
    "wikipedia_dump_url",
]
