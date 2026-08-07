"""Tests for the corpus drivers.

The Wikipedia driver is run against genuine bzip2-compressed MediaWiki
export XML built in the test, so the decompression, the incremental
parse, the markup stripping and the entity unescaping all execute. The
only substituted component is the byte-stream opener, itself a real
implementation of the production protocol.
"""

from __future__ import annotations

import bz2
from collections.abc import Generator

import pytest

from turkic_translit.corpus import _test_hooks
from turkic_translit.corpus.drivers import (
    stream_source,
    stream_wikipedia,
    wikipedia_dump_url,
)
from turkic_translit.corpus.errors import CorpusStreamError
from turkic_translit.corpus.sources import get_source_spec

EXPORT_XML = """<?xml version="1.0" encoding="utf-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Qazaqstan</title>
    <revision>
      <text xml:space="preserve">Qazaqstan &amp;mdash; &lt;b&gt;memleket&lt;/b&gt;. Astana astanasy! Üshinshi söylem?</text>
    </revision>
  </page>
  <page>
    <title>Bos bet</title>
    <revision>
      <text xml:space="preserve" />
    </revision>
  </page>
</mediawiki>
"""


@pytest.fixture
def restore_hooks() -> Generator[None, None, None]:
    """Restore the production hooks after a test rebinds them.

    Yields:
        None, once, with the originals captured.
    """
    streams, texts = _test_hooks.byte_streams, _test_hooks.dataset_texts
    yield
    _test_hooks.byte_streams = streams
    _test_hooks.dataset_texts = texts


def test_dump_url_names_the_latest_articles_export() -> None:
    """The URL points at the language's newest pages-articles dump."""
    assert wikipedia_dump_url("kk") == (
        "https://dumps.wikimedia.org/kkwiki/latest/kkwiki-latest-pages-articles.xml.bz2"
    )


def test_wikipedia_driver_splits_unescaped_text_into_fragments(
    restore_hooks: None,
) -> None:
    """Markup is dropped, entities resolved, and sentences split apart.

    The wikitext is escaped inside the XML exactly as a real dump escapes
    it, so the ``<b>`` tag reaches the driver as literal text and is
    stripped there, and ``&amp;mdash;`` survives XML decoding as
    ``&mdash;`` for the HTML unescape to resolve.

    The empty trailing fragment and the leading spaces are left in place
    deliberately: the driver extracts and does not normalise, so that
    normalisation happens once, in the run layer.
    """
    opener = _test_hooks.MappingByteStreamOpener(
        {wikipedia_dump_url("kk"): bz2.compress(EXPORT_XML.encode("utf-8"))}
    )
    _test_hooks.byte_streams = opener

    assert list(stream_wikipedia("kk")) == [
        "Qazaqstan —  memleket ",
        " Astana astanasy",
        " Üshinshi söylem",
        "",
    ]
    assert opener.requests == [wikipedia_dump_url("kk")]


def test_wikipedia_driver_yields_nothing_for_an_empty_dump(
    restore_hooks: None,
) -> None:
    """A dump whose only page has no text produces no fragments."""
    empty_export = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
        "<page><title>Bos</title><revision><text /></revision></page>"
        "</mediawiki>\n"
    )
    _test_hooks.byte_streams = _test_hooks.MappingByteStreamOpener(
        {wikipedia_dump_url("ky"): bz2.compress(empty_export.encode("utf-8"))}
    )
    assert list(stream_wikipedia("ky")) == []


def test_wikipedia_driver_surfaces_an_unreadable_dump(restore_hooks: None) -> None:
    """A dump host that will not answer raises rather than yielding nothing."""
    _test_hooks.byte_streams = _test_hooks.MappingByteStreamOpener({})
    with pytest.raises(CorpusStreamError) as excinfo:
        list(stream_wikipedia("tr"))
    assert excinfo.value.url == wikipedia_dump_url("tr")


def test_stream_source_routes_an_oscar_spec_to_the_dataset(
    restore_hooks: None,
) -> None:
    """An OSCAR source streams its dataset under the requested language."""
    streamer = _test_hooks.MappingDatasetTextStreamer({"uz": ["salom", "dunyo"]})
    _test_hooks.dataset_texts = streamer

    lines = list(stream_source(get_source_spec("oscar-2301"), "uz", "hf-token"))

    assert lines == ["salom", "dunyo"]
    assert streamer.requests == [("oscar-corpus/OSCAR-2301", "uz", "hf-token")]


def test_stream_source_routes_a_wikipedia_spec_to_the_dump(
    restore_hooks: None,
) -> None:
    """A Wikipedia source reads the dump and ignores any access token."""
    _test_hooks.byte_streams = _test_hooks.MappingByteStreamOpener(
        {wikipedia_dump_url("az"): bz2.compress(EXPORT_XML.encode("utf-8"))}
    )
    lines = list(stream_source(get_source_spec("wikipedia"), "az", "unused-token"))
    assert lines[0] == "Qazaqstan —  memleket "
