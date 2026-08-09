"""Tests for the language-model sentence stream.

The stream deliberately does no language filtering of its own, so these
check that it yields exactly the normalised text the corpus drivers
produce. Filtered training data comes from a corpus that was downloaded
with a named classifier, whose manifest records which one.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from turkic_translit import _test_hooks
from turkic_translit.corpus import _test_hooks as corpus_hooks
from turkic_translit.corpus.errors import UnknownCorpusSourceError
from turkic_translit.lm.data import DatasetStream

LINES = ["  salom\tdunyo  ", "   ", "ikkinchi qator", "", "uchinchi\nqator"]


@pytest.fixture
def oscar_lines() -> Generator[None, None, None]:
    """Serve :data:`LINES` as the OSCAR ``uz`` configuration.

    Yields:
        None, once, with the original streamer captured.
    """
    original = corpus_hooks.dataset_texts
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer({"uz": LINES})
    yield
    corpus_hooks.dataset_texts = original


def test_stream_yields_normalised_non_blank_sentences(oscar_lines: None) -> None:
    """Whitespace is collapsed and blank fragments never reach the model."""
    assert list(DatasetStream("oscar-2301", "uz")) == [
        "salom dunyo",
        "ikkinchi qator",
        "uchinchi qator",
    ]


def test_stream_honours_the_sentence_cap(oscar_lines: None) -> None:
    """The cap counts sentences yielded, not fragments read."""
    assert list(DatasetStream("oscar-2301", "uz", 2)) == [
        "salom dunyo",
        "ikkinchi qator",
    ]


def test_to_list_returns_the_same_sentences(oscar_lines: None) -> None:
    """The list helper is the stream, materialised."""
    stream = DatasetStream("oscar-2301", "uz", 2)
    assert stream.to_list() == list(stream)


def test_an_unknown_source_names_the_registered_ones(oscar_lines: None) -> None:
    """A bad source id fails with the code and the valid alternatives."""
    with pytest.raises(UnknownCorpusSourceError) as excinfo:
        list(DatasetStream("oscar-2201", "uz"))
    assert excinfo.value.known == ("oscar-2301", "wikipedia")


def test_the_access_token_reaches_the_dataset_streamer() -> None:
    """A gated dataset receives the credential the environment carries."""
    original_streamer = corpus_hooks.dataset_texts
    original_environment = _test_hooks.environment
    streamer = corpus_hooks.MappingDatasetTextStreamer({"kk": ["salem alem"]})
    corpus_hooks.dataset_texts = streamer
    _test_hooks.environment = _test_hooks.MappingEnvironment({"HF_TOKEN": "secret-token"})
    try:
        sentences = DatasetStream("oscar-2301", "kk").to_list()
    finally:
        corpus_hooks.dataset_texts = original_streamer
        _test_hooks.environment = original_environment

    assert sentences == ["salem alem"]
    assert streamer.requests == [("oscar-corpus/OSCAR-2301", "kk", "secret-token")]


def test_an_absent_token_reaches_the_streamer_as_absent() -> None:
    """An ungated dataset is streamed with no credential at all."""
    original_streamer = corpus_hooks.dataset_texts
    original_environment = _test_hooks.environment
    streamer = corpus_hooks.MappingDatasetTextStreamer({"kk": ["salem alem"]})
    corpus_hooks.dataset_texts = streamer
    _test_hooks.environment = _test_hooks.MappingEnvironment({})
    try:
        sentences = DatasetStream("oscar-2301", "kk").to_list()
    finally:
        corpus_hooks.dataset_texts = original_streamer
        _test_hooks.environment = original_environment

    assert sentences == ["salem alem"]
    assert streamer.requests == [("oscar-corpus/OSCAR-2301", "kk", None)]
