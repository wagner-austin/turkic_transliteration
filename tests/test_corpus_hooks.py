"""Tests for the corpus package's boundary adapters.

Nothing here is mocked. The production adapters are run for real: the
URL adapters against ``file://`` URLs, and the dataset adapters against a
local directory that the ``datasets`` package loads without a network.
The SiteMatrix parser runs against a response captured from the live
Wikimedia API and stored in ``tests/data``, so the shape it parses is the
shape the API actually returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turkic_translit.corpus._test_hooks import (
    HuggingFaceDatasetTextStreamer,
    MappingByteStreamOpener,
    MappingDatasetTextStreamer,
    MappingLanguageCatalogue,
    MappingReachabilityProbe,
    RemoteLanguageCatalogue,
    UrlByteStreamOpener,
    UrlReachabilityProbe,
)
from turkic_translit.corpus.errors import ERR_STREAM_FAILED, CorpusStreamError

SITEMATRIX_FIXTURE = Path(__file__).parent / "data" / "sitematrix.json"


@pytest.fixture
def local_dataset(tmp_path: Path) -> Path:
    """Build a one-file dataset directory the ``datasets`` package can read.

    One row carries a null text field, which is the case the streamer has
    to drop rather than turn into an empty line.

    Args:
        tmp_path: Directory to build the dataset in.

    Returns:
        Path of the dataset directory.
    """
    directory = tmp_path / "dataset"
    directory.mkdir()
    rows = [{"text": "salom dunyo"}, {"text": None}, {"text": "ikkinchi qator"}]
    (directory / "train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return directory


def test_url_opener_streams_a_file_url(tmp_path: Path) -> None:
    """The production opener reads real bytes from a real URL."""
    payload = tmp_path / "dump.bin"
    payload.write_bytes(b"salom dunyo")
    with UrlByteStreamOpener().open(payload.as_uri()) as stream:
        assert stream.read() == b"salom dunyo"


def test_url_opener_reports_a_missing_target_with_its_code(tmp_path: Path) -> None:
    """An unreadable URL raises the stream code naming the URL."""
    missing = (tmp_path / "absent.bin").as_uri()
    with pytest.raises(CorpusStreamError) as excinfo:
        UrlByteStreamOpener().open(missing)
    assert excinfo.value.code == ERR_STREAM_FAILED
    assert excinfo.value.url == missing


def test_mapping_opener_serves_registered_bytes() -> None:
    """The in-memory opener returns the bytes filed under the URL."""
    opener = MappingByteStreamOpener({"https://example.invalid/a": b"data"})
    with opener.open("https://example.invalid/a") as stream:
        assert stream.read() == b"data"
    assert opener.requests == ["https://example.invalid/a"]


def test_mapping_opener_refuses_an_unregistered_url() -> None:
    """A URL with no payload is an error, not an empty stream."""
    with pytest.raises(CorpusStreamError) as excinfo:
        MappingByteStreamOpener({}).open("https://example.invalid/b")
    assert excinfo.value.detail == "no payload registered for this URL"


def test_url_probe_reports_a_real_file_as_reachable(tmp_path: Path) -> None:
    """The production probe answers True for a target it can retrieve."""
    target = tmp_path / "present.bin"
    target.write_bytes(b"x")
    assert UrlReachabilityProbe().reachable(target.as_uri()) is True


def test_url_probe_reports_a_missing_target_as_unreachable(tmp_path: Path) -> None:
    """The production probe answers False rather than raising."""
    assert UrlReachabilityProbe().reachable((tmp_path / "absent.bin").as_uri()) is False


def test_mapping_probe_answers_from_its_set_and_logs_the_probe() -> None:
    """The in-memory probe answers from membership and records requests."""
    probe = MappingReachabilityProbe(["https://example.invalid/up"])
    assert probe.reachable("https://example.invalid/up") is True
    assert probe.reachable("https://example.invalid/down") is False
    assert probe.requests == [
        "https://example.invalid/up",
        "https://example.invalid/down",
    ]


def test_dataset_streamer_reads_a_local_dataset(local_dataset: Path) -> None:
    """The production streamer yields text and drops rows that carry none."""
    streamer = HuggingFaceDatasetTextStreamer()
    assert list(streamer.texts(str(local_dataset), "default", None)) == [
        "salom dunyo",
        "ikkinchi qator",
    ]


def test_mapping_streamer_serves_a_configuration_and_records_the_call() -> None:
    """The in-memory streamer yields its table and logs what was asked."""
    streamer = MappingDatasetTextStreamer({"kk": ["bir", "eki"]})
    assert list(streamer.texts("oscar-corpus/OSCAR-2301", "kk", "token")) == [
        "bir",
        "eki",
    ]
    assert streamer.requests == [("oscar-corpus/OSCAR-2301", "kk", "token")]


def test_mapping_streamer_refuses_an_unknown_configuration() -> None:
    """An unlisted configuration raises rather than yielding nothing."""
    streamer = MappingDatasetTextStreamer({"kk": ["bir"]})
    with pytest.raises(CorpusStreamError) as excinfo:
        list(streamer.texts("oscar-corpus/OSCAR-2301", "ky", None))
    assert excinfo.value.detail == "no configuration in the table"


def test_remote_catalogue_lists_a_local_dataset_configuration(
    local_dataset: Path,
) -> None:
    """The production catalogue enumerates configurations without a network."""
    catalogue = RemoteLanguageCatalogue()
    assert catalogue.oscar_configurations(str(local_dataset)) == ("default",)


def test_remote_catalogue_excludes_closed_wikipedia_editions() -> None:
    """A captured SiteMatrix yields open editions and drops closed ones.

    ``aa`` is the closed case in the fixture: its ``wiki`` entry carries a
    ``closed`` key whose value is the empty string. Testing truthiness
    rather than membership is what previously let it through.
    """
    catalogue = RemoteLanguageCatalogue(SITEMATRIX_FIXTURE.as_uri())
    assert catalogue.wikipedia_editions() == (
        "ab",
        "az",
        "en",
        "fi",
        "kk",
        "ky",
        "ru",
        "tr",
        "ug",
        "uz",
    )


def test_remote_catalogue_rejects_a_response_without_a_sitematrix(
    tmp_path: Path,
) -> None:
    """An API response of the wrong shape is an error, not an empty list."""
    document = tmp_path / "bad.json"
    document.write_text(json.dumps({"error": "quota"}), encoding="utf-8")
    catalogue = RemoteLanguageCatalogue(document.as_uri())
    with pytest.raises(CorpusStreamError) as excinfo:
        catalogue.wikipedia_editions()
    assert excinfo.value.detail == "response carries no sitematrix mapping"


def test_remote_catalogue_skips_entries_it_cannot_read(tmp_path: Path) -> None:
    """Non-block keys and malformed blocks are skipped, not fatal.

    ``count`` and ``specials`` are documented siblings of the numbered
    blocks, and a block missing its code or its site list carries nothing
    to extract, so the only entry that survives here is the well-formed
    one.
    """
    document = tmp_path / "partial.json"
    document.write_text(
        json.dumps(
            {
                "sitematrix": {
                    "count": 4,
                    "specials": [{"code": "commons"}],
                    "0": "not-a-block",
                    "1": {"code": 17, "site": [{"code": "wiki"}]},
                    "2": {"code": "xx", "site": "not-a-list"},
                    "3": {"code": "kk", "site": [{"code": "wiki"}]},
                }
            }
        ),
        encoding="utf-8",
    )
    catalogue = RemoteLanguageCatalogue(document.as_uri())
    assert catalogue.wikipedia_editions() == ("kk",)


def test_remote_catalogue_ignores_sister_projects(tmp_path: Path) -> None:
    """A language with a wiktionary but no Wikipedia is not listed."""
    document = tmp_path / "sisters.json"
    document.write_text(
        json.dumps({"sitematrix": {"0": {"code": "xx", "site": [{"code": "wiktionary"}]}}}),
        encoding="utf-8",
    )
    assert RemoteLanguageCatalogue(document.as_uri()).wikipedia_editions() == ()


def test_mapping_catalogue_answers_both_questions_from_its_tables() -> None:
    """The in-memory catalogue sorts what it was given."""
    catalogue = MappingLanguageCatalogue({"ds": ["ky", "kk"]}, ["uz", "tr"])
    assert catalogue.oscar_configurations("ds") == ("kk", "ky")
    assert catalogue.wikipedia_editions() == ("tr", "uz")


def test_mapping_catalogue_refuses_an_unknown_dataset() -> None:
    """An unlisted dataset raises rather than reporting no languages."""
    with pytest.raises(CorpusStreamError) as excinfo:
        MappingLanguageCatalogue({}, []).oscar_configurations("absent")
    assert excinfo.value.detail == "no dataset in the table"
