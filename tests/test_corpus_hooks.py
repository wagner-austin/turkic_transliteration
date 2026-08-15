"""Tests for the corpus package's boundary adapters.

Nothing here is mocked. The production adapters are run for real: the
URL adapters against ``file://`` URLs, and the Hub adapters against a
repository written to disk — its file listing where the endpoint would
be, its shards where the resolver would put them, Zstandard-compressed
as the real ones are. The SiteMatrix parser runs against a response
captured from the live Wikimedia API and stored in ``tests/data``, so
the shape it parses is the shape the API actually returns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from zstandard import ZstdCompressor

from turkic_translit.corpus._test_hooks import (
    HubShardTextStreamer,
    MappingByteStreamOpener,
    MappingDatasetTextStreamer,
    MappingLanguageCatalogue,
    MappingReachabilityProbe,
    RemoteLanguageCatalogue,
    UrlByteStreamOpener,
    UrlReachabilityProbe,
    repository_files,
)
from turkic_translit.corpus.errors import ERR_STREAM_FAILED, CorpusStreamError

SITEMATRIX_FIXTURE = Path(__file__).parent / "data" / "sitematrix.json"


DATASET = "oscar-corpus/OSCAR-2301"


def build_repository(
    root: Path, shards: Mapping[str, Sequence[Mapping[str, str | None] | str]]
) -> tuple[str, str]:
    """Write a dataset repository, listing and shards, onto disk.

    The listing is served from a file whose path is what the Hub's
    endpoint would be, and the shards from files whose paths are what
    the resolver would return, so the adapter performs its real reads
    over ``file://`` rather than against a substitute.

    Args:
        root: Directory to build the repository under.
        shards: Repository path of each shard, mapped to the JSON
            documents it holds.

    Returns:
        The endpoint listing the repository, and the template resolving
        one of its files.
    """
    listing = root / "api" / DATASET
    listing.parent.mkdir(parents=True)
    listing.write_text(
        json.dumps(
            {"siblings": [{"rfilename": "README.md"}] + [{"rfilename": path} for path in shards]}
        ),
        encoding="utf-8",
    )

    for path, documents in shards.items():
        shard = root / "files" / DATASET / path
        shard.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(f"{json.dumps(document)}\n" for document in documents)
        shard.write_bytes(ZstdCompressor().compress(lines.encode("utf-8")))

    return (root / "api").as_uri(), f"{(root / 'files').as_uri()}/{{name}}/{{path}}"


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


def test_shard_streamer_reads_every_shard_in_part_order(tmp_path: Path) -> None:
    """Documents arrive shard by shard, numbered rather than alphabetised.

    Part 10 is here because it is the case text ordering gets wrong: as
    strings ``_part_10`` sorts before ``_part_2``, which would deliver
    the corpus in an order the publisher never wrote it in.

    Two lines are dropped rather than yielded: one whose ``content`` is
    null, which would otherwise become a blank line indistinguishable
    from a real but empty document, and one that is not an object at
    all, which has no content field to read.
    """
    api, resolve = build_repository(
        tmp_path,
        {
            "kk_meta/kk_meta_part_1.jsonl.zst": [
                {"content": "bir"},
                {"content": None},
                "not-an-object",
            ],
            "kk_meta/kk_meta_part_2.jsonl.zst": [{"content": "eki"}],
            "kk_meta/kk_meta_part_10.jsonl.zst": [{"content": "on"}],
        },
    )

    streamer = HubShardTextStreamer(api, resolve)

    assert list(streamer.texts(DATASET, "kk", None)) == ["bir", "eki", "on"]


def test_shard_streamer_refuses_a_language_the_repository_lacks(tmp_path: Path) -> None:
    """A language with no shards is an error, not an empty corpus."""
    api, resolve = build_repository(
        tmp_path, {"kk_meta/kk_meta_part_1.jsonl.zst": [{"content": "bir"}]}
    )

    with pytest.raises(CorpusStreamError) as excinfo:
        list(HubShardTextStreamer(api, resolve).texts(DATASET, "ky", None))

    assert excinfo.value.detail == "no shards for language ky"


def test_repository_files_reports_every_path_the_listing_names(tmp_path: Path) -> None:
    """The listing is read as the Hub returns it, README included."""
    api, _resolve = build_repository(
        tmp_path, {"kk_meta/kk_meta_part_1.jsonl.zst": [{"content": "bir"}]}
    )

    assert repository_files(f"{api}/{DATASET}") == (
        "README.md",
        "kk_meta/kk_meta_part_1.jsonl.zst",
    )


def test_repository_files_rejects_a_response_that_lists_nothing(tmp_path: Path) -> None:
    """A response of the wrong shape is an error, not an empty repository."""
    document = tmp_path / "moved.json"
    document.write_text(json.dumps({"error": "not found"}), encoding="utf-8")

    with pytest.raises(CorpusStreamError) as excinfo:
        repository_files(document.as_uri())

    assert excinfo.value.detail == "response carries no file listing"


def test_repository_files_skips_entries_carrying_no_name(tmp_path: Path) -> None:
    """A malformed sibling is skipped, and the well-formed ones survive."""
    document = tmp_path / "partial.json"
    document.write_text(
        json.dumps({"siblings": ["not-a-mapping", {"size": 12}, {"rfilename": "kk_meta/a.zst"}]}),
        encoding="utf-8",
    )

    assert repository_files(document.as_uri()) == ("kk_meta/a.zst",)


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


def test_remote_catalogue_lists_the_languages_the_repository_holds(tmp_path: Path) -> None:
    """Languages come from the shard directories, not from a loading script.

    The README and the checksum file are in the listing and are not
    languages, so the codes are taken from the shards themselves.
    """
    api, _resolve = build_repository(
        tmp_path,
        {
            "kk_meta/kk_meta_part_1.jsonl.zst": [{"content": "bir"}],
            "ky_meta/ky_meta.jsonl.zst": [{"content": "bir"}],
            "kk_meta/checksum.sha256": [],
        },
    )

    catalogue = RemoteLanguageCatalogue(SITEMATRIX_FIXTURE.as_uri(), api)

    assert catalogue.oscar_configurations(DATASET) == ("kk", "ky")


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
