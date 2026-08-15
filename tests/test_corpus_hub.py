"""Tests for OSCAR's shard-naming rules.

Every function under test is pure, so the file lists below are written
out rather than fetched. They are shaped like the real listing: one
directory per language, a checksum beside each language's shards, and a
README at the top, all of which the real repository carries.
"""

from __future__ import annotations

from turkic_translit.corpus import hub

LISTING: tuple[str, ...] = (
    "README.md",
    ".gitattributes",
    "af_meta/af_meta.jsonl.zst",
    "af_meta/checksum.sha256",
    "kk_meta/kk_meta_part_1.jsonl.zst",
    "kk_meta/kk_meta_part_2.jsonl.zst",
    "kk_meta/kk_meta_part_10.jsonl.zst",
    "kk_meta/checksum.sha256",
    "ky_meta/ky_meta.jsonl.zst",
)


def test_the_endpoint_names_the_repository() -> None:
    """A dataset's listing lives under its own name on the API host."""
    assert (
        hub.dataset_api_url("oscar-corpus/OSCAR-2301")
        == "https://huggingface.co/api/datasets/oscar-corpus/OSCAR-2301"
    )


def test_languages_come_from_the_directories_holding_shards() -> None:
    """A language is one that has data, not one that has a directory."""
    assert hub.languages(LISTING) == ("af", "kk", "ky")


def test_a_repository_with_no_shards_offers_no_languages() -> None:
    """Documentation alone is not a corpus."""
    assert hub.languages(("README.md", "LICENSE")) == ()


def test_shards_are_ordered_by_part_number_not_by_name() -> None:
    """Part 10 follows part 2, which text ordering gets backwards."""
    assert hub.shard_paths(LISTING, "kk") == (
        "kk_meta/kk_meta_part_1.jsonl.zst",
        "kk_meta/kk_meta_part_2.jsonl.zst",
        "kk_meta/kk_meta_part_10.jsonl.zst",
    )


def test_a_checksum_is_not_a_shard() -> None:
    """Only the compressed JSON lines are read, whatever else is filed."""
    assert hub.shard_paths(LISTING, "af") == ("af_meta/af_meta.jsonl.zst",)


def test_a_language_the_repository_lacks_has_no_shards() -> None:
    """An unheld language selects nothing rather than everything."""
    assert hub.shard_paths(LISTING, "xx") == ()


def test_a_languages_shards_exclude_another_languages() -> None:
    """The directory prefix is exact, so ``kk`` never catches ``kk2``."""
    listing = ("kk_meta/kk_meta.jsonl.zst", "kk2_meta/kk2_meta.jsonl.zst")

    assert hub.shard_paths(listing, "kk") == ("kk_meta/kk_meta.jsonl.zst",)
