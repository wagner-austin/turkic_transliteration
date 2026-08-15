"""Tests for a whole corpus run.

These check the property the package exists for: the file on disk and the
manifest beside it describe the same run, and the manifest names the
classifier that actually judged the lines in that file.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from turkic_translit.corpus import _test_hooks as corpus_hooks
from turkic_translit.corpus.errors import UnknownCorpusSourceError
from turkic_translit.corpus.filtering import LidFilterRequest
from turkic_translit.corpus.manifest import (
    manifest_path_for,
    read_corpus_run_manifest,
)
from turkic_translit.corpus.run import PROGRESS_INTERVAL, download_corpus
from turkic_translit.lid import _test_hooks as lid_hooks
from turkic_translit.lid.locations import default_search_dirs

LINES = [
    "salom dunyo",
    "   ",
    "privet mir",
    "ikkinchi qator",
    "",
    "uchinchi\tqator",
]

ANSWERS = {
    "salom dunyo": [("__label__uzn_Latn", 0.99)],
    "privet mir": [("__label__rus_Cyrl", 0.99)],
    "ikkinchi qator": [("__label__uzn_Latn", 0.97)],
    "uchinchi qator": [("__label__uzn_Latn", 0.60)],
}


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


@pytest.fixture
def installed_weights() -> Generator[None, None, None]:
    """Present ``lid218e`` as installed and back it with a table model.

    Yields:
        None, once, with the original hooks captured.
    """
    probe, loader = lid_hooks.probe, lid_hooks.model_loader
    lid_hooks.probe = lid_hooks.MappingFileProbe(
        {default_search_dirs()[0] / "lid218e.bin": 1176355829}
    )
    lid_hooks.model_loader = lid_hooks.FixedModelLoader(lid_hooks.TableFastTextModel(ANSWERS))
    yield
    lid_hooks.probe = probe
    lid_hooks.model_loader = loader


def test_the_run_normalizes_and_folds_each_line_as_it_arrives(tmp_path: Path) -> None:
    """A downloaded corpus is born normalised and misencoding-repaired.

    The Turkish fragment carries a soft hyphen inside a word and the
    cp1254 mojibake the raw corpus was measured to hold; the file on
    disk must carry neither.
    """
    original = corpus_hooks.dataset_texts
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer(
        {"tr": ["sat\u00fdn al\u00add\u0131  \t \u00feekilde"]}
    )
    try:
        output = tmp_path / "tr.txt"
        download_corpus("oscar-2301", "tr", output, None, None, None)
    finally:
        corpus_hooks.dataset_texts = original

    assert output.read_text(encoding="utf-8") == "sat\u0131n ald\u0131 \u015fekilde\n"


def test_unfiltered_run_writes_every_non_blank_line(oscar_lines: None, tmp_path: Path) -> None:
    """Blank fragments are dropped and everything else is kept."""
    output = tmp_path / "uz.txt"

    manifest, manifest_file = download_corpus("oscar-2301", "uz", output, None, None, None)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "salom dunyo",
        "privet mir",
        "ikkinchi qator",
        "uchinchi qator",
    ]
    assert manifest["lines_seen"] == 4
    assert manifest["lines_written"] == 4
    assert manifest["filter_language"] is None
    assert manifest["language_identification"] is None
    assert manifest_file == manifest_path_for(output)


def test_unfiltered_run_records_the_source_it_read(oscar_lines: None, tmp_path: Path) -> None:
    """The manifest carries the source, driver, licence and language."""
    manifest, _path = download_corpus("oscar-2301", "uz", tmp_path / "uz.txt", None, None, None)
    assert manifest["source_id"] == "oscar-2301"
    assert manifest["driver"] == "oscar"
    assert manifest["license"] == "CC0-1.0"
    assert manifest["language"] == "uz"


def test_filtered_run_keeps_only_confident_lines_of_the_language(
    oscar_lines: None, installed_weights: None, tmp_path: Path
) -> None:
    """Russian is dropped by label and low confidence is dropped by threshold."""
    output = tmp_path / "uz.txt"

    manifest, _path = download_corpus(
        "oscar-2301",
        "uz",
        output,
        None,
        None,
        LidFilterRequest(language="uzn", model_id="lid218e", threshold=0.95),
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "salom dunyo",
        "ikkinchi qator",
    ]
    assert manifest["lines_seen"] == 4
    assert manifest["lines_written"] == 2


def test_filtered_run_names_the_classifier_in_the_manifest_on_disk(
    oscar_lines: None, installed_weights: None, tmp_path: Path
) -> None:
    """A corpus and its manifest together identify the filter that made it.

    This is the whole point of the package: reading the manifest back
    tells a later session exactly which weights, at which threshold,
    decided the contents of the file beside it.
    """
    output = tmp_path / "uz.txt"

    _manifest, manifest_file = download_corpus(
        "oscar-2301",
        "uz",
        output,
        None,
        None,
        LidFilterRequest(language="uzn", model_id="lid218e", threshold=0.95),
    )

    stored = read_corpus_run_manifest(manifest_file)
    assert stored["filter_language"] == "uzn"
    assert stored["language_identification"] == {
        "model_id": "lid218e",
        "weights_path": str(default_search_dirs()[0] / "lid218e.bin"),
        "weights_bytes": 1176355829,
        "threshold": 0.95,
        "script_aware": True,
    }
    assert stored["output_path"] == str(output)


def test_max_lines_stops_the_run_at_the_requested_count(oscar_lines: None, tmp_path: Path) -> None:
    """The cap counts lines written, and the manifest agrees with the file."""
    output = tmp_path / "uz.txt"

    manifest, _path = download_corpus("oscar-2301", "uz", output, 2, None, None)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "salom dunyo",
        "privet mir",
    ]
    assert manifest["lines_written"] == 2


def test_the_run_creates_missing_parent_directories(oscar_lines: None, tmp_path: Path) -> None:
    """An output path inside a new directory tree is written, not refused."""
    output = tmp_path / "corpora" / "uz" / "train.txt"
    download_corpus("oscar-2301", "uz", output, 1, None, None)
    assert output.read_text(encoding="utf-8") == "salom dunyo\n"


def test_a_long_run_reports_progress_and_still_counts_correctly(
    tmp_path: Path,
) -> None:
    """A run past the progress interval keeps its counts exact.

    The interval is crossed twice here, which is what exercises the
    periodic progress report alongside the ordinary path.
    """
    total = PROGRESS_INTERVAL * 2
    original = corpus_hooks.dataset_texts
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer(
        {"kk": [f"line {index}" for index in range(total)]}
    )
    try:
        output = tmp_path / "kk.txt"
        manifest, _path = download_corpus("oscar-2301", "kk", output, None, None, None)
    finally:
        corpus_hooks.dataset_texts = original

    assert manifest["lines_written"] == total
    assert output.read_text(encoding="utf-8").count("\n") == total


def test_an_unknown_source_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    """A bad source id fails without leaving a partial file behind."""
    output = tmp_path / "nope.txt"
    with pytest.raises(UnknownCorpusSourceError) as excinfo:
        download_corpus("oscar-2201", "uz", output, None, None, None)
    assert excinfo.value.source_id == "oscar-2201"
    assert output.exists() is False


def test_the_access_token_reaches_the_dataset_streamer(tmp_path: Path) -> None:
    """A gated dataset receives the credential it was given."""
    original = corpus_hooks.dataset_texts
    streamer = corpus_hooks.MappingDatasetTextStreamer({"uz": ["salom dunyo"]})
    corpus_hooks.dataset_texts = streamer
    try:
        download_corpus("oscar-2301", "uz", tmp_path / "uz.txt", None, "hf-token", None)
    finally:
        corpus_hooks.dataset_texts = original

    assert streamer.requests == [("oscar-corpus/OSCAR-2301", "uz", "hf-token")]
