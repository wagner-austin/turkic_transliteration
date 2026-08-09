"""Tests for the corpus-download command line.

Every test drives the real Click surface. The network is absent, not
mocked out: the dataset streamer, the reachability probe and the language
catalogue are bound to in-memory implementations of the same protocols
production uses, so option parsing, validation, the run itself and the
manifest write all execute for real.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from click.testing import CliRunner

from turkic_translit import _test_hooks
from turkic_translit.cli import download_corpus as cli_module
from turkic_translit.corpus import _test_hooks as corpus_hooks
from turkic_translit.corpus.catalogue import health_check_url
from turkic_translit.corpus.manifest import manifest_path_for
from turkic_translit.corpus.sources import SOURCE_REGISTRY
from turkic_translit.lid import _test_hooks as lid_hooks
from turkic_translit.lid.locations import default_search_dirs

ANSWERS = {
    "salom dunyo": [("__label__uzn_Latn", 0.99)],
    "privet mir": [("__label__rus_Cyrl", 0.99)],
}


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click runner with a UTF-8 environment.

    Returns:
        The runner every command test drives.
    """
    return CliRunner(env={"PYTHONIOENCODING": "utf8", "HF_TOKEN": ""})


@pytest.fixture
def corpus_lines() -> Generator[None, None, None]:
    """Serve two lines as the OSCAR ``uz`` configuration.

    Yields:
        None, once, with the original streamer captured.
    """
    original = corpus_hooks.dataset_texts
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer(
        {"uz": ["salom dunyo", "privet mir"]}
    )
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


def test_list_sources_reports_driver_and_licence(runner: CliRunner) -> None:
    """Each registered source is printed with what reads it and its licence."""
    result = runner.invoke(cli_module.cli, ["list-sources"])
    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "oscar-2301    oscar       CC0-1.0",
        "wikipedia     wikipedia   CC-BY-SA-3.0",
    ]


def test_license_reports_the_registered_licence(runner: CliRunner) -> None:
    """The licence command answers from the registry."""
    result = runner.invoke(cli_module.cli, ["license", "--source", "wikipedia"])
    assert result.exit_code == 0
    assert result.output == "wikipedia: CC-BY-SA-3.0\n"


def test_list_langs_enumerates_a_dataset_configuration(runner: CliRunner) -> None:
    """An OSCAR source lists the dataset's configuration names."""
    original = corpus_hooks.languages
    corpus_hooks.languages = corpus_hooks.MappingLanguageCatalogue(
        {"oscar-corpus/OSCAR-2301": ["uz", "kk"]}, []
    )
    try:
        result = runner.invoke(cli_module.cli, ["list-langs", "--source", "oscar-2301"])
    finally:
        corpus_hooks.languages = original

    assert result.exit_code == 0
    assert result.output == "kk uz\n"


def test_list_langs_enumerates_open_wikipedia_editions(runner: CliRunner) -> None:
    """A Wikipedia source lists the editions that are open."""
    original = corpus_hooks.languages
    corpus_hooks.languages = corpus_hooks.MappingLanguageCatalogue({}, ["tr", "az"])
    try:
        result = runner.invoke(cli_module.cli, ["list-langs", "--source", "wikipedia"])
    finally:
        corpus_hooks.languages = original

    assert result.exit_code == 0
    assert result.output == "az tr\n"


def test_doctor_reports_every_source_reachable(runner: CliRunner) -> None:
    """When both hosts answer, nothing is named as broken."""
    original = corpus_hooks.reachability
    corpus_hooks.reachability = corpus_hooks.MappingReachabilityProbe(
        [health_check_url(spec) for spec in SOURCE_REGISTRY.values()]
    )
    try:
        result = runner.invoke(cli_module.cli, ["doctor"])
    finally:
        corpus_hooks.reachability = original

    assert result.exit_code == 0
    assert result.output == "All sources reachable\n"


def test_doctor_names_the_sources_that_did_not_answer(runner: CliRunner) -> None:
    """An unreachable host is named, and only the unreachable one."""
    original = corpus_hooks.reachability
    corpus_hooks.reachability = corpus_hooks.MappingReachabilityProbe(
        [health_check_url(SOURCE_REGISTRY["wikipedia"])]
    )
    try:
        result = runner.invoke(cli_module.cli, ["doctor"])
    finally:
        corpus_hooks.reachability = original

    assert result.exit_code == 0
    assert result.output == "Unreachable sources: oscar-2301\n"


def test_download_without_a_filter_writes_every_line(
    runner: CliRunner, corpus_lines: None, tmp_path: Path
) -> None:
    """An unfiltered run keeps both lines and says so on stdout."""
    output = tmp_path / "uz.txt"

    result = runner.invoke(
        cli_module.cli,
        ["download", "--source", "oscar-2301", "--lang", "uz", "--out", str(output)],
    )

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8").splitlines() == [
        "salom dunyo",
        "privet mir",
    ]
    assert "2 lines of 2 seen" in result.output


def test_download_writes_a_manifest_beside_the_corpus(
    runner: CliRunner, corpus_lines: None, tmp_path: Path
) -> None:
    """Every run leaves a manifest, filtered or not."""
    output = tmp_path / "uz.txt"

    result = runner.invoke(
        cli_module.cli,
        ["download", "--lang", "uz", "--out", str(output), "--max-lines", "1"],
    )

    assert result.exit_code == 0
    stored = json.loads(manifest_path_for(output).read_text(encoding="utf-8"))
    assert stored["lines_written"] == 1
    assert stored["language_identification"] is None


def test_download_with_a_filter_records_the_model_that_applied_it(
    runner: CliRunner, corpus_lines: None, installed_weights: None, tmp_path: Path
) -> None:
    """Naming a classifier both filters the corpus and enters the manifest.

    This is the behaviour the option exists for: the file holds only the
    Uzbek line, and the manifest beside it names ``lid218e`` and the
    threshold that kept it.
    """
    output = tmp_path / "uz.txt"

    result = runner.invoke(
        cli_module.cli,
        [
            "download",
            "--lang",
            "uz",
            "--out",
            str(output),
            "--filter-langid",
            "uzn",
            "--lid-model",
            "lid218e",
        ],
    )

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8") == "salom dunyo\n"
    stored = json.loads(manifest_path_for(output).read_text(encoding="utf-8"))
    assert stored["filter_language"] == "uzn"
    assert stored["language_identification"]["model_id"] == "lid218e"
    assert stored["language_identification"]["threshold"] == 0.95


def test_a_stricter_threshold_is_recorded_as_given(
    runner: CliRunner, corpus_lines: None, installed_weights: None, tmp_path: Path
) -> None:
    """The threshold in the manifest is the one the run actually applied."""
    output = tmp_path / "uz.txt"

    runner.invoke(
        cli_module.cli,
        [
            "download",
            "--lang",
            "uz",
            "--out",
            str(output),
            "--filter-langid",
            "uzn",
            "--lid-model",
            "lid218e",
            "--lid-threshold",
            "0.995",
        ],
    )

    stored = json.loads(manifest_path_for(output).read_text(encoding="utf-8"))
    assert stored["language_identification"]["threshold"] == 0.995
    assert stored["lines_written"] == 0


def test_filtering_without_naming_a_model_is_refused(runner: CliRunner, tmp_path: Path) -> None:
    """A language to keep with no classifier is exactly the old silent gap."""
    result = runner.invoke(
        cli_module.cli,
        [
            "download",
            "--lang",
            "uz",
            "--out",
            str(tmp_path / "uz.txt"),
            "--filter-langid",
            "uzn",
        ],
    )
    assert result.exit_code == 2
    assert "must be given together" in result.output


def test_naming_a_model_without_a_language_is_refused(runner: CliRunner, tmp_path: Path) -> None:
    """A classifier with nothing to filter is equally meaningless."""
    result = runner.invoke(
        cli_module.cli,
        [
            "download",
            "--lang",
            "uz",
            "--out",
            str(tmp_path / "uz.txt"),
            "--lid-model",
            "lid218e",
        ],
    )
    assert result.exit_code == 2
    assert "must be given together" in result.output


def test_an_unregistered_model_is_rejected_by_the_option(runner: CliRunner, tmp_path: Path) -> None:
    """The choice of classifier is closed, so a typo cannot reach the run."""
    result = runner.invoke(
        cli_module.cli,
        [
            "download",
            "--lang",
            "uz",
            "--out",
            str(tmp_path / "uz.txt"),
            "--filter-langid",
            "uzn",
            "--lid-model",
            "lid999",
        ],
    )
    assert result.exit_code == 2
    assert "lid999" in result.output


def test_a_threshold_outside_the_unit_interval_reports_its_code(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``--lid-threshold 95`` fails with the range code, not a traceback."""
    result = runner.invoke(
        cli_module.cli,
        [
            "download",
            "--lang",
            "uz",
            "--out",
            str(tmp_path / "uz.txt"),
            "--filter-langid",
            "uzn",
            "--lid-model",
            "lid218e",
            "--lid-threshold",
            "95",
        ],
    )
    assert result.exit_code == 1
    assert "TURKIC_FIELD_004_RANGE" in result.output


def test_a_source_that_cannot_be_read_reports_its_code(runner: CliRunner, tmp_path: Path) -> None:
    """A stream failure becomes a one-line message carrying the error code."""
    original = corpus_hooks.dataset_texts
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer({})
    try:
        result = runner.invoke(
            cli_module.cli,
            ["download", "--lang", "kk", "--out", str(tmp_path / "kk.txt")],
        )
    finally:
        corpus_hooks.dataset_texts = original

    assert result.exit_code == 1
    assert "TURKIC_CORPUS_002_STREAM_FAILED" in result.output


def test_the_hugging_face_token_is_read_from_the_environment(tmp_path: Path) -> None:
    """A gated dataset receives the token the environment carries."""
    original_streamer = corpus_hooks.dataset_texts
    original_environment = _test_hooks.environment
    streamer = corpus_hooks.MappingDatasetTextStreamer({"uz": ["salom dunyo"]})
    corpus_hooks.dataset_texts = streamer
    _test_hooks.environment = _test_hooks.MappingEnvironment({"HF_TOKEN": "secret-token"})
    try:
        result = CliRunner().invoke(
            cli_module.cli,
            ["download", "--lang", "uz", "--out", str(tmp_path / "uz.txt")],
        )
    finally:
        corpus_hooks.dataset_texts = original_streamer
        _test_hooks.environment = original_environment

    assert result.exit_code == 0
    assert streamer.requests == [("oscar-corpus/OSCAR-2301", "uz", "secret-token")]


def test_an_ungated_dataset_is_streamed_without_a_token(tmp_path: Path) -> None:
    """With no credential configured, none is passed to the streamer."""
    original_streamer = corpus_hooks.dataset_texts
    original_environment = _test_hooks.environment
    streamer = corpus_hooks.MappingDatasetTextStreamer({"uz": ["salom dunyo"]})
    corpus_hooks.dataset_texts = streamer
    _test_hooks.environment = _test_hooks.MappingEnvironment({})
    try:
        result = CliRunner().invoke(
            cli_module.cli,
            ["download", "--lang", "uz", "--out", str(tmp_path / "uz.txt")],
        )
    finally:
        corpus_hooks.dataset_texts = original_streamer
        _test_hooks.environment = original_environment

    assert result.exit_code == 0
    assert streamer.requests == [("oscar-corpus/OSCAR-2301", "uz", None)]


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error", "critical"])
def test_every_log_level_the_group_offers_is_accepted(runner: CliRunner, level: str) -> None:
    """Each documented level configures logging and runs the subcommand.

    Args:
        runner: The Click runner driving the group.
        level: One of the levels ``--log-level`` advertises.
    """
    result = runner.invoke(cli_module.cli, ["--log-level", level, "list-sources"])

    assert result.exit_code == 0
    assert "oscar-2301" in result.output


def test_a_level_the_group_does_not_offer_is_rejected(runner: CliRunner) -> None:
    """An unlisted level fails at parse time rather than defaulting."""
    result = runner.invoke(cli_module.cli, ["--log-level", "trace", "list-sources"])

    assert result.exit_code == 2
    assert "trace" in result.output
