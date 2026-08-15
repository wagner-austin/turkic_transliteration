"""Tests for the Gradio application shell and its housekeeping.

The application is built for real — every tab registers, every widget is
constructed — with the corpus source and the classifier bound to
table-backed implementations so nothing reaches the network. The janitor
is driven directly rather than left to a daemon thread, so its sweeping,
its waiting and its error reporting are all observable.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import pytest
from click.testing import CliRunner

from turkic_translit import _test_hooks
from turkic_translit.cli import main as cli_group
from turkic_translit.cli.web import cli as web_command
from turkic_translit.corpus import _test_hooks as corpus_hooks
from turkic_translit.lid import _test_hooks as lid_hooks
from turkic_translit.lid.locations import default_search_dirs
from turkic_translit.web import _test_hooks as web_hooks
from turkic_translit.web import web_demo, web_utils
from turkic_translit.web.tabs import corpus as corpus_tab

ANSWERS = {"merhaba dunya": [("__label__tr", 0.99)]}
WEIGHTS = default_search_dirs()[0] / "lid.176.bin"


@pytest.fixture
def cron_dir(tmp_path: Path) -> Iterator[Path]:
    """Point the download directory at a disposable path.

    Yields:
        The directory downloads and sweeps operate on.
    """
    previous = _test_hooks.environment
    target = tmp_path / "cronjob"
    _test_hooks.environment = _test_hooks.MappingEnvironment({"TURKIC_CRON_DIR": str(target)})
    yield target
    _test_hooks.environment = previous


@pytest.fixture
def installed_ui(cron_dir: Path) -> Iterator[None]:
    """Bind everything the shell touches while it is being built.

    Yields:
        None, once, with the original hooks captured.
    """
    previous = (
        corpus_hooks.dataset_texts,
        corpus_hooks.languages,
        lid_hooks.probe,
        lid_hooks.model_loader,
    )
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer({"tr": ["merhaba dunya"]})
    corpus_hooks.languages = corpus_hooks.MappingLanguageCatalogue(
        {"oscar-corpus/OSCAR-2301": ["tr"]}, ["tr"]
    )
    lid_hooks.probe = lid_hooks.MappingFileProbe({WEIGHTS: 131266198})
    lid_hooks.model_loader = lid_hooks.FixedModelLoader(lid_hooks.TableFastTextModel(ANSWERS))
    web_utils.installed_classifier.cache_clear()
    corpus_tab._fasttext_langs.cache_clear()
    corpus_tab._lang_choices.cache_clear()
    yield
    (
        corpus_hooks.dataset_texts,
        corpus_hooks.languages,
        lid_hooks.probe,
        lid_hooks.model_loader,
    ) = previous
    web_utils.installed_classifier.cache_clear()
    corpus_tab._fasttext_langs.cache_clear()
    corpus_tab._lang_choices.cache_clear()


def test_an_installed_model_is_reported_with_its_size(installed_ui: None) -> None:
    """A present model produces no warning and names its file.

    Args:
        installed_ui: The bound hooks.
    """
    warning, info = web_demo._model_check()

    assert warning == ""
    assert WEIGHTS.name in info
    assert "MB" in info


def test_a_missing_model_is_reported_as_missing(cron_dir: Path) -> None:
    """An absent model warns and says it will be fetched on first use.

    Args:
        cron_dir: The bound download directory.
    """
    previous = lid_hooks.probe
    lid_hooks.probe = lid_hooks.MappingFileProbe({})
    try:
        warning, info = web_demo._model_check()
    finally:
        lid_hooks.probe = previous

    assert "Model file(s) missing" in warning
    assert "downloaded on first use" in warning
    assert "not installed" in info


def test_the_application_builds_with_both_tabs(installed_ui: None) -> None:
    """Building the shell registers every widget without error.

    Args:
        installed_ui: The bound hooks.
    """
    application = web_demo.build_ui()

    tabs = [block.label for block in application.blocks.values() if isinstance(block, gr.Tab)]
    assert application.title == "Turkic Transliteration Suite"
    assert tabs == ["📝 Transliterate to IPA", "📥 Download Corpus"]


def test_the_entry_point_builds_and_serves(installed_ui: None) -> None:
    """The console script assembles the application and hands it over.

    Args:
        installed_ui: The bound hooks.
    """
    previous = web_hooks.server
    recording = web_hooks.RecordingServer()
    web_hooks.server = recording
    try:
        web_demo.main()
    finally:
        web_hooks.server = previous

    assert [served.title for served in recording.served] == ["Turkic Transliteration Suite"]


def test_the_web_subcommand_serves_the_same_application(installed_ui: None) -> None:
    """``turkic-translit web`` reaches the entry point through the group.

    The subcommand is the only place Gradio is imported, which is why it
    imports inside the function body rather than at module scope, and
    that body was the one part of the console script no test entered:
    ``--help`` exercises the decorator and stops there. Driving it
    through the real group proves the registration, the deferred import
    and the call, and the recording server is what makes it safe to run
    a command whose production behaviour is to block forever.

    Args:
        installed_ui: The bound hooks.
    """
    previous = web_hooks.server
    recording = web_hooks.RecordingServer()
    web_hooks.server = recording
    try:
        result = CliRunner().invoke(cli_group, ["web"])
    finally:
        web_hooks.server = previous

    assert result.exit_code == 0, result.output
    assert [served.title for served in recording.served] == ["Turkic Transliteration Suite"]


def test_the_web_subcommand_is_registered_under_its_own_name() -> None:
    """The group offers ``web``, so the test above invokes a real command.

    Click reports exit code 2 and prints usage for a name it does not
    know, which a test that only asserted on the recording server would
    not distinguish from a command that ran and served nothing.
    """
    assert "web" in cli_group.commands
    assert cli_group.commands["web"] is web_command


def free_port() -> int:
    """Find a port nothing is listening on.

    Gradio otherwise scans a fixed range, which several parallel test
    workers exhaust between them.

    Returns:
        A port number that was free a moment ago.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        bound: int = probe.getsockname()[1]
    return bound


def test_the_production_server_really_serves() -> None:
    """The server production binds starts a reachable HTTP server.

    It is constructed to return rather than block and to bind a named
    port; the queuing, the theme, the stylesheet, the launch and the
    socket are all the production ones.
    """
    with gr.Blocks() as application:
        gr.Markdown("served for one assertion")

    port = free_port()
    web_hooks.GradioServer(block=False, port=port).serve(application)
    try:
        assert application.local_url == f"http://127.0.0.1:{port}/"
    finally:
        application.close()


def test_the_shared_log_handler_is_created_once() -> None:
    """The UI handler is attached to the package logger and reused."""
    first = web_utils.get_ui_log_handler()
    second = web_utils.get_ui_log_handler()

    assert first is second
    assert first in logging.getLogger("turkic_translit").handlers


def test_the_log_handler_buffers_and_drains() -> None:
    """Records accumulate until dumped, and the buffer then empties."""
    handler = web_utils.GradioLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    handler.emit(logging.LogRecord("t", logging.INFO, "p", 1, "first", None, None))
    handler.emit(logging.LogRecord("t", logging.INFO, "p", 1, "second", None, None))

    assert handler.dump() == "first\nsecond"
    assert handler.dump() == ""


@pytest.mark.parametrize(
    ("message", "kept"),
    [
        ("corpus download complete", True),
        ("HTTP Request: GET /x", False),
        ("turkic_model.model not found", False),
    ],
)
def test_the_ui_filter_drops_only_housekeeping(message: str, kept: bool) -> None:
    """Noise is filtered from the browser log; real messages survive.

    Args:
        message: The message being emitted.
        kept: Whether it should reach the browser.
    """
    emitted = logging.LogRecord("t", logging.INFO, "p", 1, message, None, None)

    assert web_utils.UiPrettyLogFilter().filter(emitted) is kept


def test_language_codes_become_label_value_pairs() -> None:
    """The dropdown gets a readable label beside each code."""
    assert web_utils.labelise(["tr", "kk"]) == [("Turkish (tr)", "tr"), ("Kazakh (kk)", "kk")]


def test_a_sweep_removes_only_expired_downloads(cron_dir: Path) -> None:
    """Files older than the age are deleted and newer ones are kept.

    Args:
        cron_dir: The bound download directory.
    """
    directory = web_utils.download_dir()
    stale = directory / "stale.txt"
    fresh = directory / "fresh.txt"
    stale.write_text("old", encoding="utf-8")
    fresh.write_text("new", encoding="utf-8")
    import os
    import time

    os.utime(stale, (time.time() - 100, time.time() - 100))

    assert web_utils.purge_expired_downloads(50) == 1
    assert not stale.exists()
    assert fresh.exists()


def test_a_sweep_ignores_directories(cron_dir: Path) -> None:
    """Only files are removed; a nested directory survives.

    Args:
        cron_dir: The bound download directory.
    """
    directory = web_utils.download_dir()
    (directory / "nested").mkdir()

    assert web_utils.purge_expired_downloads(0) == 0
    assert (directory / "nested").is_dir()


def test_a_failing_sweep_is_reported_and_counted_as_nothing(cron_dir: Path) -> None:
    """A filesystem error stops the sweep without stopping the janitor.

    Args:
        cron_dir: The bound download directory, replaced by a file so the
            sweep's directory listing fails.
    """
    previous = _test_hooks.environment
    blocked = cron_dir.parent / "not-a-directory"
    blocked.write_text("this is a file", encoding="utf-8")
    _test_hooks.environment = _test_hooks.MappingEnvironment({"TURKIC_CRON_DIR": str(blocked)})
    try:
        assert web_utils.sweep_once(0) == 0
    finally:
        _test_hooks.environment = previous


def test_a_successful_sweep_reports_what_it_removed(cron_dir: Path) -> None:
    """The wrapper passes the count through unchanged.

    Args:
        cron_dir: The bound download directory.
    """
    import os
    import time

    directory = web_utils.download_dir()
    stale = directory / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    os.utime(stale, (time.time() - 100, time.time() - 100))

    assert web_utils.sweep_once(50) == 1


def test_the_janitor_sweeps_and_waits_each_round(cron_dir: Path) -> None:
    """Each round performs one sweep and one wait of the given length.

    Args:
        cron_dir: The bound download directory.
    """
    previous = _test_hooks.clock
    recording = _test_hooks.RecordingClock()
    _test_hooks.clock = recording
    rounds = iter([True, True, False])
    try:
        swept = web_utils.run_janitor(30, lambda: next(rounds))
    finally:
        _test_hooks.clock = previous

    assert swept == 2
    assert recording.slept == [30, 30]


def test_the_real_loop_condition_never_stops() -> None:
    """A production run sweeps for the life of the process."""
    assert web_utils.forever() is True


def test_starting_the_janitor_runs_it_on_another_thread(cron_dir: Path) -> None:
    """The thread is a daemon and performs the sweeps asked of it.

    Args:
        cron_dir: The bound download directory.
    """
    previous = _test_hooks.clock
    _test_hooks.clock = _test_hooks.RecordingClock()
    rounds = iter([True, False])
    try:
        thread = web_utils.start_janitor(1, lambda: next(rounds))
        thread.join(timeout=10)
    finally:
        _test_hooks.clock = previous

    assert thread.daemon is True
    assert thread.name == "cron-janitor"
    assert not thread.is_alive()
