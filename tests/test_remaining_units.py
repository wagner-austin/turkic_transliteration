"""Tests for the boundaries and error paths the other modules leave open.

Everything here drives real code: the GitHub release index reads a
captured API response over a ``file:`` URL rather than a stand-in, the
interpreter facts come from the interpreter running the test, and pip is
asked to install a wheel that does not exist so its failure is the real
one.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest
from click.testing import CliRunner

from turkic_translit import _test_hooks
from turkic_translit.cli.translit import _open_input, _open_output, translit
from turkic_translit.lid.errors import (
    ERR_MODEL_FILE_MISSING,
    LidModelFileMissingError,
)
from turkic_translit.lid.registry import get_spec, resolve_model_path
from turkic_translit.lid.spec import decode_lid_model_spec, encode_lid_model_spec
from turkic_translit.logging_config import JSON_FORMAT, default_level, setup
from turkic_translit.tokenizer import DEFAULT_MODEL_NAME, default_model_path


def as_url(path: Path) -> str:
    """Render a local path as a URL urllib can open.

    Args:
        path: File to address.

    Returns:
        A ``file:`` URL naming that file.
    """
    return path.resolve().as_uri()


@pytest.fixture
def rich_logging() -> Iterator[None]:
    """Ask for the human-readable log format.

    The configuration cache is cleared on the way in and out, because it
    holds one configured logger for the life of the process.

    Yields:
        None, once, with the original environment captured.
    """
    previous = _test_hooks.environment
    _test_hooks.environment = _test_hooks.MappingEnvironment(
        {"TURKIC_LOG_FORMAT": "rich", "TURKIC_LOG_LEVEL": "WARNING"}
    )
    setup.cache_clear()
    yield
    _test_hooks.environment = previous
    setup.cache_clear()
    logging.getLogger().handlers.clear()


def test_the_packaged_tokenizer_path_is_named_whether_or_not_it_exists() -> None:
    """The path is derived from the package, not probed for."""
    path = default_model_path()

    assert path.name == DEFAULT_MODEL_NAME
    assert path.parent.name == "turkic_translit"


def test_the_default_log_level_is_info_when_unset() -> None:
    """With nothing configured the project logs at INFO."""
    previous = _test_hooks.environment
    _test_hooks.environment = _test_hooks.MappingEnvironment({})
    try:
        assert default_level() == "INFO"
    finally:
        _test_hooks.environment = previous


def test_the_configured_log_level_is_upper_cased() -> None:
    """A lower-case level from the environment still resolves."""
    previous = _test_hooks.environment
    _test_hooks.environment = _test_hooks.MappingEnvironment({"TURKIC_LOG_LEVEL": "debug"})
    try:
        assert default_level() == "DEBUG"
    finally:
        _test_hooks.environment = previous


def test_the_rich_format_installs_a_rich_handler(rich_logging: None) -> None:
    """Asking for the human format gets the colourised handler.

    Args:
        rich_logging: The bound environment asking for ``rich``.
    """
    from rich.logging import RichHandler

    setup("WARNING")

    installed = logging.getLogger().handlers
    assert [type(handler).__name__ for handler in installed] == [RichHandler.__name__]


def test_the_default_format_is_json() -> None:
    """The structured handler is what an unconfigured process gets."""
    previous = _test_hooks.environment
    _test_hooks.environment = _test_hooks.MappingEnvironment({})
    setup.cache_clear()
    try:
        logger = setup("WARNING")
        assert logger.name == "turkic_translit"
        assert JSON_FORMAT == "json"
    finally:
        _test_hooks.environment = previous
        setup.cache_clear()
        logging.getLogger().handlers.clear()


def test_a_model_missing_from_every_directory_names_the_last_one(tmp_path: Path) -> None:
    """The failure names a path the caller can go and look at."""
    with pytest.raises(LidModelFileMissingError) as raised:
        resolve_model_path("lid.176", [tmp_path / "a", tmp_path / "b"])

    assert raised.value.code == ERR_MODEL_FILE_MISSING
    assert raised.value.path == tmp_path / "b" / get_spec("lid.176")["filename"]


def test_a_model_with_nowhere_to_look_names_the_bare_filename() -> None:
    """With no search directories the message still names the file."""
    with pytest.raises(LidModelFileMissingError) as raised:
        resolve_model_path("lid.176", [])

    assert raised.value.path == Path(get_spec("lid.176")["filename"])


def test_a_model_specification_round_trips() -> None:
    """Encoding then decoding a specification returns the same one."""
    spec = get_spec("lid.176")

    assert decode_lid_model_spec(encode_lid_model_spec(spec)) == spec


def test_the_encoded_specification_carries_every_field() -> None:
    """All five fields survive the encoding."""
    encoded = encode_lid_model_spec(get_spec("lid.176"))

    assert sorted(encoded) == [
        "filename",
        "label_prefix",
        "model_id",
        "script_aware",
        "url",
    ]


def test_the_input_sentinel_selects_standard_input() -> None:
    """``-`` names stdin, and the stack does not close it."""
    with ExitStack() as stack:
        stream = _open_input(stack, "-", "utf-8")
        assert stream is sys.stdin

    assert not sys.stdin.closed


def test_an_input_path_is_opened_and_closed_by_the_stack(tmp_path: Path) -> None:
    """A real path is opened for reading and released with the stack."""
    source = tmp_path / "in.txt"
    source.write_text("мектеп\n", encoding="utf-8")

    with ExitStack() as stack:
        stream = _open_input(stack, str(source), "utf-8")
        assert stream.read() == "мектеп\n"

    assert stream.closed


def test_the_output_sentinel_selects_standard_output() -> None:
    """``-`` names stdout, and the stack does not close it."""
    with ExitStack() as stack:
        stream = _open_output(stack, "-", "utf-8")
        assert stream is sys.stdout

    assert not sys.stdout.closed


def test_an_unrequested_output_mode_opens_nothing() -> None:
    """``None`` means the caller did not ask for this output at all."""
    with ExitStack() as stack:
        assert _open_output(stack, None, "utf-8") is None


def test_an_output_path_is_opened_and_closed_by_the_stack(tmp_path: Path) -> None:
    """A real path is opened for writing and released with the stack."""
    target = tmp_path / "out.txt"

    with ExitStack() as stack:
        stream = _open_output(stack, str(target), "utf-8")
        if stream is None:
            raise AssertionError("a real path must open a writable stream")
        stream.write("mektep\n")

    assert target.read_text(encoding="utf-8") == "mektep\n"


def test_the_benchmark_flag_reports_a_rate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Asking for a benchmark logs the line count and the rate.

    Args:
        tmp_path: Directory holding the input and output.
        caplog: Captures the command's own log records.
    """
    source = tmp_path / "in.txt"
    source.write_text("мектеп\nмектеп\n", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="turkic_translit.cli.translit"):
        result = CliRunner().invoke(
            translit,
            [
                "--lang",
                "kk",
                "--in",
                str(source),
                "--out-ipa",
                str(tmp_path / "out.txt"),
                "--benchmark",
            ],
        )

    assert result.exit_code == 0, result.output
    assert any("Benchmark: 2 lines" in record.message for record in caplog.records)


def test_asking_for_latin_where_there_is_none_names_the_alternatives() -> None:
    """A language with no Latin rules is rejected before any output."""
    result = CliRunner().invoke(translit, ["--lang", "az", "--out-latin", "-"])

    assert result.exit_code == 2
    assert "has no Latin rules" in result.output


def test_asking_for_ipa_where_there_is_none_names_the_alternatives() -> None:
    """A language with no IPA rules is rejected before any output."""
    result = CliRunner().invoke(translit, ["--lang", "ar", "--out-ipa", "-"])

    assert result.exit_code == 2
    assert "has no IPA rules" in result.output
