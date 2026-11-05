"""Real seam implementations, made to fail the way the world fails them.

Every ``_test_hooks`` seam that reaches outside the process has a real
implementation bound in production and a stand-in the other tests bind.
The stand-ins prove the callers; only the real implementation can prove
what the library or the network actually does when it fails. Each case
here runs a real implementation against a real failure: an HTTP server
answering 404, a file that is not a fastText model, a port another socket
holds, a malformed Sentry DSN, a model path that does not exist and a
model directory that holds no tokenizer, for the language-model trainer
and evaluator alike, and a terminal prompt whose standard input is
closed. The failure each library raises was measured on pendragon before
these cases asserted it: the first four on 2026-11-04, the rest on
2026-11-05, when the empty directory was found to raise ValueError where
the hooks' docstrings had promised OSError.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

import gradio as gr
import pytest
from sentry_sdk.utils import BadDsn

from scripts.guard import PROJECT_ROOT
from turkic_translit import _test_hooks
from turkic_translit.lid import _test_hooks as lid_hooks
from turkic_translit.lm import _test_hooks as lm_hooks
from turkic_translit.web import _test_hooks as web_hooks

#: transformers' words for a path it cannot treat as a directory or a Hub id.
NOT_A_REPOSITORY = "Repo id must be in the form 'repo_name' or 'namespace/repo_name'"

#: transformers' words for a directory with nothing to build a tokenizer from.
NO_TOKENIZER = "Couldn't instantiate the backend tokenizer"


@pytest.fixture
def http_root(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """Serve a directory over real HTTP on a free local port.

    Args:
        tmp_path: Directory served; a file absent from it answers 404.

    Yields:
        The server's base URL and the directory it serves.
    """
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_a_download_the_server_answers_404_raises_and_leaves_no_file(
    http_root: tuple[str, Path], tmp_path: Path
) -> None:
    """A missing model on the server is an HTTPError, and nothing lands on disk.

    Args:
        http_root: The served base URL and directory.
        tmp_path: Where the download would have been written.
    """
    base, _served = http_root
    destination = tmp_path / "weights" / "lid.176.bin"
    destination.parent.mkdir()

    with pytest.raises(HTTPError) as excinfo:
        lid_hooks.UrlDownloader().fetch(f"{base}/absent.bin", destination)

    assert excinfo.value.code == 404
    assert sorted(destination.parent.iterdir()) == []


def test_a_file_that_is_not_a_fasttext_model_is_refused_by_the_loader(tmp_path: Path) -> None:
    """fastText reads the header, not the extension, and says so.

    Args:
        tmp_path: Where the impostor file is written.
    """
    impostor = tmp_path / "lid.176.bin"
    impostor.write_bytes(b"this is not a fastText model\n" * 10)

    with pytest.raises(ValueError, match="has wrong file format"):
        lid_hooks.FastTextLoader().load(impostor)


def test_the_production_server_refuses_a_port_another_socket_holds() -> None:
    """Gradio does not fall back to another port when the named one is taken."""
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    with gr.Blocks() as application:
        gr.Markdown("held port")
    try:
        with pytest.raises(OSError, match=f"Cannot find empty port in range: {port}-{port}"):
            web_hooks.GradioServer(block=False, port=port).serve(application)
    finally:
        holder.close()
        application.close()


def test_the_production_reporter_refuses_a_malformed_dsn() -> None:
    """sentry_sdk validates the DSN at initialisation rather than at the first event."""
    with pytest.raises(BadDsn, match="Unsupported scheme"):
        _test_hooks.SentryReporter().initialise("not-a-dsn", "test", 0.0, None)


def test_the_production_trainer_refuses_a_base_model_that_does_not_exist(tmp_path: Path) -> None:
    """A path that is no directory is read as a Hub id, and refused as one.

    Args:
        tmp_path: Holds the absent base model and the output directory.
    """
    output = tmp_path / "out"

    with pytest.raises(OSError, match=NOT_A_REPOSITORY):
        lm_hooks.TransformersTrainer().fine_tune(
            str(tmp_path / "absent-model"), 1, iter(["a b c"]), str(output)
        )

    assert not output.exists()


def test_the_production_trainer_refuses_a_directory_without_a_tokenizer(tmp_path: Path) -> None:
    """An existing directory with no tokenizer files fails before training.

    Args:
        tmp_path: Holds the empty base model and the output directory.
    """
    empty = tmp_path / "empty-model"
    empty.mkdir()
    output = tmp_path / "out"

    with pytest.raises(ValueError, match=NO_TOKENIZER):
        lm_hooks.TransformersTrainer().fine_tune(str(empty), 1, iter(["a b c"]), str(output))

    assert not output.exists()


def test_the_production_evaluator_refuses_a_model_that_does_not_exist(tmp_path: Path) -> None:
    """Scoring a path that is no directory fails the same way loading does.

    Args:
        tmp_path: Holds the absent model.
    """
    with pytest.raises(OSError, match=NOT_A_REPOSITORY):
        lm_hooks.TransformersEvaluator().perplexity(str(tmp_path / "absent-model"), ["a b c"])


def test_the_production_evaluator_refuses_a_directory_without_a_tokenizer(tmp_path: Path) -> None:
    """Scoring an empty directory fails at the tokenizer, before any model loads.

    Args:
        tmp_path: Holds the empty model directory.
    """
    empty = tmp_path / "empty-model"
    empty.mkdir()

    with pytest.raises(ValueError, match=NO_TOKENIZER):
        lm_hooks.TransformersEvaluator().perplexity(str(empty), ["a b c"])


def test_the_production_prompt_fails_on_a_closed_input() -> None:
    """With nothing left on standard input, the prompt raises EOFError.

    Run in a child whose standard input is the null device, because the
    thing under test is ``input`` reading a real stream that has ended.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts._test_hooks import TerminalPrompt\nTerminalPrompt().ask('question? ')",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stderr.strip().splitlines()[-1] == "EOFError: EOF when reading a line"
