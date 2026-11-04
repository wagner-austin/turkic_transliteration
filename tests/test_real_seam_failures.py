"""Real seam implementations, made to fail the way the world fails them.

Every ``_test_hooks`` seam that reaches outside the process has a real
implementation bound in production and a stand-in the other tests bind.
The stand-ins prove the callers; only the real implementation can prove
what the library or the network actually does when it fails. Each case
here runs a real implementation against a real failure: an HTTP server
answering 404, a file that is not a fastText model, a port another socket
holds, and a malformed Sentry DSN. The failure each library raises was
measured on 2026-11-04 on pendragon before these cases asserted it.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

import gradio as gr
import pytest
from sentry_sdk.utils import BadDsn

from turkic_translit import _test_hooks
from turkic_translit.lid import _test_hooks as lid_hooks
from turkic_translit.web import _test_hooks as web_hooks


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
