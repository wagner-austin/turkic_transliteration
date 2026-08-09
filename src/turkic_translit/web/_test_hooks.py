"""Injection seam for the one effect the web entry point performs.

Production binds the hook to its real implementation at import time and
never rebinds it. Tests bind a real implementation of the same protocol
that records what it was handed. Entry-point code calls the hook
unconditionally, so no branch exists purely to support testing.

Serving is a seam because it blocks: :meth:`gradio.Blocks.launch` starts
an HTTP server and does not return, so nothing that calls it can be
exercised. Everything the entry point does *before* serving — building
the interface, registering both tabs, starting the janitor — then is.

The module is private because the seam is internal to this package.
"""

from __future__ import annotations

from typing import Protocol

import gradio as gr


class Server(Protocol):
    """Serves the assembled application to a browser."""

    def serve(self, application: gr.Blocks) -> None:
        """Start serving and block until the process ends.

        Args:
            application: The interface to serve.
        """
        ...


STYLESHEET = """
.container { margin: 0 auto; }
.tab-content { padding: 10px 15px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 5px 5px; }
.examples-row { margin-top: 10px; }
.file-info { margin-top: -5px; font-size: 0.85em; color: #555; }
footer { margin-top: 20px; text-align: center; font-size: 0.8em; color: #666; }
"""


class GradioServer:
    """Server backed by Gradio's own queue and launcher.

    The theme and the stylesheet are applied here because Gradio 6 takes
    both at launch rather than at construction.

    Args:
        block: Whether to hold the calling thread until the server ends,
            which is what a console script wants. Constructed with
            ``False`` the server starts and returns, which is what a
            caller that intends to stop it again wants.
        port: Port to bind, or ``None`` to let Gradio choose one from
            its own range. Naming it matters wherever that range is not
            free or not published, which is most containers.
    """

    def __init__(self, block: bool = True, port: int | None = None) -> None:
        """Record how this server should be started."""
        self._block = block
        self._port = port

    def serve(self, application: gr.Blocks) -> None:
        """Queue the application and serve it.

        Args:
            application: The interface to serve. Queuing is what lets a
                long download report progress while it runs.
        """
        application.queue().launch(
            theme=gr.themes.Soft(),
            css=STYLESHEET,
            server_port=self._port,
            prevent_thread_lock=not self._block,
        )


class RecordingServer:
    """Server that records what it was asked to serve and serves nothing.

    A real implementation of :class:`Server`, not a mock: it holds no
    assertion helpers, so a test can only read the log it kept.
    """

    def __init__(self) -> None:
        """Start an empty log of served applications."""
        self.served: list[gr.Blocks] = []

    def serve(self, application: gr.Blocks) -> None:
        """Record the application without starting a server.

        Args:
            application: The interface that would have been served.
        """
        self.served.append(application)


server: Server = GradioServer()

__all__ = ["STYLESHEET", "GradioServer", "RecordingServer", "Server", "server"]
