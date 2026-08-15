"""``turkic-translit web`` — serve the Gradio demo.

The demo used to be reachable three ways and discoverable none: a
``turkic-web`` console script, a ``turkic_tools.py web`` runner, and a
``make web`` target wrapping the second. Someone who installed the
package and ran ``turkic-translit --help`` — the one thing a new user
predictably tries — saw seven subcommands and no mention that a web
interface existed at all.

Gradio is imported inside the command rather than at module import, so
naming this subcommand in the group costs ``--help`` nothing. The
import is not cheap, and every other subcommand would have paid for it.
"""

from __future__ import annotations

import click


@click.command("web", help="Serve the web demo in a browser.")
def cli() -> None:
    """Build the interface and serve it until interrupted."""
    from turkic_translit.web.web_demo import main as serve

    serve()


__all__ = ["cli"]
