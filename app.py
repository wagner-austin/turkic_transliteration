"""Entry point for the Hugging Face Space.

The Space serves through the same function ``turkic-translit web``
calls, and for the same reason the Space's card and requirements are
rendered from this repository: two paths that are meant to produce one
interface will not keep producing one interface.

They had already diverged. This file built the interface and launched it
itself, which skipped the theme and the stylesheet — Gradio 6 takes both
at ``launch`` rather than at construction, and only the server hook
passes them — so the Space served an unthemed page whose inputs were
near-invisible against their own background. Nothing here is configured
now, because there is nothing here to configure.
"""

from turkic_translit.web.web_demo import main

main()
