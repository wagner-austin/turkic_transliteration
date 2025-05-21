#!/usr/bin/env python
"""
Entry point for the Turkic Transliteration Web Application.
This module provides a clean interface to launch the web UI.
"""

from web_demo import build_ui

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        inbrowser=True,
    )
