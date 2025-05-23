# Web UI Documentation

This document covers the web interface for the Turkic Transliteration Suite.

## Russian-masking debug output

The Filter-Russian tab receives a tuple `(masked_text, confidence_table_md)` from the helper `do_mask`.
The Markdown table is built by `_to_md_table`, which parses the `<!--debug ... -->` block emitted by `mask_russian`.
Widgets therefore always expect two outputs; if you add new consumers remember to wire both.

Note that ANSI removal now happens in `mask_russian`, so downstream code no longer needs to scrub colour codes.
