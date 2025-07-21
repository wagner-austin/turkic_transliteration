"""
Patches for third-party libraries to fix encoding issues on Windows.
This module is imported automatically at startup.
"""

import logging
import os
import sys

from .logging_config import setup

setup()

log = logging.getLogger(__name__)
_PATCH_DONE = False
_PATCHED_FILES = set()


def _fix_broken_ssl_cert_env() -> None:
    """
    If the user (often Conda on Windows) left SSL_CERT_FILE pointing at a
    non-existent bundle, httpx ⇢ gradio will crash on import.  When the file
    is missing we delete the env-var so Python falls back to the system
    certificates.
    """
    import logging
    import os
    import pathlib

    log = logging.getLogger(__name__)
    bundle = os.environ.get("SSL_CERT_FILE")
    if bundle and not pathlib.Path(bundle).exists():
        log.warning(
            "SSL_CERT_FILE=%s does not exist – removing the variable so "
            "httpx can create a default context",
            bundle,
        )
        del os.environ["SSL_CERT_FILE"]


def apply_patches() -> None:
    """Apply all necessary patches for third-party libraries."""
    global _PATCH_DONE
    _fix_broken_ssl_cert_env()  # ← new line: ensure SSL_CERT_FILE is valid before any third-party import
    # Skip if patches have already been applied
    if _PATCH_DONE:
        log.debug("Patches already applied, skipping")
        return

    _PATCH_DONE = True
    # Fix encoding issues on Windows for all libraries
    if sys.platform == "win32":
        try:
            from typing import Any

            original_open: Any = open

            # Monkey patch the built-in open function for encoding issues
            def patched_open_with_encoding(
                file: Any, mode: Any = "r", *args: Any, **kwargs: Any
            ) -> Any:
                # Add explicit UTF-8 encoding for text files when no encoding is specified
                if (
                    mode in ("r", "rt", "w", "wt", "a", "at")
                    and isinstance(file, str)
                    and "encoding" not in kwargs
                    and "b" not in mode
                ):
                    # Check if it's being called from problematic libraries
                    import traceback

                    stack = traceback.extract_stack()
                    for frame in stack:
                        if any(
                            lib in frame.filename
                            for lib in [
                                "panphon",
                                "transformers",
                                "datasets",
                                "evaluate",
                            ]
                        ):
                            kwargs["encoding"] = "utf-8"
                            # Only log the first time per unique file
                            if file not in _PATCHED_FILES:
                                log.debug(
                                    f"Applied UTF-8 encoding patch for {file} from {frame.filename}"
                                )
                                _PATCHED_FILES.add(file)
                            break
                return original_open(file, mode, *args, **kwargs)

            # Set the environment variable for good measure
            os.environ["PYTHONUTF8"] = "1"
            # Apply the patch
            import builtins
            from typing import cast

            builtins.open = cast(Any, patched_open_with_encoding)
            log.info("Applied UTF-8 encoding patch for Windows")
            # We've already applied the patch above
        except ImportError:
            log.warning("Could not patch panphon (not installed)")


# Apply patches when module is imported
apply_patches()
log.debug("Patches module initialized")
