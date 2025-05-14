"""
Patches for third-party libraries to fix encoding issues on Windows.
This module is imported automatically at startup.
"""
import os
import sys
import functools
import logging
from .logging_config import setup; setup()

log = logging.getLogger(__name__)
_PATCH_DONE = False
_PATCHED_FILES = set()

def _fix_broken_ssl_cert_env():
    """
    If the user (often Conda on Windows) left SSL_CERT_FILE pointing at a
    non-existent bundle, httpx ⇢ gradio will crash on import.  When the file
    is missing we delete the env-var so Python falls back to the system
    certificates.
    """
    import os, pathlib, logging
    log = logging.getLogger(__name__)
    bundle = os.environ.get("SSL_CERT_FILE")
    if bundle and not pathlib.Path(bundle).exists():
        log.warning(
            "SSL_CERT_FILE=%s does not exist – removing the variable so "
            "httpx can create a default context", bundle)
        del os.environ["SSL_CERT_FILE"]

def apply_patches():
    """Apply all necessary patches for third-party libraries."""
    global _PATCH_DONE
    _fix_broken_ssl_cert_env()  # ← new line: ensure SSL_CERT_FILE is valid before any third-party import
    # Skip if patches have already been applied
    if _PATCH_DONE:
        log.debug("Patches already applied, skipping")
        return
        
    _PATCH_DONE = True
    # Fix panphon encoding issues on Windows
    if sys.platform == 'win32':
        try:
            import panphon.featuretable
            import io
            import csv
            
            # Save the original open function
            original_open = open
            
            # Monkey patch the built-in open function when used by panphon
            def patched_open_for_panphon(file, mode='r', *args, **kwargs):
                # Add explicit UTF-8 encoding for CSV files opened by panphon
                if 'panphon' in sys.modules and mode == 'r' and isinstance(file, str) and file.endswith('.csv'):
                    if 'encoding' not in kwargs:
                        kwargs['encoding'] = 'utf-8'
                        # Only log the first time per unique file
                        if file not in _PATCHED_FILES:
                            log.debug(f"Applied UTF-8 encoding patch for {file}")
                            _PATCHED_FILES.add(file)
                return original_open(file, mode, *args, **kwargs)
            
            # Set the environment variable for good measure
            os.environ['PYTHONUTF8'] = '1'
            
            # Apply the patch
            import builtins
            builtins.open = patched_open_for_panphon
            log.info("Applied panphon UTF-8 patch for Windows")

            # We've already applied the patch above
        except ImportError:
            log.warning("Could not patch panphon (not installed)")

# Apply patches when module is imported
apply_patches()
log.debug("Patches module initialized")
