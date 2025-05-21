# IMPORTANT: This file is automatically imported by Python if it's in the PYTHONPATH.
# DO NOT MOVE THIS FILE FROM THE PROJECT ROOT.
#
# Purpose: Ensures UTF-8 encoding is used consistently across all platforms,
# which is critical for working with non-ASCII Turkic text characters.

import os

# Force UTF-8 mode for Python
os.environ.setdefault("PYTHONUTF8", "1")
import logging

# Log at debug level instead of printing to stdout
logging.getLogger("sitecustomize").debug("PYTHONUTF8=%s", os.environ["PYTHONUTF8"])
