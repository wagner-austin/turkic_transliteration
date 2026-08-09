"""Process-startup configuration, imported automatically by Python.

IMPORTANT: this file must stay at the project root. Python imports it for
every interpreter that has the root on its path, which is how both
settings below reach child processes as well as the main one.

Two things are arranged here:

UTF-8 mode, because this project's subject matter is non-ASCII and a
Windows console defaults to a codepage that cannot represent it.

Coverage measurement in subprocesses. Several tests exercise the real
console scripts by spawning them, and without this their execution is
invisible to coverage — the code runs, and the report calls it untested.
"""

import logging
import os

# Force UTF-8 mode for Python.
os.environ.setdefault("PYTHONUTF8", "1")

# Start coverage in this process when the parent asked for it. The
# variable is set only by a coverage-enabled run, so its presence is also
# the guarantee that the package is importable here.
if os.environ.get("COVERAGE_PROCESS_START"):
    import coverage

    coverage.process_startup()

logging.getLogger("sitecustomize").debug("PYTHONUTF8=%s", os.environ["PYTHONUTF8"])
