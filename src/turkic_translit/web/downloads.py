"""The temporary download directory and the janitor that empties it.

The web UI writes transliterated files and corpus extracts here for the
browser to fetch, and a daemon thread removes them once they are older
than :data:`JANITOR_INTERVAL_SECONDS`. This was part of
``web/web_utils.py`` until 2026-11-04, when it moved here unchanged so
that each concern of the web helpers stays under the 600-line limit.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from .. import _test_hooks

# How long a temporary download survives, and how long the janitor waits
# between sweeps.
JANITOR_INTERVAL_SECONDS = 600


def download_dir() -> Path:
    """Return the directory temporary corpus downloads are written to.

    Resolved on each call rather than at import, so nothing is created as
    a side effect of importing this module and a test can point the
    directory somewhere disposable without touching the process.

    Args:
        None.

    Returns:
        The directory named by ``TURKIC_CRON_DIR``, or ``cronjob`` under
        the working directory. It is created if it does not yet exist.
    """
    configured = _test_hooks.environment.get("TURKIC_CRON_DIR")
    directory = Path(configured) if configured else Path.cwd() / "cronjob"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def purge_expired_downloads(max_age_sec: int) -> int:
    """Delete temporary downloads older than ``max_age_sec``.

    A file vanishing between the listing and the delete is the normal
    outcome of two sweeps overlapping, so ``missing_ok`` covers it and
    nothing is caught. Any other filesystem error is a real problem and
    propagates.

    Args:
        max_age_sec: Age beyond which a file is removed.

    Returns:
        The number of files deleted.

    Raises:
        OSError: When the download directory cannot be created, listed or
            read, or an expired file cannot be deleted.
    """
    cutoff = time.time() - max_age_sec
    removed = 0
    for entry in download_dir().glob("*"):
        if entry.is_file() and entry.stat().st_mtime < cutoff:
            entry.unlink(missing_ok=True)
            removed += 1
    return removed


def forever() -> bool:
    """Report that the janitor should keep sweeping.

    Returns:
        True, always. This is the loop condition a real run uses; a test
        supplies one that stops.
    """
    return True


def run_janitor(max_age_sec: int, running: Callable[[], bool]) -> int:
    """Sweep expired downloads until the loop condition says to stop.

    A sweep's filesystem error ends the loop. On the thread
    :func:`start_janitor` runs it on, that error reaches the thread's own
    boundary, :func:`threading.excepthook`, which reports it with its
    traceback as ``Exception in thread cron-janitor``. It used to be
    logged and the next sweep tried anyway, on the grounds that a file
    held open by the browser could not be deleted; that is Windows
    behaviour, and on the platforms this project supports deleting an
    open file succeeds, so an error here is a real fault that repeating
    the sweep every interval would only hide.

    Args:
        max_age_sec: Age beyond which a download is removed, and the
            interval between sweeps.
        running: Consulted before each sweep. A real run passes
            :func:`forever`.

    Returns:
        The number of sweeps performed.

    Raises:
        OSError: From :func:`purge_expired_downloads`, ending the loop.
    """
    sweeps = 0
    while running():
        purge_expired_downloads(max_age_sec)
        sweeps += 1
        _test_hooks.clock.sleep(max_age_sec)
    return sweeps


def start_janitor(
    max_age_sec: int = JANITOR_INTERVAL_SECONDS,
    running: Callable[[], bool] = forever,
) -> threading.Thread:
    """Start the background thread that purges temporary downloads.

    Started by the application rather than by importing this module.
    Sweeping on import meant every process that touched a web helper —
    including every test run — spawned a thread and created a download
    directory wherever it happened to be running.

    Args:
        max_age_sec: Age beyond which a download is removed, and the
            interval between sweeps.
        running: Loop condition, defaulting to sweeping indefinitely.

    Returns:
        The started daemon thread.
    """
    thread = threading.Thread(
        target=run_janitor,
        args=(max_age_sec, running),
        daemon=True,
        name="cron-janitor",
    )
    thread.start()
    return thread


__all__ = [
    "JANITOR_INTERVAL_SECONDS",
    "download_dir",
    "forever",
    "purge_expired_downloads",
    "run_janitor",
    "start_janitor",
]
