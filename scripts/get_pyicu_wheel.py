"""
One-shot helper for Windows users on Python ≥3.12:
    python scripts/get_pyicu_wheel.py
"""
import sys, urllib.request, subprocess, platform, pathlib
import logging

# Setup logging
logging.basicConfig(format='%(name)s: %(message)s', level=logging.INFO)
log = logging.getLogger("pyicu_wheel")

MAJOR, MINOR = sys.version_info[:2]
if platform.system() != "Windows":
    sys.exit("Not needed – PyICU wheels are on PyPI for non-Windows.")
tag = f"cp{MAJOR}{MINOR}"
if tag not in {"cp310", "cp311", "cp312", "cp313"}:
    sys.exit(f"No pre-built PyICU wheel yet for {tag}")

# All versions now use 2.15 consistently
VERSIONS = {"cp310": "2.15", "cp311": "2.15", "cp312": "2.15", "cp313": "2.15"}
version = VERSIONS[tag]
wheel = f"pyicu-{version}-{tag}-{tag}-win_amd64.whl"
url = f"https://github.com/cgohlke/pyicu-build/releases/download/v{version}/{wheel}"

log.info("→ %s", url)
if not pathlib.Path(wheel).exists():
    log.info("Downloading %s", wheel)
    urllib.request.urlretrieve(url, wheel)
subprocess.check_call([sys.executable, "-m", "pip", "install", wheel])
log.info("✓ PyICU %s installed", wheel)
