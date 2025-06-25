import os
import subprocess
import sys

import pytest

SCRIPTS = os.path.join(sys.prefix, "Scripts")
SUFFIX = ".cmd" if os.name == "nt" else ""


@pytest.mark.usefixtures("tmp_path")
def test_turkic_filter_russian_help() -> None:
    subprocess.run(
        [os.path.join(SCRIPTS, f"turkic-filter-russian{SUFFIX}"), "--help"], check=True
    )


@pytest.mark.usefixtures("tmp_path")
def test_turkic_build_spm_help() -> None:
    subprocess.run(
        [os.path.join(SCRIPTS, f"turkic-build-spm{SUFFIX}"), "--help"], check=True
    )


@pytest.mark.usefixtures("tmp_path")
def test_turkic_pyicu_install_help() -> None:
    subprocess.run(
        [os.path.join(SCRIPTS, f"turkic-pyicu-install{SUFFIX}"), "--help"], check=True
    )


@pytest.mark.usefixtures("tmp_path")
def test_turkic_leven_help() -> None:
    subprocess.run(
        [os.path.join(SCRIPTS, f"turkic-leven{SUFFIX}"), "--help"], check=True
    )
