import os
import subprocess
import sys

import pytest

SCRIPTS = os.path.join(sys.prefix, "Scripts")


@pytest.mark.usefixtures("tmp_path")
def test_turkic_filter_russian_help() -> None:
    subprocess.run(
        [os.path.join(SCRIPTS, "turkic-filter-russian"), "--help"], check=True
    )


@pytest.mark.usefixtures("tmp_path")
def test_turkic_build_spm_help() -> None:
    subprocess.run([os.path.join(SCRIPTS, "turkic-build-spm"), "--help"], check=True)


@pytest.mark.usefixtures("tmp_path")
def test_turkic_pyicu_install_help() -> None:
    subprocess.run(
        [os.path.join(SCRIPTS, "turkic-pyicu-install"), "--help"], check=True
    )


@pytest.mark.usefixtures("tmp_path")
def test_turkic_leven_help() -> None:
    subprocess.run([os.path.join(SCRIPTS, "turkic-leven"), "--help"], check=True)
