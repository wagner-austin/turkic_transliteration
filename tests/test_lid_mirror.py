"""Tests for the script the ``mirror-lid-models`` workflow reads the registry through.

The workflow downloads each line of the manifest from Meta, refuses a
file whose SHA-256 is not the line's, and uploads the rest to the repo
``--print-repo`` names. These tests check that what it is told is the
registry's own facts, so the files it uploads are the files a download
accepts, at the URLs a download reads.
"""

from __future__ import annotations

import runpy

import pytest

from scripts import lid_mirror
from turkic_translit.lid.registry import MIRROR_REPO, get_spec


def test_the_manifest_names_every_model_with_its_digest_and_origin() -> None:
    """One line per registered model, in registry order, read from its spec."""
    assert lid_mirror.manifest_lines() == [
        "lid.176.bin "
        "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e "
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        "lid218e.bin "
        "8ded5749a2ad79ae9ab7c9190c7c8b97ff20d54ad8b9527ffa50107238fc7f6a "
        "https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin",
    ]


def test_each_uploaded_file_lands_where_the_package_downloads_it() -> None:
    """The repo the workflow fills, plus a filename, is the spec's URL."""
    for line in lid_mirror.manifest_lines():
        filename = line.split(" ")[0]
        model_id = filename.removesuffix(".bin")
        assert get_spec(model_id)["url"] == (
            f"https://huggingface.co/{MIRROR_REPO}/resolve/main/{filename}"
        )


def test_the_repo_can_be_asked_for(capsys: pytest.CaptureFixture[str]) -> None:
    """``--print-repo`` prints the registry's mirror repo and nothing else."""
    status = lid_mirror.main(["--print-repo"])

    assert status == 0
    assert capsys.readouterr().out == "AustinWagner/turkic-translit-lid\n"


def test_the_manifest_can_be_asked_for(capsys: pytest.CaptureFixture[str]) -> None:
    """``--print-manifest`` prints the manifest's lines, one per output line."""
    status = lid_mirror.main(["--print-manifest"])

    assert status == 0
    assert capsys.readouterr().out.splitlines() == lid_mirror.manifest_lines()


def test_a_command_line_asking_for_nothing_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Neither piece of output was requested, so neither is guessed at."""
    status = lid_mirror.main([])

    assert status == 2
    assert capsys.readouterr().err == f"{lid_mirror.USAGE}\n"


def test_the_module_runs_as_a_script() -> None:
    """The ``__main__`` guard turns the returned status into an exit code.

    Executed through runpy so the guard itself runs. Pytest's own command
    line names neither option, so the status is the usage error.
    """
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(lid_mirror.__file__, run_name="__main__")

    assert excinfo.value.code == 2
