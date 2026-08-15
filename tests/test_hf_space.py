"""Tests for the script that writes the Hugging Face Space.

The test that matters is :func:`test_the_cards_sdk_is_one_the_package_accepts`,
and it reads the real card and the real pyproject. Every other test here
builds a small repository under ``tmp_path`` and checks one behaviour of
the renderer against it.

The Space's build was broken for a day by a pair of files that disagreed
— a card pinning the Gradio 5.29.0 SDK, a package requiring Gradio 6 —
and neither file was wrong on its own. Only the pair was, which is why
the pair is what gets asserted.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from scripts import hf_space

CARD = """\
---
title: Demo
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
---

# Demo
"""

PYPROJECT = """\
[project]
name = "turkic-translit"
version = "9.9.9"
dependencies = [
    "click>=8.1",
    "gradio>=6.0,<7",
]
"""

APP = "from turkic_translit.web.web_demo import build_ui\n"


def build_repository(root: Path, card: str = CARD, pyproject: str = PYPROJECT) -> Path:
    """Write a repository the renderer can read.

    Args:
        root: Directory to build under.
        card: Contents of the Space card.
        pyproject: Contents of ``pyproject.toml``.

    Returns:
        The repository root, so a caller can pass it straight on.
    """
    (root / hf_space.CARD).parent.mkdir(parents=True, exist_ok=True)
    (root / hf_space.CARD).write_text(card, encoding="utf-8")
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (root / "app.py").write_text(APP, encoding="utf-8")
    return root


def test_the_cards_sdk_is_one_the_package_accepts() -> None:
    """The real card's SDK sits inside the real package's Gradio range.

    This is the check the broken build did not have. It fails on the
    commit that raises the floor in pyproject without moving the card,
    rather than eight minutes into a Space build that nobody watches.
    """
    root = hf_space.PROJECT_ROOT
    card = (root / hf_space.CARD).read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert hf_space.sdk_matches_package(card, pyproject) is True


def test_the_card_names_an_entry_point_this_repository_has() -> None:
    """The file Hugging Face runs is one this repository can copy to it.

    The renderer copies whatever the card's ``app_file`` names, so a card
    naming a file that is not here would push a Space with no entry
    point — which Hugging Face reports as ``NO_APP_FILE`` after the
    build, not before it.
    """
    root = hf_space.PROJECT_ROOT
    card = (root / hf_space.CARD).read_text(encoding="utf-8")

    app_file = hf_space.front_matter_value(card, "app_file")

    assert "build_ui" in (root / app_file).read_text(encoding="utf-8")


def test_a_field_is_read_from_the_front_matter() -> None:
    """A declared field comes back without its quotes."""
    assert hf_space.front_matter_value('sdk_version: "6.22.0"\n', "sdk_version") == "6.22.0"


def test_an_undeclared_field_is_refused() -> None:
    """A card missing a field is an error, not a default."""
    with pytest.raises(ValueError, match="declares no sdk_version"):
        hf_space.front_matter_value("---\ntitle: Demo\n---\n", "sdk_version")


def test_the_pinned_version_comes_from_pyproject() -> None:
    """The pin is read from the one place the version is declared."""
    assert hf_space.package_version(PYPROJECT) == "9.9.9"


def test_a_pyproject_without_a_project_table_is_refused() -> None:
    """Nothing can be pinned from a file that declares no version."""
    with pytest.raises(ValueError, match=r"declares no \[project\] version"):
        hf_space.package_version("")


def test_the_gradio_requirement_is_read_as_written() -> None:
    """The whole dependency entry is returned, bounds included."""
    assert hf_space.gradio_requirement(PYPROJECT) == "gradio>=6.0,<7"


def test_a_pyproject_without_dependencies_is_refused() -> None:
    """With nothing to check the card against, the check is not skipped."""
    with pytest.raises(ValueError, match="declares no gradio dependency"):
        hf_space.gradio_requirement('[project]\nversion = "9.9.9"\n')


def test_a_pyproject_without_gradio_is_refused() -> None:
    """A dependency list that names no SDK is refused the same way."""
    with pytest.raises(ValueError, match="declares no gradio dependency"):
        hf_space.gradio_requirement('[project]\ndependencies = ["click>=8.1"]\n')


def test_an_sdk_outside_the_requirement_does_not_match() -> None:
    """The exact pair that broke the build reports as mismatched."""
    stale = CARD.replace("6.22.0", "5.29.0")

    assert hf_space.sdk_matches_package(stale, PYPROJECT) is False


def test_the_requirements_name_the_package_and_nothing_else() -> None:
    """The rendered pin is exact and stands alone under its header."""
    rendered = hf_space.requirements_text("9.9.9")

    assert rendered.splitlines()[-1] == "turkic-translit==9.9.9"
    assert rendered == f"{hf_space.REQUIREMENTS_HEADER}turkic-translit==9.9.9\n"


def test_rendering_writes_the_card_the_pin_and_the_entry_point(tmp_path: Path) -> None:
    """A Space checkout ends up holding exactly what this repository says."""
    root = build_repository(tmp_path / "repo")
    space = tmp_path / "space"
    space.mkdir()

    written = hf_space.render(root, space)

    assert written == (space / "README.md", space / "requirements.txt", space / "app.py")
    assert (space / "README.md").read_text(encoding="utf-8") == CARD
    assert (space / "requirements.txt").read_text(encoding="utf-8") == hf_space.requirements_text(
        "9.9.9"
    )
    assert (space / "app.py").read_text(encoding="utf-8") == APP


def test_a_mismatched_pair_is_never_written(tmp_path: Path) -> None:
    """The renderer refuses the pair rather than publishing a broken Space."""
    root = build_repository(tmp_path / "repo", card=CARD.replace("6.22.0", "5.29.0"))
    space = tmp_path / "space"
    space.mkdir()

    with pytest.raises(ValueError, match="sdk_version 5.29.0"):
        hf_space.render(root, space)

    assert sorted(path.name for path in space.iterdir()) == []


def test_the_version_can_be_asked_for_on_its_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The workflow reads the version through the same parser as the pin."""
    root = build_repository(tmp_path / "repo")

    status = hf_space.main(["--print-version"], root=root)

    assert status == 0
    assert capsys.readouterr().out == "9.9.9\n"


def test_the_command_line_renders_into_the_named_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--space`` writes the three files and names each one it wrote."""
    root = build_repository(tmp_path / "repo")
    space = tmp_path / "space"
    space.mkdir()

    status = hf_space.main(["--space", str(space)], root=root)

    assert status == 0
    assert capsys.readouterr().out.splitlines() == [
        f"wrote {space / 'README.md'}",
        f"wrote {space / 'requirements.txt'}",
        f"wrote {space / 'app.py'}",
    ]


def test_a_directoryless_option_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--space`` with nothing after it names no directory to write."""
    root = build_repository(tmp_path / "repo")

    status = hf_space.main(["--space"], root=root)

    assert status == 2
    assert capsys.readouterr().err == f"{hf_space.USAGE}\n"


def test_a_command_line_asking_for_nothing_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Neither piece of work was requested, so neither is guessed at."""
    root = build_repository(tmp_path / "repo")

    status = hf_space.main([], root=root)

    assert status == 2
    assert capsys.readouterr().err == f"{hf_space.USAGE}\n"


def test_the_module_runs_as_a_script() -> None:
    """The ``__main__`` guard turns the returned status into an exit code.

    Executed through runpy so the guard itself runs. Pytest's own
    command line names neither piece of work, so the status is the
    usage error.
    """
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(hf_space.__file__, run_name="__main__")

    assert excinfo.value.code == 2
