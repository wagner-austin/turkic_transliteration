"""Write the Hugging Face Space's files from this repository.

The Space is a deployment target, not a source. Every file it holds is
rendered here and pushed by the ``sync-to-hf-space`` workflow, because
the two copies drifted the moment they were allowed to.

That drift is what took the demo down. The package raised its Gradio
floor to 6.0 for the reasons given beside the dependency in pyproject,
while the Space's card still declared ``sdk_version: 5.29.0``. A Space
build installs the SDK named on its card alongside the packages in
``requirements.txt``, so pip was handed ``gradio==5.29.0`` and
``gradio>=6.0`` in one resolution and refused both.

Three things follow, and each is enforced rather than described. The
card lives at ``.github/hf-space/README.md`` in this repository, so a
push updates it. :func:`render` refuses to write a card whose SDK falls
outside the range pyproject declares, so a bad pair cannot be published.
And ``tests/test_hf_space.py`` asserts the same thing about the real
files, so the pair is checked before anything is pushed at all.

Run with:

    python -m scripts.hf_space --space <checkout of the Space>
    python -m scripts.hf_space --print-version
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CARD = Path(".github") / "hf-space" / "README.md"

PYPROJECT = "pyproject.toml"

CARD_NAME = "README.md"

REQUIREMENTS_NAME = "requirements.txt"

GRADIO = "gradio"

USAGE = "usage: python -m scripts.hf_space (--space <directory> | --print-version)"

# The Space installs nothing this package does not already declare, so
# the file names the package and stops. The previous one also listed
# epitran and panphon, which nothing under src imports, and a bare
# `gradio`, which added a second unconstrained requirement to the same
# resolution that the card's SDK pin was already deciding.
REQUIREMENTS_HEADER = """\
# Written by scripts/hf_space.py from pyproject.toml. Editing this file
# on the Space accomplishes nothing: the next sync overwrites it.
#
# Everything the demo needs is a dependency of the package, so this
# names the package and nothing else. The pin is exact rather than a
# floor because Hugging Face reuses a cached wheel when given a range.
"""


def front_matter_value(card_text: str, key: str) -> str:
    """Read one field from the Space card's YAML front matter.

    Args:
        card_text: The card, front matter included.
        key: Field to read, such as ``sdk_version``.

    Returns:
        The field's value, without surrounding quotes.

    Raises:
        ValueError: If the card does not declare the field. Hugging Face
            would fall back to a default rather than say so, which is
            how a card can be wrong without looking wrong.
    """
    pattern = re.compile(rf"^{re.escape(key)}:[ \t]*(?P<value>[^\n#]+)$", re.MULTILINE)
    match = pattern.search(card_text)
    if match is None:
        raise ValueError(f"the Space card declares no {key}")
    return match.group("value").strip().strip('"')


def _project_string(pyproject_text: str, key: str) -> str:
    """Read one string field from pyproject's ``[project]`` table.

    Args:
        pyproject_text: Contents of ``pyproject.toml``.
        key: Field to read, such as ``version``.

    Returns:
        The field's value.

    Raises:
        ValueError: If the table or the field is absent.
    """
    document = tomllib.loads(pyproject_text)
    project = document.get("project")
    value = project.get(key) if isinstance(project, dict) else None
    if not isinstance(value, str):
        raise ValueError(f"pyproject.toml declares no [project] {key}")
    return value


def package_version(pyproject_text: str) -> str:
    """Read the version the Space should pin.

    Args:
        pyproject_text: Contents of ``pyproject.toml``.

    Returns:
        The declared version, such as ``0.5.1``.
    """
    return _project_string(pyproject_text, "version")


def gradio_requirement(pyproject_text: str) -> str:
    """Read the package's Gradio requirement as it is written.

    Args:
        pyproject_text: Contents of ``pyproject.toml``.

    Returns:
        The dependency entry, such as ``gradio>=6.0,<7``.

    Raises:
        ValueError: If no dependency names Gradio. The Space's SDK is
            checked against this one entry, so its absence would leave
            the card unchecked rather than merely undocumented.
    """
    document = tomllib.loads(pyproject_text)
    project = document.get("project")
    declared = project.get("dependencies") if isinstance(project, dict) else None
    entries = [str(entry) for entry in declared] if isinstance(declared, list) else []
    for entry in entries:
        if Requirement(entry).name == GRADIO:
            return entry
    raise ValueError(f"pyproject.toml declares no {GRADIO} dependency")


def sdk_matches_package(card_text: str, pyproject_text: str) -> bool:
    """Report whether the card's SDK satisfies the package's requirement.

    Args:
        card_text: The Space card, front matter included.
        pyproject_text: Contents of ``pyproject.toml``.

    Returns:
        True when the Gradio version the Space installs is one the
        package accepts, which is the condition the failed build broke.
    """
    specifier = Requirement(gradio_requirement(pyproject_text)).specifier
    return Version(front_matter_value(card_text, "sdk_version")) in specifier


def requirements_text(version: str) -> str:
    """Render the Space's ``requirements.txt``.

    Args:
        version: Version of this package to pin.

    Returns:
        The file's full contents, header comment included.
    """
    return f"{REQUIREMENTS_HEADER}turkic-translit=={version}\n"


def _write(path: Path, text: str) -> Path:
    """Write one rendered file.

    Args:
        path: Destination, whose parent already exists.
        text: Contents to write.

    Returns:
        The path written, so the caller can report it.
    """
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def render(root: Path, space: Path) -> tuple[Path, ...]:
    """Write the card, the pin and the entry point into a Space checkout.

    The entry point is named by the card rather than assumed, so the
    file Hugging Face runs and the file this copies cannot disagree.

    Args:
        root: This repository's root.
        space: Checkout of the Space to write into.

    Returns:
        The paths written, in the order they were written.

    Raises:
        ValueError: If the card's SDK falls outside the package's Gradio
            requirement. Publishing that pair produces a Space that
            cannot install, so it is refused here as well as in the
            tests.
    """
    card_text = (root / CARD).read_text(encoding="utf-8")
    pyproject_text = (root / PYPROJECT).read_text(encoding="utf-8")

    if not sdk_matches_package(card_text, pyproject_text):
        sdk = front_matter_value(card_text, "sdk_version")
        raise ValueError(
            f"the Space card declares sdk_version {sdk}, which "
            f"{gradio_requirement(pyproject_text)} does not admit; "
            f"update {CARD.as_posix()} to a version the package accepts"
        )

    app_file = front_matter_value(card_text, "app_file")
    return (
        _write(space / CARD_NAME, card_text),
        _write(space / REQUIREMENTS_NAME, requirements_text(package_version(pyproject_text))),
        _write(space / app_file, (root / app_file).read_text(encoding="utf-8")),
    )


def option_value(argv: Sequence[str], name: str) -> str | None:
    """Extract one ``--option value`` pair from a command line.

    Args:
        argv: Arguments after the program name.
        name: Option to look for, including its dashes.

    Returns:
        The value that follows the option, or ``None`` when the option
        is absent or ends the command line.
    """
    index = 0
    while index < len(argv):
        if argv[index] == name and index + 1 < len(argv):
            return argv[index + 1]
        index += 1
    return None


def main(argv: Sequence[str], root: Path = PROJECT_ROOT) -> int:
    """Render the Space, or report the version it should pin.

    Args:
        argv: Arguments after the program name. Passed explicitly rather
            than read from ``sys.argv`` here, so the entry point is the
            only place that touches process state.
        root: Repository to read from. Defaults to this checkout.

    Returns:
        0 when the requested work was done, and 2 when the command line
        named neither piece of work.
    """
    if "--print-version" in argv:
        print(package_version((root / PYPROJECT).read_text(encoding="utf-8")))
        return 0

    space = option_value(argv, "--space")
    if space is None:
        print(USAGE, file=sys.stderr)
        return 2

    for path in render(root, Path(space)):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
