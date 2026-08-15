"""Tests for the development-environment setup script.

Nothing is patched and nothing is mocked. Every hook is rebound to a real
implementation of the same protocol — one that records what it was asked
and answers only from what it was given — so the whole flow runs without
installing a package, reading a terminal, or ending the process.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from collections.abc import Generator

import pytest

from scripts import _test_hooks, setup_dev

LINUX = "Linux"
INTERPRETER = "/usr/bin/python3"


@pytest.fixture
def hooks() -> Generator[None, None, None]:
    """Restore the production hooks after a test rebinds them.

    Yields:
        None, once, with the originals captured.
    """
    commands, prompt, exiter, interpreter = (
        _test_hooks.commands,
        _test_hooks.prompt,
        _test_hooks.exiter,
        _test_hooks.interpreter,
    )
    yield
    _test_hooks.commands = commands
    _test_hooks.prompt = prompt
    _test_hooks.exiter = exiter
    _test_hooks.interpreter = interpreter


def describe(platform_name: str = LINUX, isolated: bool = True) -> None:
    """Bind the interpreter hook to stated facts.

    Args:
        platform_name: Operating system the script should see.
        isolated: Whether the interpreter should look isolated.
    """
    _test_hooks.interpreter = _test_hooks.DescribedInterpreter(
        platform_name=platform_name, isolated=isolated, executable=INTERPRETER
    )


def test_an_isolated_interpreter_is_not_questioned(hooks: None) -> None:
    """Inside a virtual environment the developer is not interrupted."""
    describe(isolated=True)
    scripted = _test_hooks.ScriptedPrompt([])
    _test_hooks.prompt = scripted

    setup_dev.confirm_outside_virtual_env()

    assert scripted.questions == []


def test_a_bare_interpreter_proceeds_when_confirmed(hooks: None) -> None:
    """Answering yes continues without ending the process."""
    describe(isolated=False)
    scripted = _test_hooks.ScriptedPrompt(["y"])
    _test_hooks.prompt = scripted

    setup_dev.confirm_outside_virtual_env()

    assert scripted.questions == ["Continue anyway? [y/N]: "]


def test_a_bare_interpreter_aborts_when_declined(hooks: None) -> None:
    """Anything other than yes ends the run with a non-zero status."""
    describe(isolated=False)
    _test_hooks.prompt = _test_hooks.ScriptedPrompt([""])

    with pytest.raises(SystemExit) as excinfo:
        setup_dev.confirm_outside_virtual_env()

    assert excinfo.value.code == 1


def test_install_names_the_project_root_and_dev_extra(hooks: None) -> None:
    """The editable install points at this checkout with its dev extras."""
    describe()
    runner = _test_hooks.RecordingCommandRunner({})
    _test_hooks.commands = runner

    setup_dev.install_editable()

    assert runner.required == [
        (INTERPRETER, "-m", "pip", "install", "-e", f"{setup_dev.PROJECT_ROOT}[dev]")
    ]


def test_tool_report_names_only_the_missing_ones(hooks: None) -> None:
    """Every tool is probed and the unusable ones are returned."""
    describe()
    runner = _test_hooks.RecordingCommandRunner({"ruff": True, "pytest": True})
    _test_hooks.commands = runner

    missing = setup_dev.report_tools(("ruff", "mypy", "pytest"))

    assert missing == ["mypy"]
    assert runner.probed == [
        ("ruff", "--version"),
        ("mypy", "--version"),
        ("pytest", "--version"),
    ]


def test_next_steps_describe_make_when_it_is_available(hooks: None) -> None:
    """With Make present the documented targets are the advice given."""
    describe()
    runner = _test_hooks.RecordingCommandRunner({"make": True})
    _test_hooks.commands = runner

    setup_dev.report_next_steps()

    assert runner.probed == [("make", "--version")]


def test_next_steps_explain_how_to_install_make_when_absent(hooks: None) -> None:
    """Without Make the advice is how to get it."""
    describe()
    runner = _test_hooks.RecordingCommandRunner({})
    _test_hooks.commands = runner

    setup_dev.report_next_steps()

    assert runner.probed == [("make", "--version")]


def test_the_whole_setup_runs_end_to_end(hooks: None) -> None:
    """Main performs the install, the probes, and nothing else."""
    describe()
    runner = _test_hooks.RecordingCommandRunner({"ruff": True, "mypy": True})
    _test_hooks.commands = runner

    setup_dev.main()

    assert [probe[0] for probe in runner.probed] == ["ruff", "mypy", "pytest", "make"]
    assert len(runner.required) == 1


def test_running_the_module_as_a_script_performs_the_setup(hooks: None) -> None:
    """``python -m scripts.setup_dev`` is the documented invocation.

    Executed through runpy so the module-level ``__main__`` guard runs
    for real rather than being exempted from coverage. The hooks live in
    a module both copies import, so rebinding them reaches the fresh
    copy without any patching.

    The file is named by path rather than by module name: this test
    module already imported ``scripts.setup_dev``, and re-running it by
    name warns that a module is being executed after it is already in
    ``sys.modules``, which is exactly the ambiguity the warning exists
    to report.
    """
    describe()
    runner = _test_hooks.RecordingCommandRunner({"ruff": True, "mypy": True, "pytest": True})
    _test_hooks.commands = runner

    runpy.run_path(setup_dev.__file__, run_name="__main__")

    assert len(runner.required) == 1
    assert [probe[0] for probe in runner.probed] == ["ruff", "mypy", "pytest", "make"]


def test_the_production_runner_reports_a_real_program() -> None:
    """The subprocess-backed runner probes an actual process."""
    runner = _test_hooks.SubprocessCommandRunner()

    assert runner.succeeds([sys.executable, "-c", "raise SystemExit(0)"]) is True
    assert runner.succeeds([sys.executable, "-c", "raise SystemExit(3)"]) is False


def test_the_production_runner_requires_success() -> None:
    """The require path raises when the program fails."""
    runner = _test_hooks.SubprocessCommandRunner()

    runner.require([sys.executable, "-c", "raise SystemExit(0)"])
    with pytest.raises(subprocess.CalledProcessError):
        runner.require([sys.executable, "-c", "raise SystemExit(4)"])


def test_the_scripted_prompt_runs_out_loudly() -> None:
    """Asking more questions than scripted is an error, not an empty string."""
    scripted = _test_hooks.ScriptedPrompt(["only one"])

    assert scripted.ask("first? ") == "only one"
    with pytest.raises(IndexError):
        scripted.ask("second? ")


def test_the_terminal_prompt_reads_a_real_line_of_input() -> None:
    """The production prompt reads from actual standard input.

    Run in a child process with a real pipe on its stdin, because the
    thing under test is ``input`` itself: substituting it would leave
    the one line this class contains unexercised.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts._test_hooks import TerminalPrompt\n"
            "print(f'answered={TerminalPrompt().ask(\"question? \")}')",
        ],
        input="typed reply\n",
        capture_output=True,
        text=True,
        cwd=setup_dev.PROJECT_ROOT,
        check=True,
    )

    assert completed.stdout.strip() == "question? answered=typed reply"


def test_the_production_interpreter_reports_this_process() -> None:
    """The live interpreter hook reports facts about this very process.

    Isolation is asserted against the same two mechanisms the hook
    consults, so the test holds whether the suite runs inside a virtual
    environment or against a base installation.
    """
    import os
    import platform

    running = _test_hooks.RunningInterpreter()
    isolated = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or bool(os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"))
    )

    assert running.executable() == sys.executable
    assert running.platform_name() == platform.system()
    assert running.is_isolated() is isolated


def test_the_described_interpreter_reports_what_it_was_given() -> None:
    """The stated interpreter answers exactly the three facts it holds."""
    described = _test_hooks.DescribedInterpreter(
        platform_name="Haiku", isolated=False, executable="/opt/py"
    )

    assert described.platform_name() == "Haiku"
    assert described.is_isolated() is False
    assert described.executable() == "/opt/py"
