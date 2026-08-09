"""Injection seam for the effects the developer scripts perform.

Production binds each hook to its real implementation at import time and
never rebinds it. Tests bind them to recording implementations of the
same protocols. Script code calls the hooks unconditionally, so no branch
exists purely to support testing.

The three effects here are the ones a test must not perform for real:
running a subprocess, reading from a terminal, and ending the process.

The module is private because the seam belongs to ``scripts`` and is not
part of anything this project publishes.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import NoReturn, Protocol


class CommandRunner(Protocol):
    """Runs a subprocess and reports whether it succeeded."""

    def succeeds(self, command: Sequence[str]) -> bool:
        """Run ``command`` and report its exit status as a boolean.

        Args:
            command: Program and arguments. Output is discarded.

        Returns:
            True when the program ran and exited zero.
        """
        ...

    def require(self, command: Sequence[str]) -> None:
        """Run ``command`` and raise unless it exits zero.

        Args:
            command: Program and arguments. Output is inherited.

        Raises:
            CalledProcessError: If the program exits non-zero.
            FileNotFoundError: If the program is not on PATH.
        """
        ...


class SubprocessCommandRunner:
    """Runner backed by :mod:`subprocess`."""

    def succeeds(self, command: Sequence[str]) -> bool:
        """Probe whether a program exists and exits zero.

        Args:
            command: Program and arguments. Output is discarded because
                only the exit status is being asked about.

        Returns:
            True when the program ran and exited zero. A missing program
            and a failing one are both reported as False, because the
            caller is asking one question — is this usable — and both
            answers are no.
        """
        completed = subprocess.run(
            list(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode == 0

    def require(self, command: Sequence[str]) -> None:
        """Run a program that the setup cannot continue without.

        Args:
            command: Program and arguments. Output is inherited so the
                developer sees the installer's progress.

        Raises:
            CalledProcessError: If the program exits non-zero.
            FileNotFoundError: If the program is not on PATH.
        """
        subprocess.check_call(list(command))


class RecordingCommandRunner:
    """Runner answering from a table and logging every invocation.

    A real implementation of :class:`CommandRunner`, not a mock: it has
    no assertion helpers, so a test can only check the commands it
    recorded or the value it returned.

    Args:
        succeeding: Programs, by their first argument, that this runner
            reports as succeeding. Anything else reports False.
    """

    def __init__(self, succeeding: Mapping[str, bool]) -> None:
        """Store the outcome table and start empty invocation logs."""
        self._succeeding = dict(succeeding)
        self.probed: list[tuple[str, ...]] = []
        self.required: list[tuple[str, ...]] = []

    def succeeds(self, command: Sequence[str]) -> bool:
        """Record the probe and answer from the table.

        Args:
            command: Program and arguments.

        Returns:
            The recorded outcome for the program, or False when it is
            not in the table.
        """
        self.probed.append(tuple(command))
        return self._succeeding.get(command[0], False)

    def require(self, command: Sequence[str]) -> None:
        """Record the invocation without running anything.

        Args:
            command: Program and arguments.
        """
        self.required.append(tuple(command))


class Prompt(Protocol):
    """Reads a line of input from the developer."""

    def ask(self, question: str) -> str:
        """Put a question to the developer and read the answer.

        Args:
            question: Text shown before the cursor.

        Returns:
            The line typed, without its terminator.
        """
        ...


class TerminalPrompt:
    """Prompt backed by :func:`input`."""

    def ask(self, question: str) -> str:
        """Read one line from the terminal.

        Args:
            question: Text shown before the cursor.

        Returns:
            The line typed, without its terminator.
        """
        return input(question)


class ScriptedPrompt:
    """Prompt answering from a fixed script of replies.

    Args:
        answers: Replies to give, in order.
    """

    def __init__(self, answers: Sequence[str]) -> None:
        """Store the replies and start an empty question log."""
        self._answers = list(answers)
        self.questions: list[str] = []

    def ask(self, question: str) -> str:
        """Record the question and return the next scripted reply.

        Args:
            question: Text that would have been shown.

        Returns:
            The next reply.

        Raises:
            IndexError: If the script has run out of replies, which
                means the code asked more questions than the test
                anticipated.
        """
        self.questions.append(question)
        return self._answers.pop(0)


class Interpreter(Protocol):
    """The facts about the running interpreter this script branches on."""

    def platform_name(self) -> str:
        """Name the operating system.

        Returns:
            The platform name, e.g. ``Windows`` or ``Linux``.
        """
        ...

    def is_isolated(self) -> bool:
        """Report whether this interpreter is an isolated environment.

        Returns:
            True inside a virtual environment or a conda environment.
        """
        ...

    def executable(self) -> str:
        """Name the interpreter to invoke for child processes.

        Returns:
            Path to the running interpreter.
        """
        ...


class RunningInterpreter:
    """Interpreter facts read from the live process."""

    def platform_name(self) -> str:
        """Name the operating system.

        Returns:
            The value :func:`platform.system` reports.
        """
        return platform.system()

    def is_isolated(self) -> bool:
        """Report whether this interpreter is isolated from the system one.

        Both mechanisms count: a virtual environment moves ``sys.prefix``
        away from the base installation, and conda advertises itself
        through the environment instead.

        Returns:
            True when the interpreter is isolated.
        """
        virtualenv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )
        conda = bool(os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"))
        return virtualenv or conda

    def executable(self) -> str:
        """Name the running interpreter.

        Returns:
            The value of ``sys.executable``.
        """
        return sys.executable


class DescribedInterpreter:
    """Interpreter facts stated outright rather than probed.

    Args:
        platform_name: Operating system to report.
        isolated: Whether to report the interpreter as isolated.
        executable: Interpreter path to report.
    """

    def __init__(self, platform_name: str, isolated: bool, executable: str) -> None:
        """Store the three facts this interpreter reports."""
        self._platform_name = platform_name
        self._isolated = isolated
        self._executable = executable

    def platform_name(self) -> str:
        """Return the stated operating system.

        Returns:
            The name given at construction.
        """
        return self._platform_name

    def is_isolated(self) -> bool:
        """Return the stated isolation.

        Returns:
            The flag given at construction.
        """
        return self._isolated

    def executable(self) -> str:
        """Return the stated interpreter path.

        Returns:
            The path given at construction.
        """
        return self._executable


class Exit(Protocol):
    """Ends the process with a status code."""

    def fail(self, status: int) -> NoReturn:
        """End the process.

        Args:
            status: Exit status to report.

        Raises:
            SystemExit: Always.
        """
        ...


class SystemExiter:
    """Exit backed by :func:`sys.exit`."""

    def fail(self, status: int) -> NoReturn:
        """End the process with ``status``.

        Args:
            status: Exit status to report.

        Raises:
            SystemExit: Always.
        """
        sys.exit(status)


commands: CommandRunner = SubprocessCommandRunner()
interpreter: Interpreter = RunningInterpreter()
prompt: Prompt = TerminalPrompt()
exiter: Exit = SystemExiter()

__all__ = [
    "CommandRunner",
    "DescribedInterpreter",
    "Exit",
    "Interpreter",
    "Prompt",
    "RecordingCommandRunner",
    "RunningInterpreter",
    "ScriptedPrompt",
    "SubprocessCommandRunner",
    "SystemExiter",
    "TerminalPrompt",
    "commands",
    "exiter",
    "interpreter",
    "prompt",
]
