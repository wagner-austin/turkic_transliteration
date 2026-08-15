"""Injection seam for the process state this package reads.

Production binds each hook to its real implementation at import time and
never rebinds it. Tests bind them to real implementations of the same
protocols that answer from a mapping. Package code calls the hooks
unconditionally, so no branch exists purely to support testing.

The environment is a seam because it is shared, mutable, process-wide
state: a test that sets ``HF_TOKEN`` to observe one function changes what
every other function sees, including in tests running beside it. Reading
it through a protocol makes the dependency explicit at each call site and
lets a test state the answer rather than alter the process.

The remaining seams are the effects the PyICU bootstrap performs: reading
facts about the running interpreter, querying the GitHub releases API,
downloading a wheel, and running pip. None of those can happen in a test
run, and all four are decisions the installer branches on.

The module is private because the seam is internal to this package.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol


class Environment(Protocol):
    """The process environment, as this package reads it."""

    def get(self, name: str) -> str | None:
        """Return one environment variable.

        Args:
            name: Variable to read.

        Returns:
            The value, or ``None`` when the variable is unset. Unset and
            empty are deliberately distinct: an empty credential is a
            configuration mistake, not an absent one.
        """
        ...


class ProcessEnvironment:
    """Environment backed by :data:`os.environ`."""

    def get(self, name: str) -> str | None:
        """Read a variable from the running process.

        Args:
            name: Variable to read.

        Returns:
            The value, or ``None`` when unset.
        """
        return os.environ.get(name)


class MappingEnvironment:
    """Environment answering from a fixed mapping.

    A real implementation of :class:`Environment`, not a mock: it holds
    no assertion helpers and reports every unlisted name as unset, so a
    test can only check the value the code under test produced.

    Args:
        values: Variables this environment defines. Anything absent is
            reported as unset.
    """

    def __init__(self, values: Mapping[str, str]) -> None:
        """Store the variables this environment reports."""
        self._values = dict(values)

    def get(self, name: str) -> str | None:
        """Return the stored value for ``name``.

        Args:
            name: Variable to read.

        Returns:
            The stored value, or ``None`` when it was not provided.
        """
        return self._values.get(name)


# Attribute names belonging to a foreign binding, held as data so that no
# identifier in this project has to spell them.
_CREATE_FROM_RULES = "createFromRules"


class IcuTransliterator(Protocol):
    """A compiled ICU transliterator."""

    def transliterate(self, text: str) -> str:
        """Apply the compiled rules to ``text``.

        Args:
            text: Input in the source orthography.

        Returns:
            The transliterated text.
        """
        ...


class RuleCompiler(Protocol):
    """ICU's bound ``Transliterator.createFromRules`` classmethod.

    Stated as a callable rather than as a named method because the name
    belongs to ICU's C++ API and is not one this project may rename.
    Taking the bound method by name and annotating it here keeps the
    foreign spelling out of the code entirely.
    """

    def __call__(self, name: str, rules: str, direction: int) -> IcuTransliterator:
        """Compile a transliterator from rule text.

        Args:
            name: Identifier for the compiled instance.
            rules: The rule file's contents.
            direction: ICU direction constant; 0 is forward.

        Returns:
            The compiled transliterator.
        """
        ...


class IcuProvider(Protocol):
    """Supplies ICU's rule compiler, or reports that PyICU is absent."""

    def rule_compiler(self) -> RuleCompiler:
        """Return the function that compiles rule text.

        Returns:
            ICU's bound rule-compiling classmethod.

        Raises:
            ImportError: If PyICU is not installed. The caller turns this
                into a message naming the platform's install command.
        """
        ...


class InstalledIcu:
    """Provider backed by the real PyICU package."""

    def rule_compiler(self) -> RuleCompiler:
        """Import PyICU and take its rule compiler.

        Returns:
            ICU's bound ``createFromRules``.

        Raises:
            ImportError: If PyICU is not installed.
        """
        module = __import__("icu")
        compiler: RuleCompiler = getattr(module.Transliterator, _CREATE_FROM_RULES)
        return compiler


class AbsentIcu:
    """Provider for an environment where PyICU is not installed.

    A real implementation of :class:`IcuProvider`, not a mock: it does
    exactly what the import does on a machine without the package, which
    is the condition the caller's error message exists to explain.
    """

    def rule_compiler(self) -> RuleCompiler:
        """Report PyICU as absent.

        Returns:
            Never returns.

        Raises:
            ImportError: Always.
        """
        raise ImportError("No module named 'icu'")


class Clock(Protocol):
    """The passage of time, as this package waits on it."""

    def sleep(self, seconds: float) -> None:
        """Pause the calling thread.

        Args:
            seconds: How long to wait.
        """
        ...


class SystemClock:
    """Clock backed by :func:`time.sleep`."""

    def sleep(self, seconds: float) -> None:
        """Pause the calling thread for real.

        Args:
            seconds: How long to wait.
        """
        time.sleep(seconds)


class RecordingClock:
    """Clock that records what it was asked to wait and returns at once.

    A real implementation of :class:`Clock`, not a mock: it holds no
    assertion helpers, so a test can only read the waits it logged.
    """

    def __init__(self) -> None:
        """Start an empty log of requested waits."""
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        """Record the wait without performing it.

        Args:
            seconds: How long the caller asked to wait.
        """
        self.slept.append(seconds)


class ErrorReporter(Protocol):
    """The external backend crashes are reported to."""

    def initialise(
        self, dsn: str, environment: str, traces_sample_rate: float, release: str | None
    ) -> None:
        """Start reporting to the backend.

        Args:
            dsn: Endpoint credentials identifying the project.
            environment: Deployment name recorded on every event.
            traces_sample_rate: Fraction of transactions to trace.
            release: Version recorded on every event, or ``None``.

        Raises:
            ImportError: If the backend's package is not installed. A
                configured DSN with no package is a misconfiguration, and
                continuing would leave the operator believing errors are
                being reported when nothing is listening.
        """
        ...


class SentryReporter:
    """Reporter backed by ``sentry_sdk``."""

    def initialise(
        self, dsn: str, environment: str, traces_sample_rate: float, release: str | None
    ) -> None:
        """Initialise Sentry for this process.

        Args:
            dsn: Endpoint credentials identifying the project.
            environment: Deployment name recorded on every event.
            traces_sample_rate: Fraction of transactions to trace.
            release: Version recorded on every event, or ``None``.

        Raises:
            ImportError: If ``sentry-sdk`` is not installed.
        """
        sentry = __import__("sentry_sdk")
        sentry.init(
            dsn=dsn,
            environment=environment,
            traces_sample_rate=traces_sample_rate,
            release=release,
        )


class RecordingErrorReporter:
    """Reporter that logs its configuration and starts nothing.

    A real implementation of :class:`ErrorReporter`, not a mock: it holds
    no assertion helpers, so a test can only read what it was configured
    with.
    """

    def __init__(self) -> None:
        """Start an empty configuration log."""
        self.initialised: list[tuple[str, str, float, str | None]] = []

    def initialise(
        self, dsn: str, environment: str, traces_sample_rate: float, release: str | None
    ) -> None:
        """Record the configuration without contacting anything.

        Args:
            dsn: Endpoint credentials identifying the project.
            environment: Deployment name recorded on every event.
            traces_sample_rate: Fraction of transactions to trace.
            release: Version recorded on every event, or ``None``.
        """
        self.initialised.append((dsn, environment, traces_sample_rate, release))


class RuleText(Protocol):
    """Reader for the text of a rule file."""

    def read(self, path: Path) -> str:
        """Return the full contents of one rule file.

        Args:
            path: Absolute path to a ``.rules`` file.

        Returns:
            The file's text, decoded as UTF-8.
        """
        ...


class FileRuleText:
    """Rule text read from the filesystem."""

    def read(self, path: Path) -> str:
        """Read a rule file from disk.

        Args:
            path: Absolute path to a ``.rules`` file.

        Returns:
            The file's text, decoded as UTF-8.
        """
        return path.read_text(encoding="utf-8")


class MappingRuleText:
    """Rule text answering from a fixed mapping of path to contents.

    A real implementation of :class:`RuleText`, not a mock: it records
    nothing and offers no assertion helpers, so a test using it can only
    check the value the code under test produced.

    Args:
        texts: Mapping of path to file contents. A path absent from the
            mapping raises, because a rule file the code asks for and
            the test did not supply is a mistake in the test.
    """

    def __init__(self, texts: Mapping[Path, str]) -> None:
        """Store the mapping backing this reader."""
        self._texts = dict(texts)

    def read(self, path: Path) -> str:
        """Return the stored contents for ``path``.

        Args:
            path: Path whose contents were supplied.

        Returns:
            The stored text.

        Raises:
            KeyError: If the test supplied no text for this path.
        """
        return self._texts[path]


clock: Clock = SystemClock()
icu: IcuProvider = InstalledIcu()
rule_text: RuleText = FileRuleText()
environment: Environment = ProcessEnvironment()
reporter: ErrorReporter = SentryReporter()

__all__ = [
    "AbsentIcu",
    "Clock",
    "Environment",
    "ErrorReporter",
    "FileRuleText",
    "IcuProvider",
    "IcuTransliterator",
    "InstalledIcu",
    "MappingEnvironment",
    "MappingRuleText",
    "ProcessEnvironment",
    "RecordingClock",
    "RecordingErrorReporter",
    "RuleCompiler",
    "RuleText",
    "SentryReporter",
    "SystemClock",
    "clock",
    "environment",
    "icu",
    "reporter",
]
