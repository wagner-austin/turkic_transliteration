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

import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from turkic_translit.wheels import ReleaseAsset, decode_release_asset


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


class Interpreter(Protocol):
    """The facts about the running interpreter the bootstrap branches on."""

    def platform_name(self) -> str:
        """Name the operating system.

        Returns:
            The platform name, e.g. ``Windows`` or ``Linux``.
        """
        ...

    def version(self) -> tuple[int, int]:
        """Report the running Python version.

        Returns:
            Major and minor version numbers.
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

    def version(self) -> tuple[int, int]:
        """Report the running Python version.

        Returns:
            The major and minor components of ``sys.version_info``.
        """
        major, minor = sys.version_info[:2]
        return (major, minor)

    def executable(self) -> str:
        """Name the running interpreter.

        Returns:
            The value of ``sys.executable``.
        """
        return sys.executable


class DescribedInterpreter:
    """Interpreter facts stated outright rather than probed.

    A real implementation of :class:`Interpreter`, not a mock: it answers
    the three questions and records nothing, so a test can only observe
    what the code under test did with the answers.

    Args:
        platform_name: Operating system to report.
        version: Major and minor Python version to report.
        executable: Interpreter path to report.
    """

    def __init__(self, platform_name: str, version: tuple[int, int], executable: str) -> None:
        """Store the three facts this interpreter reports."""
        self._platform_name = platform_name
        self._version = version
        self._executable = executable

    def platform_name(self) -> str:
        """Return the stated operating system.

        Returns:
            The name given at construction.
        """
        return self._platform_name

    def version(self) -> tuple[int, int]:
        """Return the stated Python version.

        Returns:
            The version given at construction.
        """
        return self._version

    def executable(self) -> str:
        """Return the stated interpreter path.

        Returns:
            The path given at construction.
        """
        return self._executable


class ReleaseIndex(Protocol):
    """The GitHub releases the bootstrap takes wheels from."""

    def latest_assets(self, api_url: str) -> tuple[ReleaseAsset, ...]:
        """List every asset attached to the latest release.

        Args:
            api_url: Releases endpoint to query.

        Returns:
            The assets, in the order the API published them.

        Raises:
            FieldError: If an asset is missing its name or its URL.
        """
        ...

    def download(self, url: str, destination: Path) -> None:
        """Fetch a wheel to a local path.

        Args:
            url: Absolute URL of the wheel.
            destination: Path to write the bytes to.

        Raises:
            URLError: If the URL is unreachable or does not exist.
        """
        ...


class GitHubReleaseIndex:
    """Release index backed by the GitHub API and plain HTTP downloads."""

    def latest_assets(self, api_url: str) -> tuple[ReleaseAsset, ...]:
        """Read the latest release's assets from the API.

        A response that is not an object, or whose ``assets`` is not a
        list, yields no assets rather than an exception: the caller
        reports "this release publishes no wheel for you", which is the
        same outcome and names the interpreter that needs one.

        Args:
            api_url: Releases endpoint to query.

        Returns:
            The validated assets, in API order.

        Raises:
            FieldError: If an asset is missing its name or its URL.
        """
        with urllib.request.urlopen(api_url) as response:
            document = json.load(response)
        if not isinstance(document, Mapping):
            return ()
        assets = document.get("assets")
        if not isinstance(assets, list):
            return ()
        return tuple(decode_release_asset(asset) for asset in assets if isinstance(asset, Mapping))

    def download(self, url: str, destination: Path) -> None:
        """Fetch a wheel to a local path.

        Args:
            url: Absolute URL of the wheel.
            destination: Path to write the bytes to.

        Raises:
            URLError: If the URL is unreachable or does not exist.
        """
        urllib.request.urlretrieve(url, destination)


class TableReleaseIndex:
    """Release index answering from stated assets and recording downloads.

    A real implementation of :class:`ReleaseIndex`, not a mock: it holds
    no assertion helpers, so a test can only inspect the wheels it was
    asked for and the files it wrote.

    Args:
        assets: The assets this index reports for any queried endpoint.
        contents: Bytes to write for a downloaded wheel.
    """

    def __init__(self, assets: Sequence[ReleaseAsset], contents: bytes = b"wheel") -> None:
        """Store the assets and start an empty download log."""
        self._assets = tuple(assets)
        self._contents = contents
        self.queried: list[str] = []
        self.downloaded: list[tuple[str, Path]] = []

    def latest_assets(self, api_url: str) -> tuple[ReleaseAsset, ...]:
        """Record the query and return the stated assets.

        Args:
            api_url: Releases endpoint the caller asked about.

        Returns:
            The assets given at construction.
        """
        self.queried.append(api_url)
        return self._assets

    def download(self, url: str, destination: Path) -> None:
        """Record the download and write the stated bytes.

        Args:
            url: Absolute URL the caller asked for.
            destination: Path the bytes are written to.
        """
        self.downloaded.append((url, destination))
        destination.write_bytes(self._contents)


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


class Installer(Protocol):
    """Installs a wheel into an interpreter."""

    def install(self, interpreter: str, wheel: Path) -> None:
        """Install one wheel.

        Args:
            interpreter: Path of the interpreter to install into.
            wheel: Path of the wheel file.

        Raises:
            CalledProcessError: If the installation exits non-zero.
        """
        ...


class PipInstaller:
    """Installer backed by ``python -m pip install``."""

    def install(self, interpreter: str, wheel: Path) -> None:
        """Install one wheel with pip.

        Args:
            interpreter: Path of the interpreter to install into.
            wheel: Path of the wheel file. Output is inherited so the
                developer sees pip's progress.

        Raises:
            CalledProcessError: If pip exits non-zero.
        """
        subprocess.check_call([interpreter, "-m", "pip", "install", str(wheel)])


class RecordingInstaller:
    """Installer that logs what it was asked to install and installs nothing.

    A real implementation of :class:`Installer`, not a mock: it holds no
    assertion helpers, so a test can only read the log it kept.
    """

    def __init__(self) -> None:
        """Start an empty installation log."""
        self.installed: list[tuple[str, Path]] = []

    def install(self, interpreter: str, wheel: Path) -> None:
        """Record the request without running anything.

        Args:
            interpreter: Path of the interpreter that was named.
            wheel: Path of the wheel that was named.
        """
        self.installed.append((interpreter, wheel))


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
interpreter: Interpreter = RunningInterpreter()
releases: ReleaseIndex = GitHubReleaseIndex()
installer: Installer = PipInstaller()
reporter: ErrorReporter = SentryReporter()

__all__ = [
    "AbsentIcu",
    "Clock",
    "DescribedInterpreter",
    "Environment",
    "ErrorReporter",
    "GitHubReleaseIndex",
    "IcuProvider",
    "IcuTransliterator",
    "InstalledIcu",
    "Installer",
    "Interpreter",
    "FileRuleText",
    "MappingEnvironment",
    "MappingRuleText",
    "PipInstaller",
    "ProcessEnvironment",
    "RecordingClock",
    "RuleText",
    "RecordingErrorReporter",
    "RecordingInstaller",
    "ReleaseIndex",
    "RuleCompiler",
    "RunningInterpreter",
    "SentryReporter",
    "SystemClock",
    "TableReleaseIndex",
    "clock",
    "environment",
    "icu",
    "installer",
    "interpreter",
    "releases",
    "reporter",
]
