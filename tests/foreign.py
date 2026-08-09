"""Typed access to the untyped libraries the verification tests use.

``icu`` and ``panphon`` ship no type information. Rather than exempt the
tests that use them from strict checking, each is imported here by name
and immediately bound to a protocol stating the part of it the tests
touch — the same treatment the package gives every other foreign
surface, applied where the tests need it.

The bound names are taken with :func:`getattr` because they are spelled
in a foreign convention this project may not rename, and reaching them
that way keeps those spellings out of every identifier here.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import IO, Literal, Protocol

from turkic_translit._test_hooks import IcuTransliterator

# Names belonging to a foreign binding, held as data.
_TRANSLITERATOR = "Transliterator"
_CREATE_INSTANCE = "createInstance"
_FEATURE_TABLE = "FeatureTable"
_FEATURETABLE_MODULE = "panphon.featuretable"
_FILES = "files"


class TransliteratorFactory(Protocol):
    """ICU's bound ``Transliterator.createInstance`` classmethod."""

    def __call__(self, identifier: str) -> IcuTransliterator:
        """Build one of ICU's own published transliterators.

        Args:
            identifier: An ICU transform id, e.g. ``Any-Latin; NFC``.

        Returns:
            The transliterator that id names.
        """
        ...


class FeatureTable(Protocol):
    """panphon's phonological feature table, as the tests read it."""

    def word_to_vector_list(self, word: str) -> Sequence[Sequence[str]]:
        """Describe each segment of an IPA string as a feature vector.

        Args:
            word: An IPA transcription.

        Returns:
            One vector per segment, in order. Each value is ``+``, ``-``
            or ``0`` — panphon reports feature values as characters, not
            as numbers.
        """
        ...


class ResourceDirectory(Protocol):
    """A package's data files, as panphon walks and opens them.

    Stated as the two operations panphon performs rather than as
    ``importlib.abc.Traversable``, because a replacement supplied by a
    test satisfies these two and is not a subclass of that abstract
    class. This is the interface the swap actually has to honour.
    """

    def joinpath(self, *parts: str) -> ResourceDirectory:
        """Descend into a child path.

        Args:
            *parts: Path components to append.

        Returns:
            The child, addressable the same way.
        """
        ...

    def open(self, mode: Literal["r"] = "r", encoding: str = "utf-8") -> IO[str]:
        """Open this resource as text.

        Args:
            mode: File mode; panphon reads text at every call site.
            encoding: Codec to decode the bytes with.

        Returns:
            The opened text stream.
        """
        ...


class ResourceLocator(Protocol):
    """panphon's bound ``importlib.resources.files``.

    panphon opens its packaged CSVs through this function, which is the
    seam the UTF-8 workaround replaces.
    """

    def __call__(self, package: str) -> ResourceDirectory:
        """Locate a package's data files.

        Args:
            package: Dotted name of the package to read from.

        Returns:
            The package's data files.
        """
        ...


def panphon_resource_module() -> ModuleType:
    """Return the panphon module whose file access is intercepted.

    Returns:
        ``panphon.featuretable``.
    """
    return __import__(_FEATURETABLE_MODULE, fromlist=[_FILES])


def replace_panphon_resource_locator(locator: ResourceLocator) -> ResourceLocator:
    """Install a resource locator into panphon, returning the old one.

    The swap lives here rather than in the test because the attribute
    being replaced is panphon's, and reaching it by name is what keeps
    the module's untyped surface from leaking into the test.

    Args:
        locator: The locator panphon should use from now on.

    Returns:
        The locator that was in place, so the caller can restore it.
    """
    module = panphon_resource_module()
    previous: ResourceLocator = getattr(module, _FILES)
    setattr(module, _FILES, locator)
    return previous


def icu_transliterator_factory() -> TransliteratorFactory:
    """Return ICU's factory for its own published transliterators.

    Returns:
        The bound ``Transliterator.createInstance``.
    """
    module = __import__("icu")
    factory: TransliteratorFactory = getattr(getattr(module, _TRANSLITERATOR), _CREATE_INSTANCE)
    return factory


def panphon_feature_table() -> FeatureTable:
    """Build panphon's feature table.

    Returns:
        The table, narrowed to the one method the tests call.
    """
    module = __import__("panphon")
    table: FeatureTable = getattr(module, _FEATURE_TABLE)()
    return table
