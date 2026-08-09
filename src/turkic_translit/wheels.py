"""Selection of the pre-built PyICU wheel a Windows interpreter needs.

PyICU publishes no Windows wheel to PyPI, so the bootstrap installer
takes one from the ``cgohlke/pyicu-build`` GitHub releases. Everything
here is the decision about *which* wheel: naming it, recognising it in a
release listing, and rejecting the interpreters no wheel exists for. The
fetching and installing are effects and live behind the hooks in
:mod:`turkic_translit._test_hooks`.

Keeping the decision separate from the effects is what lets it be
exercised exactly: every rejection below is a distinct error code, and a
test states a release listing rather than reaching the network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, TypedDict

from turkic_translit.validation import require_non_empty_str, require_present

ERR_UNSUPPORTED_PLATFORM: Final = "TURKIC_WHEEL_001_UNSUPPORTED_PLATFORM"
ERR_UNSUPPORTED_PYTHON: Final = "TURKIC_WHEEL_002_UNSUPPORTED_PYTHON"
ERR_NO_MATCHING_ASSET: Final = "TURKIC_WHEEL_003_NO_MATCHING_ASSET"

WINDOWS: Final = "Windows"
PLATFORM_TAG: Final = "win_amd64"
RELEASES_API_URL: Final = "https://api.github.com/repos/cgohlke/pyicu-build/releases/latest"
DOWNLOAD_URL_TEMPLATE: Final = (
    "https://github.com/cgohlke/pyicu-build/releases/download/v{version}/{wheel_name}"
)

SUPPORTED_PYTHON_TAGS: Final[tuple[str, ...]] = ("cp310", "cp311", "cp312", "cp313")


class WheelError(Exception):
    """Base class for wheel-selection failures.

    Args:
        code: Stable error code from this module.
        message: Human-readable description naming the offending value.
    """

    def __init__(self, code: str, message: str) -> None:
        """Store the code and render ``code: message`` as the string form."""
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class UnsupportedPlatformError(WheelError):
    """Raised on a platform that has no wheel to install."""

    def __init__(self, platform_name: str) -> None:
        """Name the platform and say why nothing needs installing there.

        Args:
            platform_name: Operating system the interpreter reports.
        """
        super().__init__(
            ERR_UNSUPPORTED_PLATFORM,
            f"this installer only applies to {WINDOWS}, not {platform_name!r}; "
            f"PyICU publishes wheels to PyPI for every other platform, so "
            f"pip install pyicu is the whole procedure there",
        )
        self.platform_name = platform_name


class UnsupportedPythonError(WheelError):
    """Raised when no wheel is built for the running Python version."""

    def __init__(self, tag: str) -> None:
        """Name the rejected tag and list every tag that would have worked.

        Args:
            tag: Interpreter tag derived from the running version.
        """
        super().__init__(
            ERR_UNSUPPORTED_PYTHON,
            f"no pre-built PyICU wheel exists for {tag}; "
            f"wheels are published for {', '.join(SUPPORTED_PYTHON_TAGS)}",
        )
        self.tag = tag


class NoMatchingAssetError(WheelError):
    """Raised when a release carries no wheel for this interpreter."""

    def __init__(self, tag: str, available: tuple[str, ...]) -> None:
        """Name the tag looked for and every asset the release published.

        Args:
            tag: Interpreter tag the wheel had to carry.
            available: Names of every asset in the release, in order.
        """
        super().__init__(
            ERR_NO_MATCHING_ASSET,
            f"the latest pyicu-build release publishes no {PLATFORM_TAG} wheel "
            f"for {tag}; it publishes {', '.join(available) or '<nothing>'}",
        )
        self.tag = tag
        self.available = available


class ReleaseAsset(TypedDict):
    """One downloadable file attached to a GitHub release.

    Attributes:
        name: File name of the asset, e.g. ``pyicu-2.15-cp313-...whl``.
        download_url: Absolute URL the asset's bytes are served from.
    """

    name: str
    download_url: str


def encode_release_asset(asset: ReleaseAsset) -> dict[str, str]:
    """Render an asset in the shape the GitHub API publishes it in.

    Args:
        asset: The asset to encode.

    Returns:
        A mapping using the API's own field names, so a captured
        response and an encoded asset are interchangeable.
    """
    return {"name": asset["name"], "browser_download_url": asset["download_url"]}


def decode_release_asset(source: Mapping[str, str | int | float | bool | None]) -> ReleaseAsset:
    """Validate one entry of a release's ``assets`` list.

    Args:
        source: One asset object as decoded from the API response.

    Returns:
        The validated asset.

    Raises:
        FieldError: If either field is missing, is not a string, or is
            empty. An asset with no name or no URL cannot be installed,
            so it is rejected rather than skipped.
    """
    return ReleaseAsset(
        name=require_non_empty_str("name", require_present("name", source)),
        download_url=require_non_empty_str(
            "browser_download_url", require_present("browser_download_url", source)
        ),
    )


def python_tag(version: tuple[int, int]) -> str:
    """Render an interpreter version as the tag wheels are named with.

    Args:
        version: Major and minor version of the running interpreter.

    Returns:
        The tag, e.g. ``cp313`` for Python 3.13.
    """
    major, minor = version
    return f"cp{major}{minor}"


def require_installable(platform_name: str, tag: str) -> None:
    """Raise unless a wheel exists for this platform and interpreter.

    Args:
        platform_name: Operating system the interpreter reports.
        tag: Interpreter tag from :func:`python_tag`.

    Raises:
        UnsupportedPlatformError: If the platform is not Windows.
        UnsupportedPythonError: If no wheel is built for the tag.
    """
    if platform_name != WINDOWS:
        raise UnsupportedPlatformError(platform_name)
    if tag not in SUPPORTED_PYTHON_TAGS:
        raise UnsupportedPythonError(tag)


def matches(asset_name: str, tag: str) -> bool:
    """Report whether an asset is the wheel this interpreter needs.

    Args:
        asset_name: File name of the asset.
        tag: Interpreter tag the wheel must carry.

    Returns:
        True when the name carries both the interpreter tag and the
        platform tag.
    """
    return tag in asset_name and PLATFORM_TAG in asset_name


def select_asset(assets: Sequence[ReleaseAsset], tag: str) -> ReleaseAsset:
    """Pick the wheel for this interpreter out of a release's assets.

    Args:
        assets: Every asset the release publishes, in API order.
        tag: Interpreter tag the wheel must carry.

    Returns:
        The first matching asset, which is the only one a release
        publishes per interpreter.

    Raises:
        NoMatchingAssetError: If no asset carries both tags.
    """
    for asset in assets:
        if matches(asset["name"], tag):
            return asset
    raise NoMatchingAssetError(tag, tuple(asset["name"] for asset in assets))


def pinned_asset(version: str, tag: str) -> ReleaseAsset:
    """Name the wheel of an explicitly requested PyICU version.

    The release tag and file name follow a fixed scheme, so a pinned
    version is resolved by construction rather than by querying the API.
    A version that was never released fails at download time, naming the
    URL that did not exist.

    Args:
        version: PyICU version requested, e.g. ``2.15``.
        tag: Interpreter tag from :func:`python_tag`.

    Returns:
        The asset that version and interpreter combination refers to.
    """
    wheel_name = f"pyicu-{version}-{tag}-{tag}-{PLATFORM_TAG}.whl"
    return ReleaseAsset(
        name=wheel_name,
        download_url=DOWNLOAD_URL_TEMPLATE.format(version=version, wheel_name=wheel_name),
    )


__all__ = [
    "DOWNLOAD_URL_TEMPLATE",
    "ERR_NO_MATCHING_ASSET",
    "ERR_UNSUPPORTED_PLATFORM",
    "ERR_UNSUPPORTED_PYTHON",
    "PLATFORM_TAG",
    "RELEASES_API_URL",
    "SUPPORTED_PYTHON_TAGS",
    "WINDOWS",
    "NoMatchingAssetError",
    "ReleaseAsset",
    "UnsupportedPlatformError",
    "UnsupportedPythonError",
    "WheelError",
    "decode_release_asset",
    "encode_release_asset",
    "matches",
    "pinned_asset",
    "python_tag",
    "require_installable",
    "select_asset",
]
