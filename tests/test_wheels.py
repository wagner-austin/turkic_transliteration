"""Tests for the PyICU wheel-selection decision.

Nothing here reaches the network or the filesystem: wheel selection is a
pure decision over an interpreter's version, its platform, and a release
listing, and every rejection carries its own error code.
"""

from __future__ import annotations

import pytest

from turkic_translit.validation import ERR_FIELD_EMPTY, ERR_FIELD_MISSING, FieldError
from turkic_translit.wheels import (
    DOWNLOAD_URL_TEMPLATE,
    ERR_NO_MATCHING_ASSET,
    ERR_UNSUPPORTED_PLATFORM,
    ERR_UNSUPPORTED_PYTHON,
    PLATFORM_TAG,
    SUPPORTED_PYTHON_TAGS,
    WINDOWS,
    NoMatchingAssetError,
    ReleaseAsset,
    UnsupportedPlatformError,
    UnsupportedPythonError,
    decode_release_asset,
    encode_release_asset,
    matches,
    pinned_asset,
    python_tag,
    require_installable,
    select_asset,
)


def wheel(name: str) -> ReleaseAsset:
    """Build an asset whose URL is irrelevant to the test.

    Args:
        name: File name the asset carries.

    Returns:
        An asset named ``name`` with a plausible download URL.
    """
    return ReleaseAsset(name=name, download_url=f"https://example.invalid/{name}")


@pytest.mark.parametrize(
    ("version", "expected"),
    [((3, 10), "cp310"), ((3, 13), "cp313"), ((4, 0), "cp40")],
)
def test_python_tag_renders_the_version(version: tuple[int, int], expected: str) -> None:
    """The tag is the major and minor numbers run together after ``cp``.

    Args:
        version: Interpreter version to render.
        expected: The tag wheels for that version are named with.
    """
    assert python_tag(version) == expected


def test_every_supported_tag_is_installable() -> None:
    """Each published tag passes the platform and interpreter check."""
    for tag in SUPPORTED_PYTHON_TAGS:
        require_installable(WINDOWS, tag)


def test_non_windows_is_rejected_with_its_own_code() -> None:
    """Another platform is told to use PyPI rather than this installer."""
    with pytest.raises(UnsupportedPlatformError) as raised:
        require_installable("Linux", "cp313")

    assert raised.value.code == ERR_UNSUPPORTED_PLATFORM
    assert raised.value.platform_name == "Linux"
    assert "pip install pyicu" in raised.value.message


def test_unbuilt_python_is_rejected_with_the_tags_that_work() -> None:
    """An unsupported interpreter learns the whole supported vocabulary."""
    with pytest.raises(UnsupportedPythonError) as raised:
        require_installable(WINDOWS, "cp39")

    assert raised.value.code == ERR_UNSUPPORTED_PYTHON
    assert raised.value.tag == "cp39"
    for tag in SUPPORTED_PYTHON_TAGS:
        assert tag in raised.value.message


def test_platform_is_checked_before_the_interpreter() -> None:
    """A Linux run is told about the platform, not about its Python."""
    with pytest.raises(UnsupportedPlatformError):
        require_installable("Linux", "cp39")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (f"pyicu-2.15-cp313-cp313-{PLATFORM_TAG}.whl", True),
        ("pyicu-2.15-cp312-cp312-win_amd64.whl", False),
        ("pyicu-2.15-cp313-cp313-win32.whl", False),
        ("Source code (zip)", False),
    ],
)
def test_matches_requires_both_tags(name: str, expected: bool) -> None:
    """A wheel matches only when it carries the interpreter and platform.

    Args:
        name: Asset file name to test.
        expected: Whether it is the wheel a cp313 Windows run needs.
    """
    assert matches(name, "cp313") is expected


def test_select_asset_returns_the_matching_wheel() -> None:
    """Selection picks the one asset built for this interpreter."""
    wanted = wheel(f"pyicu-2.15-cp313-cp313-{PLATFORM_TAG}.whl")
    assets = [wheel("Source code (zip)"), wheel("pyicu-2.15-cp312-cp312-win32.whl"), wanted]

    assert select_asset(assets, "cp313") == wanted


def test_select_asset_takes_the_first_of_several_matches() -> None:
    """Releases publish one wheel per interpreter; the first is it."""
    first = wheel(f"pyicu-2.15-cp313-cp313-{PLATFORM_TAG}.whl")
    second = wheel(f"pyicu-2.16-cp313-cp313-{PLATFORM_TAG}.whl")

    assert select_asset([first, second], "cp313") == first


def test_select_asset_reports_what_the_release_did_publish() -> None:
    """A release with no matching wheel names everything it does carry."""
    assets = [wheel("Source code (zip)"), wheel("pyicu-2.15-cp312-cp312-win32.whl")]

    with pytest.raises(NoMatchingAssetError) as raised:
        select_asset(assets, "cp313")

    assert raised.value.code == ERR_NO_MATCHING_ASSET
    assert raised.value.tag == "cp313"
    assert raised.value.available == ("Source code (zip)", "pyicu-2.15-cp312-cp312-win32.whl")
    assert "Source code (zip)" in raised.value.message


def test_select_asset_reports_an_empty_release_readably() -> None:
    """A release with no assets at all still produces a legible message."""
    with pytest.raises(NoMatchingAssetError) as raised:
        select_asset([], "cp313")

    assert raised.value.available == ()
    assert "<nothing>" in raised.value.message


def test_pinned_asset_builds_the_published_name_and_url() -> None:
    """A pinned version resolves by construction, not by an API call."""
    asset = pinned_asset("2.15", "cp313")

    assert asset["name"] == f"pyicu-2.15-cp313-cp313-{PLATFORM_TAG}.whl"
    assert asset["download_url"] == DOWNLOAD_URL_TEMPLATE.format(
        version="2.15", wheel_name=asset["name"]
    )


def test_pinned_asset_matches_its_own_interpreter_tag() -> None:
    """What the pin builds is what selection would have accepted."""
    asset = pinned_asset("2.15", "cp311")

    assert matches(asset["name"], "cp311")


def test_release_asset_round_trips_through_the_api_shape() -> None:
    """Encoding then decoding an asset returns the same asset."""
    asset = wheel(f"pyicu-2.15-cp313-cp313-{PLATFORM_TAG}.whl")

    assert decode_release_asset(encode_release_asset(asset)) == asset


def test_encode_release_asset_uses_the_api_field_names() -> None:
    """The encoded form is interchangeable with a captured response."""
    encoded = encode_release_asset(wheel("pyicu.whl"))

    assert sorted(encoded) == ["browser_download_url", "name"]


def test_decode_release_asset_requires_a_name() -> None:
    """An asset with no name cannot be installed and is rejected."""
    with pytest.raises(FieldError) as raised:
        decode_release_asset({"browser_download_url": "https://example.invalid/x.whl"})

    assert raised.value.code == ERR_FIELD_MISSING
    assert raised.value.field == "name"


def test_decode_release_asset_requires_a_download_url() -> None:
    """An asset with no URL cannot be fetched and is rejected."""
    with pytest.raises(FieldError) as raised:
        decode_release_asset({"name": "pyicu.whl"})

    assert raised.value.code == ERR_FIELD_MISSING
    assert raised.value.field == "browser_download_url"


def test_decode_release_asset_rejects_an_empty_name() -> None:
    """A present-but-blank name is a malformed release, not an absent one."""
    with pytest.raises(FieldError) as raised:
        decode_release_asset({"name": "  ", "browser_download_url": "https://example.invalid/"})

    assert raised.value.code == ERR_FIELD_EMPTY
    assert raised.value.field == "name"
