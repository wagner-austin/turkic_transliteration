"""Tests for typed language-identification model selection.

These exercise the real registry, the real decoder, and the real
resolution logic. The only injected component is the filesystem probe,
bound to :class:`MappingFileProbe`, which is a real implementation of the
:class:`FileProbe` protocol rather than a mock: it answers from a
dictionary instead of from disk, and every assertion below checks a
returned value or a raised error code.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from turkic_translit.lid import _test_hooks
from turkic_translit.lid.errors import (
    ERR_LABEL_MALFORMED,
    ERR_MODEL_FILE_EMPTY,
    ERR_MODEL_FILE_MISSING,
    ERR_UNKNOWN_MODEL,
    LidLabelError,
    LidModelFileEmptyError,
    LidModelFileMissingError,
    UnknownLidModelError,
)
from turkic_translit.lid.registry import (
    MIRROR_REPO,
    REGISTRY,
    get_spec,
    known_model_ids,
    resolve_model_path,
)
from turkic_translit.lid.spec import (
    decode_lid_model_spec,
    encode_lid_model_spec,
    strip_label_prefix,
)
from turkic_translit.validation import (
    ERR_FIELD_EMPTY,
    ERR_FIELD_MISSING,
    ERR_FIELD_TYPE,
    FieldError,
)


def _spec_fields(**changes: str | bool) -> dict[str, str | bool]:
    """Build a complete, valid specification mapping, with some fields changed.

    Args:
        **changes: Fields to replace in the valid mapping.

    Returns:
        A mapping every field of which decodes, apart from those changed.
    """
    fields: dict[str, str | bool] = {
        "model_id": "x",
        "filename": "x.bin",
        "url": "https://example.org/x.bin",
        "origin_url": "https://example.org/origin/x.bin",
        "sha256": "0" * 64,
        "label_prefix": "__label__",
        "script_aware": True,
    }
    fields.update(changes)
    return fields


@pytest.fixture
def restore_probe() -> Generator[None, None, None]:
    """Restore the production filesystem probe after a test rebinds it.

    Yields:
        None, once, with the original probe captured.
    """
    original = _test_hooks.probe
    yield
    _test_hooks.probe = original


def test_registry_exposes_both_classifiers() -> None:
    """Both the script-blind and script-aware models are registered."""
    assert known_model_ids() == ("lid.176", "lid218e")


def test_lid176_is_not_script_aware() -> None:
    """fastText's 176-language model does not encode script in labels."""
    spec = get_spec("lid.176")
    assert spec["filename"] == "lid.176.bin"
    assert spec["script_aware"] is False


def test_lid218e_is_script_aware() -> None:
    """NLLB's model encodes script, which is why Uzbek needs it."""
    spec = get_spec("lid218e")
    assert spec["filename"] == "lid218e.bin"
    assert spec["script_aware"] is True


def test_unknown_model_names_the_alternatives() -> None:
    """An unregistered id reports its code and every valid id."""
    with pytest.raises(UnknownLidModelError) as excinfo:
        get_spec("lid999")
    assert excinfo.value.code == ERR_UNKNOWN_MODEL
    assert excinfo.value.known == ("lid.176", "lid218e")


def test_spec_round_trips_through_encode_decode() -> None:
    """Encoding a spec and decoding it again yields an equal spec."""
    original = get_spec("lid218e")
    assert decode_lid_model_spec(encode_lid_model_spec(original)) == original


def test_decode_rejects_missing_field() -> None:
    """A specification missing a required key fails with a field code."""
    with pytest.raises(FieldError) as excinfo:
        decode_lid_model_spec(
            {key: value for key, value in _spec_fields().items() if key != "label_prefix"}
        )
    assert excinfo.value.code == ERR_FIELD_MISSING
    assert excinfo.value.field == "label_prefix"


def test_decode_rejects_wrong_type() -> None:
    """A non-boolean ``script_aware`` fails with a type code."""
    with pytest.raises(FieldError) as excinfo:
        decode_lid_model_spec(_spec_fields(script_aware="yes"))
    assert excinfo.value.code == ERR_FIELD_TYPE
    assert excinfo.value.field == "script_aware"


def test_decode_rejects_empty_string() -> None:
    """A whitespace-only field fails with an empty-value code."""
    with pytest.raises(FieldError) as excinfo:
        decode_lid_model_spec(_spec_fields(model_id="  "))
    assert excinfo.value.code == ERR_FIELD_EMPTY
    assert excinfo.value.field == "model_id"


def test_resolve_returns_first_directory_that_holds_the_weights(
    restore_probe: None,
) -> None:
    """Search order is honoured and the first hit is returned."""
    second = Path("/models/b")
    _test_hooks.probe = _test_hooks.MappingFileProbe({second / "lid218e.bin": 4096})
    resolved = resolve_model_path("lid218e", [Path("/models/a"), second])
    assert resolved == second / "lid218e.bin"


def test_resolve_never_substitutes_a_different_model(restore_probe: None) -> None:
    """Having lid.176 on disk does not satisfy a request for lid218e."""
    directory = Path("/models")
    _test_hooks.probe = _test_hooks.MappingFileProbe({directory / "lid.176.bin": 4096})
    with pytest.raises(LidModelFileMissingError) as excinfo:
        resolve_model_path("lid218e", [directory])
    assert excinfo.value.code == ERR_MODEL_FILE_MISSING
    assert excinfo.value.model_id == "lid218e"


def test_resolve_rejects_zero_byte_weights(restore_probe: None) -> None:
    """A truncated download is an error, not a silent cache miss."""
    directory = Path("/models")
    _test_hooks.probe = _test_hooks.MappingFileProbe({directory / "lid218e.bin": 0})
    with pytest.raises(LidModelFileEmptyError) as excinfo:
        resolve_model_path("lid218e", [directory])
    assert excinfo.value.code == ERR_MODEL_FILE_EMPTY
    assert excinfo.value.path == directory / "lid218e.bin"


def test_resolve_with_no_search_directories_reports_the_filename(
    restore_probe: None,
) -> None:
    """An empty search path still names the file that was wanted."""
    _test_hooks.probe = _test_hooks.MappingFileProbe({})
    with pytest.raises(LidModelFileMissingError) as excinfo:
        resolve_model_path("lid.176", [])
    assert excinfo.value.path == Path("lid.176.bin")


def test_resolve_rejects_unknown_model_before_touching_disk(
    restore_probe: None,
) -> None:
    """Validation precedes IO, so a bad id never probes the filesystem."""
    _test_hooks.probe = _test_hooks.MappingFileProbe({})
    with pytest.raises(UnknownLidModelError):
        resolve_model_path("lid999", [Path("/models")])


def test_strip_label_prefix_yields_the_bare_language_tag() -> None:
    """A script-aware label loses only its prefix, keeping the script."""
    assert strip_label_prefix(get_spec("lid218e"), "__label__uzn_Latn") == "uzn_Latn"


def test_strip_label_prefix_rejects_a_foreign_label() -> None:
    """A label without the declared prefix means the wrong weights loaded."""
    with pytest.raises(LidLabelError) as excinfo:
        strip_label_prefix(get_spec("lid.176"), "kk")
    assert excinfo.value.code == ERR_LABEL_MALFORMED
    assert excinfo.value.expected_prefix == "__label__"


def test_real_probe_reads_a_written_file(tmp_path: Path) -> None:
    """The production probe reports real existence and real size."""
    target = tmp_path / "weights.bin"
    target.write_bytes(b"0123456789")
    real = _test_hooks.RealFileProbe()
    assert real.exists(target) is True
    assert real.size_bytes(target) == 10


def test_real_probe_reports_absence(tmp_path: Path) -> None:
    """The production probe reports a missing path as non-existent."""
    assert _test_hooks.RealFileProbe().exists(tmp_path / "absent.bin") is False


def test_decode_rejects_non_string_where_string_required() -> None:
    """A boolean supplied for a string field fails with a type code."""
    with pytest.raises(FieldError) as excinfo:
        decode_lid_model_spec(_spec_fields(model_id=True))
    assert excinfo.value.code == ERR_FIELD_TYPE
    assert excinfo.value.field == "model_id"


def test_every_model_downloads_from_the_mirror_and_names_meta_as_its_origin() -> None:
    """No registered model is downloaded from the host that crashed the demo.

    On 2026-11-15 the Space failed at startup because its download from
    ``dl.fbaipublicfiles.com`` failed certificate verification. That host
    stays in the registry as each file's origin, the place the mirror is
    copied from, and nowhere a download reads.
    """
    for spec in REGISTRY.values():
        assert spec["url"] == (
            f"https://huggingface.co/{MIRROR_REPO}/resolve/main/{spec['filename']}"
        )
        assert spec["origin_url"].startswith("https://dl.fbaipublicfiles.com/")
        assert spec["origin_url"].endswith(f"/{spec['filename']}")


def test_the_pinned_digests_are_the_ones_measured_from_meta() -> None:
    """The digests are the SHA-256 of Meta's files, read on 2026-11-15.

    Measured by streaming each ``origin_url`` through ``sha256sum``; the
    byte counts, 131,266,198 and 1,176,355,829, matched the
    ``Content-Length`` Meta's CDN reported.
    """
    assert get_spec("lid.176")["sha256"] == (
        "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
    )
    assert get_spec("lid218e")["sha256"] == (
        "8ded5749a2ad79ae9ab7c9190c7c8b97ff20d54ad8b9527ffa50107238fc7f6a"
    )


@pytest.mark.parametrize("digest", ["0" * 63, "A" * 64, "g" * 64])
def test_decode_rejects_a_digest_that_is_not_a_sha256(digest: str) -> None:
    """A short, uppercase or non-hexadecimal digest is refused by name.

    A digest that cannot be a SHA-256 compares equal to nothing, so it
    would refuse every download for a reason nobody could read off it.

    Args:
        digest: The malformed digest.
    """
    with pytest.raises(FieldError) as excinfo:
        decode_lid_model_spec(_spec_fields(sha256=digest))
    assert excinfo.value.code == ERR_FIELD_TYPE
    assert excinfo.value.field == "sha256"
