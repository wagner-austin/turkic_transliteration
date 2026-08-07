"""Tests for the SentencePiece training command.

SentencePiece is trained for real on small synthetic corpora, so the
trainer arguments this command builds are proven to be arguments
SentencePiece accepts. Only the corpus source is substituted, by an
in-memory implementation of the production streaming protocol, which is
what keeps the test offline and fast.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from click.testing import CliRunner

from turkic_translit.cli.train_spm import (
    build_manifest_document,
    build_trainer_arguments,
    main,
    sha256_of_file,
)
from turkic_translit.corpus import _test_hooks as corpus_hooks
from turkic_translit.lid import _test_hooks as lid_hooks
from turkic_translit.lid.locations import default_search_dirs

_KAZAKH = [f"qazaq tili {index} salem alem birinshi" for index in range(120)]
_KYRGYZ = [f"kyrgyz tili {index} salam dune birinchi" for index in range(120)]

ANSWERS = dict.fromkeys(_KAZAKH, ("__label__kk", 0.99)) | dict.fromkeys(
    _KYRGYZ, ("__label__ky", 0.99)
)


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click runner with a UTF-8 environment.

    Returns:
        The runner every command test drives.
    """
    return CliRunner(env={"PYTHONIOENCODING": "utf8", "HF_TOKEN": ""})


@pytest.fixture
def two_language_source() -> Generator[None, None, None]:
    """Serve Kazakh and Kyrgyz configurations from memory.

    Yields:
        None, once, with the original streamer captured.
    """
    original = corpus_hooks.dataset_texts
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer(
        {"kk": _KAZAKH, "ky": _KYRGYZ}
    )
    yield
    corpus_hooks.dataset_texts = original


@pytest.fixture
def installed_weights() -> Generator[None, None, None]:
    """Present ``lid.176`` as installed and back it with a table model.

    Yields:
        None, once, with the original hooks captured.
    """
    probe, loader = lid_hooks.probe, lid_hooks.model_loader
    lid_hooks.probe = lid_hooks.MappingFileProbe(
        {default_search_dirs()[0] / "lid.176.bin": 131266198}
    )
    lid_hooks.model_loader = lid_hooks.FixedModelLoader(lid_hooks.TableFastTextModel(ANSWERS))
    yield
    lid_hooks.probe = probe
    lid_hooks.model_loader = loader


def test_sha256_matches_a_known_digest(tmp_path: Path) -> None:
    """The chunked digest equals the published digest of the empty string."""
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")
    assert sha256_of_file(target) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_reads_a_file_larger_than_one_chunk(tmp_path: Path) -> None:
    """A multi-chunk read produces the same digest as a single hash call."""
    import hashlib

    payload = b"a" * (1 << 21)
    target = tmp_path / "large.bin"
    target.write_bytes(payload)
    assert sha256_of_file(target) == hashlib.sha256(payload).hexdigest()


def test_trainer_arguments_join_every_corpus_path() -> None:
    """SentencePiece receives one comma-joined input list."""
    arguments = build_trainer_arguments(
        [Path("a/kk.txt"), Path("a/ky.txt")],
        "spm/turkic",
        1000,
        "unigram",
        1.0,
        ("<lang_kk>", "<lang_ky>"),
        False,
        None,
    )
    assert arguments["input"] == f"{Path('a/kk.txt')},{Path('a/ky.txt')}"
    assert arguments["user_defined_symbols"] == ["<lang_kk>", "<lang_ky>"]
    assert "input_sentence_size" not in arguments


def test_trainer_arguments_carry_a_sentence_sample_when_given() -> None:
    """A sample size is passed through rather than dropped."""
    arguments = build_trainer_arguments(
        [Path("kk.txt")], "spm/t", 1000, "bpe", 0.9995, (), True, 5000
    )
    assert arguments["input_sentence_size"] == 5000
    assert arguments["hard_vocab_limit"] is True


def test_training_writes_a_model_and_an_unfiltered_manifest(
    runner: CliRunner, two_language_source: None, tmp_path: Path
) -> None:
    """A run with no classifier records that no filter was applied."""
    prefix = tmp_path / "turkic"
    manifest = tmp_path / "manifest.json"

    result = runner.invoke(
        main,
        [
            "--langs",
            "kk,ky",
            "--model-prefix",
            str(prefix),
            "--vocab-size",
            "200",
            "--manifest",
            str(manifest),
        ],
    )

    assert result.exit_code == 0
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["languages"] == ["kk", "ky"]
    assert document["spm_args"]["vocab_size"] == 200
    assert document["model_sha256"] == sha256_of_file(Path(f"{prefix}.model"))
    assert [corpus["language_identification"] for corpus in document["corpora"]] == [
        None,
        None,
    ]


def test_training_records_the_classifier_that_filtered_each_language(
    runner: CliRunner,
    two_language_source: None,
    installed_weights: None,
    tmp_path: Path,
) -> None:
    """The model's manifest names the weights that shaped its training text.

    Each language is filtered against its own code, so the Kazakh corpus
    keeps only Kazakh and the Kyrgyz corpus only Kyrgyz, and both name
    the single classifier the run was given.
    """
    prefix = tmp_path / "turkic"
    manifest = tmp_path / "manifest.json"

    result = runner.invoke(
        main,
        [
            "--langs",
            "kk,ky",
            "--model-prefix",
            str(prefix),
            "--vocab-size",
            "200",
            "--lid-model",
            "lid.176",
            "--manifest",
            str(manifest),
        ],
    )

    assert result.exit_code == 0
    corpora = json.loads(manifest.read_text(encoding="utf-8"))["corpora"]
    assert [corpus["filter_language"] for corpus in corpora] == ["kk", "ky"]
    assert [corpus["language_identification"]["model_id"] for corpus in corpora] == [
        "lid.176",
        "lid.176",
    ]
    assert [corpus["lines_written"] for corpus in corpora] == [120, 120]


def test_training_reserves_one_tag_per_language_by_default(
    runner: CliRunner, two_language_source: None, tmp_path: Path
) -> None:
    """Without explicit symbols, each language gets a reserved tag."""
    prefix = tmp_path / "turkic"
    manifest = tmp_path / "manifest.json"

    runner.invoke(
        main,
        [
            "--langs",
            "kk,ky",
            "--model-prefix",
            str(prefix),
            "--vocab-size",
            "200",
            "--manifest",
            str(manifest),
        ],
    )

    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["spm_args"]["user_defined_symbols"] == ["<lang_kk>", "<lang_ky>"]


def test_explicit_symbols_replace_the_default_tags(
    runner: CliRunner, two_language_source: None, tmp_path: Path
) -> None:
    """Symbols given on the command line are used verbatim."""
    prefix = tmp_path / "turkic"
    manifest = tmp_path / "manifest.json"

    runner.invoke(
        main,
        [
            "--langs",
            "kk",
            "--model-prefix",
            str(prefix),
            "--vocab-size",
            "200",
            "--user-symbols",
            "<mask>,<sep>",
            "--manifest",
            str(manifest),
        ],
    )

    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["spm_args"]["user_defined_symbols"] == ["<mask>", "<sep>"]


def test_training_without_a_manifest_still_writes_the_model(
    runner: CliRunner, two_language_source: None, tmp_path: Path
) -> None:
    """The manifest is optional; the model is not."""
    prefix = tmp_path / "turkic"

    result = runner.invoke(
        main,
        ["--langs", "kk", "--model-prefix", str(prefix), "--vocab-size", "200"],
    )

    assert result.exit_code == 0
    assert Path(f"{prefix}.model").read_bytes()[:1] != b""


def test_an_empty_language_list_is_refused(runner: CliRunner, tmp_path: Path) -> None:
    """A comma with no codes around it names no language."""
    result = runner.invoke(main, ["--langs", " , ", "--model-prefix", str(tmp_path / "t")])
    assert result.exit_code == 2
    assert "at least one language code" in result.output


def test_manifest_document_embeds_each_corpus_manifest(tmp_path: Path) -> None:
    """The embedded corpus manifests are the ones the runs produced."""
    from turkic_translit.corpus.manifest import CorpusRunManifest

    model = tmp_path / "t.model"
    model.write_bytes(b"model-bytes")
    corpus = CorpusRunManifest(
        source_id="oscar-2301",
        driver="oscar",
        license="CC0-1.0",
        language="kk",
        output_path=str(tmp_path / "kk.txt"),
        lines_seen=10,
        lines_written=7,
        filter_language=None,
        language_identification=None,
    )

    document = build_manifest_document(
        ("kk",), "oscar-2301", [(tmp_path / "kk.txt", corpus)], {"vocab_size": 200}, model
    )

    assert document["languages"] == ["kk"]
    assert document["model_sha256"] == sha256_of_file(model)
    assert document["corpora"] == [
        {
            "source_id": "oscar-2301",
            "driver": "oscar",
            "license": "CC0-1.0",
            "language": "kk",
            "output_path": str(tmp_path / "kk.txt"),
            "lines_seen": 10,
            "lines_written": 7,
            "filter_language": None,
            "language_identification": None,
        }
    ]
