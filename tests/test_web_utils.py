"""Tests for the web helpers the tabs call into.

SentencePiece is trained for real on a small synthetic corpus, the
pipeline runs against a table-backed classifier, and the corpus streamer
answers from memory. Nothing is substituted for the code under test.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from turkic_translit import _test_hooks
from turkic_translit.corpus import _test_hooks as corpus_hooks
from turkic_translit.lid import _test_hooks as lid_hooks
from turkic_translit.lid.locations import default_search_dirs
from turkic_translit.tokenizer import (
    DEFAULT_MODEL_NAME,
    MODEL_PATH_VARIABLE,
    TurkicTokenizer,
    default_model_path,
    sentencepiece_trainer,
)
from turkic_translit.web import web_utils

CORPUS = [f"salem alem {index} qazaq tili birinshi" for index in range(120)]
ANSWERS = {
    "salem": [("__label__kk", 0.99)],
    "privet": [("__label__ru", 0.99)],
    "мир": [("__label__ru", 0.98)],
    "hello": [("__label__en", 0.97)],
}
WEIGHTS = default_search_dirs()[0] / "lid.176.bin"

# The exact strings the pipeline tests hand to the tokenizer, so the
# classifier table can be built to cover every piece they produce.
PIPELINE_INPUTS = ("salem privet", "salem")


class RecordingProgress:
    """A progress reporter that keeps what it was told.

    A real implementation of
    :class:`~turkic_translit.web.web_utils.ProgressReporter`, not a mock:
    it holds no assertion helpers, so a test can only read the reports
    the code under test produced.
    """

    def __init__(self) -> None:
        """Start with nothing reported."""
        self.reports: list[tuple[float | None, str]] = []

    def __call__(self, progress: float | None, desc: str = "") -> None:
        """Record one progress report.

        Args:
            progress: Completed fraction, or ``None`` when the total is
                unknown.
            desc: Text that would have been shown beside the bar.
        """
        self.reports.append((progress, desc))


@dataclass(frozen=True)
class UploadedFile:
    """A file chosen in the browser, as the web module reads it.

    Args:
        name: Path to the uploaded file's contents on disk.
    """

    name: str


@pytest.fixture
def classifier() -> Iterator[None]:
    """Present ``lid.176`` as installed and back it with a table model.

    Yields:
        None, once, with the original hooks captured.
    """
    probe, loader = lid_hooks.probe, lid_hooks.model_loader
    lid_hooks.probe = lid_hooks.MappingFileProbe({WEIGHTS: 131266198})
    lid_hooks.model_loader = lid_hooks.FixedModelLoader(lid_hooks.TableFastTextModel(ANSWERS))
    web_utils.installed_classifier.cache_clear()
    yield
    lid_hooks.probe = probe
    lid_hooks.model_loader = loader
    web_utils.installed_classifier.cache_clear()


@pytest.fixture
def tokenizer_model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Train a real tokenizer model and point the project at it.

    The pipeline needs a SentencePiece model, which is not shipped. One
    is trained here so the tokenising helpers run for real rather than
    reporting the model as absent.

    Yields:
        None, once, with the original hooks captured.
    """
    directory = tmp_path_factory.mktemp("tokenizer")
    corpus = directory / "corpus.txt"
    corpus.write_text("\n".join(CORPUS) + "\n", encoding="utf-8", newline="\n")
    prefix = directory / "tiny"
    sentencepiece_trainer().train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=38,
        model_type="unigram",
        character_coverage=1.0,
    )

    # The pipeline classifies sub-word pieces, not whole words, so the
    # table is built from what this model actually produces rather than
    # from words the tokenizer would never hand the classifier.
    model_file = prefix.with_suffix(".model")
    pieces = TurkicTokenizer(str(model_file)).tokenize(" ".join(PIPELINE_INPUTS))
    piece_answers = {
        form: [("__label__kk", 0.99)]
        for piece in pieces
        for form in (piece, piece.lower(), piece.lstrip("▁"))
    }

    previous_environment = _test_hooks.environment
    probe, loader = lid_hooks.probe, lid_hooks.model_loader
    _test_hooks.environment = _test_hooks.MappingEnvironment({MODEL_PATH_VARIABLE: str(model_file)})
    lid_hooks.probe = lid_hooks.MappingFileProbe({WEIGHTS: 131266198})
    lid_hooks.model_loader = lid_hooks.FixedModelLoader(lid_hooks.TableFastTextModel(piece_answers))
    web_utils._lazy_pipeline.cache_clear()
    yield
    _test_hooks.environment = previous_environment
    lid_hooks.probe, lid_hooks.model_loader = probe, loader
    web_utils._lazy_pipeline.cache_clear()


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    """Write a corpus large enough for SentencePiece to train on.

    Returns:
        Path of the written corpus.
    """
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(CORPUS) + "\n", encoding="utf-8", newline="\n")
    return path


@pytest.mark.parametrize("out_fmt", ["LATIN", "Ipa", "latin", "ipa"])
def test_the_output_format_is_matched_case_insensitively(out_fmt: str) -> None:
    """The dropdown's label casing does not decide the format.

    Args:
        out_fmt: The format as the UI might spell it.
    """
    result, stats = web_utils.direct_transliterate("сәлем", "kk", False, out_fmt)

    assert result
    assert out_fmt.upper() in stats


def test_an_unknown_output_format_is_rejected() -> None:
    """Only the two documented formats are accepted."""
    with pytest.raises(ValueError, match="out_fmt must be 'latin' or 'ipa'"):
        web_utils.direct_transliterate("сәлем", "kk", False, "runes")


def test_the_statistics_report_both_byte_counts() -> None:
    """The summary names the input and output sizes in bytes."""
    _result, stats = web_utils.direct_transliterate("сәлем", "kk", False, "latin")

    assert f"Cyrillic : {len('сәлем'.encode())}" in stats


def test_an_unknown_pipeline_mode_is_rejected() -> None:
    """The pipeline is asked for one of two modes, or nothing."""
    with pytest.raises(ValueError, match="mode must be 'latin' or 'ipa'"):
        web_utils.pipeline_transliterate("сәлем", "runes")


def test_a_missing_tokenizer_model_is_explained_not_raised(tmp_path: Path) -> None:
    """The token table says how to produce the model it needs.

    The SentencePiece model is not shipped, so its absence on a fresh
    checkout is an ordinary first-run state.

    Args:
        tmp_path: Directory the absent model is pointed at.
    """
    previous = _test_hooks.environment
    _test_hooks.environment = _test_hooks.MappingEnvironment(
        {MODEL_PATH_VARIABLE: str(tmp_path / "absent.model")}
    )
    try:
        table = web_utils.token_table_markdown("сәлем әлем")
    finally:
        _test_hooks.environment = previous

    assert "Tokenizer model file missing" in table
    assert "turkic-build-spm" in table
    assert MODEL_PATH_VARIABLE in table


def test_the_packaged_location_is_used_when_nothing_is_configured() -> None:
    """With no variable set the model is looked for inside the package."""
    previous = _test_hooks.environment
    _test_hooks.environment = _test_hooks.MappingEnvironment({})
    try:
        located = default_model_path()
    finally:
        _test_hooks.environment = previous

    assert located.name == DEFAULT_MODEL_NAME
    assert located.parent.name == "turkic_translit"


def test_a_configured_model_path_is_used_as_given(tmp_path: Path) -> None:
    """The variable names the model outright, wherever it lives."""
    previous = _test_hooks.environment
    _test_hooks.environment = _test_hooks.MappingEnvironment(
        {MODEL_PATH_VARIABLE: str(tmp_path / "mine.model")}
    )
    try:
        assert default_model_path() == tmp_path / "mine.model"
    finally:
        _test_hooks.environment = previous


def test_the_token_table_names_each_token_and_its_language(tokenizer_model: None) -> None:
    """Every token is tabulated beside the language assigned to it.

    Args:
        tokenizer_model: The bound tokenizer model and classifier.
    """
    table = web_utils.token_table_markdown("salem privet")

    rows = [line for line in table.splitlines() if line.startswith("|")]
    header, separator, *body = rows
    tokens = "".join(row.split("|")[1].strip() for row in body)

    assert "Token" in header
    assert "Lang" in header
    assert separator.startswith("|:")
    # This vocabulary is small enough that pieces are single characters,
    # so the token column reassembles the input once SentencePiece's
    # word-boundary marker is read back as the space it stands for.
    assert tokens.replace("▁", " ").strip() == "salem privet"
    assert all(row.split("|")[2].strip() in {"kk", ""} for row in body)


@pytest.mark.parametrize("mode", ["latin", "ipa", "IPA"])
def test_the_pipeline_transliterates_in_either_mode(tokenizer_model: None, mode: str) -> None:
    """The pipeline runs end to end and reports the output length.

    Args:
        tokenizer_model: The bound tokenizer model and classifier.
        mode: The requested output mode, in any casing.
    """
    result, stats = web_utils.pipeline_transliterate("salem", mode)

    assert result
    assert stats == f"**{len(result)} chars**"


def test_russian_tokens_are_masked(classifier: None) -> None:
    """Tokens the classifier calls Russian are replaced.

    Args:
        classifier: The bound table-backed classifier.
    """
    masked = web_utils.mask_russian("privet мир hello", thr=0.5, min_len=3)

    assert masked == "<RU> <RU> hello"


def test_masking_reports_each_decision_when_asked(classifier: None) -> None:
    """Debug output carries one entry per token, in order.

    Args:
        classifier: The bound table-backed classifier.
    """
    import json

    masked = web_utils.mask_russian("privet hello", thr=0.5, min_len=3, debug=True)
    start = masked.find("<!--debug ") + len("<!--debug ")
    entries = json.loads(masked[start : masked.find(" -->", start)])

    assert [entry["tok"] for entry in entries] == ["privet", "hello"]
    assert [entry["ru"] for entry in entries] == [True, False]


def test_masking_with_truncated_weights_explains_itself() -> None:
    """A model file with no bytes produces guidance, not a traceback."""
    probe = lid_hooks.probe
    lid_hooks.probe = lid_hooks.MappingFileProbe({WEIGHTS: 0})
    web_utils.installed_classifier.cache_clear()
    try:
        masked = web_utils.mask_russian("privet мир", thr=0.5, min_len=3)
    finally:
        lid_hooks.probe = probe
        web_utils.installed_classifier.cache_clear()

    assert "Language-identification model unavailable" in masked
    assert web_utils.LANGUAGE_MODEL_ID in masked


def test_progress_is_reported_every_tenth_line() -> None:
    """Only every tenth line moves the bar, and it names the total."""
    reporter = RecordingProgress()

    for written in range(1, 21):
        web_utils._report_progress(reporter, written, 20)

    assert [progress for progress, _desc in reporter.reports] == [0.5, 1.0]
    assert reporter.reports[0][1] == "10/20 lines kept"


def test_progress_without_a_total_reports_a_running_count() -> None:
    """An unbounded run has no fraction to report, only a count."""
    reporter = RecordingProgress()

    web_utils._report_progress(reporter, 10, None)

    assert reporter.reports == [(None, "10 lines kept")]


def test_progress_never_exceeds_completion() -> None:
    """More lines than requested still reports a full bar, not more."""
    reporter = RecordingProgress()

    web_utils._report_progress(reporter, 30, 20)

    assert [progress for progress, _desc in reporter.reports] == [1.0]


def test_the_silent_reporter_discards_everything() -> None:
    """Outside the web UI there is nowhere to report progress to."""
    assert web_utils.SilentProgress()(0.5, desc="ignored") is None


def test_an_unknown_source_is_reported_without_writing_a_file() -> None:
    """The download helper rejects a source it does not have."""
    path, info = web_utils.download_corpus_to_file("oscar-9999", "kk")

    assert path == ""
    assert "invalid_source" in info
    assert "oscar-2301" in info


def test_the_summary_omits_the_filter_line_when_nothing_filtered(tmp_path: Path) -> None:
    """A run with no filter reports no removals, rather than zero."""
    summary = web_utils._download_summary(
        source="oscar-2301",
        lang="kk",
        written=2,
        seen=3,
        removed=None,
        prob_threshold=0.95,
        path=tmp_path / "kk.txt",
    )

    assert "Lines removed" not in summary
    assert "Lines written:** 2" in summary


def test_the_summary_reports_removals_when_a_filter_ran(tmp_path: Path) -> None:
    """A filtered run names how many lines the classifier rejected."""
    summary = web_utils._download_summary(
        source="oscar-2301",
        lang="kk",
        written=2,
        seen=5,
        removed=3,
        prob_threshold=0.95,
        path=tmp_path / "kk.txt",
    )

    assert "Lines removed by LangID filter:** 3 (p ≥ 0.95)" in summary


def test_training_needs_text_or_a_file() -> None:
    """Training on nothing is rejected before SentencePiece is invoked."""
    with pytest.raises(ValueError, match="Either input text or training file"):
        web_utils.train_sentencepiece_model("   ")


def test_training_from_pasted_text_produces_a_model() -> None:
    """Text typed into the box is enough to train on."""
    path, info = web_utils.train_sentencepiece_model(
        "\n".join(CORPUS), vocab_size=40, model_type="unigram"
    )

    assert Path(path).is_file()
    assert "Model Training Complete" in info
    assert "Vocabulary Size: 40 tokens" in info


def test_training_from_an_uploaded_file_produces_a_model(corpus_file: Path) -> None:
    """An upload is passed to SentencePiece by path, not by content.

    Args:
        corpus_file: The corpus the upload points at.
    """
    path, info = web_utils.train_sentencepiece_model(
        "", training_file=UploadedFile(str(corpus_file)), vocab_size=40
    )

    assert Path(path).is_file()
    assert "Model Type: unigram" in info


def test_training_ignores_an_upload_with_no_path(corpus_file: Path) -> None:
    """An upload naming nothing leaves the pasted text as the corpus.

    Args:
        corpus_file: Unused; present so the fixture's corpus exists.
    """
    path, _info = web_utils.train_sentencepiece_model(
        "\n".join(CORPUS), training_file=UploadedFile(""), vocab_size=40
    )

    assert Path(path).is_file()


def test_a_filtered_download_keeps_only_the_named_language(tmp_path: Path) -> None:
    """The classifier decides each line, and removals are counted."""
    previous_streamer = corpus_hooks.dataset_texts
    previous_environment = _test_hooks.environment
    previous_probe, previous_loader = lid_hooks.probe, lid_hooks.model_loader
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer(
        {"kk": ["salem", "privet", "", "hello"]}
    )
    _test_hooks.environment = _test_hooks.MappingEnvironment(
        {"TURKIC_CRON_DIR": str(tmp_path / "cronjob")}
    )
    lid_hooks.probe = lid_hooks.MappingFileProbe({WEIGHTS: 131266198})
    lid_hooks.model_loader = lid_hooks.FixedModelLoader(lid_hooks.TableFastTextModel(ANSWERS))
    web_utils.installed_classifier.cache_clear()
    try:
        path, info = web_utils.download_corpus_to_file("oscar-2301", "kk", None, True, 0.95)
    finally:
        corpus_hooks.dataset_texts = previous_streamer
        _test_hooks.environment = previous_environment
        lid_hooks.probe, lid_hooks.model_loader = previous_probe, previous_loader
        web_utils.installed_classifier.cache_clear()

    assert Path(path).read_text(encoding="utf-8") == "salem\n"
    assert "Lines removed by LangID filter:** 2" in info


def test_a_filtered_download_gives_up_after_too_many_rejections(tmp_path: Path) -> None:
    """A stream that keeps almost nothing stops rather than hanging.

    The guard trips at fifty times the requested line count, so asking
    for one line from a source that yields no matches stops after fifty.
    """
    previous_streamer = corpus_hooks.dataset_texts
    previous_environment = _test_hooks.environment
    previous_probe, previous_loader = lid_hooks.probe, lid_hooks.model_loader
    corpus_hooks.dataset_texts = corpus_hooks.MappingDatasetTextStreamer({"kk": ["privet"] * 200})
    _test_hooks.environment = _test_hooks.MappingEnvironment(
        {"TURKIC_CRON_DIR": str(tmp_path / "cronjob")}
    )
    lid_hooks.probe = lid_hooks.MappingFileProbe({WEIGHTS: 131266198})
    lid_hooks.model_loader = lid_hooks.FixedModelLoader(lid_hooks.TableFastTextModel(ANSWERS))
    web_utils.installed_classifier.cache_clear()
    try:
        path, info = web_utils.download_corpus_to_file("oscar-2301", "kk", 1, True, 0.95)
    finally:
        corpus_hooks.dataset_texts = previous_streamer
        _test_hooks.environment = previous_environment
        lid_hooks.probe, lid_hooks.model_loader = previous_probe, previous_loader
        web_utils.installed_classifier.cache_clear()

    assert Path(path).read_text(encoding="utf-8") == ""
    assert "Total sentences processed:** 50" in info
