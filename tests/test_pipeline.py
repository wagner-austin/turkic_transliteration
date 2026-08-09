"""Tests for the tokenise-identify-transliterate pipeline.

SentencePiece is trained for real on a tiny corpus, so the tokeniser is
genuine and its word-boundary pieces are the ones the pipeline actually
has to cope with. Only the classifier is table-backed, and that is a real
implementation of the production protocol.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from turkic_translit.lid import _test_hooks
from turkic_translit.pipeline import TurkicTransliterationPipeline
from turkic_translit.tokenizer import sentencepiece_trainer

ANSWERS = {
    "salem": [("__label__kk", 0.99)],
    "alem": [("__label__kk", 0.98)],
    "privet": [("__label__ru", 0.97)],
}


@pytest.fixture(scope="module")
def tokenizer_model(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Train a small SentencePiece model the pipeline can tokenise with.

    Args:
        tmp_path_factory: Factory for a module-scoped temporary directory.

    Returns:
        Path of the trained ``.model`` file.
    """
    directory = tmp_path_factory.mktemp("spm")
    corpus = directory / "corpus.txt"
    corpus.write_text(
        "\n".join(
            ["salem alem privet dunie qajyt bugin kobik zhuz"] * 60 + ["salem alem privet"] * 60
        )
        + "\n",
        encoding="utf-8",
    )
    prefix = directory / "tiny"
    sentencepiece_trainer().train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=32,
        model_type="unigram",
        character_coverage=1.0,
        hard_vocab_limit=False,
    )
    return str(Path(f"{prefix}.model"))


@pytest.fixture
def table_classifier() -> Generator[None, None, None]:
    """Present ``lid.176`` as installed and back it with a table model.

    Yields:
        None, once, with the original hooks captured.
    """
    from turkic_translit.lid.locations import default_search_dirs

    probe, loader = _test_hooks.probe, _test_hooks.model_loader
    _test_hooks.probe = _test_hooks.MappingFileProbe(
        {default_search_dirs()[0] / "lid.176.bin": 131266198}
    )
    _test_hooks.model_loader = _test_hooks.FixedModelLoader(_test_hooks.TableFastTextModel(ANSWERS))
    yield
    _test_hooks.probe = probe
    _test_hooks.model_loader = loader


def test_each_token_is_labelled_with_its_language(
    tokenizer_model: str, table_classifier: None
) -> None:
    """Every token receives a label from the named classifier."""
    pipeline = TurkicTransliterationPipeline(tokenizer_model)
    assert pipeline.predict_tokens(["▁salem", "▁privet"]) == ["kk", "ru"]


def test_a_bare_word_boundary_piece_is_reported_as_no_language(
    tokenizer_model: str, table_classifier: None
) -> None:
    """A piece with no text is not sent to the classifier, which would raise.

    SentencePiece emits a lone boundary marker for some inputs, and the
    classifier refuses empty text by design, so the pipeline has to
    recognise the case rather than let it become an exception.
    """
    pipeline = TurkicTransliterationPipeline(tokenizer_model)
    assert pipeline.predict_tokens(["▁", "▁salem"]) == ["", "kk"]


def test_the_pipeline_names_the_classifier_it_loaded(
    tokenizer_model: str, table_classifier: None
) -> None:
    """The classifier is chosen explicitly, not by resolution order."""
    pipeline = TurkicTransliterationPipeline(tokenizer_model)
    assert pipeline.classifier.model_id == "lid.176"


def test_processing_round_trips_text_through_the_tokeniser(
    tokenizer_model: str, table_classifier: None
) -> None:
    """Tokenising, labelling and detokenising reproduces the input.

    The answer table is built from the pieces the real tokeniser
    produces, so the classifier is asked exactly what the pipeline asks
    it and an unlisted piece would surface as a failure rather than
    being answered by a default.
    """
    from turkic_translit.tokenizer import TurkicTokenizer

    pieces = TurkicTokenizer(tokenizer_model).tokenize("salem alem")
    answers = {
        cleaned: [("__label__kk", 0.99)]
        for cleaned in (piece.replace("▁", "").strip() for piece in pieces)
        if cleaned != ""
    }
    _test_hooks.model_loader = _test_hooks.FixedModelLoader(_test_hooks.TableFastTextModel(answers))

    pipeline = TurkicTransliterationPipeline(tokenizer_model)

    assert pipeline.process("salem alem") == "salem alem"
