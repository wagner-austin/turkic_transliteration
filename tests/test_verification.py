from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal

import pytest

from tests.foreign import (
    ResourceDirectory,
    icu_transliterator_factory,
    panphon_feature_table,
    replace_panphon_resource_locator,
)
from turkic_translit.core import to_ipa
from turkic_translit.lid.classifier import LidClassifier
from turkic_translit.lid.factory import load_installed_classifier
from turkic_translit.lid.locations import default_search_dirs
from turkic_translit.lid.registry import find_model_path
from turkic_translit.tokenizer import sentencepiece_processor, sentencepiece_trainer
from turkic_translit.web.web_utils import LANGUAGE_MODEL_ID, train_sentencepiece_model


@dataclass(frozen=True)
class UploadedFile:
    """A file chosen in the browser, as the trainer reads it.

    Args:
        name: Path to the uploaded file's contents on disk.
    """

    name: str


@contextmanager
def _panphon_utf8_resources() -> Iterator[None]:
    """Force panphon's packaged data files to be read as UTF-8.

    Works around a persistent upstream bug: panphon opens its packaged
    CSVs without an ``encoding=`` argument, so on Windows the default
    cp1252 codec trips over the IPA characters in the data.

    Which call has to be intercepted changed with the library. Up to
    0.21.2 panphon used ``pkg_resources`` and plain ``open``, so patching
    ``builtins.open`` was enough. 0.22.2 uses
    ``importlib.resources.files(...).joinpath(...).open()``, which never
    reaches ``builtins.open``, so the traversable is wrapped instead.
    Staying on 0.21.2 is not an option: it imports ``pkg_resources``,
    which setuptools 81 removed, so the whole module fails to import.

    Yields:
        None, once, with panphon's resource loader wrapped.
    """

    class _Utf8Traversable:
        """A traversable whose ``open`` defaults to UTF-8.

        Args:
            inner: The traversable being wrapped.
        """

        def __init__(self, inner: ResourceDirectory) -> None:
            """Store the traversable this one delegates to."""
            self._inner = inner

        def joinpath(self, *parts: str) -> _Utf8Traversable:
            """Descend, keeping the UTF-8 default.

            Args:
                parts: Path components to append.

            Returns:
                The wrapped child traversable.
            """
            return _Utf8Traversable(self._inner.joinpath(*parts))

        def open(self, mode: Literal["r"] = "r", encoding: str = "utf-8") -> IO[str]:
            """Open the resource as UTF-8 text.

            Args:
                mode: File mode; text mode at every panphon call site.
                encoding: Defaulted to UTF-8, which is the whole point.

            Returns:
                The opened text stream.
            """
            stream: IO[str] = self._inner.open(mode, encoding=encoding)
            return stream

    def utf8_files(package: str) -> ResourceDirectory:
        """Locate a package's data files, opening them as UTF-8.

        Args:
            package: Dotted name of the package to read from.

        Returns:
            The wrapped traversable.
        """
        return _Utf8Traversable(real_files(package))

    real_files = replace_panphon_resource_locator(utf8_files)
    try:
        yield
    finally:
        replace_panphon_resource_locator(real_files)


# 1. Test PyICU transliteration
def test_icu_transliteration() -> None:
    """ICU's generic Any-Latin romanises Kazakh Cyrillic.

    The exact romanisation of Ғ varies by ICU version (G, Ğ or Ġ), so
    that letter is checked as a set while the rest of the word is
    pinned exactly.
    """
    result = icu_transliterator_factory()("Any-Latin; NFC").transliterate("Ғылым")

    assert result[1:] == "ylym"
    assert result[0] in {"G", "Ğ", "Ġ"}, f"ICU result: {result}"


# 2. Test epitran + panphon IPA for Kazakh
def test_epitran_panphon_ipa() -> None:
    """
    Test ICU-based IPA conversion and panphon's phonological features.
    This test ensures that the IPA conversion pipeline is working correctly.
    """
    test_word = "Ғылым"  # "Knowledge" in Kazakh
    ipa = to_ipa(test_word, "kk")
    with _panphon_utf8_resources():
        vec = panphon_feature_table().word_to_vector_list(ipa)

    # Print for inspection during test runs
    print(f"\nTest word: {test_word}")
    print(f"IPA transcription: {ipa}")
    print(f"Phonological features count: {len(vec)}")

    # Check actual content - Ғ should be ʁ in IPA
    assert "ʁ" in ipa, f"Expected 'ʁ' in IPA transcription, got: {ipa}"
    assert len(ipa) >= 4, f"Expected at least 4 characters in IPA, got: {len(ipa)}"

    # Check feature extraction results
    assert len(vec) >= 4, f"Expected at least 4 feature vectors (one per sound), got: {len(vec)}"

    # Check that the feature vectors have the proper structure
    # panphon returns arrays of feature values, not dictionaries
    for i, segment in enumerate(vec):
        # Each segment should have at least 20 features
        assert len(segment) >= 20, f"Segment {i} has too few features: {len(segment)}"
        # Each segment should be a list of feature values (+/-/0)
        assert all(val in ["+", "-", "0"] for val in segment), (
            f"Invalid feature values in segment {i}: {segment}"
        )


# 3. Test SentencePiece encode/decode round-trip
def test_sentencepiece_roundtrip() -> None:
    """
    Test SentencePiece tokenizer model training and round-trip encoding/decoding.

    This test ensures the tokenization pipeline works for downstream tasks.
    """
    # Use a mix of Latin, Cyrillic and special chars to test encoding
    samples = ["Ğalamdyq jeli", "Kitap bar", "Ülken söz", "Мысал текст"]
    model_file = "mini.model"
    vocab_file = "mini.vocab"

    with tempfile.NamedTemporaryFile("w", encoding="utf8", delete=False) as temp_file:
        # Write sample text to temporary file
        for sample in samples:
            temp_file.write(f"{sample}\n")
        temp_path = temp_file.name

    try:
        # Train with the exact vocab size needed for this corpus (33)
        # This value was determined from the error message
        sentencepiece_trainer().train(
            input=temp_path,
            model_prefix="mini",
            vocab_size=33,  # Exactly what SentencePiece can handle with this corpus
            model_type="unigram",
            character_coverage=0.9995,
        )

        # Test encoding and decoding
        sample = samples[0]
        proc = sentencepiece_processor(model_file)
        pieces = proc.encode(sample, out_type=str)
        decoded = proc.decode(pieces)

        # The round trip is the assertion: it can only hold if encode
        # produced a usable id sequence.
        assert decoded == sample

    finally:
        # Clean up files
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        if os.path.exists(model_file):
            os.unlink(model_file)
        if os.path.exists(vocab_file):
            os.unlink(vocab_file)


# 4. Test fastText LID logic
def test_fasttext_lid() -> None:
    """The registered classifier identifies Russian from real weights.

    Resolution goes through the model registry rather than a ladder of
    guessed paths, and the answer comes back as a typed prediction
    rather than a tuple whose shape depends on which fastText build is
    installed. Both of those were what the old version of this test
    worked around, and both are now the library's job.
    """
    if find_model_path(LANGUAGE_MODEL_ID, default_search_dirs()) is None:
        pytest.skip(f"{LANGUAGE_MODEL_ID} weights are not installed")

    prediction = load_installed_classifier(LANGUAGE_MODEL_ID).classify(
        "Пример текста на русском языке"
    )

    assert prediction["label"] == "ru"
    assert prediction["probability"] > 0.5


CORPUS_MODEL_ID = "lid218e"

# Each line is written in the script the corresponding OSCAR slice is
# actually sourced in, because the point of a script-aware classifier is
# that the script is part of the answer.
SCRIPT_AWARE_CASES = [
    ("tur_Latn", "Türkiye Cumhuriyeti'nin başkenti Ankara şehridir."),
    ("azj_Latn", "Azərbaycan Respublikasının paytaxtı Bakı şəhəridir."),
    ("kaz_Cyrl", "Қазақстан Республикасының астанасы Астана қаласы болып табылады."),
    ("kir_Cyrl", "Кыргыз Республикасынын борбору Бишкек шаары."),
    ("uzn_Latn", "Ozbekiston Respublikasining poytaxti Toshkent shahridir."),
    ("uig_Arab", "شىنجاڭ ئۇيغۇر ئاپتونوم رايونى جۇڭگونىڭ غەربىدە جايلاشقان."),
    ("fin_Latn", "Suomen tasavallan pääkaupunki on Helsinki ja siellä asuu paljon ihmisiä."),
]


def _installed_corpus_classifier() -> LidClassifier:
    """Load the corpus-building classifier, or skip if it is absent.

    The weights are 1.18 GB and are not distributed with the package, so
    every test here is opt-in: present on a machine that has built a
    corpus, skipped everywhere else.

    Returns:
        The loaded classifier.
    """
    if find_model_path(CORPUS_MODEL_ID, default_search_dirs()) is None:
        pytest.skip(f"{CORPUS_MODEL_ID} weights are not installed")
    return load_installed_classifier(CORPUS_MODEL_ID)


# 5. Test the script-aware classifier that builds the corpora
@pytest.mark.parametrize(("expected_label", "text"), SCRIPT_AWARE_CASES)
def test_lid218e_identifies_each_corpus_language(expected_label: str, text: str) -> None:
    """Real lid218e weights label every corpus language and its script.

    This is the one thing the hook-backed unit tests cannot establish.
    They answer from a table, so they prove the registry, the resolver
    and the prefix stripping are wired correctly while saying nothing
    about the model those parts exist to reach. The corpora behind the
    published results were filtered by these weights at ``p >= 0.95``,
    so the threshold is asserted too rather than merely the ranking.
    """
    prediction = _installed_corpus_classifier().classify(text)

    assert prediction["label"] == expected_label
    assert prediction["probability"] >= 0.95


def test_lid218e_cannot_label_cyrillic_uzbek() -> None:
    """Cyrillic Uzbek is unreachable, and confidently mislabelled.

    ``uzb_Cyrl`` is absent from the label set, so no threshold can
    recover Cyrillic Uzbek: the model does not degrade to low confidence
    but assigns a different language above the corpus threshold, which
    is why those lines are dropped silently rather than flagged. This is
    a documented property of the corpus rather than a defect, and it is
    pinned here so that a future model revision adding the label fails
    this test instead of changing the corpus unnoticed.
    """
    classifier = _installed_corpus_classifier()

    labels = classifier.known_labels()
    assert "uzn_Latn" in labels
    assert "uzb_Cyrl" not in labels

    prediction = classifier.classify("Ўзбекистон Республикасининг пойтахти Тошкент шаҳридир.")

    assert prediction["label"] != "uzn_Latn"
    assert prediction["probability"] >= 0.95


# 6. Test SentencePiece training in web interface
def test_web_sentencepiece_training() -> None:
    """
    Test the SentencePiece training functionality used in the web interface.

    This test ensures that the train_sentencepiece_model function correctly trains
    a SentencePiece model with the provided text and parameters and returns the
    expected output format.
    """
    # Test text content with mixed languages
    test_text = """менің атым Айдар
сәлем әлем
Қазақстан республикасы
қалың елім қазағым
кыргыз тилинде сүйлөйм"""

    # Create a test file for training
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as test_file:
        # Different content for the file to verify both are used
        test_file.write("тестовый текст\nбіз қазақша сөйлейміз\nкыргызча сүйлөйбүз")
        test_path = test_file.name

    # The upload as Gradio hands it over: an object carrying the path it
    # wrote the contents to, which is all the trainer reads.
    test_file_obj = UploadedFile(test_path)

    try:
        # Train using both text content and file upload
        model_path, info = train_sentencepiece_model(
            input_text=test_text,
            training_file=test_file_obj,
            vocab_size=50,  # Must be smaller than the max vocab size the corpus can support
            model_type="unigram",
            character_coverage=1.0,
            user_symbols="<test>,<kk>,<ky>",
        )

        # A path that resolves to a real file is a stronger claim than
        # the string type it necessarily has.
        assert Path(model_path).is_file()

        # Check if the info contains expected details
        assert "Model Training Complete" in info
        assert "Vocabulary Size:" in info
        assert "unigram" in info

        # Verify the model works by loading it and using it
        proc = sentencepiece_processor(model_path)

        # Test encoding/decoding
        test_phrase = "менің атым Айдар"
        pieces = proc.encode(test_phrase, out_type=str)
        decoded = proc.decode(pieces)

        assert decoded == test_phrase

        # Test using only text content, no file
        model_path2, info2 = train_sentencepiece_model(
            input_text=test_text,
            vocab_size=40,  # Even smaller vocab size for second test
            model_type="bpe",
            character_coverage=0.9995,
        )

        assert Path(model_path2).exists()
        assert "bpe" in info2

    finally:
        # Clean up
        if os.path.exists(test_path):
            os.unlink(test_path)

        # Clean up model files
        if "model_path" in locals() and os.path.exists(model_path):
            os.unlink(model_path)
        if "model_path2" in locals() and os.path.exists(model_path2):
            os.unlink(model_path2)
