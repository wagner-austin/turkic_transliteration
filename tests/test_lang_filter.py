"""Tests for the Russian-token decision and its web integration.

The classifier is a real :class:`LidClassifier` over a table-backed
model, so labels arrive stripped and confidences arrive as plain floats
exactly as they do in production. The previous version of these tests
built a mock returning NumPy arrays, which is precisely the coupling that
pinned the project to NumPy 1.

This file replaces the near-identical ``test_lang_filter_new.py``, which
duplicated every case here and added only the web integration test at the
end.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence

import pytest

from turkic_translit.lang_filter import KZ_EXTRA, RU_ONLY, is_russian_token
from turkic_translit.lid import _test_hooks
from turkic_translit.lid.classifier import LidClassifier
from turkic_translit.lid.locations import default_search_dirs
from turkic_translit.lid.registry import get_spec


def classifier_for(
    answers: Mapping[str, Sequence[tuple[str, float]]],
) -> LidClassifier:
    """Build a classifier answering from a ranked table.

    Args:
        answers: Mapping of token to its ranked predictions, labels still
            carrying the ``__label__`` prefix.

    Returns:
        A classifier over that table.
    """
    return LidClassifier(get_spec("lid.176"), _test_hooks.TableFastTextModel(answers))


def test_a_token_shorter_than_the_minimum_is_never_russian() -> None:
    """Length is checked before the classifier is consulted."""
    lid = classifier_for({"пр": [("__label__ru", 0.9)]})
    assert is_russian_token("пр", thr=0.5, min_len=3, lid=lid) is False


def test_the_same_token_passes_once_the_minimum_allows_it() -> None:
    """Only the length rule was rejecting it."""
    lid = classifier_for({"пр": [("__label__ru", 0.9)]})
    assert is_russian_token("пр", thr=0.5, min_len=2, lid=lid) is True


def test_a_stoplisted_token_is_exempt_even_when_classified_russian() -> None:
    """The stoplist overrides a confident Russian prediction."""
    lid = classifier_for(
        {
            "привет": [("__label__ru", 0.9)],
            "здравствуйте": [("__label__ru", 0.9)],
        }
    )
    stoplist = {"привет", "мир"}
    assert is_russian_token("привет", thr=0.5, min_len=3, lid=lid, stoplist=stoplist) is False
    assert is_russian_token("здравствуйте", thr=0.5, min_len=3, lid=lid, stoplist=stoplist) is True


def test_a_kazakh_specific_letter_rules_the_token_out() -> None:
    """``ә`` cannot occur in Russian, whatever the classifier says."""
    lid = classifier_for({"сәлем": [("__label__ru", 0.9)]})
    assert "ә" in KZ_EXTRA
    assert is_russian_token("сәлем", thr=0.5, min_len=3, lid=lid) is False


def test_russian_wins_outright_above_the_threshold() -> None:
    """A confident top-ranked Russian label is accepted."""
    lid = classifier_for(
        {"привет": [("__label__ru", 0.8), ("__label__uk", 0.1), ("__label__bg", 0.05)]}
    )
    assert is_russian_token("привет", thr=0.5, min_len=3, lid=lid) is True


def test_russian_wins_but_below_the_threshold_is_refused() -> None:
    """Winning is not enough; the confidence bar still applies."""
    lid = classifier_for(
        {"привет": [("__label__ru", 0.4), ("__label__uk", 0.3), ("__label__bg", 0.2)]}
    )
    assert is_russian_token("привет", thr=0.5, min_len=3, lid=lid) is False


def test_russian_as_a_close_second_is_accepted_within_the_margin() -> None:
    """A near-miss counts when it is inside the margin."""
    lid = classifier_for(
        {"привет": [("__label__uk", 0.55), ("__label__ru", 0.5), ("__label__bg", 0.2)]}
    )
    assert is_russian_token("привет", thr=0.5, min_len=3, lid=lid, margin=0.1) is True
    assert is_russian_token("привет", thr=0.5, min_len=3, lid=lid, margin=0.01) is False


def test_russian_as_a_distant_second_is_refused() -> None:
    """Below the threshold, rank two does not rescue it."""
    lid = classifier_for(
        {"привет": [("__label__uk", 0.6), ("__label__ru", 0.4), ("__label__bg", 0.2)]}
    )
    assert is_russian_token("привет", thr=0.5, min_len=3, lid=lid) is False


def test_the_lowest_threshold_accepts_pure_cyrillic_orthography() -> None:
    """At ``thr`` zero the orthography test is the documented behaviour."""
    lid = classifier_for(
        {"привет": [("__label__uk", 0.5), ("__label__bg", 0.3), ("__label__ru", 0.2)]}
    )
    assert RU_ONLY.findall("привет") == ["привет"]
    assert is_russian_token("привет", thr=0.0, min_len=3, lid=lid) is True
    assert is_russian_token("привет", thr=0.1, min_len=3, lid=lid) is False


def test_mixed_script_defeats_the_orthography_test() -> None:
    """A token with Latin letters is not pure Cyrillic."""
    lid = classifier_for(
        {
            "приветworld": [
                ("__label__uk", 0.5),
                ("__label__bg", 0.3),
                ("__label__ru", 0.2),
            ]
        }
    )
    assert is_russian_token("приветworld", thr=0.0, min_len=3, lid=lid) is False


@pytest.fixture
def installed_weights() -> Iterator[None]:
    """Present ``lid.176`` as installed and back it with a table model.

    The web helper resolves its classifier through the same registry as
    everything else, so binding the probe and the loader exercises the
    real resolution path without a 126 MB model on disk. The helper's
    cache is cleared on the way in and out, because a classifier built
    from another test's loader would otherwise survive here.

    Yields:
        None, once, with the original hooks captured.
    """
    from turkic_translit.web import web_utils

    probe, loader = _test_hooks.probe, _test_hooks.model_loader
    _test_hooks.probe = _test_hooks.MappingFileProbe(
        {default_search_dirs()[0] / "lid.176.bin": 131266198}
    )
    _test_hooks.model_loader = _test_hooks.FixedModelLoader(
        _test_hooks.TableFastTextModel(
            {
                "привет": [("__label__ru", 0.9)],
                "мир": [("__label__ru", 0.8)],
                "hello": [("__label__en", 0.95)],
            }
        )
    )
    web_utils.installed_classifier.cache_clear()
    yield
    _test_hooks.probe = probe
    _test_hooks.model_loader = loader
    web_utils.installed_classifier.cache_clear()


@pytest.mark.usefixtures("installed_weights")
def test_masking_replaces_russian_tokens_and_reports_each_decision() -> None:
    """The web helper masks Russian and explains itself when asked."""
    from turkic_translit.web import web_utils

    result = web_utils.mask_russian(text="привет мир hello", thr=0.5, min_len=3, debug=True)

    assert "<RU> <RU> hello" in result
    debug_start = result.find("<!--debug ") + len("<!--debug ")
    debug_data = json.loads(result[debug_start : result.find(" -->", debug_start)])
    assert [entry["tok"] for entry in debug_data] == ["привет", "мир", "hello"]
    assert [entry["ru"] for entry in debug_data] == [True, True, False]
