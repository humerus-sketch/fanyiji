import pytest

from fanyiji.language_policy import (
    AsrResult, continuation_language, context_language, normalize_detected_language,
    validate_asr_language, validate_translation_source,
)


def test_recognition_and_translation_languages_have_separate_contracts():
    assert validate_asr_language("auto") == "auto"
    assert validate_translation_source("mixed") == "mixed"
    for validate, value in ((validate_asr_language, "mixed"), (validate_translation_source, "auto")):
        with pytest.raises(ValueError):
            validate(value)
    assert AsrResult("原文").detected_language is None


@pytest.mark.parametrize("value,expected", [
    ("Japanese", "ja"), ("Chinese", "zh"), (" English ", "en"),
    ("auto", None), ("mixed", None), ("Korean", None), (None, None),
])
def test_detection_normalizes_only_real_supported_language_labels(value, expected):
    assert normalize_detected_language(value) == expected


@pytest.mark.parametrize("text,primary,detected,expected", [
    ("森の。", "ja", None, "ja"),
    ("この HP の。", "ja", None, "ja"),
    ("I need more ammo の。", "ja", None, None),
    ("如果没弹药的话。", "ja", "zh", "zh"),
    ("如果没弹药的话。", "ja", None, None),
    ("森林。", "ja", "ja", "ja"),
    ("因为。", "zh", None, "zh"),
    ("はい。", "ja", "zh", None),
    ("Inventory", "ja", None, None),
    ("森の。", "ja", "en", None),
])
def test_multilingual_continuation_retains_uncertain_or_switched_boundaries(text, primary, detected, expected):
    assert continuation_language(text, primary, detected, automatic=True) == expected


def test_fixed_mode_preserves_legacy_continuation_rules():
    assert continuation_language("森林。", "ja", "zh", automatic=False) == "ja"


@pytest.mark.parametrize("text,detected,expected", [
    ("Keep the red key.", None, "en"), ("Use it here.", "en", "en"),
    ("Inventory", None, None), ("Inventory", "en", "en"),
    ("Keep the red key の。", None, None), ("Use it here.", "ja", None),
    ("森の。", "en", None), ("森の。", None, "ja"),
])
def test_context_policy_keeps_clear_english_without_unsafe_cjk_joins(text, detected, expected):
    assert context_language(text, "ja", detected, automatic=True) == expected
