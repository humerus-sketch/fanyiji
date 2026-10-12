"""Signal properties and bounded selection policy; no model or microphone use."""
from __future__ import annotations

import threading
from unittest.mock import Mock

import numpy as np
import pytest

from fanyiji.recognition_quality import (
    QualityRecognizer, RecognitionCancelled, time_stretch_wsola,
)
from fanyiji.language_policy import AsrResult
from fanyiji.asr_protocol import AsrProtocolError


def tone(seconds=2., frequency=220.):
    timeline = np.arange(round(seconds * 16000), dtype=np.float64) / 16000
    return (.3 * np.sin(2 * np.pi * frequency * timeline)
            + .08 * np.sin(2 * np.pi * 2 * frequency * timeline)).astype(np.float32)


def test_residual_protocol_is_hard_rejected_before_density_or_slowdown():
    engine = ScriptedRecognizer(["language None<asr_text>", "正常正文"])
    with pytest.raises(AsrProtocolError):
        QualityRecognizer(engine).recognize(tone(3.), "auto")
    assert len(engine.calls) == 1
    # Only the affected segment is rejected; the next segment remains usable.
    assert QualityRecognizer(engine).recognize(tone(3.), "auto").text == "正常正文"


def test_short_protocol_only_result_cannot_be_preserved_as_uncertain_text():
    engine = ScriptedRecognizer(["language None<asr_text>"])
    with pytest.raises(AsrProtocolError):
        QualityRecognizer(engine).recognize(tone(.054), "auto")
    assert len(engine.calls) == 1


def test_protocol_in_speed_review_cannot_recover_an_empty_source():
    engine = ScriptedRecognizer(["", "language None<asr_text>"])
    result = QualityRecognizer(engine).recognize(tone(2.), "auto")
    assert result.text == "" and result.metadata["status"] == "review_failed_original_preserved"
    assert len(engine.calls) == 2


def test_protocol_in_generation_recovery_never_emits_partial_text():
    engine = ScriptedRecognizer([RuntimeError("Qwen识别接近128 token生成上限"),
                                "正常正文", "language None<asr_text>"])
    result = QualityRecognizer(engine).recognize(tone(6.), "auto")
    assert result.text == "" and result.metadata["status"] == "generation_limit_part_failed"
    assert len(engine.calls) == 3


class ScriptedRecognizer:
    def __init__(self, results, *, event=None, mutate=False):
        self.results = iter(results)
        self.calls = []
        self.event = event
        self.mutate = mutate

    def recognize(self, samples, language):
        self.calls.append((samples.copy(), language))
        if self.mutate:
            samples[:] = 0
        if self.event is not None:
            self.event.set()
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def spectral_peak(samples):
    # Avoid the endpoint join when checking the stationary harmonic signal.
    center = samples[4000:-4000].astype(np.float64)
    magnitudes = np.abs(np.fft.rfft(center * np.hanning(center.size)))
    return np.fft.rfftfreq(center.size, 1 / 16000)[np.argmax(magnitudes)]


@pytest.mark.parametrize("frequency", [100., 220., 440.])
def test_wsola_preserves_pitch_duration_and_original_samples(frequency):
    source = tone(2., frequency)
    original = source.copy()
    slowed = time_stretch_wsola(source, .92)
    assert slowed.dtype == np.float32 and slowed.ndim == 1
    assert slowed.size == round(source.size / .92)
    assert abs(spectral_peak(slowed) - frequency) < 2.
    # A sample-rate trick would shift a 440 Hz tone by about 35 Hz.
    assert abs(spectral_peak(slowed) - spectral_peak(source)) < 2.
    np.testing.assert_array_equal(source, original)
    assert not np.shares_memory(source, slowed)
    assert np.isfinite(slowed).all()
    assert np.max(np.abs(slowed)) <= np.max(np.abs(source)) + 1e-6
    np.testing.assert_array_equal(slowed[:320], source[:320])
    np.testing.assert_array_equal(slowed[-320:], source[-320:])


def test_wsola_keeps_silence_and_neutral_speed_exact():
    silent = np.zeros(16000, np.float32)
    stretched = time_stretch_wsola(silent)
    assert not stretched.any() and stretched.size == round(16000 / .92)
    source = tone()
    np.testing.assert_array_equal(time_stretch_wsola(source, 1.), source)


@pytest.mark.parametrize("speed", [.0, .84, 1.01, float("nan"), float("inf"), True])
def test_invalid_speed_rejected(speed):
    with pytest.raises(ValueError):
        time_stretch_wsola(tone(), speed)
    with pytest.raises(ValueError):
        QualityRecognizer(ScriptedRecognizer([]), speed=speed)


@pytest.mark.parametrize("source", [np.zeros((2, 2), np.float32),
                                     np.zeros(1600, np.float64),
                                     np.array([float("nan")], np.float32)])
def test_invalid_input_fails_before_decode(source):
    recognizer = ScriptedRecognizer([])
    with pytest.raises(ValueError):
        QualityRecognizer(recognizer).recognize(source, "ja")
    assert not recognizer.calls


def test_short_wsola_rejected_without_pitch_changing_fallback():
    with pytest.raises(ValueError, match="80 ms"):
        time_stretch_wsola(tone(.04))


def test_normal_clear_text_decodes_once_even_when_fast_inference_available():
    recognizer = ScriptedRecognizer(["今日は天気がいいです。"])
    result = QualityRecognizer(recognizer).recognize(tone(), "ja")
    assert result.text == "今日は天気がいいです。"
    assert len(recognizer.calls) == 1
    assert result.metadata["suspicions"] == []
    assert result.metadata["status"] == "original"
    assert result.metadata["speed_adjustment"] is None


def test_empty_audio_has_no_decode_or_speed_review():
    recognizer = ScriptedRecognizer([])
    result = QualityRecognizer(recognizer).recognize(np.zeros(0, np.float32), "ja")
    assert not result.text and not recognizer.calls
    assert result.metadata["status"] == "empty_input"


def test_empty_text_on_silence_is_not_guessed_or_rerun():
    recognizer = ScriptedRecognizer([""])
    result = QualityRecognizer(recognizer).recognize(np.zeros(32000, np.float32), "ja")
    assert not result.text and len(recognizer.calls) == 1
    assert not result.metadata["uncertain"]


def test_empty_with_audio_recovers_once_but_keeps_unverified_label():
    source = tone()
    recognizer = ScriptedRecognizer(["", "今日は雨です。"], mutate=True)
    result = QualityRecognizer(recognizer).recognize(source, "ja")
    assert result.text == "今日は雨です。"
    assert result.metadata["original_text"] == ""
    assert result.metadata["review_text"] == "今日は雨です。"
    assert result.metadata["selected"] == "speed_review"
    assert result.metadata["uncertain"] and result.metadata["disagreement"]
    assert len(recognizer.calls) == 2
    assert recognizer.calls[1][0].size == round(source.size / .92)
    np.testing.assert_array_equal(source, tone())
    assert result.metadata["attempts"][1]["sample_rate"] == 16000
    assert result.metadata["speed_adjustment"]["method"] == "WSOLA"
    assert result.metadata["elapsed_seconds"] >= 0


def test_disagreement_cannot_silently_replace_plausible_original():
    original = "今日は雨です。" * 5
    recognizer = ScriptedRecognizer([original, "明日は晴れです。"])
    result = QualityRecognizer(recognizer).recognize(tone(), "ja")
    assert result.text == original
    assert "repetitive_output" in result.metadata["suspicions"]
    assert result.metadata["disagreement"]
    assert result.metadata["status"] == "review_disagrees_original_preserved"
    assert result.metadata["review_text"] == "明日は晴れです。"


def test_high_text_density_triggers_one_review_without_claiming_true_speech_rate():
    original = "本日は午後三時から図書館の会議室で新しい計画について詳しく説明します"
    recognizer = ScriptedRecognizer([original, original])
    result = QualityRecognizer(recognizer).recognize(tone(2.), "ja")
    assert len(recognizer.calls) == 2
    assert "high_text_density" in result.metadata["suspicions"]
    assert result.metadata["text_density_characters_per_second"] > 12.
    assert result.metadata["text_density_is_speaking_rate"] is False
    assert result.metadata["uncertain"]


def test_uncertain_real_repetition_is_never_deleted_by_speed_review():
    original = "はい、はい、はい、はい、はい、はい、はい、はい。"
    recognizer = ScriptedRecognizer([original, "はい。"])
    result = QualityRecognizer(recognizer).recognize(tone(2.), "ja")
    assert result.text == original and result.metadata["selected"] == "original"
    assert result.metadata["disagreement"] and result.metadata["uncertain"]


def test_agreement_is_not_confidence_or_proof_of_correctness():
    repeated = "違います。" * 6
    result = QualityRecognizer(ScriptedRecognizer([repeated, repeated])).recognize(tone(), "ja")
    assert result.metadata["status"] == "review_agrees_unverified"
    assert result.metadata["uncertain"]
    assert "confidence" not in result.metadata


def test_decode_error_during_speed_review_preserves_original_and_records_error():
    recognizer = ScriptedRecognizer(["", OSError("native failure")])
    result = QualityRecognizer(recognizer).recognize(tone(), "ja")
    assert result.text == "" and len(recognizer.calls) == 2
    assert result.metadata["status"] == "review_failed_original_preserved"
    assert result.metadata["attempts"][1]["error"]["type"] == "OSError"


def test_original_load_error_propagates_without_retry():
    recognizer = ScriptedRecognizer([OSError("missing model")])
    with pytest.raises(OSError, match="missing model"):
        QualityRecognizer(recognizer).recognize(tone(), "ja")
    assert len(recognizer.calls) == 1


def test_short_suspicious_original_kept_without_speedup():
    recognizer = ScriptedRecognizer(["今日は�"])
    result = QualityRecognizer(recognizer).recognize(tone(.2), "ja")
    assert result.text == "今日は�" and len(recognizer.calls) == 1
    assert result.metadata["status"] == "short_suspicious_original_preserved"


@pytest.mark.parametrize(("language", "seconds", "text"), [
    ("ja", .02, "本日は午後三時から図書館です。"),
    ("ja", .399, "本日は午後三時から図書館の会議室で説明します。"),
    ("zh", .05, "今天下午三点开始在图书馆会议室说明计划。"),
])
def test_extreme_short_tail_density_marks_uncertain_without_redecode(language, seconds, text):
    source = tone(seconds)
    recognizer = ScriptedRecognizer([text])
    result = QualityRecognizer(recognizer).recognize(source, language)
    assert result.text == text and len(recognizer.calls) == 1
    assert result.metadata["suspicions"] == ["short_audio_text_density"]
    assert result.metadata["uncertain"] and not result.metadata["disagreement"]
    assert result.metadata["selected"] == "original"
    assert result.metadata["status"] == "short_suspicious_original_preserved"
    assert result.metadata["text_density_is_speaking_rate"] is False
    assert result.metadata["speed_adjustment"] is None
    np.testing.assert_array_equal(source, tone(seconds))


@pytest.mark.parametrize(("language", "seconds", "text"), [
    ("ja", .02, "はい。"),
    ("ja", .2, "いいえ。"),
    ("ja", .4, "今日は行きません。"),
    ("zh", .02, "对。"),
    ("zh", .2, "不去。"),
    ("zh", .4, "不是今天，是明天。"),
])
def test_short_answers_and_plausible_negation_keep_single_normal_decode(language, seconds, text):
    recognizer = ScriptedRecognizer([text])
    result = QualityRecognizer(recognizer).recognize(tone(seconds), language)
    assert result.text == text and len(recognizer.calls) == 1
    assert result.metadata["status"] == "original"
    assert not result.metadata["uncertain"] and not result.metadata["suspicions"]
    assert result.metadata["speed_adjustment"] is None


@pytest.mark.parametrize(("language", "threshold"), [("ja", 32), ("zh", 24)])
def test_short_density_equal_threshold_is_not_flagged_but_above_is(language, threshold):
    text = "本日は午後三時から図書館の会議室で新しい計画について詳しく説明します"
    allowed = text[:threshold]
    result = QualityRecognizer(ScriptedRecognizer([allowed])).recognize(tone(1.), language)
    assert len(allowed) == threshold
    assert result.metadata["text_density_characters_per_second"] == threshold
    assert not result.metadata["uncertain"] and not result.metadata["suspicions"]
    suspicious = text[:threshold + 1]
    recognizer = ScriptedRecognizer([suspicious, suspicious])
    result = QualityRecognizer(recognizer).recognize(tone(1.), language)
    assert result.metadata["suspicions"] == ["short_audio_text_density"]
    assert result.metadata["status"] == "review_agrees_unverified"
    assert result.metadata["uncertain"] and len(recognizer.calls) == 2


def test_short_density_at_400ms_uses_bounded_review_and_preserves_disagreement():
    text = "本日は午後三時から図書館の会議室で新しい計画について詳しく説明します"
    recognizer = ScriptedRecognizer([text, "明日は晴れです。"])
    result = QualityRecognizer(recognizer).recognize(tone(.4), "ja")
    assert result.text == text and len(recognizer.calls) == 2
    assert result.metadata["suspicions"] == ["short_audio_text_density"]
    assert result.metadata["uncertain"] and result.metadata["disagreement"]
    assert result.metadata["selected"] == "original"
    assert result.metadata["status"] == "review_disagrees_original_preserved"
    assert result.metadata["speed_adjustment"]["method"] == "WSOLA"
    assert result.metadata["speed_adjustment"]["speed"] == .92
    assert result.metadata["text_density_is_speaking_rate"] is False


def cap_error():
    return RuntimeError("Qwen识别接近128 token生成上限，可能不完整")


def test_generation_cap_uses_three_exact_nonoverlapping_parts_without_speed_review():
    source = tone(10.)
    source[52000:55200] = 0
    source[104000:108000] = 0
    original = source.copy()
    recognizer = ScriptedRecognizer([cap_error(), "前の文です。", "次の文です。", "最後の文です。"])
    result = QualityRecognizer(recognizer).recognize(source, "ja")
    assert result.text == "前の文です。 次の文です。 最後の文です。"
    assert len(recognizer.calls) == 4
    reconstructed = np.concatenate([call[0] for call in recognizer.calls[1:]])
    np.testing.assert_array_equal(reconstructed, original)
    np.testing.assert_array_equal(source, original)
    assert result.metadata["split_overlap_samples"] == 0
    assert result.metadata["speed_adjustment"] is None
    assert result.metadata["status"] == "generation_limit_recovered_unverified"
    assert result.metadata["uncertain"]
    boundaries = result.metadata["split_boundaries_samples"]
    assert boundaries[0] == 0 and boundaries[-1] == source.size
    assert max(np.diff(boundaries)) < 5 * 16000


def test_generation_cap_recovery_failure_does_not_recurse_or_translate_partial():
    recognizer = ScriptedRecognizer([cap_error(), "前です。", cap_error()])
    result = QualityRecognizer(recognizer).recognize(tone(6.), "ja")
    assert result.text == "" and len(recognizer.calls) == 3
    assert result.metadata["status"] == "generation_limit_part_failed"
    assert result.metadata["selected"] == "none"
    assert result.metadata["attempts"][1]["text"] == "前です。"


def test_generation_cap_empty_piece_never_drops_it_and_presents_other_piece_as_full():
    recognizer = ScriptedRecognizer([cap_error(), ""])
    result = QualityRecognizer(recognizer).recognize(tone(6.), "ja")
    assert result.text == "" and len(recognizer.calls) == 2
    assert result.metadata["status"] == "generation_limit_part_empty"


@pytest.mark.parametrize("seconds", [.3, 14.])
def test_generation_cap_outside_split_budget_never_runs_unbounded(seconds):
    recognizer = ScriptedRecognizer([cap_error()])
    result = QualityRecognizer(recognizer).recognize(tone(seconds), "ja")
    assert result.text == "" and len(recognizer.calls) == 1
    assert result.metadata["status"] == "generation_limit_recovery_outside_budget"


def test_cancel_before_native_decode_and_after_native_decode():
    event = threading.Event()
    event.set()
    recognizer = ScriptedRecognizer([])
    with pytest.raises(RecognitionCancelled):
        QualityRecognizer(recognizer, cancel_event=event).recognize(tone(), "ja")
    assert not recognizer.calls
    event.clear()
    recognizer = ScriptedRecognizer(["今日は晴れです。"], event=event)
    with pytest.raises(RecognitionCancelled):
        QualityRecognizer(recognizer, cancel_event=event).recognize(tone(), "ja")
    assert len(recognizer.calls) == 1


def test_wsola_checks_cancellation_inside_processing():
    checks = 0
    def cancelled():
        nonlocal checks
        checks += 1
        return checks > 3
    with pytest.raises(RecognitionCancelled):
        time_stretch_wsola(tone(), cancelled=cancelled)
    assert checks == 4


@pytest.mark.parametrize(("language", "text"), [
    ("auto", "Please proceed to the next checkpoint and save your progress."),
    ("en", "This extraordinary achievement belongs to everyone who worked here."),
    ("ja", "Please reload your weapon before entering the next area."),
    ("auto", "次の checkpoint まで進んでください。"),
    ("auto", "请前往 checkpoint，然后 save your progress。"),
])
def test_normal_latin_words_and_mixed_scripts_do_not_trigger_character_density(language, text):
    recognizer = ScriptedRecognizer([text])
    result = QualityRecognizer(recognizer).recognize(tone(2.), language)
    assert result.text == text and len(recognizer.calls) == 1
    assert result.metadata["suspicions"] == []
    assert result.metadata["text_density_policy"] == "script_and_word_count"
    assert result.metadata["text_density_words_per_second"] > 0
    assert result.metadata["language_policy"] == language
    assert result.metadata["detected_language"] is None


def test_extreme_latin_word_density_still_gets_one_bounded_review():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen"
    recognizer = ScriptedRecognizer([text, text])
    result = QualityRecognizer(recognizer).recognize(tone(2.), "auto")
    assert result.metadata["suspicions"] == ["high_text_density"]
    assert result.metadata["text_density_words_per_second"] == 8.5
    assert result.metadata["text_density_non_latin_characters_per_second"] == 0
    assert len(recognizer.calls) == 2 and all(language == "auto" for _, language in recognizer.calls)


def test_short_english_hallucination_density_is_flagged_without_unbounded_review():
    text = "one two three four five six seven eight nine ten"
    recognizer = ScriptedRecognizer([text])
    result = QualityRecognizer(recognizer).recognize(tone(.1), "auto")
    assert result.metadata["suspicions"] == ["short_audio_text_density"]
    assert len(recognizer.calls) == 1 and result.metadata["uncertain"]


class DetailedScriptedRecognizer(ScriptedRecognizer):
    def recognize_detailed(self, samples, language):
        return super().recognize(samples, language)

    def recognize(self, samples, language):
        raise AssertionError("The type-defined detailed API should be used")


def test_legacy_mock_does_not_auto_create_and_invoke_a_detailed_api():
    recognizer = Mock()
    recognizer.recognize.return_value = "英語と中文です。"
    result = QualityRecognizer(recognizer).recognize(tone(2.), "auto")
    recognizer.recognize.assert_called_once()
    recognizer.recognize_detailed.assert_not_called()
    assert result.text == "英語と中文です。"
    assert result.metadata["detected_language"] is None and result.metadata["detection_source"] == "unknown"


def test_rich_original_detection_kept_when_different_review_disagrees():
    original = "違います。" * 6
    recognizer = DetailedScriptedRecognizer([
        AsrResult(original, "ja", "native_result"), AsrResult("今天下雨。", "zh", "qwen_protocol")])
    result = QualityRecognizer(recognizer).recognize(tone(2.), "auto")
    assert result.text == original
    assert result.metadata["detected_language"] == "ja" and result.metadata["detection_source"] == "native_result"
    assert [attempt["detected_language"] for attempt in result.metadata["attempts"]] == ["ja", "zh"]
    assert all(language == "auto" for _, language in recognizer.calls)
    assert "confidence" not in result.metadata


def test_recovered_rich_review_detection_follows_the_selected_text():
    recognizer = DetailedScriptedRecognizer([AsrResult(""), AsrResult("Please wait.", "en", "native_result")])
    result = QualityRecognizer(recognizer).recognize(tone(2.), "auto")
    assert result.text == "Please wait." and result.metadata["selected"] == "speed_review"
    assert result.metadata["detected_language"] == "en" and result.metadata["detection_source"] == "native_result"
    assert len(recognizer.calls) == 2 and all(language == "auto" for _, language in recognizer.calls)


@pytest.mark.parametrize(("parts", "detected", "source"), [
    ([AsrResult("前です。", "ja", "native_result"), AsrResult("後です。", "ja", "native_result")], "ja", "native_result"),
    ([AsrResult("前です。", "ja", "qwen_protocol"), AsrResult("Then continue.", "en", "native_result")], None, "mixed_parts"),
    ([AsrResult("前です。", "ja", "native_result"), AsrResult("Then continue.")], None, "unknown"),
])
def test_auto_split_recovery_preserves_per_part_language_without_fabricating_one(parts, detected, source):
    recognizer = DetailedScriptedRecognizer([cap_error(), *parts])
    result = QualityRecognizer(recognizer).recognize(tone(6.), "auto")
    assert result.metadata["selected"] == "split_recovery"
    assert result.metadata["detected_language"] == detected and result.metadata["detection_source"] == source
    assert len(recognizer.calls) == 3 and all(language == "auto" for _, language in recognizer.calls)
    assert [attempt["detected_language"] for attempt in result.metadata["attempts"][1:]] == [part.detected_language for part in parts]


def test_rich_api_rejects_incompatible_result_instead_of_silently_guessing():
    recognizer = DetailedScriptedRecognizer(["legacy text"])
    with pytest.raises(TypeError, match="AsrResult"):
        QualityRecognizer(recognizer).recognize(tone(2.), "auto")
    assert len(recognizer.calls) == 1


def test_rich_api_rejects_nonstring_text_before_quality_policy():
    recognizer = DetailedScriptedRecognizer([AsrResult(None)])
    with pytest.raises(TypeError, match="AsrResult.text"):
        QualityRecognizer(recognizer).recognize(tone(2.), "auto")
    assert len(recognizer.calls) == 1


def test_auto_density_uses_reported_japanese_for_ambiguous_han_script():
    text = "日本国内空港到着後手荷物受取場所確認完了"
    assert 18 < len(text) <= 24
    recognizer = DetailedScriptedRecognizer([AsrResult(text, "Japanese", "native_result")])
    result = QualityRecognizer(recognizer).recognize(tone(2.), "auto")
    assert result.metadata["detected_language"] == "ja"
    assert not result.metadata["suspicions"] and len(recognizer.calls) == 1
