import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fanyiji.config import AppConfig
from fanyiji.pipeline import FairQueue, SubtitlePipeline
from fanyiji.pipeline import RecognitionJob, TranslationJob, _Run
from fanyiji.recognition_quality import QualityRecognition
from fanyiji.recognition_quality import RecognitionCancelled
from fanyiji.speech_filter import FilterDecision


def test_queue_fairness_and_audio_duration_limit():
    queue = FairQueue(max_seconds=3)
    assert queue.put("mic", "mic1", 1)
    assert queue.put("mic", "mic2", 1)
    assert queue.put("system", "system1", 1)
    assert not queue.put("system", "overflow", 0.1)
    assert [queue.get(), queue.get(), queue.get()] == ["mic1", "system1", "mic2"]


def test_close_drains_accepted_work():
    queue = FairQueue()
    queue.put("system", "accepted", 1)
    queue.close()
    assert not queue.put("mic", "new", 1)
    assert queue.get() == "accepted"
    assert queue.get() is None


def test_configuration_switch_can_cancel_queued_work():
    queue = FairQueue()
    queue.put("mic", "old-direction", 1)
    queue.close(discard=True)
    assert queue.get() is None


def pipeline_models(tmp_path):
    for relative in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    return tmp_path


def test_pause_keeps_resident_translator_but_shutdown_closes_it_and_blocks_restart(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    translator = pipeline._translator = Mock()
    pipeline.stop()
    translator.close.assert_not_called()
    pipeline.shutdown()
    pipeline.shutdown()
    translator.close.assert_called_once()
    with patch.object(pipeline, "_initialize") as initialize:
        pipeline.start(AppConfig(), 1)
        initialize.assert_not_called()
    assert pipeline._run is None


def test_shutdown_cancels_a_published_translator_while_preload_is_in_flight(tmp_path):
    loaded, released = threading.Event(), threading.Event()
    translator = Mock()

    def preload():
        loaded.set()
        assert released.wait(3)

    translator.preload.side_effect = preload
    translator.close.side_effect = released.set
    pipeline = SubtitlePipeline(pipeline_models(tmp_path), Mock())
    with patch("fanyiji.engines.RecognizerEngine"), patch(
        "fanyiji.engines.create_translation_engine", return_value=translator
    ), patch("fanyiji.audio.AudioCaptureManager") as capture:
        pipeline.start(AppConfig(), 1)
        run = pipeline._run
        assert loaded.wait(2)
        started = time.monotonic()
        pipeline.shutdown()
        assert time.monotonic() - started < 1.5
        for thread in run.threads:
            thread.join(timeout=2)
        assert run.cancel.is_set()
        translator.close.assert_called_once()
        capture.assert_not_called()


def test_shutdown_before_factory_return_closes_unpublished_engine_without_preload(tmp_path):
    constructing, released = threading.Event(), threading.Event()
    translator = Mock()

    def create(*args, **kwargs):
        constructing.set()
        assert released.wait(3)
        return translator

    pipeline = SubtitlePipeline(pipeline_models(tmp_path), Mock())
    with patch("fanyiji.engines.RecognizerEngine"), patch(
        "fanyiji.engines.create_translation_engine", side_effect=create
    ), patch("fanyiji.audio.AudioCaptureManager") as capture:
        pipeline.start(AppConfig(), 1)
        run = pipeline._run
        assert constructing.wait(2)
        pipeline.shutdown()
        released.set()
        for thread in run.threads:
            thread.join(timeout=2)
        translator.close.assert_called_once()
        translator.preload.assert_not_called()
        capture.assert_not_called()


def recognition_run(speech_only=True):
    run = _Run(session=7, config=AppConfig(speech_only=speech_only))
    job = RecognitionJob("system", "7:system:1", np.full(32000, .25, np.float32),
                         time.monotonic(), ("ja", "zh"))
    run.asr_queue.put("system", job, 2)
    run.asr_queue.close()
    return run, job


def test_shutdown_during_filter_factory_closes_unpublished_filter(tmp_path):
    constructing, released = threading.Event(), threading.Event()
    speech_filter = Mock()

    def create(*args, **kwargs):
        constructing.set()
        assert released.wait(3)
        return speech_filter

    pipeline = SubtitlePipeline(pipeline_models(tmp_path), Mock())
    with patch("fanyiji.engines.RecognizerEngine"), patch(
        "fanyiji.engines.create_translation_engine"
    ), patch("fanyiji.speech_filter.SpeechOnlyFilter", side_effect=create), patch(
        "fanyiji.audio.AudioCaptureManager"
    ) as capture:
        pipeline.start(AppConfig(), 1)
        run = pipeline._run
        assert constructing.wait(2)
        pipeline.shutdown()
        released.set()
        for thread in run.threads:
            thread.join(timeout=2)
        assert pipeline._speech_filter is None
        speech_filter.close.assert_called_once()
        speech_filter.preload.assert_not_called()
        capture.assert_not_called()


def test_music_skip_never_reaches_recognition_or_translation(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    pipeline._recognizer = Mock()
    pipeline._speech_filter = Mock()
    run, job = recognition_run()
    pipeline._speech_filter.process.return_value = FilterDecision(np.zeros_like(job.samples),
                                                                 {"action": "skip", "reason": "music"})
    pipeline._recognize(run)
    pipeline._recognizer.recognize.assert_not_called()
    assert run.mt_queue.get() is None
    assert [event["kind"] for event in events] == ["speech_filter"]


def test_masked_audio_is_the_actual_recognition_input_and_raw_stays_intact(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    pipeline._recognizer = Mock()
    pipeline._recognizer.recognize.return_value = "今日は晴れ。"
    pipeline._speech_filter = Mock()
    pipeline.on_filtered = Mock()
    run, job = recognition_run()
    clean = job.samples.copy()
    clean[:16000] = 0
    decision = FilterDecision(clean, {"action": "masked"})
    pipeline._speech_filter.process.return_value = decision
    pipeline._recognize(run)
    args = pipeline._recognizer.recognize.call_args.args
    np.testing.assert_array_equal(args[0], clean)
    assert args[1] == "ja"
    np.testing.assert_array_equal(job.samples, np.full(32000, .25, np.float32))
    pipeline.on_filtered.assert_called_once_with(job, decision)
    assert run.mt_queue.get().text == "今日は晴れ。"
    assert run.mt_queue.get() is None


def test_disabled_filter_has_no_tagger_inference(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    pipeline._recognizer = Mock()
    pipeline._recognizer.recognize.return_value = "こんにちは"
    pipeline._speech_filter = Mock()
    run, job = recognition_run(speech_only=False)
    pipeline._recognize(run)
    pipeline._speech_filter.process.assert_not_called()
    np.testing.assert_array_equal(pipeline._recognizer.recognize.call_args.args[0], job.samples)


def test_filter_failure_does_not_silently_translate_unfiltered_audio(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    pipeline._recognizer = Mock()
    pipeline._speech_filter = Mock()
    pipeline._speech_filter.process.side_effect = RuntimeError("bad classification")
    run, _ = recognition_run()
    pipeline._recognize(run)
    pipeline._recognizer.recognize.assert_not_called()
    assert run.mt_queue.get() is None
    assert any(event["kind"] == "notice" for event in events)


@pytest.mark.parametrize(("configured", "capability", "automatic"), [
    (True, True, True), (False, True, False), (True, False, False),
    (True, None, False), (True, Mock(), False),
])
def test_multilingual_preload_requires_explicit_translation_support(tmp_path, configured, capability, automatic):
    events = []
    pipeline = SubtitlePipeline(pipeline_models(tmp_path), events.append)
    recognizer = SimpleNamespace(preload=Mock(), backend_info={"model_id": "asr-stub"})
    translator = SimpleNamespace(preload=Mock(), backend_info={"model_kind": "translation-stub"})
    if capability is not None:
        translator.supports_mixed_sources = capability
    config = AppConfig(speech_only=False, multilingual_asr=configured)
    run = pipeline._run = _Run(1, config)
    with patch("fanyiji.engines.RecognizerEngine", return_value=recognizer), patch(
        "fanyiji.engines.create_translation_engine", return_value=translator
    ), patch("fanyiji.segmentation.SpeechSegmenter"), patch(
        "fanyiji.audio.AudioCaptureManager"
    ), patch.object(pipeline, "_recognize"), patch.object(pipeline, "_translate"):
        try:
            pipeline._initialize(run)
            for worker in run.threads:
                worker.join(timeout=2)
            recognizer.preload.assert_called_once_with(("auto",) if automatic else ("ja", "zh"))
            assert run.multilingual_recognition is automatic
            ready = next(event for event in events if event["kind"] == "models_ready")
            assert ready["models"]["asr"]["recognition_mode"] == ("auto" if automatic else "fixed")
            assert ready["models"]["asr"]["recognition_language"] == ("auto" if automatic else "multiple")
            assert config.playback.target_lang == "zh" and config.microphone.target_lang == "ja"
            fallback = ready["models"]["asr"].get("multilingual_fallback")
            assert bool(fallback) is (configured and not automatic)
        finally:
            pipeline.shutdown()


def mixed_job(identifier, start, detected_language=None, *, ended_by="silence"):
    samples = np.full(16000, .1, np.float32)
    return RecognitionJob("system", identifier, samples, time.monotonic(), ("ja", "zh"),
        start, start + samples.size, 1, ended_by, detected_language=detected_language,
        detection_source="native" if detected_language else "unknown")


def drain_translation_jobs(run):
    run.mt_queue.close()
    jobs = []
    while (job := run.mt_queue.get()) is not None:
        jobs.append(job)
    return jobs


def test_automatic_worker_keeps_native_language_metadata_and_fixed_targets(tmp_path, caplog):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig(speech_only=False), multilingual_recognition=True)
    jobs = [mixed_job(identifier, index * 16000, ended_by="natural_pause")
            for index, identifier in enumerate(("japanese", "chinese", "english"))]
    quality = Mock()
    quality.recognize.side_effect = [
        QualityRecognition("今日は晴れです。", {"detected_language": "ja", "detection_source": "native"}),
        QualityRecognition("今天很晴朗。", {"detected_language": None, "detection_source": "unknown"}),
        QualityRecognition("The game will start soon.", {"detected_language": "en", "detection_source": "native"}),
    ]
    for job in jobs:
        run.asr_queue.put("system", job, 1)
    run.asr_queue.close()
    caplog.set_level("INFO", logger="fanyiji.pipeline")
    with patch("fanyiji.recognition_quality.QualityRecognizer", return_value=quality):
        pipeline._recognize(run)
    assert [call.args[1] for call in quality.recognize.call_args_list] == ["auto"] * 3
    recognition_events = [event for event in events if event["kind"] == "recognition"]
    assert [event["detected_language"] for event in recognition_events] == ["ja", None, "en"]
    assert all(event["selected_asr_language"] == "auto" for event in recognition_events)
    assert "selected_asr_language=auto detected_language=unknown detection_source=unknown" in caplog.text
    queued = drain_translation_jobs(run)
    assert [job.segment_id for job in queued] == ["japanese", "chinese", "english"]
    assert all(job.direction == ("ja", "zh") for job in queued)
    assert all(not job.context for job in queued)


@pytest.mark.parametrize(("provider", "batch", "expected_source"), [
    ("codex", False, "mixed"), ("codex", True, "mixed"), ("local", False, "mixed"),
])
def test_mixed_translation_requests_keep_the_configured_target(tmp_path, provider, batch, expected_source):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    run = _Run(1, AppConfig(translation_provider=provider), multilingual_recognition=True)
    translator = Mock(supports_mixed_sources=True)
    translator.backend_info = {"supports_source_context": True}
    translator.translate.return_value = "译文"
    translator.translate_batch.return_value = {"first": "译文", "second": "第二句"}
    jobs = [TranslationJob("system", "first", "今日は中文 and English", 100, ("ja", "zh"), 1)]
    if batch:
        jobs.append(replace(jobs[0], segment_id="second", text="可以開始です"))
    pipeline._translation_request(run, translator, jobs)
    method = translator.translate_batch if batch else translator.translate
    assert method.call_args.args[1:] == (expected_source, "zh")
    if provider == "codex":
        assert method.call_args.kwargs["cancellation_event"] is run.cancel
    assert all(job.direction == ("ja", "zh") for job in jobs)


def test_fixed_translation_adapter_never_receives_mixed_source(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    run = _Run(1, AppConfig(), multilingual_recognition=True)
    translator = Mock()
    translator.backend_info = {}
    translator.translate.return_value = "译文"
    job = TranslationJob("system", "fixed", "こんにちは", 100, ("ja", "zh"), 1)
    pipeline._translation_request(run, translator, [job])
    assert translator.translate.call_args.args == (job.text, "ja", "zh")


def test_automatic_japanese_continuation_rechecks_in_auto_mode(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig(), multilingual_recognition=True)
    first, second = mixed_job("first", 0, "ja"), mixed_job("second", 16000, "ja")
    quality = Mock()
    quality.recognize.return_value = QualityRecognition("森の妖精です。",
        {"detected_language": "ja", "detection_source": "native"})
    pipeline._accept_text(run, first, "森の。", first.samples, quality)
    pipeline._accept_text(run, second, "妖精です。", second.samples, quality)
    assert quality.recognize.call_args.args[1] == "auto"
    queued = drain_translation_jobs(run)
    assert len(queued) == 1 and queued[0].revision == 2
    assert queued[0].source_segment_ids == ("first", "second")
    assert queued[0].text == "森の妖精です。" and queued[0].direction == ("ja", "zh")
    assert first.detected_language == "ja" and first.detection_source == "native"


@pytest.mark.parametrize(("next_text", "native_language"), [
    ("今天天气很好。", "zh"), ("The game starts now.", "en"),
    ("今天天气很好。", None), ("Play the game と言いました。", "ja"),
])
def test_language_changes_and_ambiguous_mixed_text_preserve_boundaries(tmp_path, next_text, native_language):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig(), multilingual_recognition=True)
    first, second = mixed_job("first", 0, "ja"), mixed_job("second", 16000, native_language)
    quality = Mock()
    pipeline._accept_text(run, first, "森の。", first.samples, quality)
    pipeline._accept_text(run, second, next_text, second.samples, quality)
    pipeline._flush_pending(run, force=True)
    queued = drain_translation_jobs(run)
    assert [job.text for job in queued] == ["森の。", next_text]
    assert queued[1].context == ""
    assert all(job.source_segment_ids in (("first",), ("second",)) for job in queued)
    quality.recognize.assert_not_called()
    assert not any(event["kind"] == "sentence_join" for event in events)


def test_native_chinese_uses_chinese_continuation_rules_on_japanese_channel(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    run = _Run(1, AppConfig(), multilingual_recognition=True)
    first, second = mixed_job("first", 0, "zh"), mixed_job("second", 20000, "zh")
    quality = Mock()
    pipeline._accept_text(run, first, "如果明天下雨。", first.samples, quality)
    assert "system" in run.pending_text
    pipeline._accept_text(run, second, "我们就不去了。", second.samples, quality)
    queued = drain_translation_jobs(run)
    assert len(queued) == 1 and queued[0].text == "如果明天下雨我们就不去了。"
    assert queued[0].direction == ("ja", "zh")
    quality.recognize.assert_not_called()


def test_unknown_han_text_on_japanese_channel_does_not_claim_chinese_or_wait(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    run = _Run(1, AppConfig(), multilingual_recognition=True)
    first = mixed_job("unknown", 0)
    pipeline._accept_text(run, first, "因为", first.samples, Mock())
    assert not run.pending_text
    assert drain_translation_jobs(run)[0].text == "因为"
    assert first.detected_language is None and first.detection_source == "unknown"
    assert run.source_context.get("system", ("ja", "zh")) == ""


@pytest.mark.parametrize("native_language", ["en", None])
def test_consecutive_english_keeps_reference_context_without_japanese_phrase_join(tmp_path, native_language):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    run = _Run(1, AppConfig(), multilingual_recognition=True)
    first, second = mixed_job("first", 0, native_language), mixed_job("second", 16000, native_language)
    quality = Mock()
    pipeline._accept_text(run, first, "Keep the red key.", first.samples, quality)
    pipeline._accept_text(run, second, "Use it here.", second.samples, quality)
    queued = drain_translation_jobs(run)
    assert [job.segment_id for job in queued] == ["first", "second"]
    assert queued[0].context == "" and queued[1].context == "Keep the red key."
    assert not run.pending_text
    quality.recognize.assert_not_called()
    assert first.detected_language == native_language


def test_english_to_japanese_does_not_reuse_english_context(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    run = _Run(1, AppConfig(), multilingual_recognition=True)
    first, second = mixed_job("first", 0, "en"), mixed_job("second", 16000, "ja")
    pipeline._accept_text(run, first, "Keep the red key.", first.samples, Mock())
    pipeline._accept_text(run, second, "今日は晴れです。", second.samples, Mock())
    queued = drain_translation_jobs(run)
    assert [job.text for job in queued] == ["Keep the red key.", "今日は晴れです。"]
    assert queued[1].context == ""


@pytest.mark.parametrize("text", [
    "language None<asr_text>", "语言 None<asr_text>",
    "language null", "&lt;asr_text&gt;识别文本", "<|im_start|>assistant",
])
def test_protocol_source_never_emits_original_or_enters_translation_queue(tmp_path, text):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig())
    job = mixed_job("protocol", 0)
    run.source_context.remember("system", ("ja", "zh"), "安全前文", now=job.ended_at)
    run.source_languages["system"] = "ja"
    pipeline._enqueue_translation(run, job, text)
    assert drain_translation_jobs(run) == []
    assert not any(event["kind"] == "original" for event in events)
    assert any(event["kind"] == "notice" for event in events)
    assert run.source_context.get("system", ("ja", "zh"), now=job.ended_at) == ""
    assert "system" not in run.source_languages


def test_protocol_fragment_cannot_emit_a_provisional_original(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig())
    job = mixed_job("protocol", 0)
    pipeline._accept_text(run, job, "language None<asr_text>森の。", job.samples, Mock())
    assert not run.pending_text and drain_translation_jobs(run) == []
    assert not any(event["kind"] == "original" for event in events)


def test_protocol_missing_segment_finishes_the_prior_safe_pending_fragment(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig())
    first, second = mixed_job("safe", 0), mixed_job("bad", 16000)
    pipeline._accept_text(run, first, "森の。", first.samples, Mock())
    pipeline._accept_text(run, second, "language None<asr_text>", second.samples, Mock())
    queued = drain_translation_jobs(run)
    assert len(queued) == 1 and queued[0].text == "森の。"
    assert not run.pending_text
    assert all("<asr_text>" not in event.get("text", "") for event in events)


def test_protocol_cannot_be_remembered_as_source_context(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    run = _Run(1, AppConfig())
    job = mixed_job("protocol", 0)
    run.source_context.remember("system", ("ja", "zh"), "安全前文", now=job.ended_at)
    run.source_languages["system"] = "ja"
    pipeline._remember_source(run, job, "<asr_text>污染")
    assert run.source_context.get("system", ("ja", "zh"), now=job.ended_at) == ""
    assert "system" not in run.source_languages


@pytest.mark.parametrize("provider", ["codex", "local"])
@pytest.mark.parametrize("field", ["text", "context"])
def test_independent_translation_job_protocol_never_reaches_any_model(tmp_path, provider, field):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig(translation_provider=provider))
    translator = Mock()
    translator.backend_info = {"supports_source_context": True}
    job = TranslationJob("system", "bad", "安全原文", 100, ("ja", "zh"), 1)
    job = replace(job, **{field: "语言 None<asr_text>"})
    pipeline._translation_request(run, translator, [job])
    translator.translate.assert_not_called()
    translator.translate_batch.assert_not_called()
    translated = next(event for event in events if event["kind"] == "translation")
    assert translated["failed"] is True and translated["protocol_rejected"] is True
    assert "<asr_text>" not in translated["text"] + translated["context"]
    assert any(event["kind"] == "notice" for event in events)


def test_protocol_input_in_batch_rejects_only_the_bad_job(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig(translation_provider="codex"), multilingual_recognition=True)
    translator = Mock(supports_mixed_sources=True)
    translator.backend_info = {"supports_source_context": True}
    translator.translate_batch.return_value = {"safe1": "安全译文一", "safe2": "安全译文二"}
    safe = TranslationJob("system", "safe1", "今日は晴れ。", 100, ("ja", "zh"), 1)
    bad = replace(safe, segment_id="bad", context="language None<asr_text>")
    following = replace(safe, segment_id="safe2", text="明日は休み。")
    pipeline._translation_request(run, translator, [safe, bad, following])
    translator.translate.assert_not_called()
    sent = translator.translate_batch.call_args.args[0]
    assert [job["id"] for job in sent] == ["safe1", "safe2"]
    assert translator.translate_batch.call_args.args[1:] == ("mixed", "zh")
    published = [event for event in events if event["kind"] == "translation"]
    assert {event["segment_id"] for event in published} == {"safe1", "safe2", "bad"}
    assert next(event for event in published if event["segment_id"] == "bad")["failed"] is True
    assert all("<asr_text>" not in event["text"] + event["context"] for event in published)


@pytest.mark.parametrize(("provider", "batch"), [("codex", False), ("codex", True), ("local", False)])
def test_protocol_model_output_becomes_failed_placeholder_and_next_good_output_survives(tmp_path, provider, batch):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig(translation_provider=provider))
    translator = Mock()
    translator.backend_info = {}
    translator.translate.return_value = "语言 None<asr_text>"
    translator.translate_batch.return_value = {"bad": "语言 None<asr_text>", "safe": "安全译文"}
    jobs = [TranslationJob("system", "bad", "今天很晴朗。", 100, ("ja", "zh"), 1)]
    if batch:
        jobs.append(replace(jobs[0], segment_id="safe"))
    pipeline._translation_request(run, translator, jobs)
    published = [event for event in events if event["kind"] == "translation"]
    rejected = next(event for event in published if event["segment_id"] == "bad")
    assert rejected["failed"] is True and rejected["protocol_rejected"] is True
    assert "<asr_text>" not in rejected["text"]
    assert any(event["kind"] == "notice" for event in events)
    if batch:
        accepted = next(event for event in published if event["segment_id"] == "safe")
        assert accepted["text"] == "安全译文" and accepted["failed"] is False


def test_normal_recognition_cancellation_is_not_an_asr_error(tmp_path, caplog):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run, _ = recognition_run(speech_only=False)
    quality = Mock()

    def cancel(*_args):
        run.cancel.set()
        raise RecognitionCancelled("识别复核已取消")

    quality.recognize.side_effect = cancel
    with patch("fanyiji.recognition_quality.QualityRecognizer", return_value=quality):
        pipeline._recognize(run)
    assert not any(record.levelname == "ERROR" for record in caplog.records)
    assert not any(event["kind"] in ("notice", "original", "translation") for event in events)
    assert drain_translation_jobs(run) == []


def test_cancelled_joint_recheck_cannot_publish_failure_or_new_subtitle(tmp_path, caplog):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig())
    first, second = mixed_job("first", 0), mixed_job("second", 16000)
    quality = Mock()
    pipeline._accept_text(run, first, "森の。", first.samples, quality)
    before = len(events)

    def cancel(*_args):
        run.cancel.set()
        raise RecognitionCancelled("识别复核已取消")

    quality.recognize.side_effect = cancel
    pipeline._accept_text(run, second, "妖精です。", second.samples, quality)
    assert events[before:] == [] and drain_translation_jobs(run) == []
    assert not any(record.levelname == "ERROR" for record in caplog.records)
