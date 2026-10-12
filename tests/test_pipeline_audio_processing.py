from dataclasses import replace
from types import SimpleNamespace
import threading
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fanyiji.audio import AudioChunk
from fanyiji.config import AppConfig
from fanyiji.pipeline import RecognitionJob, SubtitlePipeline, TranslationJob, _Run
from fanyiji.recognition_quality import QualityRecognition
from fanyiji.speech_filter import FilterDecision


def audio_job(*, seconds=1.0, room="21399689", generation=4, identifier="one", start=0):
    samples = np.full(round(seconds * 16000), .1, np.float32)
    return RecognitionJob("system", identifier, samples, time.monotonic(), ("ja", "zh"),
        start, start + samples.size, generation, "natural_pause", source_room=room)


def worker(tmp_path, jobs=None, *, mode="gtcrn", speech_only=False, validator=None):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append, source_validator=validator)
    run = _Run(1, AppConfig(speech_only=speech_only, audio_processing=mode))
    quality = Mock()
    quality.recognize.return_value = QualityRecognition("今日は晴れです。", {})
    for job in jobs or [audio_job()]:
        run.asr_queue.put(job.channel, job, job.samples.size / 16000)
    run.asr_queue.close()
    return pipeline, run, quality, events


def process_result(samples, mode="gtcrn"):
    return SimpleNamespace(samples=samples, metadata={"mode": mode,
        "action": "enhanced" if mode == "gtcrn" else "separated", "reason": "processed"})


def recognize(pipeline, run, quality):
    with patch("fanyiji.recognition_quality.QualityRecognizer", return_value=quality):
        pipeline._recognize(run)


@pytest.mark.parametrize("mode", ["gtcrn", "uvr"])
def test_processing_uses_speech_filtered_pcm_keeps_raw_and_clocks(tmp_path, mode):
    job = audio_job()
    original = job.samples.copy()
    pipeline, run, quality, events = worker(tmp_path, [job], mode=mode, speech_only=True)
    filtered = original.copy()
    filtered[:8000] = 0
    enhanced = filtered * .5
    pipeline._speech_filter = Mock()
    pipeline._speech_filter.process.return_value = FilterDecision(filtered, {"action": "masked"})
    run.audio_processor = Mock()
    run.audio_processor.process.return_value = process_result(enhanced, mode)
    recognize(pipeline, run, quality)
    np.testing.assert_array_equal(run.audio_processor.process.call_args.args[0], filtered)
    np.testing.assert_array_equal(quality.recognize.call_args.args[0], enhanced)
    np.testing.assert_array_equal(job.samples, original)
    translated = run.mt_queue.get()
    assert translated.source_room == job.source_room and translated.source_generation == job.generation
    assert translated.seconds == 1 and (job.start_sample, job.end_sample) == (0, 16000)
    timing = translated.timing
    assert timing["filter_finished_at"] <= timing["audio_processing_started_at"]
    assert timing["audio_processing_started_at"] <= timing["audio_processing_finished_at"] <= timing["asr_started_at"]
    assert [event["kind"] for event in events][:3] == ["speech_filter", "audio_processing", "recognition"]


def test_off_does_not_construct_or_process_model(tmp_path):
    pipeline, run, quality, _ = worker(tmp_path, mode="off")
    with patch("fanyiji.audio_processing.AudioProcessor") as factory:
        assert pipeline._prepare_audio_processor(run)["enabled"] is False
    factory.assert_not_called()
    run.audio_processor = Mock(side_effect=AssertionError("off must not infer"))
    recognize(pipeline, run, quality)
    run.audio_processor.process.assert_not_called()
    assert run.mt_queue.get().timing["audio_processing_seconds"] == 0


def test_no_processing_model_and_very_short_reply_do_not_discard_speech(tmp_path):
    job = audio_job(seconds=.1)
    pipeline, run, quality, events = worker(tmp_path, [job])
    recognize(pipeline, run, quality)
    np.testing.assert_array_equal(quality.recognize.call_args.args[0], job.samples)
    assert run.mt_queue.get().segment_id == job.segment_id
    decision = next(event for event in events if event["kind"] == "audio_processing")
    assert decision["action"] == "bypass" and decision["reason"] == "processor_unavailable"


@pytest.mark.parametrize("bad", ["exception", "inplace_exception", "length", "dtype", "nan"])
def test_model_failure_and_invalid_output_fall_back_to_intact_pcm(tmp_path, bad):
    job = audio_job()
    pipeline, run, quality, events = worker(tmp_path, [job])
    run.audio_processor = Mock()
    def failing(samples):
        if bad == "inplace_exception":
            samples[:] = 0
            raise RuntimeError("native failure")
        if bad == "exception":
            raise RuntimeError("native failure")
        if bad == "length":
            return process_result(samples[:-1])
        if bad == "dtype":
            return process_result(samples.astype(np.float64))
        return process_result(np.full_like(samples, np.nan))
    run.audio_processor.process.side_effect = failing
    recognize(pipeline, run, quality)
    np.testing.assert_array_equal(quality.recognize.call_args.args[0], job.samples)
    assert run.mt_queue.get().text == "今日は晴れです。"
    assert any(event.get("action") == "bypass" for event in events)


@pytest.mark.parametrize("stage", ["queued", "filter", "processing", "asr"])
def test_navigation_invalidates_audio_before_original_or_translation(tmp_path, stage):
    state = {"generation": 4}
    def current(job):
        return getattr(job, "source_generation", getattr(job, "generation", 0)) == state["generation"]
    pipeline, run, quality, events = worker(tmp_path, speech_only=stage == "filter", validator=current)
    if stage == "queued":
        state["generation"] = 5
    elif stage == "filter":
        pipeline._speech_filter = Mock()
        def filtering(samples):
            state["generation"] = 5
            return FilterDecision(samples, {"action": "keep"})
        pipeline._speech_filter.process.side_effect = filtering
    elif stage == "processing":
        run.audio_processor = Mock()
        def processing(samples):
            state["generation"] = 5
            return process_result(samples)
        run.audio_processor.process.side_effect = processing
    else:
        def asr(samples, language):
            state["generation"] = 5
            return QualityRecognition("古い音声です。", {})
        quality.recognize.side_effect = asr
    recognize(pipeline, run, quality)
    assert run.mt_queue.get() is None
    assert not any(event["kind"] in ("original", "recognition", "translation") for event in events)
    if stage != "asr":
        quality.recognize.assert_not_called()


@pytest.mark.parametrize("stage", ["queued", "request", "publish"])
def test_navigation_fences_translation_request_and_late_results(tmp_path, stage):
    state = {"generation": 4}
    pipeline, run, _, events = worker(tmp_path, validator=lambda item: item.source_generation == state["generation"])
    translated = TranslationJob("system", "one", "今日は晴れです。", time.monotonic(), ("ja", "zh"), 1,
        source_room="21399689", source_generation=4)
    translator = SimpleNamespace(backend_info={}, translate=Mock(return_value="今天天气晴朗"))
    if stage == "queued":
        state["generation"] = 5
    elif stage == "request":
        def translate(*args, **kwargs):
            state["generation"] = 5
            return "今天天气晴朗"
        translator.translate.side_effect = translate
    else:
        state["generation"] = 5
        pipeline._publish_translation(run, translated, "今天天气晴朗")
        assert not events
        return
    pipeline._translation_request(run, translator, [translated])
    assert not any(event["kind"] == "translation" for event in events)
    if stage == "queued":
        translator.translate.assert_not_called()


def test_room_changes_and_audio_generations_do_not_share_translation_context(tmp_path):
    pipeline, run, quality, _ = worker(tmp_path, [
        audio_job(identifier="old"), audio_job(identifier="new", start=16000, room="1698225", generation=5)])
    recognize(pipeline, run, quality)
    first, second = run.mt_queue.get(), run.mt_queue.get()
    assert first.source_room == "21399689" and second.source_room == "1698225"
    assert second.context == "" and second.context_generation > first.context_generation


def test_video_source_with_empty_room_still_translates(tmp_path):
    pipeline, run, quality, _ = worker(tmp_path, [audio_job(room="")])
    recognize(pipeline, run, quality)
    translated = run.mt_queue.get()
    assert translated.source_room == "" and translated.text == "今日は晴れです。"


def test_joint_revision_reuses_processed_audio_without_second_processing_pass(tmp_path):
    first = replace(audio_job(), ended_by="max_duration")
    second = replace(audio_job(identifier="two", start=16000), ended_by="max_duration")
    pipeline, run, quality, _ = worker(tmp_path, [first, second])
    run.audio_processor = Mock()
    run.audio_processor.process.side_effect = lambda samples: process_result(samples * .5)
    quality.recognize.side_effect = [QualityRecognition("森の。", {}),
        QualityRecognition("妖精です。", {}), QualityRecognition("森の妖精です。", {})]
    recognize(pipeline, run, quality)
    assert run.audio_processor.process.call_count == 2
    assert quality.recognize.call_count == 3
    np.testing.assert_array_equal(quality.recognize.call_args_list[-1].args[0], np.full(32000, .05, np.float32))
    translated = run.mt_queue.get()
    assert translated.revision == 2 and translated.source_segment_ids == ("one", "two")
    assert translated.timing["source_segment_count"] == 2


def test_processing_does_not_hold_capture_callback(tmp_path):
    pipeline, run, quality, _ = worker(tmp_path)
    entered, released = threading.Event(), threading.Event()
    run.audio_processor = Mock()
    def blocked(samples):
        entered.set()
        assert released.wait(3)
        return process_result(samples)
    run.audio_processor.process.side_effect = blocked
    run.segmenters["system"] = SimpleNamespace(feed=lambda chunk: [])
    with patch("fanyiji.recognition_quality.QualityRecognizer", return_value=quality):
        thread = threading.Thread(target=pipeline._recognize, args=(run,))
        thread.start()
        try:
            assert entered.wait(2)
            started = time.monotonic()
            pipeline._on_chunk(run, AudioChunk("system", np.full(512, .1, np.float32), 0, 512, time.monotonic(), 4))
            assert time.monotonic() - started < .25
        finally:
            released.set()
            thread.join(3)
        assert not thread.is_alive()


@pytest.mark.parametrize("changed", [dict(source_room="1698225"), dict(source_generation=5)])
def test_translation_batch_cannot_cross_source_even_if_context_generation_matches(changed):
    from fanyiji.translation_queue import TranslationQueue
    queue = TranslationQueue(job_factory=TranslationJob)
    first = TranslationJob("system", "first", "一句话", 100, ("ja", "zh"), 1,
        source_room="21399689", source_generation=4)
    second = replace(first, segment_id="second", **changed)
    queue.put("system", first)
    queue.put("system", second)
    queue.close()
    assert [job.segment_id for job in queue.get_batch()] == ["first"]
    assert [job.segment_id for job in queue.get_batch()] == ["second"]


def test_processor_mode_switch_preserves_inflight_engine_and_shutdown_closes_all(tmp_path):
    pipeline, run, _, _ = worker(tmp_path)
    first = SimpleNamespace(backend_info={"ready": True, "mode": "gtcrn"}, preload=Mock(), close=Mock())
    second = SimpleNamespace(backend_info={"ready": True, "mode": "uvr"}, preload=Mock(), close=Mock())
    with patch("fanyiji.audio_processing.AudioProcessor", side_effect=[first, second]) as factory:
        pipeline._prepare_audio_processor(run)
        next_run = _Run(2, replace(run.config, audio_processing="uvr"))
        pipeline._prepare_audio_processor(next_run)
        assert run.audio_processor is first and next_run.audio_processor is second
        first.close.assert_not_called()
        pipeline._prepare_audio_processor(_Run(3, run.config))
        assert factory.call_count == 2
    pipeline.shutdown()
    first.close.assert_called_once()
    second.close.assert_called_once()


def test_shutdown_during_processor_construction_closes_unpublished_backend(tmp_path):
    pipeline, run, _, _ = worker(tmp_path)
    constructing, released = threading.Event(), threading.Event()
    processor = SimpleNamespace(backend_info={"ready": True}, preload=Mock(), close=Mock())
    def create(*args, **kwargs):
        constructing.set()
        assert released.wait(3)
        return processor
    with patch("fanyiji.audio_processing.AudioProcessor", side_effect=create):
        thread = threading.Thread(target=pipeline._prepare_audio_processor, args=(run,))
        thread.start()
        try:
            assert constructing.wait(2)
            pipeline.shutdown()
        finally:
            released.set()
            thread.join(3)
    processor.preload.assert_not_called()
    processor.close.assert_called_once()
    assert not pipeline._audio_processors


def test_preload_failure_keeps_models_available_for_original_audio(tmp_path):
    pipeline, run, quality, events = worker(tmp_path)
    with patch("fanyiji.audio_processing.AudioProcessor", side_effect=RuntimeError("native construction")):
        info = pipeline._prepare_audio_processor(run)
    assert not info["available"] and not info["model_loaded"]
    recognize(pipeline, run, quality)
    assert run.mt_queue.get().text == "今日は晴れです。"
    assert any(event["kind"] == "notice" for event in events)
