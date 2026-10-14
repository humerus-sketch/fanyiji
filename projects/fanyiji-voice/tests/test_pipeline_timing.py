"""Capture clocks and independent translation timing, without audio/model access."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fanyiji.audio import AudioChunk
from fanyiji.config import AppConfig
from fanyiji.pipeline import RecognitionJob, SubtitlePipeline, TranslationJob, _Run
from fanyiji.segmentation import SegmentEvent


class Clock:
    def __init__(self, now=110.0):
        self.now = now

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def setup(tmp_path, *, hybrid=False, speech_only=False):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(3, AppConfig(speech_only=speech_only,
        translation_provider="hybrid" if hybrid else "local"))
    return pipeline, run, events


def source(identifier="a", *, start=0, end=16000, timing=None):
    return RecognitionJob("system", identifier, np.zeros(end - start, np.float32),
        101.0 if timing is None else timing["audio_ended_at"], ("ja", "zh"),
        start, end, 1, timing={} if timing is None else timing)


def translator(clock, seconds=1.0, *, codex=False, fail=False):
    def translate(*args, **kwargs):
        clock.advance(seconds)
        if fail:
            raise RuntimeError("fixture failure")
        return "有效译文"
    return SimpleNamespace(backend_info={"model_id": "fixture", "supports_source_context": True},
        translate=translate, reset_conversation=Mock() if codex else None)


def test_capture_to_recognizer_and_translation_preserves_hard_hold_and_real_work(tmp_path):
    pipeline, run, events = setup(tmp_path, speech_only=True)
    clock = Clock(108)
    segment = SegmentEvent("system", "final", "a", 1, np.zeros(32000, np.float32),
        0, 32000, "max_duration", 1, 105.25)
    run.segmenters["system"] = SimpleNamespace(feed=lambda chunk: [segment])
    chunk = AudioChunk("system", np.zeros(16000, np.float32), 48000, 64000, 104, 1)
    captured_jobs = []
    pipeline.on_segment = lambda job, accepted: captured_jobs.append(job)
    def filter_audio(samples):
        clock.advance(.1)
        return SimpleNamespace(samples=samples, skip=False, metadata={})
    def recognize(samples, language):
        clock.advance(.6)
        return SimpleNamespace(text="こんにちは。", metadata={})
    pipeline._speech_filter = SimpleNamespace(process=filter_audio)
    quality = SimpleNamespace(recognize=recognize)
    with patch("fanyiji.pipeline.time.monotonic", side_effect=clock.monotonic), patch(
        "fanyiji.recognition_quality.QualityRecognizer", return_value=quality):
        pipeline._on_chunk(run, chunk)
        run.asr_queue.close()
        clock.now = 110
        pipeline._recognize(run)
        translated = run.mt_queue.get()
        pipeline._translation_request(run, translator(clock, 1.3), [translated])
    job = captured_jobs[0]
    assert job.timing["audio_started_at"] == 100
    assert job.timing["audio_ended_at"] == 102
    assert job.timing["segment_ready_at"] == 105.25
    assert job.timing["segment_ready_at"] != job.queued_at == 108
    assert job.timing["segmentation_seconds"] == 3.25
    assert job.timing["asr_queue_seconds"] == 4.75
    assert job.timing["recognition_dequeued_at"] == 110
    assert job.timing["speech_filter_seconds"] == pytest.approx(.1)
    assert job.timing["asr_started_at"] == pytest.approx(110.1)
    assert job.timing["asr_finished_at"] == pytest.approx(110.7)
    assert job.timing["asr_seconds"] == pytest.approx(.6)
    original = next(event for event in events if event["kind"] == "original")
    recognition = next(event for event in events if event["kind"] == "recognition")
    final = next(event for event in events if event["kind"] == "translation")
    assert final["timing"]["translation_queued_at"] == pytest.approx(110.7)
    assert final["timing"]["translation_started_at"] == pytest.approx(110.7)
    assert final["timing"]["translation_finished_at"] == pytest.approx(112)
    assert final["timing"]["translation_seconds"] == pytest.approx(1.3)
    assert final["timing"]["event_emitted_at"] - final["timing"]["audio_started_at"] == pytest.approx(12)
    assert final["latency"] == pytest.approx(10)  # Legacy audio-tail latency stays intact.
    assert "translation_started_at" not in original["timing"]
    assert "translation_queued_at" not in recognition["timing"]
    assert original["timing"] is not translated.timing
    from fanyiji.latency import LatencyTracker
    tracker = LatencyTracker()
    tracker.begin_session(run.session)
    presented = {**final, "timing": {**final["timing"],
        "ui_received_at": 112.1, "ui_updated_at": 112.2, "ui_painted_at": 112.3}}
    metrics = tracker.on_presented(presented)
    assert metrics["valid"] and metrics["end_to_end_seconds"] == pytest.approx(12.3)


def test_held_final_released_by_reset_uses_its_own_generation_capture_clock(tmp_path):
    pipeline, run, _ = setup(tmp_path)
    pending = []
    run.segmenters["system"] = SimpleNamespace(feed=lambda chunk: pending)
    with patch("fanyiji.pipeline.time.monotonic", return_value=111):
        pipeline._on_chunk(run, AudioChunk("system", np.zeros(16000, np.float32),
            48000, 64000, 104, 1))
        pending.append(SegmentEvent("system", "final", "a", 1, np.zeros(16000, np.float32),
            16000, 32000, "max_duration", 1, 110))
        pipeline._on_chunk(run, AudioChunk("system", np.zeros(16000, np.float32),
            0, 16000, 110, 2))
    job = run.asr_queue.get(timeout=0)
    assert job.generation == 1
    assert job.ended_at == job.timing["audio_ended_at"] == 102
    assert job.timing["audio_started_at"] == 101
    assert job.timing["segmentation_seconds"] == 8


def test_missing_generation_clock_does_not_invent_audio_timestamps(tmp_path):
    pipeline, run, _ = setup(tmp_path)
    held = SegmentEvent("system", "final", "a", 1, np.zeros(16000, np.float32),
        16000, 32000, "max_duration", 1, 110)
    run.segmenters["system"] = SimpleNamespace(feed=lambda chunk: [held])
    pipeline._on_chunk(run, AudioChunk("system", np.zeros(16000, np.float32), 0, 16000, 110, 2))
    timing = run.asr_queue.get(timeout=0).timing
    assert "audio_started_at" not in timing and "audio_ended_at" not in timing


def test_hybrid_branches_have_independent_jobs_timestamps_and_event_snapshots(tmp_path):
    pipeline, run, events = setup(tmp_path, hybrid=True)
    clock = Clock()
    run.translator = translator(clock, 2, codex=True)
    run.draft_translator = translator(clock, .5)
    origin = source(timing={"audio_started_at": 100, "audio_ended_at": 101,
        "segment_ready_at": 102, "source_segment_count": 1})
    with patch("fanyiji.pipeline.time.monotonic", side_effect=clock.monotonic):
        pipeline._enqueue_translation(run, origin, "こんにちは。")
        final, draft = run.mt_queue.get(), run.draft_queue.get()
        pipeline._translation_request(run, run.draft_translator, [draft], stage="draft")
        draft_event = next(event for event in events if event["kind"] == "translation")
        draft_snapshot = dict(draft_event["timing"])
        clock.advance(1)
        pipeline._translation_request(run, run.translator, [final])
    assert final is not draft and final.timing is not draft.timing
    assert draft.timing["translation_started_at"] == 110
    assert draft.timing["translation_finished_at"] == 110.5
    assert final.timing["translation_started_at"] == 111.5
    assert final.timing["translation_finished_at"] == 113.5
    assert draft_event["timing"] == draft_snapshot
    assert [event["translation_stage"] for event in events if event["kind"] == "translation"] == ["draft", "final"]
    assert "translation_started_at" not in origin.timing


def test_joined_revision_uses_earliest_source_and_summed_actual_asr_work(tmp_path):
    pipeline, run, events = setup(tmp_path)
    clock = Clock(104)
    first = source(timing={"audio_started_at": 100, "audio_ended_at": 101,
        "segment_ready_at": 102, "asr_started_at": 103, "asr_finished_at": 103.5,
        "asr_seconds": .5, "asr_queue_seconds": 1, "segmentation_seconds": 1,
        "source_segment_count": 1})
    second = source("b", start=16000, end=32000, timing={"audio_started_at": 101,
        "audio_ended_at": 102, "segment_ready_at": 103, "asr_started_at": 104,
        "asr_finished_at": 104.7, "asr_seconds": .7, "asr_queue_seconds": 1,
        "segmentation_seconds": 1, "source_segment_count": 1})
    def recognize(*args):
        clock.advance(.4)
        return SimpleNamespace(text="森の妖精です。", metadata={})
    quality = SimpleNamespace(recognize=recognize)
    with patch("fanyiji.pipeline.time.monotonic", side_effect=clock.monotonic):
        pipeline._accept_text(run, first, "森の。", first.samples, quality)
        provisional_snapshot = dict(events[0]["timing"])
        clock.now = 105
        pipeline._accept_text(run, second, "妖精です。", second.samples, quality)
    translated = run.mt_queue.get(timeout=0)
    assert translated.revision == 2
    assert translated.ended_at == 102
    assert translated.timing["audio_started_at"] == 100
    assert translated.timing["audio_ended_at"] == 102
    assert translated.timing["asr_started_at"] == 103
    assert translated.timing["asr_finished_at"] == 104.7
    assert translated.timing["asr_seconds"] == pytest.approx(1.2)
    assert translated.timing["joint_asr_seconds"] == pytest.approx(.4)
    assert translated.timing["assembly_processing_seconds"] == pytest.approx(0)
    assert translated.timing["segmentation_seconds"] == 2
    assert translated.timing["source_segment_count"] == 2
    assert events[0]["timing"] == provisional_snapshot


def test_late_draft_is_candidate_and_failures_have_timing_without_success(tmp_path):
    pipeline, run, events = setup(tmp_path, hybrid=True)
    clock = Clock()
    run.translator = translator(clock, 1, codex=True)
    run.draft_translator = translator(clock, .5)
    with patch("fanyiji.pipeline.time.monotonic", side_effect=clock.monotonic):
        pipeline._enqueue_translation(run, source(), "こんにちは。")
        final, draft = run.mt_queue.get(), run.draft_queue.get()
        pipeline._translation_request(run, run.translator, [final])
        pipeline._translation_request(run, run.draft_translator, [draft], stage="draft")
        pipeline._enqueue_translation(run, source("b"), "ありがとう。")
        failed = run.mt_queue.get()
        pipeline._translation_request(run, translator(clock, fail=True, codex=True), [failed])
    candidate = next(event for event in events if event["kind"] == "translation_candidate")
    assert candidate["timing"]["translation_finished_at"] == 111.5
    status = next(event for event in events if event["kind"] == "translation_status")
    assert status["failed"] and status["timing"]["translation_finished_at"] == 112.5
    assert not any(event["kind"] == "translation" and event["segment_id"] == "b" for event in events)


def test_canceled_and_superseded_results_cannot_become_visible_finals(tmp_path):
    pipeline, run, events = setup(tmp_path)
    job = TranslationJob("system", "a", "こんにちは。", 101, ("ja", "zh"), 1,
        timing={"translation_started_at": 110, "translation_finished_at": 111})
    run.translation_versions[("system", "a")] = 2
    pipeline._publish_translation(run, job, "旧译文")
    assert events[0]["kind"] == "translation_superseded"
    assert events[0]["timing"] is not job.timing
    run.cancel.set()
    pipeline._publish_translation(run, job, "取消后的译文")
    assert len(events) == 1


def test_translation_timing_survives_disk_spool(tmp_path):
    from fanyiji.translation_queue import TranslationQueue
    queue = TranslationQueue(memory_limit=1, job_factory=TranslationJob)
    timing = {"audio_started_at": 100, "audio_ended_at": 101, "segment_ready_at": 102,
        "translation_queued_at": 103}
    for identifier in ("a", "b"):
        queue.put("system", TranslationJob("system", identifier, "こんにちは。", 101,
            ("ja", "zh"), 1, timing=dict(timing)))
    queue.close()
    assert queue.get().timing == timing
    assert queue.get().timing == timing
    assert queue.get() is None


def test_normal_three_second_pause_preserves_context_but_does_not_join_half_sentence(tmp_path):
    pipeline, run, events = setup(tmp_path)
    first, second = source(), source("b", start=64000, end=80000)
    second.ended_at = 105
    quality = Mock()
    with patch("fanyiji.pipeline.time.monotonic", return_value=110):
        pipeline._accept_text(run, first, "森の。", first.samples, quality)
        pipeline._accept_text(run, second, "妖精です。", second.samples, quality)
    run.mt_queue.close()
    translations = [run.mt_queue.get(), run.mt_queue.get()]
    assert [job.segment_id for job in translations] == ["a", "b"]
    assert all(job.revision == 1 and job.context_generation == 0 for job in translations)
    assert translations[1].context == "森の。"
    assert run.context_generations["system"] == 0
    quality.recognize.assert_not_called()
    assert not any(event["kind"] == "sentence_join" for event in events)


@pytest.mark.parametrize("elapsed,expected_generation", [(19.999, 0), (20, 1), (20.001, 1)])
def test_context_continuity_expires_at_same_twenty_second_boundary_as_history(tmp_path, elapsed, expected_generation):
    pipeline, run, _ = setup(tmp_path)
    first = source()
    end = 16000 + round(elapsed * 16000)
    second = source("b", start=end - 16000, end=end)
    second.ended_at = 101 + elapsed
    quality = Mock()
    pipeline._accept_text(run, first, "昨日は雨でした。", first.samples, quality)
    pipeline._accept_text(run, second, "今日は晴れです。", second.samples, quality)
    first_translation, second_translation = run.mt_queue.get(), run.mt_queue.get()
    assert first_translation.context_generation == 0
    assert second_translation.context_generation == expected_generation
    assert bool(second_translation.context) is (expected_generation == 0)


def test_source_generation_change_clears_context_even_after_short_pause(tmp_path):
    pipeline, run, _ = setup(tmp_path)
    first, second = source(), source("b", start=16000, end=32000)
    second.generation = 2
    second.ended_at = 102
    quality = Mock()
    pipeline._accept_text(run, first, "昨日は雨でした。", first.samples, quality)
    pipeline._accept_text(run, second, "今日は晴れです。", second.samples, quality)
    run.mt_queue.get()
    translation = run.mt_queue.get()
    assert translation.context_generation == 1 and not translation.context


def test_batch_fallback_keeps_first_start_and_totals_only_actual_request_work(tmp_path):
    from fanyiji.codex_engine import CodexBatchResponseError
    pipeline, run, events = setup(tmp_path)
    run.config = AppConfig(speech_only=False, translation_provider="codex")
    clock = Clock()
    engine = translator(clock, 1, codex=True)
    def translate_batch(*args, **kwargs):
        clock.advance(2)
        raise CodexBatchResponseError("invalid batch")
    engine.translate_batch = translate_batch
    jobs = [TranslationJob("system", identifier, "こんにちは。", 101, ("ja", "zh"), 1,
        queued_at=109, timing={"translation_queued_at": 109}) for identifier in ("a", "b")]
    with patch("fanyiji.pipeline.time.monotonic", side_effect=clock.monotonic):
        pipeline._translation_request(run, engine, jobs)
    finals = [event for event in events if event["kind"] == "translation"]
    assert len(finals) == 2 and all(not event["failed"] for event in finals)
    assert [event["timing"]["translation_started_at"] for event in finals] == [110, 110]
    assert [event["timing"]["translation_finished_at"] for event in finals] == [113, 114]
    assert [event["timing"]["translation_seconds"] for event in finals] == [3, 3]


def test_real_joined_pipeline_event_is_valid_for_latency_tracker_with_overlapping_audio(tmp_path):
    from fanyiji.latency import LatencyTracker
    pipeline, run, events = setup(tmp_path)
    clock = Clock(102.6)
    jobs = []
    for identifier, audio_start, audio_end in (("a", 100, 101), ("b", 101, 102)):
        jobs.append(source(identifier, start=round((audio_start - 100) * 16000),
            end=round((audio_end - 100) * 16000), timing={
                "audio_started_at": audio_start, "audio_ended_at": audio_end,
                "segment_ready_at": audio_end + .1, "segmentation_seconds": .1,
                "recognition_dequeued_at": audio_end + .2, "asr_queue_seconds": .1,
                "filter_started_at": audio_end + .2, "filter_finished_at": audio_end + .3,
                "speech_filter_seconds": .1, "asr_started_at": audio_end + .3,
                "asr_finished_at": audio_end + .5, "asr_seconds": .2,
                "source_segment_count": 1}))
    quality = SimpleNamespace(recognize=lambda *args: SimpleNamespace(text="森の妖精です。", metadata={}))
    with patch("fanyiji.pipeline.time.monotonic", side_effect=clock.monotonic):
        pipeline._accept_text(run, jobs[0], "森の。", jobs[0].samples, quality)
        pipeline._accept_text(run, jobs[1], "妖精です。", jobs[1].samples, quality)
        job = run.mt_queue.get()
        pipeline._translation_request(run, translator(clock, 1), [job])
    event = next(event for event in events if event["kind"] == "translation")
    timing = {**event["timing"], "ui_received_at": 103.7,
        "ui_updated_at": 103.8, "ui_painted_at": 103.9}
    tracker = LatencyTracker()
    tracker.begin_session(run.session)
    metrics = tracker.on_presented({**event, "timing": timing})
    assert timing["asr_started_at"] < timing["audio_ended_at"]
    assert metrics["valid"] and metrics["breakdown_mode"] == "overlapping_joined_audio"
    assert metrics["breakdown"]["asr_seconds"] == pytest.approx(.4)
    assert metrics["breakdown"]["speech_filter_seconds"] == pytest.approx(.2)
