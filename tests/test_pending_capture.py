"""Confirm a half-sentence only when capture proves no continuation can join."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fanyiji.audio import AudioChunk
from fanyiji.config import AppConfig
from fanyiji.pipeline import RecognitionJob, SubtitlePipeline, _Run
from fanyiji.segmentation import SegmentationConfig, SegmentationProgress, SegmentEvent, SpeechSegmenter


class QuietVad:
    def accept_waveform(self, samples):
        pass

    def is_speech_detected(self):
        return False

    def empty(self):
        return True

    def reset(self):
        pass

    def flush(self):
        pass


def setup(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig(speech_only=False))
    job = RecognitionJob("system", "a", np.zeros(16000, np.float32), 100,
        ("ja", "zh"), 0, 16000, 1, "silence")
    with patch("fanyiji.pipeline.time.monotonic", return_value=101):
        pipeline._accept_text(run, job, "森の。", job.samples, Mock())
    return pipeline, run, job, events


def progress(*, floor=64000, generation=1, active=False, held=None):
    return SegmentationProgress(generation, floor + 10240, floor, active, held)


def test_snapshot_is_immutable_and_excludes_unprocessed_tail_and_onset_padding():
    segmenter = SpeechSegmenter("system", vad=QuietVad())
    chunk = AudioChunk("system", np.zeros(40001, np.float32), 0, 40001, 103, 1)
    segmenter.feed(chunk)
    snapshot = segmenter.continuation_snapshot()
    assert snapshot.processed_sample == 39936  # Only complete 512-sample VAD windows.
    assert snapshot.earliest_future_start_sample == 29184  # 0.672s conservative onset/prefix margin.
    assert not snapshot.speech_active and snapshot.held_start_sample is None
    with pytest.raises(FrozenInstanceError):
        snapshot.processed_sample = 0
    segmenter.feed(AudioChunk("system", np.zeros(512, np.float32), 40001, 40513, 103.1, 1))
    assert snapshot.processed_sample == 39936
    assert segmenter.continuation_snapshot().processed_sample > snapshot.processed_sample


def test_v5_native_window_overlap_is_protected_at_integer_padding_boundary(tmp_path):
    config = SegmentationConfig(min_speech_seconds=.32, pre_speech_seconds=.32)
    segmenter = SpeechSegmenter("system", config=config, vad=QuietVad())
    segmenter.feed(AudioChunk("system", np.zeros(51200, np.float32), 0, 51200, 103, 1))
    snapshot = segmenter.continuation_snapshot()
    # Native v5's next detected onset may start at 39808, still within the
    # inclusive 1.8s join window. The old two-window margin gave 39936.
    assert snapshot.earliest_future_start_sample == 39424
    pipeline, _, job, _ = setup(tmp_path)
    job.end_sample = 11008
    assert not pipeline._continuation_window_closed(job, snapshot)


def test_static_guard_keeps_inclusive_join_boundary_then_allows_one_sample_beyond(tmp_path):
    pipeline, _, job, _ = setup(tmp_path)
    assert not pipeline._continuation_window_closed(job, progress(floor=44800))
    assert pipeline._continuation_window_closed(job, progress(floor=44801))


def test_confirmed_capture_silence_translates_once_without_waiting_fifteen_seconds(tmp_path):
    pipeline, run, _, events = setup(tmp_path)
    run.capture_progress["system"] = progress()
    with patch("fanyiji.pipeline.time.monotonic", return_value=101.1):
        pipeline._flush_pending(run)
    assert not run.pending_text
    translated = run.mt_queue.get(timeout=0)
    assert translated.text == "森の。" and not translated.provisional
    assert [event["provisional"] for event in events if event["kind"] == "original"] == [True, False]


def test_already_translated_fragment_is_confirmed_without_another_model_request(tmp_path):
    pipeline, run, _, events = setup(tmp_path)
    with patch("fanyiji.pipeline.time.monotonic", return_value=102.3):
        pipeline._flush_pending(run)
    assert run.pending_text["system"].enqueued
    run.capture_progress["system"] = progress()
    with patch("fanyiji.pipeline.time.monotonic", return_value=103):
        pipeline._flush_pending(run)
        pipeline._flush_pending(run)
    run.mt_queue.close()
    assert run.mt_queue.get().provisional
    assert run.mt_queue.get() is None
    assert len([event for event in events if event["kind"] == "subtitle_final"]) == 1


@pytest.mark.parametrize("value", [None, progress(active=True), progress(held=16000),
    progress(generation=2), progress(generation=None), progress(floor=44800)])
def test_active_held_unknown_generation_or_open_window_preserves_pending(tmp_path, value):
    pipeline, run, _, events = setup(tmp_path)
    if value is not None:
        run.capture_progress["system"] = value
    with patch("fanyiji.pipeline.time.monotonic", return_value=104):
        pipeline._flush_pending(run)
    assert "system" in run.pending_text
    assert not any(event["kind"] == "subtitle_final" for event in events)
    translated = run.mt_queue.get(timeout=0)
    assert translated.provisional


def test_queued_continuation_remains_protected_then_joins_normally(tmp_path):
    pipeline, run, first, events = setup(tmp_path)
    following = RecognitionJob("system", "b", np.zeros(16000, np.float32), 102,
        ("ja", "zh"), 20000, 36000, 1, "silence")
    run.asr_queue.put("system", following, 1)
    run.capture_progress["system"] = progress()
    with patch("fanyiji.pipeline.time.monotonic", return_value=104):
        pipeline._flush_pending(run)
        assert "system" in run.pending_text and not run.pending_text["system"].enqueued
        assert not run.mt_queue._queues["system"]
        assert run.asr_queue.get() is following
        pipeline._accept_text(run, following, "妖精です。", following.samples, Mock())
    translated = run.mt_queue.get(timeout=0)
    assert translated.segment_id == first.segment_id and translated.revision == 2
    assert translated.text == "森の妖精です。"
    assert len([event for event in events if event["kind"] == "sentence_join"]) == 1


def test_capture_publishes_progress_only_after_final_is_available_to_asr(tmp_path):
    pipeline, run, _, _ = setup(tmp_path)
    old_progress, new_progress = progress(floor=30000), progress()
    run.capture_progress["system"] = old_progress
    segment = SegmentEvent("system", "final", "b", 1, np.zeros(16000, np.float32),
        20000, 36000, "silence", 1, 104)
    run.segmenters["system"] = SimpleNamespace(feed=lambda chunk: [segment],
        continuation_snapshot=lambda: new_progress)
    observations = []
    def on_segment(job, accepted):
        observations.append((run.capture_progress["system"], run.asr_queue.peek("system"), accepted))
        pipeline._flush_pending(run)
    pipeline.on_segment = on_segment
    with patch("fanyiji.pipeline.time.monotonic", return_value=104):
        pipeline._on_chunk(run, AudioChunk("system", np.zeros(16000, np.float32), 64000, 80000, 104, 1))
        pipeline._flush_pending(run)
    assert observations[0][0] is old_progress
    assert observations[0][1].start_sample == 20000 and observations[0][2]
    assert run.capture_progress["system"] is new_progress
    assert "system" in run.pending_text
    assert not run.mt_queue._queues["system"]
