"""Synthetic boundary and device-lifecycle checks; never open real microphones."""
from __future__ import annotations

import ctypes
import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from fanyiji.audio import AudioCaptureManager, AudioCaptureSettings, AudioChunk, pcm_to_mono
from fanyiji.segmentation import SegmentEvent, SegmentationConfig, SpeechSegmenter, create_silero_vad


class ScriptedVad:
    """Expose already-known detector boundaries, independent of the segmenter."""

    def __init__(self, completions=()):
        self.completions = deque(completions)
        self.completed = deque()
        self.received = []
        self.position = 0
        self.reset_count = 0
        self.speech = False
        self.current_segment = SimpleNamespace(start=0, samples=[])

    def accept_waveform(self, samples):
        self.received.append(samples.copy())
        self.position += samples.size
        while self.completions and self.completions[0][0] <= self.position:
            _, start, length = self.completions.popleft()
            audio = np.concatenate(self.received)
            self.completed.append(SimpleNamespace(start=start, samples=audio[start:start + length].copy()))

    def empty(self):
        return not self.completed

    @property
    def front(self):
        return self.completed[0]

    def pop(self):
        self.completed.popleft()

    def reset(self):
        self.reset_count += 1
        self.received.clear()
        self.completed.clear()
        self.position = 0

    def flush(self):
        pass

    def is_speech_detected(self):
        return self.speech


def chunk(samples, start=0, generation=1, channel="mic"):
    samples = np.asarray(samples, dtype=np.float32)
    return AudioChunk(channel, samples, start, start + samples.size, 123.0, generation)


def test_final_uses_source_samples_and_absolute_clock_with_pre_padding():
    vad = ScriptedVad([(1536, 512, 768)])
    segmenter = SpeechSegmenter("mic", vad=vad)
    samples = np.linspace(-0.8, 0.8, 1536, dtype=np.float32)
    events = segmenter.feed(chunk(samples, start=10000))
    assert len(events) == 1
    final = events[0]
    assert final.kind == "final" and final.reason == "silence"
    assert (final.start_sample, final.end_sample) == (10000, 11280)
    np.testing.assert_array_equal(final.samples, samples[:1280])
    assert final.generation == 1 and final.utterance_id.startswith("g1-")


def test_non_window_sized_chunks_do_not_lose_samples():
    vad = ScriptedVad([(1024, 0, 1024)])
    segmenter = SpeechSegmenter("mic", vad=vad)
    audio = np.arange(1200, dtype=np.float32) / 1200
    assert segmenter.feed(chunk(audio[:321])) == []
    finals = segmenter.feed(chunk(audio[321:], start=321))
    np.testing.assert_array_equal(np.concatenate(vad.received), audio[:1024])
    np.testing.assert_array_equal(finals[0].samples, audio[:1024])


def test_short_native_answer_keeps_its_natural_boundary():
    vad = ScriptedVad([(8192, 0, 4096)])
    segmenter = SpeechSegmenter("mic", vad=vad)
    samples = np.concatenate((np.full(4096, 0.1, dtype=np.float32),
                              np.zeros(4096, dtype=np.float32)))
    events = segmenter.feed(chunk(samples))
    assert len(events) == 1 and events[0].reason == "silence"
    assert (events[0].start_sample, events[0].end_sample) == (0, 4096)
    np.testing.assert_array_equal(events[0].samples, samples[:4096])


def test_generation_and_gap_discard_unfinished_audio_without_reusing_ids():
    vad = ScriptedVad()
    segmenter = SpeechSegmenter("mic", vad=vad)
    segmenter.feed(chunk(np.ones(400)))
    reset = segmenter.feed(chunk(np.zeros(512), generation=2))
    assert [e.reason for e in reset] == ["capture_restarted"]
    assert vad.reset_count == 1
    np.testing.assert_array_equal(vad.received[0], np.zeros(512))
    reset = segmenter.feed(chunk(np.zeros(512), start=1024, generation=2))
    assert [e.reason for e in reset] == ["audio_gap"]
    assert vad.reset_count == 2


def test_reconfigure_discards_pending_and_held_audio_but_preserves_utterance_sequence(monkeypatch):
    old = ScriptedVad([(512, 0, 512)])
    segmenter = SpeechSegmenter("mic", vad=old)
    first = segmenter.feed(chunk(np.full(512, .1)))[0]
    segmenter.feed(chunk(np.full(400, .3), start=512))
    segmenter._held_hard_final = SegmentEvent("mic", "final", first.utterance_id, 2,
        np.full(512, .4, np.float32), 512, 1024, "max_duration", 1, 123)
    new = ScriptedVad([(512, 0, 512)])
    created = []
    def create(path, config):
        created.append(config)
        return new
    monkeypatch.setattr("fanyiji.segmentation.create_silero_vad", create)
    config = SegmentationConfig(max_speech_seconds=4.5, max_cut_hold_seconds=0,
                                preferred_speech_seconds=0, min_silence_seconds=.6)
    assert segmenter.reconfigure(config) is None
    assert created == [config] and segmenter._pause_vad is None
    assert segmenter.continuation_snapshot().held_start_sample is None
    second = segmenter.feed(chunk(np.full(512, .2), start=912))[0]
    assert second.utterance_id != first.utterance_id
    assert second.utterance_id == "g1-u2"
    assert second.start_sample == 912
    np.testing.assert_array_equal(second.samples, np.full(512, .2, np.float32))
    segmenter.reconfigure(config)
    assert len(created) == 1  # Repeated frames do not reload the native model.


def test_reconfigure_build_failure_preserves_existing_detector_and_pending_pcm(monkeypatch):
    old = ScriptedVad()
    segmenter = SpeechSegmenter("mic", vad=old)
    segmenter.feed(chunk(np.full(400, .3)))
    def fail(*args):
        raise RuntimeError("model load failed")
    monkeypatch.setattr("fanyiji.segmentation.create_silero_vad", fail)
    with pytest.raises(RuntimeError, match="model load failed"):
        segmenter.reconfigure(SegmentationConfig(max_speech_seconds=4.5))
    assert segmenter.config == SegmentationConfig()
    assert old.reset_count == 0 and segmenter._expected == 400
    np.testing.assert_array_equal(segmenter._pending, np.full(400, .3, np.float32))


def test_native_long_final_keeps_natural_reason_and_does_not_duplicate_padding():
    config = SegmentationConfig(max_speech_seconds=0.064, min_speech_seconds=0.01,
                                pre_speech_seconds=0.03)
    vad = ScriptedVad([(1024, 0, 1024), (2048, 1024, 1024)])
    segmenter = SpeechSegmenter("mic", config, vad=vad)
    samples = np.linspace(-1, 1, 2048, dtype=np.float32)
    events = segmenter.feed(chunk(samples))
    assert [e.reason for e in events] == ["silence", "silence"]
    assert events[0].end_sample == events[1].start_sample
    assert events[0].utterance_id != events[1].utterance_id
    np.testing.assert_array_equal(np.concatenate([e.samples for e in events]), samples)


def test_late_native_final_preserves_speech_after_history_has_evicted_its_start():
    length = 40 * 16000
    vad = ScriptedVad([(length, 0, length)])
    segmenter = SpeechSegmenter("mic", vad=vad)
    samples = np.linspace(-0.8, 0.8, length, dtype=np.float32)
    events = []
    for start in range(0, length, 512):
        events.extend(segmenter.feed(chunk(samples[start:start + 512], start=start)))
    assert len(events) == 1
    final = events[0]
    assert final.start_sample == 0 and final.end_sample == length
    assert final.samples.size == final.end_sample - final.start_sample
    np.testing.assert_array_equal(final.samples, samples)


def test_continuous_speech_is_finalized_with_preserved_pending_and_absolute_clocks():
    class ContinuousVad(ScriptedVad):
        # Model delayed speech detection after each native reset.
        segment_start = 1024

        def accept_waveform(self, samples):
            super().accept_waveform(samples)
            self.speech = self.position >= 1536

        def flush(self):
            if self.speech:
                audio = np.concatenate(self.received)
                self.completed.append(SimpleNamespace(start=self.segment_start,
                    samples=audio[self.segment_start:].copy()))
                self.segment_start = self.position
            self.speech = False

        def reset(self):
            super().reset()
            self.speech = False
            self.segment_start = 1024

    vad = ContinuousVad()
    segmenter = SpeechSegmenter("mic", vad=vad)
    length, origin, generation = 40 * 16000, 12345, 7
    samples = np.linspace(-0.8, 0.8, length, dtype=np.float32)
    events, received_at = [], []
    position, packet_sizes = 0, (311, 777, 1025, 2049)
    while position < length:
        size = min(packet_sizes[len(received_at) % len(packet_sizes)], length - position)
        packet_events = segmenter.feed(chunk(samples[position:position + size],
                                              start=origin + position, generation=generation))
        events.extend(packet_events)
        received_at.extend([position + size] * len(packet_events))
        position += size
    assert vad.reset_count == 3
    events.extend(segmenter.flush())
    assert len(events) == 4 and vad.reset_count == 4
    assert [event.reason for event in events] == ["max_duration"] * 3 + ["flush"]
    assert received_at[0] / 16000 <= (segmenter.config.max_speech_seconds
                                     + segmenter.config.max_cut_hold_seconds + .3)
    assert all(event.kind == "final" and event.generation == generation for event in events)
    assert len({event.utterance_id for event in events}) == 4
    assert events[0].start_sample == origin and events[-1].end_sample == origin + length
    assert all(left.end_sample == right.start_sample for left, right in zip(events, events[1:]))
    for event in events:
        assert event.samples.size == event.end_sample - event.start_sample
        np.testing.assert_array_equal(event.samples, samples[event.start_sample - origin:event.end_sample - origin])
    np.testing.assert_array_equal(np.concatenate([event.samples for event in events]), samples)


def test_drafts_are_disabled_by_default_and_optional_revisions_share_final_id():
    vad = ScriptedVad([(1024, 0, 1024)])
    vad.speech = True
    config = SegmentationConfig(drafts_enabled=True, draft_interval_seconds=0.032)
    segmenter = SpeechSegmenter("mic", config, vad=vad)
    draft = segmenter.feed(chunk(np.ones(512)))[0]
    final = segmenter.feed(chunk(np.ones(512), start=512))[0]
    assert draft.kind == "draft" and final.kind == "final"
    assert draft.utterance_id == final.utterance_id
    assert final.revision == draft.revision + 1
    default_vad = ScriptedVad()
    default_vad.speech = True
    assert SpeechSegmenter("mic", vad=default_vad).feed(chunk(np.ones(20000))) == []


def test_channels_are_isolated_and_invalid_input_is_rejected():
    left = SpeechSegmenter("mic", vad=ScriptedVad([(512, 0, 512)]))
    right = SpeechSegmenter("system", vad=ScriptedVad())
    assert left.feed(chunk(np.ones(512)))[0].kind == "final"
    assert right.feed(chunk(np.zeros(512), channel="system")) == []
    with pytest.raises(ValueError, match="own audio channel"):
        left.feed(chunk(np.zeros(512), channel="system"))
    with pytest.raises(ValueError, match="non-finite"):
        left.feed(chunk([float("nan")], start=512))


def test_silero_configuration_uses_expected_cpu_api_without_loading_a_model(tmp_path, monkeypatch):
    import sys
    recorded = {}

    def make_config():
        return SimpleNamespace(silero_vad=SimpleNamespace())

    def make_detector(config, buffer_size_in_seconds):
        recorded.update(config=config, buffer=buffer_size_in_seconds)
        return ScriptedVad()

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        VadModelConfig=make_config, VoiceActivityDetector=make_detector))
    model = tmp_path / "synthetic-model.onnx"
    model.write_bytes(b"placeholder for mocked constructor")
    create_silero_vad(model, SegmentationConfig())
    cfg = recorded["config"]
    assert cfg.sample_rate == 16000 and cfg.num_threads == 1 and cfg.provider == "cpu"
    assert cfg.silero_vad.model == str(model)
    assert cfg.silero_vad.max_speech_duration == pytest.approx(12.4)
    assert cfg.silero_vad.min_silence_duration == 1.1
    assert cfg.silero_vad.window_size == 512


def test_accuracy_defaults_keep_short_speech_and_pre_padding_without_drafts():
    config = SegmentationConfig()
    assert config.min_silence_seconds == 1.1
    assert config.min_speech_seconds == 0.25
    assert config.max_speech_seconds == 10.0
    assert config.pre_speech_seconds == 0.3
    assert config.drafts_enabled is False


def test_forced_cut_rewarms_vad_and_preserves_short_continuation():
    class WarmVad(ScriptedVad):
        segment_start = 0

        def accept_waveform(self, samples):
            super().accept_waveform(samples)
            self.speech = self.position >= 4096

        def flush(self):
            if self.speech:
                audio = np.concatenate(self.received)
                self.completed.append(SimpleNamespace(start=self.segment_start,
                    samples=audio[self.segment_start:].copy()))
                self.segment_start = self.position
            self.speech = False

        def reset(self):
            super().reset()
            self.segment_start = 0
            self.speech = False

    vad = WarmVad()
    config = SegmentationConfig(max_speech_seconds=0.32)
    segmenter = SpeechSegmenter("mic", config, vad=vad)
    # After the initial 0.256 s warm-up and a 0.32 s application interval, only
    # 0.096 s remains. Reinitializing a 0.25 s detector would lose that tail.
    samples = np.linspace(-0.8, 0.8, 10752, dtype=np.float32)
    events = segmenter.feed(chunk(samples, start=50000))
    assert vad.reset_count == 1
    assert events == []  # A completed hard cut is held only for its tiny tail.
    events.extend(segmenter.flush())
    assert [event.reason for event in events] == ["max_duration_tail_join"]
    assert vad.reset_count == 2
    assert events[-1].end_sample == 50000 + samples.size
    assert all(event.emitted_at > 0 for event in events)
    np.testing.assert_array_equal(np.concatenate([event.samples for event in events]), samples)


class KnownBoundaryVad(ScriptedVad):
    """An independent boundary oracle over uniquely numbered source samples.

    Native resets replay a real prefix, so local detector positions change;
    waveform values identify the original audio position without guessing the
    application's bookkeeping or replacing a gap with artificial silence.
    """

    def __init__(self, natural_end=None, natural_emit=None, natural_start=None):
        super().__init__()
        self.natural_end = natural_end
        self.natural_emit = natural_emit
        self.natural_start = natural_start
        self.ended = False

    def accept_waveform(self, samples):
        super().accept_waveform(samples)
        self.speech = not self.ended
        if (not self.ended and self.natural_emit is not None
                and int(samples[-1]) + 1 >= self.natural_emit):
            audio = np.concatenate(self.received)
            keep = self.natural_end - int(audio[0])
            start = max(0, (self.natural_start or int(audio[0])) - int(audio[0]))
            self.completed.append(SimpleNamespace(start=start, samples=audio[start:keep].copy()))
            self.ended = True
            self.speech = False

    def flush(self):
        if self.speech:
            self.completed.append(SimpleNamespace(start=0,
                samples=np.concatenate(self.received).copy()))
        self.speech = False

    def reset(self):
        super().reset()
        self.speech = False


def numbered_replay(segmenter, audio, *, origin=20000, generation=1):
    events, received = [], []
    for offset in range(0, audio.size, 512):
        block = audio[offset:offset + 512]
        emitted = segmenter.feed(chunk(block, start=origin + offset, generation=generation))
        events.extend(emitted)
        received.extend([offset + block.size] * len(emitted))
    return events, received


@pytest.mark.parametrize("tail_samples", [256, 1536, 10400])
def test_tiny_tail_finishing_before_deadline_becomes_one_unmodified_final(tail_samples):
    hard_end = 32768  # First 512-sample detection plus a window-rounded 2 s limit.
    voice_end = hard_end + tail_samples
    native_emission = hard_end + 16384
    audio = np.arange(native_emission, dtype=np.float32)
    vad = KnownBoundaryVad(voice_end, native_emission)
    segmenter = SpeechSegmenter("mic", SegmentationConfig(max_speech_seconds=2), vad=vad)
    events, received = numbered_replay(segmenter, audio)
    assert len(events) == 1 and events[0].reason == "max_duration_tail_join"
    assert events[0].start_sample == 20000 and events[0].end_sample == 20000 + voice_end
    assert received == [native_emission]
    np.testing.assert_array_equal(events[0].samples, audio[:voice_end])
    assert segmenter.flush() == []


def test_ongoing_speech_releases_hard_cut_on_sample_deadline_without_waiting_for_sentence():
    config = SegmentationConfig(max_speech_seconds=2)
    audio = np.arange(7 * 16000 + 17, dtype=np.float32)
    segmenter = SpeechSegmenter("mic", config, vad=KnownBoundaryVad())
    events, received = numbered_replay(segmenter, audio)
    assert len(events) >= 2
    for event, emission in zip(events, received):
        assert event.reason == "max_duration"
        waiting = emission - (event.end_sample - 20000)
        assert round(config.max_cut_hold_seconds * 16000) <= waiting < round(
            config.max_cut_hold_seconds * 16000) + 512
    events.extend(segmenter.flush())
    assert events[-1].end_sample == 20000 + audio.size
    assert all(left.end_sample == right.start_sample for left, right in zip(events, events[1:]))
    np.testing.assert_array_equal(np.concatenate([event.samples for event in events]), audio)


@pytest.mark.parametrize("tail_samples,emission_delay", [(10496, 16384), (1536, 30208)])
def test_long_or_late_tail_stays_separate_and_loses_no_audio(tail_samples, emission_delay):
    hard_end = 32768
    voice_end = hard_end + tail_samples
    native_emission = hard_end + emission_delay
    audio = np.arange(native_emission, dtype=np.float32)
    segmenter = SpeechSegmenter("mic", SegmentationConfig(max_speech_seconds=2),
        vad=KnownBoundaryVad(voice_end, native_emission))
    events, _ = numbered_replay(segmenter, audio)
    assert [event.reason for event in events] == ["max_duration", "silence"]
    assert events[0].end_sample == events[1].start_sample
    np.testing.assert_array_equal(np.concatenate([event.samples for event in events]), audio[:voice_end])


@pytest.mark.parametrize("emission_delay,joined", [(29184, True), (29696, False)])
def test_tail_at_rounded_deadline_can_join_but_one_later_window_cannot(emission_delay, joined):
    hard_end = 32768
    voice_end = hard_end + 1536
    audio = np.arange(hard_end + emission_delay, dtype=np.float32)
    segmenter = SpeechSegmenter("mic", SegmentationConfig(max_speech_seconds=2),
        vad=KnownBoundaryVad(voice_end, audio.size))
    events, _ = numbered_replay(segmenter, audio)
    expected = ["max_duration_tail_join"] if joined else ["max_duration", "silence"]
    assert [event.reason for event in events] == expected
    np.testing.assert_array_equal(np.concatenate([event.samples for event in events]), audio[:voice_end])


def test_independent_short_answer_after_a_real_gap_is_not_swallowed_by_held_hard_cut():
    hard_end = 32768
    natural_start = hard_end + 12800
    voice_end = natural_start + 3200
    native_emission = hard_end + 20992
    audio = np.arange(native_emission, dtype=np.float32)
    segmenter = SpeechSegmenter("mic", SegmentationConfig(max_speech_seconds=2),
        vad=KnownBoundaryVad(voice_end, native_emission, natural_start))
    events, _ = numbered_replay(segmenter, audio)
    assert [event.reason for event in events] == ["max_duration", "silence"]
    assert events[0].end_sample < events[1].start_sample
    assert events[1].samples.size <= round(segmenter.config.max_cut_tail_seconds * 16000)
    for event in events:
        np.testing.assert_array_equal(event.samples,
            audio[event.start_sample - 20000:event.end_sample - 20000])


def test_normal_independent_short_answers_are_emitted_immediately_without_hard_cut_buffering():
    vad = ScriptedVad([(4096, 0, 3200), (12288, 8192, 3200)])
    segmenter = SpeechSegmenter("mic", vad=vad)
    audio = np.arange(12288, dtype=np.float32)
    events, received = numbered_replay(segmenter, audio)
    assert [event.reason for event in events] == ["silence", "silence"]
    assert received == [4096, 12288]
    assert events[0].utterance_id != events[1].utterance_id
    assert events[0].end_sample < events[1].start_sample


@pytest.mark.parametrize("discontinuity", ["explicit", "generation", "gap"])
def test_held_completed_final_precedes_reset_and_cannot_join_new_audio(discontinuity):
    hard_end = 32768
    audio = np.arange(hard_end, dtype=np.float32)
    segmenter = SpeechSegmenter("mic", SegmentationConfig(max_speech_seconds=2),
                                vad=KnownBoundaryVad())
    events, _ = numbered_replay(segmenter, audio)
    assert events == []
    if discontinuity == "explicit":
        emitted = segmenter.reset("cancel")
        assert segmenter.flush() == []
    elif discontinuity == "generation":
        emitted = segmenter.feed(chunk(np.arange(512, dtype=np.float32), start=0, generation=2))
    else:
        emitted = segmenter.feed(chunk(np.arange(512, dtype=np.float32), start=20000 + hard_end + 512))
    assert [event.kind for event in emitted] == ["final", "reset"]
    assert emitted[0].generation == emitted[1].generation == 1
    assert emitted[0].reason == "max_duration"
    np.testing.assert_array_equal(emitted[0].samples, audio)
    assert all(event.reason != "max_duration_tail_join" for event in segmenter.flush())


@pytest.mark.parametrize("kwargs", [
    {"max_cut_hold_seconds": -.1}, {"max_cut_hold_seconds": 3.1},
    {"max_cut_hold_seconds": float("nan")}, {"max_cut_hold_seconds": float("inf")},
    {"max_cut_tail_seconds": 0}, {"max_cut_tail_seconds": 1.1},
    {"max_cut_tail_seconds": float("nan")},
])
def test_hard_cut_buffering_configuration_rejects_unbounded_or_invalid_values(kwargs):
    with pytest.raises(ValueError, match="Hard-cut buffering"):
        SegmentationConfig(**kwargs)


def test_hard_cut_buffering_can_be_disabled_for_comparison():
    audio = np.arange(32768, dtype=np.float32)
    config = SegmentationConfig(max_speech_seconds=2, max_cut_hold_seconds=0)
    segmenter = SpeechSegmenter("mic", config, vad=KnownBoundaryVad())
    events, received = numbered_replay(segmenter, audio)
    assert len(events) == 1 and events[0].reason == "max_duration"
    assert received == [audio.size]


def test_long_phrase_ends_on_confirmed_natural_pause_without_cutting_next_onset():
    source_end, voice_end = 98304, 94208
    pause = ScriptedVad([(source_end, 0, voice_end)])
    segmenter = SpeechSegmenter("mic", vad=KnownBoundaryVad(), pause_vad=pause)
    audio = np.arange(12 * 16000 + 17, dtype=np.float32)
    events, received = numbered_replay(segmenter, audio)
    assert len(events) == 1 and events[0].reason == "natural_pause"
    assert received == [source_end]
    assert events[0].end_sample == 20000 + voice_end
    assert events[0].samples.size >= 5.5 * 16000
    np.testing.assert_array_equal(events[0].samples, audio[:voice_end])
    events.extend(segmenter.flush())
    assert len(events) == 2 and events[1].reason == "flush"
    assert events[1].start_sample == events[0].end_sample
    assert events[1].end_sample == 20000 + audio.size
    assert events[0].utterance_id != events[1].utterance_id
    np.testing.assert_array_equal(np.concatenate([event.samples for event in events]), audio)


@pytest.mark.parametrize("completed_span,completed_at", [(48000, 55296), (48000, 98304)])
def test_short_or_stale_pause_does_not_split_or_backdate_an_unfinished_long_phrase(
        completed_span, completed_at):
    pause = ScriptedVad([(completed_at, 0, completed_span)])
    segmenter = SpeechSegmenter("mic", vad=KnownBoundaryVad(), pause_vad=pause)
    audio = np.arange(8 * 16000, dtype=np.float32)
    events, _ = numbered_replay(segmenter, audio)
    assert events == []
    events.extend(segmenter.flush())
    assert len(events) == 1 and events[0].reason == "flush"
    np.testing.assert_array_equal(events[0].samples, audio)


def test_continuous_long_phrase_without_vad_pause_keeps_hard_limit_and_tail_policy():
    segmenter = SpeechSegmenter("mic", vad=KnownBoundaryVad(), pause_vad=ScriptedVad())
    audio = np.arange(12 * 16000, dtype=np.float32)
    events, received = numbered_replay(segmenter, audio)
    assert len(events) == 1 and events[0].reason == "max_duration"
    assert 10 * 16000 <= events[0].samples.size < 10.1 * 16000
    assert received[0] - (events[0].end_sample - 20000) < 1.832 * 16000
    events.extend(segmenter.flush())
    np.testing.assert_array_equal(np.concatenate([event.samples for event in events]), audio)


def test_preferred_pause_can_be_disabled_without_weakening_old_configuration():
    config = SegmentationConfig(preferred_speech_seconds=0, min_silence_seconds=.2)
    pause = ScriptedVad([(98304, 0, 94208)])
    segmenter = SpeechSegmenter("mic", config, vad=KnownBoundaryVad(), pause_vad=pause)
    audio = np.arange(8 * 16000, dtype=np.float32)
    events, _ = numbered_replay(segmenter, audio)
    assert events == [] and pause.received == []
    np.testing.assert_array_equal(segmenter.flush()[0].samples, audio)


@pytest.mark.parametrize("kwargs", [
    {"preferred_speech_seconds": -1}, {"preferred_speech_seconds": float("inf")},
    {"preferred_speech_seconds": float("nan")}, {"preferred_pause_seconds": 0},
    {"preferred_pause_seconds": 3.1}, {"preferred_pause_seconds": float("nan")},
])
def test_preferred_pause_configuration_is_bounded(kwargs):
    with pytest.raises(ValueError, match="Preferred speech"):
        SegmentationConfig(**kwargs)


def test_generation_change_resets_both_vad_clocks_before_new_source_audio():
    primary, pause = KnownBoundaryVad(), ScriptedVad()
    segmenter = SpeechSegmenter("mic", vad=primary, pause_vad=pause)
    numbered_replay(segmenter, np.arange(32000, dtype=np.float32))
    events = segmenter.feed(chunk(np.arange(512, dtype=np.float32), generation=2))
    assert [event.reason for event in events] == ["capture_restarted"]
    assert primary.reset_count == pause.reset_count == 1
    assert len(pause.received) == 1
    np.testing.assert_array_equal(pause.received[0], np.arange(512, dtype=np.float32))


@pytest.mark.parametrize("change", ["generation", "gap"])
def test_natural_pause_onset_carry_never_crosses_capture_discontinuity(change):
    pause = ScriptedVad([(98304, 0, 94208)])
    segmenter = SpeechSegmenter("mic", vad=KnownBoundaryVad(), pause_vad=pause)
    audio = np.arange(98304, dtype=np.float32)
    events, _ = numbered_replay(segmenter, audio)
    assert [event.reason for event in events] == ["natural_pause"]
    new_audio = np.arange(16000, dtype=np.float32) + 1000000
    if change == "generation":
        start, generation, expected_reset = 0, 2, "capture_restarted"
    else:
        start, generation, expected_reset = 20000 + audio.size + 512, 1, "audio_gap"
    reset = segmenter.feed(chunk(new_audio, start=start, generation=generation))
    assert [event.reason for event in reset] == [expected_reset]
    final = segmenter.flush()
    assert len(final) == 1 and final[0].start_sample == start
    assert final[0].generation == generation
    np.testing.assert_array_equal(final[0].samples, new_audio)


def test_public_flush_pads_subwindow_tail_and_restores_real_clock():
    class WindowVad(ScriptedVad):
        def accept_waveform(self, samples):
            assert samples.size == 512
            super().accept_waveform(samples)
            self.speech = True

        def flush(self):
            audio = np.concatenate(self.received)
            self.completed.append(SimpleNamespace(start=0, samples=audio.copy()))
            self.speech = False

    vad = WindowVad()
    segmenter = SpeechSegmenter("mic", vad=vad)
    audio = np.linspace(-0.8, 0.8, 1201, dtype=np.float32)
    assert segmenter.feed(chunk(audio, start=3000)) == []
    final = segmenter.flush()[0]
    assert (final.start_sample, final.end_sample) == (3000, 4201)
    np.testing.assert_array_equal(final.samples, audio)
    next_audio = np.linspace(0.1, 0.2, 777, dtype=np.float32)
    assert segmenter.feed(chunk(next_audio, start=4201)) == []
    following = segmenter.flush()[0]
    assert (following.start_sample, following.end_sample) == (4201, 4978)
    np.testing.assert_array_equal(following.samples, next_audio)


def test_mono_conversion_handles_channels_and_invalid_values():
    stereo = np.array([[0.4, 0.8], [-0.8, -0.4], [np.nan, 1]], dtype=np.float32)
    actual = pcm_to_mono(stereo.tobytes(), 2)
    np.testing.assert_allclose(actual, [0.6, -0.6, 0], atol=1e-6)
    assert actual.dtype == np.float32 and actual.ndim == 1
    with pytest.raises(ValueError, match="Incomplete"):
        pcm_to_mono(np.zeros(3, dtype=np.float32).tobytes(), 2)


def test_disabled_capture_never_initializes_backends():
    def forbidden():
        raise AssertionError("Disabled capture initialized a backend")
    capture = AudioCaptureManager(lambda _: None, _backend_loader=forbidden, _probe_factory=forbidden)
    capture.start(set())
    capture.stop()
    with pytest.raises(ValueError):
        capture.start({"invalid"})


def test_default_endpoint_probe_reads_only_enabled_roles(monkeypatch):
    import fanyiji.audio as module
    probe = module._DefaultEndpointProbe.__new__(module._DefaultEndpointProbe)
    flows = []
    probe._endpoint_id = lambda flow: flows.append(flow) or f"endpoint-{flow}"
    assert probe.read({"system"}) == {"system": "endpoint-0"}
    assert flows == [0]
    assert probe.read({"mic"}) == {"mic": "endpoint-1"}


def test_endpoint_com_contract_copies_id_and_releases_native_allocations(monkeypatch):
    import fanyiji.audio as module
    allocated = ctypes.create_unicode_buffer("endpoint-new")
    calls = []

    def method(pointer, index, restype, *args):
        def invoke(this, *values):
            calls.append((index, values))
            if index == 4:
                ctypes.cast(values[2], ctypes.POINTER(ctypes.c_void_p))[0] = 222
            elif index == 5:
                ctypes.cast(values[0], ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(allocated)
            return 0
        return invoke

    freed = []
    monkeypatch.setattr(module, "_com_method", method)
    probe = module._DefaultEndpointProbe.__new__(module._DefaultEndpointProbe)
    probe._enumerator = ctypes.c_void_p(111)
    probe._ole = SimpleNamespace(CoTaskMemFree=lambda p: freed.append(p.value))
    assert probe._endpoint_id(1) == "endpoint-new"
    assert calls[0][1][:2] == (1, 1)  # capture, multimedia
    assert [index for index, _ in calls] == [4, 5, 2]
    assert freed == [ctypes.addressof(allocated)]


def test_default_change_recreates_context_and_generation_without_recording():
    contexts = []
    statuses = []
    reconnected = threading.Event()

    class Probe:
        count = 0
        def read(self, channels):
            self.count += 1
            return {channel: "old" if self.count == 1 else "new" for channel in channels}
        def close(self):
            pass

    class Stream:
        active = False
        def start_stream(self):
            self.active = True
        def is_active(self):
            return self.active
        def stop_stream(self):
            self.active = False
        def close(self):
            self.active = False

    class Context:
        terminated = False
        def __init__(self):
            contexts.append(self)
        def get_host_api_info_by_type(self, kind):
            return {"defaultInputDevice": 0}
        def get_device_info_by_index(self, index):
            return {"index": 0, "name": "synthetic mic", "defaultSampleRate": 16000, "maxInputChannels": 1}
        def get_default_wasapi_loopback(self):
            raise AssertionError("Disabled system source was initialized")
        def open(self, **kwargs):
            return Stream()  # Deliberately never invoke capture callbacks.
        def terminate(self):
            self.terminated = True

    def status(event):
        statuses.append(event)
        if event.state == "running" and event.generation >= 2:
            reconnected.set()

    backend = SimpleNamespace(PyAudio=Context, paWASAPI=13, paFloat32=1, paContinue=0, paComplete=1)
    capture = AudioCaptureManager(lambda _: pytest.fail("No synthetic callback expected"), status,
        AudioCaptureSettings(default_device_poll_seconds=0.01, reconnect_seconds=0.01),
        _probe_factory=Probe, _backend_loader=lambda: (backend, None))
    try:
        capture.start({"mic"})
        assert reconnected.wait(3.0), [(s.state, s.message) for s in statuses]
    finally:
        capture.stop()
    assert len(contexts) >= 2 and all(c.terminated for c in contexts)
    assert {s.channel for s in statuses} == {"mic"}
    assert any(s.state == "reconnecting" for s in statuses)
    assert statuses[-1].state == "stopped"


def test_one_missing_source_does_not_restart_or_interrupt_healthy_source():
    chunks = []
    final_segments = []
    failures = []
    contexts = []
    portaudio_threads = []
    complete = threading.Event()
    segmenter = SpeechSegmenter("mic", vad=ScriptedVad([(2048, 0, 2048)]))
    samples = np.full(512, 0.125, dtype=np.float32)

    class Probe:
        def read(self, channels):
            return {channel: "stable" for channel in channels}
        def close(self):
            pass

    class Stream:
        active = False
        def __init__(self, callback):
            self.callback = callback
        def start_stream(self):
            portaudio_threads.append(threading.get_ident())
            self.active = True
        def is_active(self):
            portaudio_threads.append(threading.get_ident())
            if self.active:
                self.callback(samples.tobytes(), 512,
                    {"current_time": 10.0, "input_buffer_adc_time": 10.0 - 512 / 16000}, 0)
            return self.active
        def stop_stream(self):
            portaudio_threads.append(threading.get_ident())
            self.active = False
        def close(self):
            portaudio_threads.append(threading.get_ident())
            self.active = False

    class Context:
        def __init__(self):
            contexts.append(self)
            portaudio_threads.append(threading.get_ident())
        def get_host_api_info_by_type(self, kind):
            portaudio_threads.append(threading.get_ident())
            return {"defaultInputDevice": 0}
        def get_device_info_by_index(self, index):
            return {"index": 0, "name": "fake mic", "defaultSampleRate": 16000, "maxInputChannels": 1}
        def get_default_wasapi_loopback(self):
            portaudio_threads.append(threading.get_ident())
            raise LookupError("synthetic missing playback device")
        def open(self, **kwargs):
            portaudio_threads.append(threading.get_ident())
            return Stream(kwargs["stream_callback"])
        def terminate(self):
            portaudio_threads.append(threading.get_ident())

    def on_chunk(packet):
        chunks.append(packet)
        final_segments.extend(segmenter.feed(packet))
        if len(failures) >= 2 and final_segments:
            complete.set()

    def on_status(status):
        if status.channel == "system" and status.state == "error":
            failures.append(status)

    backend = SimpleNamespace(PyAudio=Context, paWASAPI=13, paFloat32=1, paContinue=0, paComplete=1)
    capture = AudioCaptureManager(on_chunk, on_status,
        AudioCaptureSettings(default_device_poll_seconds=0.01, reconnect_seconds=0.06),
        _probe_factory=Probe, _backend_loader=lambda: (backend, None))
    try:
        capture.start({"mic", "system"})
        assert complete.wait(3.0), (len(failures), len(chunks))
    finally:
        capture.stop()
    assert len(contexts) == 1
    assert len({packet.generation for packet in chunks}) == 1
    assert len(set(portaudio_threads)) == 1  # All COM/PortAudio operations stay on the supervisor.
    assert final_segments[0].samples.size == 2048
    assert final_segments[0].kind == "final"
    assert all(left.end_sample == right.start_sample for left, right in zip(chunks, chunks[1:]))
