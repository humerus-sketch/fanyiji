"""Opt-in CPU replay checks against a saved WAV; never capture audio devices.

Run with FANYIJI_NATIVE_VAD_TEST=1. CI does not require local weights or
private evaluation audio, so these checks are skipped unless explicitly enabled.
"""
from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from fanyiji.audio import AudioChunk, SAMPLE_RATE
from fanyiji.segmentation import SegmentationConfig, SpeechSegmenter


pytestmark = pytest.mark.skipif(os.environ.get("FANYIJI_NATIVE_VAD_TEST") != "1",
    reason="Opt-in local CPU VAD replay; requires a saved audio fixture and model")

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = 12345


@pytest.fixture(scope="module")
def saved_audio():
    path = ROOT / "artifacts/diagnostics/all-model-evaluation-20261009/corpus/audio/recent-ja-1100.wav"
    model = ROOT / "models/vad/silero_vad.onnx"
    if not path.is_file() or not model.is_file():
        pytest.skip("Local saved audio or VAD weights unavailable")
    with wave.open(str(path), "rb") as audio:
        assert audio.getframerate() == SAMPLE_RATE and audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768


def replay(audio, config=None):
    segmenter = SpeechSegmenter("system", config,
        vad_model_path=ROOT / "models/vad/silero_vad.onnx")
    events = []
    for offset in range(0, audio.size, 512):
        block = audio[offset:offset + 512]
        events.extend(segmenter.feed(AudioChunk("system", block, ORIGIN + offset,
            ORIGIN + offset + block.size, 123.0, 1)))
    events.extend(segmenter.flush())
    assert all(left.end_sample <= right.start_sample for left, right in zip(events, events[1:]))
    for event in events:
        assert event.samples.size == event.end_sample - event.start_sample
        np.testing.assert_array_equal(event.samples,
            audio[event.start_sample - ORIGIN:event.end_sample - ORIGIN])
    return events


def test_native_default_hard_cuts_keep_entire_continuous_audio_and_partial_window(saved_audio):
    audio = np.tile(saved_audio, 5)
    events = replay(audio)
    assert len(events) >= 4
    assert events[0].start_sample == ORIGIN
    assert events[-1].end_sample == ORIGIN + audio.size
    assert all(left.end_sample == right.start_sample for left, right in zip(events, events[1:]))
    assert all(event.reason == "max_duration" for event in events[:-1])
    assert events[-1].reason == "flush"
    np.testing.assert_array_equal(np.concatenate([event.samples for event in events]), audio)


def test_native_hard_cut_preserves_96ms_voice_tail(saved_audio):
    continuous = np.tile(saved_audio, 5)
    boundary = replay(continuous)[0].end_sample - ORIGIN
    voice_end = boundary + 1536
    audio = np.concatenate((continuous[:voice_end], np.zeros(32000, dtype=np.float32)))
    original = replay(audio, SegmentationConfig(max_cut_hold_seconds=0))
    events = replay(audio)
    assert [event.reason for event in original] == ["max_duration", "silence"]
    assert original[0].end_sample == ORIGIN + boundary
    assert original[1].end_sample >= ORIGIN + voice_end
    assert len(events) == 1 and events[0].reason == "max_duration_tail_join"
    assert events[0].start_sample == original[0].start_sample
    assert events[0].end_sample == original[-1].end_sample
    np.testing.assert_array_equal(events[0].samples,
        np.concatenate([event.samples for event in original]))


def test_native_continuous_speech_hard_cut_expires_with_at_most_one_window_rounding(saved_audio):
    config = SegmentationConfig()
    audio = np.tile(saved_audio, 3)
    segmenter = SpeechSegmenter("system", config,
        vad_model_path=ROOT / "models/vad/silero_vad.onnx")
    events, emitted = [], []
    for offset in range(0, audio.size, 512):
        block = audio[offset:offset + 512]
        output = segmenter.feed(AudioChunk("system", block, ORIGIN + offset,
            ORIGIN + offset + block.size, 123.0, 1))
        events.extend(output)
        emitted.extend([ORIGIN + offset + block.size] * len(output))
    assert events and all(event.reason == "max_duration" for event in events)
    for event, source_end in zip(events, emitted):
        waiting = source_end - event.end_sample
        assert round(config.max_cut_hold_seconds * SAMPLE_RATE) <= waiting < round(
            config.max_cut_hold_seconds * SAMPLE_RATE) + config.window_size
        np.testing.assert_array_equal(event.samples,
            audio[event.start_sample - ORIGIN:event.end_sample - ORIGIN])
    events.extend(segmenter.flush())
    assert events[-1].end_sample == ORIGIN + audio.size
    assert all(left.end_sample == right.start_sample for left, right in zip(events, events[1:]))
    np.testing.assert_array_equal(np.concatenate([event.samples for event in events]), audio)


def test_native_accuracy_default_joins_thinking_pause(saved_audio):
    speech = saved_audio[16000:64000]
    audio = np.concatenate((speech, np.zeros(17600, dtype=np.float32), speech,
                            np.zeros(48000, dtype=np.float32)))
    original = replay(audio, SegmentationConfig(min_silence_seconds=0.6))
    accuracy = replay(audio, SegmentationConfig())
    assert len(original) == 2
    assert len(accuracy) == 1 and accuracy[0].reason in ("silence", "natural_pause")
    assert accuracy[0].start_sample <= original[0].start_sample
    assert accuracy[0].end_sample >= original[1].end_sample


def test_native_preferred_pause_separates_long_phrase_earlier_and_keeps_next_onset(saved_audio):
    speech = np.tile(saved_audio, 2)[:6 * SAMPLE_RATE]
    audio = np.concatenate((speech, np.zeros(SAMPLE_RATE, dtype=np.float32),
        saved_audio[:4 * SAMPLE_RATE], np.zeros(3 * SAMPLE_RATE, dtype=np.float32)))
    baseline = replay(audio, SegmentationConfig(preferred_speech_seconds=0))
    events = replay(audio)
    assert baseline[0].reason == "max_duration"
    assert len(events) == 2 and events[0].reason == "natural_pause"
    assert 5.5 <= events[0].samples.size / SAMPLE_RATE < 7
    assert events[0].end_sample < baseline[0].end_sample
    assert events[1].start_sample == events[0].end_sample
    assert events[-1].end_sample >= ORIGIN + 11 * SAMPLE_RATE
    np.testing.assert_array_equal(np.concatenate([event.samples for event in events]),
        audio[events[0].start_sample - ORIGIN:events[-1].end_sample - ORIGIN])
