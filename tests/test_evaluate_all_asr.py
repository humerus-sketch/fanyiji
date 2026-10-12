"""Frozen-input and single-native-call guarantees for the evaluation harness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_all_asr import RecordingRecognizer, acceptance, load_corpus, model_spec


def corpus(tmp_path, *, duplicate=False):
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16)
    item = {"id": "test-ja", "source_lang": "ja", "audio_path": str(audio),
            "sha256": hashlib.sha256(audio.read_bytes()).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "frozen", "samples": [item, item] if duplicate else [item]}), encoding="utf-8")
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(), audio


def test_frozen_pcm_is_verified_without_model_loading(tmp_path):
    path, expected, _ = corpus(tmp_path)
    _, samples = load_corpus(path, expected)
    assert samples[0]["verified_frames"] == 16
    assert samples[0]["verified_audio_seconds"] == .001


def test_corrupted_manifest_and_waveform_are_rejected(tmp_path):
    path, expected, audio = corpus(tmp_path)
    with pytest.raises(ValueError, match="manifest SHA"):
        load_corpus(path, "0" * 64)
    with audio.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="WAV SHA"):
        load_corpus(path, expected)


def test_duplicate_frozen_clip_ids_are_rejected(tmp_path):
    path, expected, _ = corpus(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="unique"):
        load_corpus(path, expected)


def test_native_capture_does_not_repeat_decode_and_survives_generation_rejection():
    class Native:
        count = 0
        def decode_stream(self, stream):
            self.count += 1
            stream.result = SimpleNamespace(text="unfinished raw", tokens=["x"] * 125)
    native, captured = Native(), {}
    proxy = RecordingRecognizer(native, captured)
    proxy.decode_stream(SimpleNamespace())
    assert native.count == 1
    assert captured["native_text"] == "unfinished raw"
    assert captured["exposed_token_count"] == 125
    assert acceptance(None, RuntimeError("production guard"), captured,
                      {"generation": {"cleaned_token_guard": 112}}) == "rejected_generation_guard"
    assert acceptance("normal", None, captured, {}) == "accepted_nonempty"


def test_small_uses_production_whisper_paths_without_writing_profile(tmp_path):
    _, directory, profile = model_spec("whisper-small-int8", tmp_path / "models", tmp_path / "model-cache" / "asr-evaluation")
    assert directory.name == "reference-whisper-small"
    assert profile == {"backend": "whisper", "model_id": "whisper-small-int8",
                       "encoder": "small-encoder.int8.onnx", "decoder": "small-decoder.int8.onnx", "tokens": "small-tokens.txt"}
    assert not directory.exists()
