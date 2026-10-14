"""Frozen-input and whole-worker drain guarantees; no models or CLI calls."""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_e2e_replay import (DrainCheck, QueueWorkTracker, compose_audio,
                                 NoCollection, feed_jitter_summary, isolate_replay_controller,
                                 load_asr_profile, load_inputs, read_wav, replay_source_valid,
                                 work_snapshot)


def wav(path, samples=160):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(np.arange(samples, dtype="<i2").tobytes())
    return path


def manifest(tmp_path):
    japanese, chinese = wav(tmp_path / "japanese.wav"), wav(tmp_path / "chinese.wav")
    def row(name, language, audio):
        return {"id": name, "source_lang": language, "audio_path": audio.name,
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(), "human_truth": None}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"status": "frozen", "samples": [
        row("japanese", "ja", japanese), row("chinese", "zh", chinese)]}), encoding="utf-8")
    return path, japanese


def test_manifest_selects_only_verified_japanese_without_writing(tmp_path):
    path, audio = manifest(tmp_path)
    before = path.read_bytes()
    inputs, metadata, reference = load_inputs(manifest=path)
    assert len(inputs) == 1 and metadata[0]["id"] == "japanese"
    assert metadata[0]["human_truth"] is None
    assert reference["sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
    with audio.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_inputs(manifest=path)


@pytest.mark.parametrize("clips", [["chinese"], ["missing"], ["japanese", "japanese"]])
def test_manifest_rejects_non_japanese_unknown_or_duplicate_selection(tmp_path, clips):
    path, _ = manifest(tmp_path)
    with pytest.raises(ValueError):
        load_inputs(manifest=path, clips=clips)


def test_continuous_recording_has_no_internal_gaps_and_retains_tail(tmp_path):
    path = wav(tmp_path / "continuous.wav", samples=17000)
    inputs, metadata, frozen = load_inputs(audio_file=path)
    samples, locations = compose_audio(inputs, gap=0)
    assert frozen is None and metadata[0]["audio_seconds"] == 17000 / 16000
    assert locations == [{"start_sample": 4800, "end_sample": 21800}]
    np.testing.assert_array_equal(samples[4800:21800], inputs[0][1])
    assert len(samples) == 4800 + 17000 + 48000
    assert samples[21800:].sum() == 0
    with pytest.raises(ValueError, match="cannot be combined"):
        load_inputs(audio_file=path, clips=["continuous"])


@pytest.mark.parametrize("gap", [-1, float("nan"), float("inf")])
def test_invalid_replay_gaps_are_rejected(gap):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compose_audio([(Path("unused"), np.zeros(10, np.float32))], gap)


def test_asr_override_is_memory_only_and_does_not_rewrite_active_profile(tmp_path):
    active = tmp_path / "backend.json"
    original = b'{"backend":"qwen3_asr","model_id":"production"}'
    active.write_bytes(original)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text('{"backend":"sense_voice","model_id":"evaluation"}', encoding="utf-8")
    directory, profile, watched = load_asr_profile(tmp_path, evaluation)
    assert directory == tmp_path and profile["model_id"] == "evaluation" and watched == evaluation
    assert active.read_bytes() == original
    with pytest.raises(ValueError, match="requires"):
        load_asr_profile(None, evaluation)


def test_missing_explicit_profile_is_rejected_not_silently_sensevoice(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_asr_profile(tmp_path, tmp_path / "missing.json")
    _, profile, path = load_asr_profile(tmp_path)
    assert profile == {"backend": "sense_voice", "model_id": "sensevoice-int8"} and path is None


def run_state():
    asr = SimpleNamespace(_condition=threading.Condition(), _queues={"mic": deque(), "system": deque()})
    return SimpleNamespace(asr_queue=asr,
        mt_queue=SimpleNamespace(snapshot=lambda: {"pending": 0}),
        draft_queue=SimpleNamespace(snapshot=lambda: {"pending": 0}),
        pending_text={}, capture_progress={}, draft_active=0, overloaded=False)


def test_dequeued_native_job_remains_busy_until_next_getter():
    tracker, run = QueueWorkTracker(), run_state()
    items = deque([object()])
    def get():
        if not items:
            raise TimeoutError
        return items.popleft()
    run.asr_queue.get = get
    tracker.wrap(run.asr_queue, "get", "asr")
    assert run.asr_queue.get() is not None
    # Queue pending==0 must not end replay while its filter/ASR job runs.
    assert work_snapshot(run, tracker)["busy"]["asr"] == 1
    assert not work_snapshot(run, tracker)["idle"]
    with pytest.raises(TimeoutError):
        run.asr_queue.get()
    assert work_snapshot(run, tracker)["idle"]


def test_mt_batch_inflight_is_not_mistaken_for_empty_queue():
    tracker, run = QueueWorkTracker(), run_state()
    calls = deque([[object(), object()], []])
    run.mt_queue.get_batch = lambda: calls.popleft()
    tracker.wrap(run.mt_queue, "get_batch", "mt")
    assert len(run.mt_queue.get_batch()) == 2
    assert not work_snapshot(run, tracker)["idle"]
    assert run.mt_queue.get_batch() == []
    assert work_snapshot(run, tracker)["idle"]


def test_pending_phrase_and_native_held_tail_prevent_premature_completion():
    tracker, run = QueueWorkTracker(), run_state()
    run.pending_text["system"] = object()
    assert not work_snapshot(run, tracker)["idle"]
    run.pending_text.clear()
    run.capture_progress["system"] = SimpleNamespace(speech_active=False, held_start_sample=123)
    assert not work_snapshot(run, tracker)["idle"]
    run.capture_progress["system"] = SimpleNamespace(speech_active=True, held_start_sample=None)
    assert not work_snapshot(run, tracker)["idle"]
    run.capture_progress["system"] = SimpleNamespace(speech_active=False, held_start_sample=None)
    assert work_snapshot(run, tracker)["idle"]


def test_completion_requires_done_feed_and_stable_idle_but_no_final_count():
    drain = DrainCheck(stable_seconds=2)
    assert not drain.update(feed_done=False, idle=True, now=1)
    assert not drain.update(feed_done=True, idle=True, now=2)
    assert not drain.update(feed_done=True, idle=False, now=3)
    assert not drain.update(feed_done=True, idle=True, now=4)
    assert not drain.update(feed_done=True, idle=True, now=5.999)
    assert drain.update(feed_done=True, idle=True, now=6)
    # Zero recognized speech is a complete run, separately scored as coverage.
    assert not drain.update(feed_done=False, idle=True, now=7)


def test_empty_wav_is_not_a_valid_fast_success(tmp_path):
    path = wav(tmp_path / "empty.wav", samples=0)
    with pytest.raises(ValueError, match="nonempty"):
        read_wav(path)


@pytest.mark.parametrize("values", [
    {"channel": "system", "generation": 1, "source_room": ""},
    {"channel": "system", "source_generation": 1, "source_room": ""},
])
def test_replay_source_accepts_pcm_jobs_and_qt_events(values):
    assert replay_source_valid(SimpleNamespace(**values))


@pytest.mark.parametrize("values", [
    {"channel": "mic", "generation": 1},
    {"channel": "system", "generation": 2},
    {"channel": "system", "generation": True},
    {"channel": "system", "generation": 1, "source_room": "1698225"},
    {"channel": "system", "generation": 1, "source_generation": 2},
    {"channel": "system"},
])
def test_replay_source_rejects_real_sources_and_wrong_epochs(values):
    assert not replay_source_valid(SimpleNamespace(**values))


def test_replay_start_never_initializes_browser_or_resets_existing_queue(tmp_path, monkeypatch):
    """Exercise real controller startup against enabled external preferences."""
    from PySide6.QtWidgets import QApplication
    from fanyiji import app as app_module, pipeline as pipeline_module
    from fanyiji.config import AppConfig, SourceConfig
    from fanyiji.ui import MainWindow

    application = QApplication.instance() or QApplication([])
    preferences = tmp_path / "browser-preferences.json"
    preferences.write_text('{"danmaku_enabled":true}', encoding="utf-8")
    queue = tmp_path / "danmaku-queue.sqlite3"
    queue.write_bytes(b"existing queue must never be opened or reset")
    before = {path: path.read_bytes() for path in (preferences, queue)}
    pipeline = Mock()
    monkeypatch.setattr(app_module, "data_root", lambda: tmp_path)
    monkeypatch.setattr(pipeline_module, "SubtitlePipeline", lambda *_args, **_kwargs: pipeline)
    config = AppConfig(translation_provider="codex", microphone=SourceConfig(enabled=False))
    window = MainWindow(config)
    controller = app_module.AppController(window, config, tmp_path / "unused-settings.json",
                                          specialized_service=NoCollection())
    replay_factory = Mock()
    try:
        isolate_replay_controller(controller, replay_factory)
        controller.start()
        # Drain its transition so the assertions cover asynchronous startup.
        controller.executor.submit(lambda: None).result(timeout=3)
        assert pipeline.start.call_count == 1
        assert pipeline.capture_factory is replay_factory
        assert controller.browser_service is None and controller.danmaku is None
        assert controller.live_source_service.validate_job(
            SimpleNamespace(channel="system", source_generation=1, source_room=""))
        with pytest.raises(AssertionError, match="forbidden"):
            controller._ensure_browser_service()
        assert all(path.read_bytes() == content for path, content in before.items())
        assert not (tmp_path / "unused-settings.json").exists()
    finally:
        controller.close()
        controller.executor.shutdown(wait=True)
        window.close()
        application.processEvents()


def test_isolation_refuses_already_initialized_danmaku():
    controller = SimpleNamespace(browser_service=None, danmaku=object())
    with pytest.raises(RuntimeError, match="before browser"):
        isolate_replay_controller(controller, object())


def test_feed_jitter_reports_actual_scheduling_lateness():
    assert feed_jitter_summary([])["p95_seconds"] is None
    result = feed_jitter_summary([0.0, .01, .02, .3])
    assert result["blocks"] == 4
    assert result["p50_seconds"] == pytest.approx(.015)
    assert result["p95_seconds"] == pytest.approx(.258)
    assert result["max_seconds"] == .3
