"""Replay frozen local WAVs through production VAD/ASR/MT and real Qt paint.

No sound device is opened. Only recognized text uses the existing Codex login.
Run variants in separate processes, serially; keep other subtitle runs closed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from fanyiji import app as app_module, audio as audio_module
from fanyiji.config import SourceConfig, load_config
from fanyiji.paths import data_root
from fanyiji.ui import MainWindow


DEFAULT_CORPUS = ROOT / "artifacts/diagnostics/all-model-evaluation-20261009/corpus/audio"
DEFAULT_CLIPS = ("recent-ja-0001", "recent-ja-0199", "recent-ja-1001")


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read_wav(path):
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getframerate(), source.getsampwidth(), source.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("Replay requires mono 16kHz PCM16 WAV")
        frames = source.getnframes()
        pcm = source.readframes(frames)
        if not frames or len(pcm) != frames * 2:
            raise ValueError("Replay requires complete nonempty WAV PCM")
        return np.frombuffer(pcm, "<i2").astype(np.float32) / 32768


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(*, manifest=None, clips=None, audio_file=None):
    """Verify a frozen Japanese subset; never infer labels or edit its manifest."""
    if audio_file is not None:
        if manifest is not None or clips:
            raise ValueError("--audio-file cannot be combined with --manifest or --clips")
        path = Path(audio_file).resolve()
        samples = read_wav(path)
        return [(path, samples)], [{"id": path.stem, "path": str(path), "sha256": sha256(path),
            "audio_seconds": len(samples) / 16000}], None
    if manifest is None:
        names = tuple(clips or DEFAULT_CLIPS)
        if len(set(names)) != len(names) or any(Path(name).name != name or name in (".", "..")
                                               for name in names):
            raise ValueError("Clip IDs must be unique file names")
        paths = [(DEFAULT_CORPUS / f"{name}.wav").resolve() for name in names]
        inputs = [(path, read_wav(path)) for path in paths]
        return inputs, [{"id": name, "path": str(path), "sha256": sha256(path),
            "audio_seconds": len(samples) / 16000} for name, (path, samples) in zip(names, inputs)], None
    manifest = Path(manifest).resolve()
    content = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if content.get("status") != "frozen" or not isinstance(content.get("samples"), list):
        raise ValueError("Replay manifest must contain frozen samples")
    indexed = {}
    for row in content["samples"]:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in indexed:
            raise ValueError("Frozen sample IDs must be unique strings")
        indexed[identifier] = row
    names = list(clips) if clips else [name for name, row in indexed.items() if row.get("source_lang") == "ja"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Replay requires a nonempty unique Japanese selection")
    inputs, metadata = [], []
    for name in names:
        row = indexed.get(name)
        if row is None or row.get("source_lang") != "ja":
            raise ValueError(f"Not a frozen Japanese sample: {name}")
        path = Path(row["audio_path"])
        path = (manifest.parent / path if not path.is_absolute() else path).resolve()
        if not path.is_relative_to(manifest.parent):
            raise ValueError("Frozen WAV must stay inside the manifest directory")
        digest = sha256(path)
        if digest != row.get("sha256"):
            raise ValueError(f"Frozen WAV SHA-256 mismatch: {name}")
        samples = read_wav(path)
        inputs.append((path, samples))
        metadata.append({"id": name, "path": str(path), "sha256": digest,
            "audio_seconds": len(samples) / 16000, "source_kind": row.get("source_kind"),
            "human_truth": row.get("human_truth")})
    return inputs, metadata, {"path": str(manifest), "sha256": sha256(manifest)}


def compose_audio(inputs, gap):
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("Replay gap must be finite and nonnegative")
    pieces = [np.zeros(4800, np.float32)]
    locations = []
    cursor = 4800
    for index, (_, samples) in enumerate(inputs):
        locations.append({"start_sample": cursor, "end_sample": cursor + len(samples)})
        pieces.append(samples)
        cursor += len(samples)
        # Always feed real silence after the final recording. This allows the
        # production VAD/continuation rule to close, without forcing a flush.
        silence = np.zeros(round((gap if index + 1 < len(inputs) else max(3.0, gap)) * 16000), np.float32)
        pieces.append(silence)
        cursor += len(silence)
    return np.concatenate(pieces), locations


def load_asr_profile(directory, profile_path=None):
    if directory is None:
        if profile_path is not None:
            raise ValueError("--asr-profile requires --asr-directory")
        return None
    directory = Path(directory).resolve()
    path = Path(profile_path).resolve() if profile_path is not None else directory / "backend.json"
    if path.is_file():
        profile = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(profile, dict):
            raise ValueError("ASR profile must be a JSON object")
    elif profile_path is not None:
        raise ValueError("Explicit ASR profile does not exist")
    else:
        profile = {"backend": "sense_voice", "model_id": "sensevoice-int8"}
    return directory, profile, path if path.is_file() else None


class QueueWorkTracker:
    """Track whole jobs, including native filter/rechecks and batch fallbacks.

    Queue pending counts exclude work already removed by a worker. The next
    getter call closes its preceding job, even if the new get blocks forever.
    This instrumentation does not change queue ordering or native inference.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._busy = {}

    def wrap(self, queue, method, label):
        original = getattr(queue, method)
        def get(*args, **kwargs):
            key = (label, threading.get_ident())
            with self._lock:
                self._busy[key] = False
            item = original(*args, **kwargs)
            with self._lock:
                self._busy[key] = item is not None and not (isinstance(item, list) and not item)
            return item
        setattr(queue, method, get)

    def snapshot(self):
        with self._lock:
            return {label: sum(value for (name, _), value in self._busy.items() if name == label)
                    for label in ("asr", "mt", "draft")}


def work_snapshot(run, tracker):
    if run is None:
        return {"initialized": False, "idle": False}
    with run.asr_queue._condition:
        asr_pending = sum(len(queue) for queue in run.asr_queue._queues.values())
    busy = tracker.snapshot()
    native_pending = any(progress.speech_active or progress.held_start_sample is not None
                         for progress in list(run.capture_progress.values()))
    result = {"initialized": True, "asr_pending": asr_pending, "busy": busy,
              "mt_pending": run.mt_queue.snapshot()["pending"],
              "draft_pending": run.draft_queue.snapshot()["pending"],
              "pending_text": len(run.pending_text), "native_vad_pending": native_pending,
              "draft_active": run.draft_active, "overloaded": run.overloaded}
    result["idle"] = not (asr_pending or any(busy.values()) or result["mt_pending"]
        or result["draft_pending"] or result["pending_text"] or native_pending or result["draft_active"])
    return result


class DrainCheck:
    def __init__(self, stable_seconds=2.0):
        self.stable_seconds = stable_seconds
        self.since = None

    def update(self, *, feed_done, idle, now):
        if not feed_done or not idle:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.stable_seconds


class NoCollection:
    busy = False
    def begin_session(self, *_): pass
    def on_segment(self, *_): pass
    def on_event(self, *_): pass
    def cancel_comparison(self): pass
    def close(self): pass


def replay_source_valid(job):
    """Only this frozen playback epoch is allowed into the replay UI."""
    generation = getattr(job, "source_generation", getattr(job, "generation", None))
    return (getattr(job, "channel", "") == "system" and type(generation) is int
            and generation == 1 and getattr(job, "source_room", "") == "")


def isolate_replay_controller(controller, capture_factory):
    """Keep production queues, browser sessions and source leases untouched.

    AppController installs a selected-source factory in its constructor. A
    module-level AudioCaptureManager substitution no longer overrides it.
    Startup can also initialize/reset danmaku storage from external preferences.
    """
    if controller.browser_service is not None or controller.danmaku is not None:
        raise RuntimeError("Replay isolation must be installed before browser initialization")
    controller._danmaku_startup_checked = True
    def forbidden_browser(*_args, **_kwargs):
        raise AssertionError("Browser and danmaku initialization are forbidden during frozen replay")
    controller._ensure_browser_service = forbidden_browser
    controller.pipeline.capture_factory = capture_factory
    controller.pipeline.source_validator = replay_source_valid
    controller.live_source_service.validate_job = replay_source_valid


def feed_jitter_summary(values):
    if not values:
        return {"blocks": 0, "p50_seconds": None, "p95_seconds": None, "max_seconds": None}
    return {"blocks": len(values), "p50_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95)), "max_seconds": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("codex", "hybrid"), required=True)
    parser.add_argument("--effort", choices=("low", "medium"))
    parser.add_argument("--audio-processing-mode", choices=("off", "gtcrn", "uvr"))
    parser.add_argument("--asr-directory", type=Path)
    parser.add_argument("--asr-profile", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--audio-file", type=Path)
    parser.add_argument("--gap", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    inputs, input_metadata, frozen_manifest = load_inputs(
        manifest=args.manifest, clips=args.clips, audio_file=args.audio_file)
    samples, locations = compose_audio(inputs, args.gap)
    for metadata, location in zip(input_metadata, locations):
        metadata.update(location)
    override = load_asr_profile(args.asr_directory, args.asr_profile)
    settings = data_root() / "settings.json"
    settings_bytes = settings.read_bytes()
    loaded_config = load_config(settings)
    if loaded_config is None:
        raise ValueError("Replay requires completed existing application settings")
    config = replace(loaded_config, translation_provider=args.provider,
        codex_effort=args.effort or loaded_config.codex_effort, multilingual_asr=True,
        audio_processing=args.audio_processing_mode or loaded_config.audio_processing,
        microphone=SourceConfig(enabled=False),
        playback=SourceConfig(enabled=True, direction="ja_to_zh"))
    watched = {settings.resolve(): settings_bytes}
    protected_paths = [data_root() / name for name in (
        "live-source-settings.json", "browser-preferences.json", "danmaku-content-filter.json",
        "danmaku-queue.sqlite3", "danmaku-queue.sqlite3-wal", "danmaku-queue.sqlite3-shm")]
    protected_before = {path.resolve(): path.read_bytes() if path.is_file() else None
                        for path in protected_paths}
    for path in (app_module.models_root() / "asr/backend.json",
                 app_module.models_root() / "translation/backend.json",
                 Path(frozen_manifest["path"]) if frozen_manifest else None,
                 override[2] if override else None):
        if path is not None and path.is_file():
            watched[path.resolve()] = path.read_bytes()
    args.output = args.output.resolve()
    if any(path.is_relative_to(args.output) for path in watched):
        raise ValueError("Replay output must not contain watched production/corpus files")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 2, "started_at": stamp(), "status": "running", "provider": args.provider,
        "model": config.codex_model, "effort": config.codex_effort, "speed": config.codex_speed,
        "language_policy": "auto", "speech_only": config.speech_only,
        "audio_processing_mode": config.audio_processing,
        "asr_override": {"directory": str(override[0]), "profile": override[1], "num_threads": 4,
                         "profile_modified": False} if override else None,
        "real_time_replay": True, "audio_devices_opened": False, "fresh_audio_collected": False,
        "user_settings_modified": False, "training_collected": False,
        "browser_initialized": False, "danmaku_initialized": False,
        "limitations": ["Frozen audio replay; human truth is supplied separately, never inferred from model agreement",
                        "No physical audio-device latency or screen scan-out is measured",
                        "Final metrics require confirmed source text and visible Qt label paint",
                        "Filtered/empty/failed or nonvisible finals are reported separately, not counted as fast successes"],
        "input_mode": "continuous_audio_file" if args.audio_file else "frozen_utterances_with_synthetic_gaps",
        "gap_seconds": args.gap, "tail_silence_seconds": max(3.0, args.gap),
        "feed_seconds": len(samples) / 16000, "timeout_seconds": args.timeout,
        "replay_float32_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        "frozen_manifest": frozen_manifest, "inputs": input_metadata,
        "configuration_sha256_before": {str(path): hashlib.sha256(content).hexdigest()
                                          for path, content in watched.items()}}
    errors, captures, events, presentations, segments = [], [], [], [], []
    feed_jitter, window_states = [], []
    tracker, drain = QueueWorkTracker(), DrainCheck()
    application = QApplication([])
    application.setQuitOnLastWindowClosed(False)
    window = MainWindow(config)
    window.resize(1964, 842)
    window.setWindowTitle(f"翻译姬评测 · {config.audio_processing.upper()}")
    # Keep the test visible without covering the user's normal desktop work.
    window.topmost_checkbox.setChecked(False)
    controller = app_module.AppController(window, config, args.output / "unused-settings.json",
                                         specialized_service=NoCollection())
    if override is not None:
        from fanyiji.asr_engine import ProfileRecognizerEngine
        # Preselect only this process's recognizer. Pipeline preload, Quality-
        # Recognizer, filter, sentence assembly and MT still execute unchanged.
        controller.pipeline._recognizer = ProfileRecognizerEngine(override[0], num_threads=4, config=override[1])
    original_segment_observer = controller.pipeline.on_segment
    def segment_observed(job, accepted):
        segments.append({"segment_id": job.segment_id, "channel": job.channel,
            "start_sample": job.start_sample, "end_sample": job.end_sample,
            "audio_seconds": job.samples.size / 16000, "accepted": accepted,
            "float32_sha256": hashlib.sha256(job.samples.tobytes()).hexdigest(),
            "ended_by": job.ended_by, "timing": dict(job.timing)})
        if original_segment_observer is not None:
            original_segment_observer(job, accepted)
    controller.pipeline.on_segment = segment_observed
    model_started = time.monotonic()

    class ReplayCapture:
        def __init__(self, on_chunk, on_status):
            self.on_chunk, self.on_status = on_chunk, on_status
            self.done = threading.Event()
            self.stop_event = threading.Event()
            self.worker = None
            captures.append(self)

        def start(self, enabled_channels):
            if enabled_channels != {"system"}:
                raise ValueError("Replay is one playback channel only")
            self.on_status(audio_module.CaptureStatus("system", "running", "冻结录音实时重放", 1))
            def feed():
                origin = time.monotonic()
                report["capture_started_monotonic"] = origin
                try:
                    for start in range(0, len(samples), 512):
                        block = samples[start:start + 512]
                        end = start + len(block)
                        target = origin + end / 16000
                        if self.stop_event.wait(max(0, target - time.monotonic())):
                            return
                        feed_jitter.append(max(0.0, time.monotonic() - target))
                        self.on_chunk(audio_module.AudioChunk("system", block, start, end, target, 1))
                    report["capture_finished_monotonic"] = time.monotonic()
                    self.done.set()
                except Exception:
                    errors.append("capture_feed_error")
            self.worker = threading.Thread(target=feed, name="frozen-audio-replay", daemon=True)
            self.worker.start()

        def stop(self):
            self.stop_event.set()
            if self.worker is not None and self.worker is not threading.current_thread():
                self.worker.join(timeout=1)

    isolate_replay_controller(controller, ReplayCapture)
    original_observer = controller.pipeline.on_event
    def observed(event):
        events.append({**event, "observed_monotonic": time.monotonic()})
        if event["kind"] == "models_ready":
            report["model_load_seconds"] = time.monotonic() - model_started
            report["actual_models"] = {role: {key: info.get(key) for key in (
                "model_id", "model_kind", "model_file", "device", "num_threads", "reasoning_effort", "speed",
                "service_tier", "recognition_language", "recognition_mode", "accuracy_processing")}
                for role, info in event["models"].items() if isinstance(info, dict)}
            run = controller.pipeline._run
            tracker.wrap(run.asr_queue, "get", "asr")
            for label, queue in (("mt", run.mt_queue), ("draft", run.draft_queue)):
                tracker.wrap(queue, "get", label)
                tracker.wrap(queue, "get_batch", label)
        elif event["kind"] == "error":
            errors.append("pipeline_error")
        original_observer(event)
    controller.pipeline.on_event = observed
    window.subtitle_presented.connect(lambda event: presentations.append(dict(event)))
    controller.start()
    window.show()
    deadline = time.monotonic() + args.timeout
    timer = QTimer()
    def poll():
        state = {"visible": window.isVisible(), "minimized": window.isMinimized()}
        if not window_states or any(window_states[-1][key] != value for key, value in state.items()):
            window_states.append({**state, "observed_monotonic": time.monotonic()})
        done = bool(captures and captures[0].done.is_set())
        run = controller.pipeline._run
        snapshot = work_snapshot(run, tracker)
        report["last_work_snapshot"] = snapshot
        if errors or snapshot.get("overloaded"):
            report["status"] = "failed"
            application.quit()
        elif drain.update(feed_done=done, idle=snapshot["idle"], now=time.monotonic()):
            report["status"] = "completed"
            application.quit()
        elif time.monotonic() >= deadline:
            report["status"] = "timeout"
            application.quit()
    timer.timeout.connect(poll)
    timer.start(100)
    application.exec()
    timer.stop()
    report["summary"] = controller.latency.summary()
    report["presented"] = presentations
    report["latency_records"] = list(controller.latency._records)
    report["segments"] = segments
    report["recognition"] = [event for event in events if event["kind"] == "recognition"]
    report["requests"] = [event for event in events if event["kind"] == "translation_request"]
    report["translations"] = [event for event in events if event["kind"] == "translation"]
    report["originals"] = [event for event in events if event["kind"] == "original"]
    report["speech_filter"] = [event for event in events if event["kind"] == "speech_filter"]
    report["audio_processing"] = [event for event in events if event["kind"] == "audio_processing"]
    report["capture_feed_jitter"] = feed_jitter_summary(feed_jitter)
    report["window_states"] = window_states
    report["window_minimized_observed"] = any(row["minimized"] for row in window_states)
    report["window_invisible_observed"] = any(not row["visible"] for row in window_states)
    report["browser_initialized"] = controller.browser_service is not None
    report["danmaku_initialized"] = controller.danmaku is not None
    report["notices"] = [event for event in events if event["kind"] in ("notice", "translation_status", "translation_superseded")]
    final_candidates = [event for event in report["translations"]
                        if event.get("translation_stage", "final") == "final" and not event.get("failed")]
    identity = lambda event: (event.get("channel"), event.get("segment_id"),
                             event.get("revision", 1), event.get("context_generation"))
    painted = {identity(event) for event in report["latency_records"] if event["stage"] == "final"}
    final_identities = {identity(event) for event in final_candidates}
    report["coverage"] = {
        "vad_segments": len(segments), "accepted_segments": sum(row["accepted"] for row in segments),
        "filter_skipped_segments": sum(event.get("action") == "skip" for event in report["speech_filter"]),
        "recognition_segments": len(report["recognition"]),
        "nonempty_recognition_segments": sum(bool(event.get("text", "").strip()) for event in report["recognition"]),
        "original_segment_identities": len({identity(event) for event in report["originals"]}),
        "successful_final_translation_identities": len(final_identities),
        "failed_translation_events": sum(bool(event.get("failed")) for event in report["translations"]),
        "failed_requests": sum(bool(event.get("failed")) for event in report["requests"]),
        "valid_painted_final_identities": len(painted),
        "successful_finals_not_validly_painted": len(final_identities - painted),
        "unpainted_final_identities": [list(key) for key in final_identities - painted]}
    report["errors"] = errors
    window.grab().save(str(args.output / "window.png"))
    controller.close()
    controller.pipeline.shutdown()
    window.close()
    report["settings_unchanged"] = settings.read_bytes() == settings_bytes
    report["protected_user_files_unchanged"] = all(
        (path.read_bytes() if path.is_file() else None) == content
        for path, content in protected_before.items())
    report["configuration_unchanged"] = all(path.is_file() and path.read_bytes() == content
                                            for path, content in watched.items())
    report["configuration_sha256_after"] = {str(path): sha256(path) if path.is_file() else None for path in watched}
    report["input_audio_unchanged"] = all(sha256(metadata["path"]) == metadata["sha256"] for metadata in input_metadata)
    report["capture_threads_stopped"] = all(capture.worker is None or not capture.worker.is_alive() for capture in captures)
    report["finished_at"] = stamp()
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "settings_unchanged": report["settings_unchanged"]}, ensure_ascii=False), flush=True)
    return int(report["status"] != "completed" or not report["configuration_unchanged"]
        or not report["input_audio_unchanged"] or not report["protected_user_files_unchanged"]
        or report["browser_initialized"] or report["danmaku_initialized"])


if __name__ == "__main__":
    raise SystemExit(main())
