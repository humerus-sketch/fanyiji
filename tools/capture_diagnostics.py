"""Explicitly requested local capture; production app stays untouched.

Continuous capture is opt-in and stops with a stop file, application exit, or
settings change. Intermediate results are atomic snapshots of finalized jobs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
import wave
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from fanyiji.config import load_config
from fanyiji.paths import data_root, models_root
from fanyiji.pipeline import FairQueue, RecognitionJob, SubtitlePipeline

TZ = timezone(timedelta(hours=8))


def timestamp() -> str:
    return datetime.now(TZ).isoformat(timespec="milliseconds")


def pcm16(samples) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def wav_file(path: Path):
    stream = wave.open(str(path), "wb")
    stream.setnchannels(1)
    stream.setsampwidth(2)
    stream.setframerate(16_000)
    return stream


class Recorder:
    def __init__(self, directory: Path, channels: set[str]):
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "segments").mkdir()
        self.directory = directory
        self.lock = threading.RLock()
        self.active = True
        self.started = time.monotonic()
        self.ready = threading.Event()
        self.events: list[dict] = []
        self.samples: dict[str, dict] = {}
        self.cleanup_errors: list[str] = []
        self.model_info = {"asr": "SenseVoiceSmall ONNX INT8", "translation": "not_loaded"}
        self.raw_counts = {channel: 0 for channel in channels}
        self.raw = {channel: wav_file(directory / f"{channel}-continuous.wav") for channel in channels}
        self.event_stream = (directory / "events.jsonl").open("w", encoding="utf-8")
        self.segment_stream = (directory / "segments.jsonl").open("w", encoding="utf-8")

    def on_chunk(self, chunk):
        with self.lock:
            if self.active and chunk.channel in self.raw:
                self.raw[chunk.channel].writeframesraw(pcm16(chunk.samples))
                self.raw_counts[chunk.channel] += chunk.samples.size
                self.ready.set()

    def event(self, event):
        if event["kind"] == "level":
            return
        with self.lock:
            if not self.active:
                return
            record = {**event, "observed_at": timestamp(), "elapsed_seconds": round(time.monotonic() - self.started, 3)}
            self.events.append(record)
            self.event_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.event_stream.flush()
            if event["kind"] in ("original", "translation", "error", "source_status"):
                print(json.dumps(record, ensure_ascii=False), flush=True)

    def segment(self, job, accepted):
        with self.lock:
            if not self.active:
                return
            filename = job.segment_id.replace(":", "-") + ".wav"
            path = self.directory / "segments" / filename
            with wav_file(path) as stream:
                stream.writeframes(pcm16(job.samples))
            record = {
                "id": job.segment_id, "channel": job.channel,
                "source_lang": job.direction[0], "target_lang": job.direction[1],
                "audio_path": str(path.resolve()), "audio_duration_seconds": job.samples.size / 16000,
                "audio_end_at_monotonic": job.ended_at, "observed_at": timestamp(), "accepted": accepted,
            }
            self.samples[job.segment_id] = record
            self.segment_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.segment_stream.flush()

    def snapshot(self, config, status):
        with self.lock:
            for event in self.events:
                item = self.samples.get(event.get("segment_id"))
                if item is None:
                    continue
                if event["kind"] == "original":
                    item["production_text"] = event["text"]
                elif event["kind"] == "translation":
                    item["production_translation"] = event["text"]
                    item["pipeline_latency_seconds"] = event.get("latency")
            report = {
                "status": status, "updated_at": timestamp(), "settings_snapshot": asdict(config),
                "collection_scope": "Local sidecar with identical source pipeline, not an export from the already running EXE",
                "recording_uploaded": False, "raw_audio_seconds": {c: n / 16000 for c, n in self.raw_counts.items()},
                "models": dict(self.model_info),
                "cleanup_errors": list(self.cleanup_errors),
                "tail_finalized": False,
                "samples": list(self.samples.values()),
            }
            if status != "collecting":
                report["finished_at"] = report["updated_at"]
            temporary = self.directory / "results.json.tmp"
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.directory / "results.json")
            print(json.dumps({"capture_status": status, "directory": str(self.directory.resolve()),
                              "segments": len(self.samples), "raw_audio_seconds": report["raw_audio_seconds"]}, ensure_ascii=False), flush=True)

    def finish(self, config, status):
        with self.lock:
            self.active = False
            for stream in (*self.raw.values(), self.event_stream, self.segment_stream):
                try:
                    stream.close()
                except Exception as exc:
                    self.cleanup_errors.append(type(exc).__name__)
            if self.cleanup_errors:
                status = "failed"
            self.snapshot(config, status)
            return status


class ApplicationWatch:
    """Keep capture tied to this exact Windows process, without PID reuse."""

    def __init__(self, pid):
        self.handle = None
        if pid is None:
            return
        if os.name != "nt":
            raise RuntimeError("--application-pid 只支持 Windows")
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x00100000, False, pid)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "无法监测字幕进程，未开始采集")

    def running(self):
        return self.handle is None or self.kernel.WaitForSingleObject(self.handle, 0) == 258

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--application-pid", type=int)
    parser.add_argument("--snapshot-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.duration <= 180:
        raise SystemExit("诊断采集时长必须在 1～180 秒以内。")
    if not 5 <= args.snapshot_seconds <= 60:
        raise SystemExit("诊断快照间隔必须在 5～60 秒以内。")
    config = load_config(data_root() / "settings.json")
    if config is None or not config.enabled_channels:
        raise SystemExit("当前没有已保存且启用的音频源，不擅自开启其他来源。")
    stop_file = (args.stop_file or args.output / "STOP").resolve()
    if stop_file.exists():
        raise SystemExit("停止标记已存在，未开始采集。")
    watch = ApplicationWatch(args.application_pid)
    recorder = Recorder(args.output.resolve(), config.enabled_channels)

    class RecordingPipeline(SubtitlePipeline):
        def _on_chunk(self, run, chunk):
            if run.cancel.is_set():
                return
            recorder.on_chunk(chunk)
            super()._on_chunk(run, chunk)

    put = FairQueue.put

    def recorded_put(queue, channel, item, seconds):
        accepted = put(queue, channel, item, seconds)
        if isinstance(item, RecognitionJob):
            recorder.segment(item, accepted)
        return accepted

    FairQueue.put = recorded_put
    pipeline = RecordingPipeline(models_root(), recorder.event)
    status = "completed"

    def stop_reason():
        if stop_file.exists():
            return "stopped_by_request"
        if not watch.running():
            return "application_closed"
        if load_config(data_root() / "settings.json") != config:
            return "settings_changed"
        run = pipeline._run
        if run is not None and run.overloaded:
            return "overloaded"
        return None

    print(json.dumps({"capture_requested_at": timestamp(), "seconds": args.duration,
                      "continuous": args.continuous, "stop_file": str(stop_file),
                      "channels": sorted(config.enabled_channels), "output": str(recorder.directory)}, ensure_ascii=False), flush=True)
    try:
        reason = stop_reason()
        if reason is None:
            pipeline.start(config, int(time.time()))
            ready_deadline = time.monotonic() + 20
            while not recorder.ready.is_set() and time.monotonic() < ready_deadline:
                reason = stop_reason()
                if reason is not None:
                    break
                recorder.ready.wait(timeout=0.2)
            reason = reason or stop_reason()
        if reason is not None:
            status = reason
        elif not recorder.ready.is_set():
            status = "no_audio_received"
        else:
            print(json.dumps({"capture_started_at": timestamp(), "seconds": args.duration,
                              "continuous": args.continuous}, ensure_ascii=False), flush=True)
            deadline = float("inf") if args.continuous else time.monotonic() + args.duration
            snapshot_at = time.monotonic()
            while time.monotonic() < deadline:
                reason = stop_reason()
                if reason is not None:
                    status = reason
                    break
                if time.monotonic() >= snapshot_at:
                    translator = pipeline._translator
                    info = getattr(translator, "backend_info", None)
                    recorder.model_info["translation"] = info if isinstance(info, dict) else "M2M100 418M CPU INT8"
                    recorder.snapshot(config, "collecting")
                    snapshot_at = time.monotonic() + args.snapshot_seconds
                time.sleep(0.2)
        run = pipeline._run
        if run is not None and run.capture is not None:
            run.capture.stop()
            run.asr_queue.close()
            for thread in run.threads:
                if thread.name in ("recognition", "translation"):
                    thread.join(timeout=15)
        # The final unfinished utterance remains available in continuous WAV;
        # do not pretend it was displayed by the normal silence-endpoint path.
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        status = "failed"
        recorder.event({"kind": "error", "text": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            pipeline.shutdown()
        except Exception as exc:
            status = "failed"
            recorder.event({"kind": "error", "text": f"清理采集失败：{type(exc).__name__}"})
        finally:
            FairQueue.put = put
            try:
                status = recorder.finish(config, status)
            finally:
                watch.close()
    return 0 if status in ("completed", "stopped_by_request", "application_closed", "settings_changed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
