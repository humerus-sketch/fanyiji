"""Exercise the real offline pipeline with WAV-backed capture and no devices.

Run with the project's runtime Python. Uses real Silero/SenseVoice/M2M100,
32 ms sample clocks, and 1.6 seconds of silence for natural VAD finalization.
The cancellation check delays already computed native results until after a
session switch. Python socket entrypoints are blocked; this is not an OS
network capture or an audio-hardware test.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import socket
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fanyiji.audio import AudioChunk, CaptureStatus
from fanyiji.config import AppConfig
from fanyiji.pipeline import RecognitionJob, SubtitlePipeline, TranslationJob
from fanyiji.self_test import _memory_snapshot, _read_sample


def verify(models: Path, output: Path) -> int:
    started = time.monotonic()
    config = AppConfig(speech_only=(models / "speech_filter/model.int8.onnx").is_file())
    report = {
        "audio_capture": False, "audio_backend": "fake_wav_sample_clock",
        "network_guard_scope": "Python socket entrypoints; not an OS-level traffic audit",
        "measurement_scope": "Two official file samples; no P50/P95 or live-device claim",
        "window_samples": 512, "sample_rate_hz": 16_000, "tail_silence_seconds": 1.6,
        "network_attempts": [], "events": [], "level_event_counts": {},
        "memory_before": _memory_snapshot(),
        "speech_filter_enabled": config.speech_only,
        "speech_filter_scope": "Uses classifier only if these explicitly supplied model assets include it",
    }
    condition = threading.Condition()
    blocked_release = threading.Event()
    pipeline = None

    def event_received(event):
        with condition:
            if event["kind"] == "level":
                channel = event["channel"]
                report["level_event_counts"][channel] = report["level_event_counts"].get(channel, 0) + 1
            else:
                report["events"].append({**event, "observed_seconds": round(time.monotonic() - started, 6)})
            condition.notify_all()

    def wait_until(predicate, description, timeout=20.0):
        deadline = time.monotonic() + timeout
        with condition:
            while not predicate():
                errors = [event for event in report["events"] if event["kind"] == "error"]
                if errors:
                    raise RuntimeError(errors[-1]["text"])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(description)
                condition.wait(min(remaining, 0.1))

    def block_network(method):
        def blocked(*args, **kwargs):
            address = args[-1] if args else kwargs
            report["network_attempts"].append({"method": method, "address": repr(address)[:200]})
            raise RuntimeError(f"Network is forbidden during offline verification: {method}")
        return blocked

    try:
        samples = {}
        descriptors = {}
        for channel, language in (("mic", "zh"), ("system", "ja")):
            samples[channel], descriptors[channel] = _read_sample(models / "self_test" / f"{language}.wav")
        report["samples"] = descriptors

        class WaveCapture:
            instances = []
            feed_enabled = True

            def __init__(self, on_chunk, on_status, **kwargs):
                self.on_chunk = on_chunk
                self.on_status = on_status
                self.stop_requested = threading.Event()
                self.finished = threading.Event()
                self.thread = None
                self.error = None
                self.feed = self.feed_enabled
                self.instances.append(self)

            def start(self, enabled_channels=None):
                channels = set(enabled_channels or ())
                if not self.feed:
                    self.finished.set()
                    return
                self.thread = threading.Thread(target=self._feed, args=(channels,), name="fake-wav-capture", daemon=True)
                self.thread.start()

            def _feed(self, channels):
                try:
                    waveforms = {
                        channel: np.concatenate((samples[channel], np.zeros(25_600, dtype=np.float32)))
                        for channel in channels
                    }
                    epoch = time.monotonic()
                    for offset in range(0, max(audio.size for audio in waveforms.values()), 512):
                        deadline = epoch + (offset + 512) / 16_000
                        if self.stop_requested.wait(max(0.0, deadline - time.monotonic())):
                            break
                        for channel in ("mic", "system"):
                            if channel not in waveforms or offset >= waveforms[channel].size:
                                continue
                            block = waveforms[channel][offset:offset + 512]
                            if block.size < 512:
                                block = np.pad(block, (0, 512 - block.size)).astype(np.float32)
                            self.on_chunk(AudioChunk(
                                channel=channel, samples=block, start_sample=offset,
                                end_sample=offset + 512, captured_at=deadline, generation=1,
                            ))
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.on_status(CaptureStatus("mic", "error", self.error, 1))
                finally:
                    self.finished.set()
                    with condition:
                        condition.notify_all()

            def stop(self):
                self.stop_requested.set()
                if self.thread is not None and self.thread is not threading.current_thread():
                    self.thread.join(timeout=2.0)

        with contextlib.ExitStack() as guards:
            socket_guards = [
                (socket.socket, "connect"), (socket.socket, "connect_ex"),
                (socket.socket, "sendto"), (socket, "create_connection"), (socket, "getaddrinfo"),
            ]
            for owner, name in socket_guards:
                guards.enter_context(patch.object(owner, name, block_network(name)))
            report["blocked_network_entrypoints"] = [name for _, name in socket_guards]
            guards.enter_context(patch("fanyiji.audio.AudioCaptureManager", WaveCapture))
            pipeline = SubtitlePipeline(models, event_received)
            pipeline.start(config, session=101)
            wait_until(lambda: bool(WaveCapture.instances), "The WAV capture manager was not created")
            first_capture = WaveCapture.instances[0]
            wait_until(first_capture.finished.is_set, "The timed WAV feed did not finish")
            if first_capture.error:
                raise RuntimeError(first_capture.error)

            def has_both_outputs():
                for channel in ("mic", "system"):
                    kinds = {
                        event["kind"] for event in report["events"]
                        if event["session"] == 101 and event.get("channel") == channel
                        and event.get("text", "").strip()
                        and event.get("text") != "[此片段翻译失败]"
                    }
                    if not {"original", "translation"} <= kinds:
                        return False
                return True

            wait_until(has_both_outputs, "Both channels did not produce original and translation text")
            report["dual_audio_outputs"] = {channel: [] for channel in ("mic", "system")}
            for event in report["events"]:
                if event["session"] == 101 and event["kind"] in {"original", "translation"}:
                    report["dual_audio_outputs"][event["channel"]].append(event)
            pipeline.stop()

            # Exercise cancellation with two real native results waiting to return.
            # The previous WAV feed already validated VAD and normal queue flow.
            WaveCapture.feed_enabled = False
            asr_waiting = threading.Event()
            mt_waiting = threading.Event()
            asr_returned = threading.Event()
            mt_returned = threading.Event()
            computed = {}
            real_recognize = pipeline._recognizer.recognize
            real_translate = pipeline._translator.translate

            def delayed_recognize(audio, language):
                computed["original"] = real_recognize(audio, language)
                asr_waiting.set()
                if not blocked_release.wait(20):
                    raise TimeoutError("The cancellation ASR gate was not released")
                asr_returned.set()
                return computed["original"]

            def delayed_translate(text, source, target):
                computed["translation"] = real_translate(text, source, target)
                mt_waiting.set()
                if not blocked_release.wait(20):
                    raise TimeoutError("The cancellation MT gate was not released")
                mt_returned.set()
                return computed["translation"]

            guards.enter_context(patch.object(pipeline._recognizer, "recognize", delayed_recognize))
            guards.enter_context(patch.object(pipeline._translator, "translate", delayed_translate))
            pipeline.start(config, session=202)
            wait_until(lambda: len(WaveCapture.instances) >= 2, "The controlled old session did not start")
            old_run = pipeline._run
            utterance_text = next(
                event["text"] for event in report["dual_audio_outputs"]["mic"] if event["kind"] == "original"
            )
            duration = samples["mic"].size / 16_000
            if not old_run.asr_queue.put("mic", RecognitionJob(
                "mic", "202:mic:delayed-asr", samples["mic"], time.monotonic(), ("zh", "ja")
            ), duration):
                raise RuntimeError("Failed to enqueue the cancellation ASR job")
            if not old_run.mt_queue.put("mic", TranslationJob(
                "mic", "202:mic:delayed-mt", utterance_text, time.monotonic(), ("zh", "ja"), duration
            ), duration):
                raise RuntimeError("Failed to enqueue the cancellation MT job")
            wait_until(lambda: asr_waiting.is_set() and mt_waiting.is_set(), "Native cancellation results were not computed")
            switch_at = time.monotonic()
            pipeline.start(config, session=303)
            wait_until(lambda: len(WaveCapture.instances) >= 3, "The replacement session did not start")
            blocked_release.set()
            wait_until(lambda: asr_returned.is_set() and mt_returned.is_set(), "The old results were not released")
            for thread in old_run.threads:
                thread.join(timeout=2.0)
            if any(thread.is_alive() for thread in old_run.threads):
                raise RuntimeError("The canceled session workers did not finish")
            late = [
                event for event in report["events"]
                if event["session"] == 202 and event["observed_seconds"] >= switch_at - started
            ]
            report["session_switch"] = {
                "old_session": 202, "new_session": 303,
                "method": "Delay real computed ASR/MT results, switch session, then release",
                "computed_original": computed["original"], "computed_translation": computed["translation"],
                "late_old_session_events": late, "old_workers_finished": True,
            }
            if late:
                raise RuntimeError("The canceled session published late events")
            if report["network_attempts"]:
                raise RuntimeError("The pipeline attempted a guarded network operation")
            report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        blocked_release.set()
        if pipeline is not None:
            pipeline.stop()
        report["total_seconds"] = round(time.monotonic() - started, 6)
        report["memory_after"] = _memory_snapshot()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if sys.stdout is not None:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            print(json.dumps({"status": report["status"], "report": str(output),
                              "network_attempts": len(report["network_attempts"]),
                              "total_seconds": report["total_seconds"], **({"error": report["error"]} if "error" in report else {})},
                             ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "offline_pipeline_verification.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return verify(args.models.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
