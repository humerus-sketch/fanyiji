"""Real accuracy pipeline replay without devices or user profile writes.

The two local official WAVs are replayed on a 32 ms clock through bounded
capture queues, the production VAD/filter/QualityRecognizer/MT/context path.
Python socket guards allow loopback helpers; this is not an OS traffic audit.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import queue
import socket
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fanyiji.audio import AudioChunk, CaptureStatus
from fanyiji.config import AppConfig, SourceConfig
from fanyiji.pipeline import RecognitionJob, SubtitlePipeline, TranslationJob, _Run
from fanyiji.self_test import _memory_snapshot, _read_sample

TAIL_SECONDS = 2.0
BLOCK = 512
RATE = 16000
EXACT_JOIN_CONTROLS = (
    ("ja", "zh", "彼女は今日は。\n歌わないと言いました。", "negation"),
    ("ja", "zh", "森の。\n妖精みたいな名前だと思います。", "forest_fairy"),
    ("zh", "ja", "如果明天不下雨。\n我们就去公园，否则在家。", "condition"),
    ("zh", "ja", "不是十五元。\n是五十元，运费另加八元。", "correction_and_yuan"),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def waveform_digest(samples):
    return hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest()


@contextlib.contextmanager
def loopback_guard(report):
    """Guard Python networking, including DNS; native subprocesses are outside scope."""
    def check(address, method):
        host = address[0] if isinstance(address, tuple) and address else address
        allowed = host in ("127.0.0.1", "::1", None)
        report["network_operations"].append({"method": method, "address": repr(address), "allowed": allowed})
        if not allowed:
            raise RuntimeError("Only loopback Python network operations are allowed during replay")

    originals = {name: getattr(socket.socket, name) for name in ("connect", "connect_ex", "sendto")}
    original_create, original_lookup = socket.create_connection, socket.getaddrinfo

    def connect(sock, address):
        check(address, "connect")
        return originals["connect"](sock, address)

    def connect_ex(sock, address):
        check(address, "connect_ex")
        return originals["connect_ex"](sock, address)

    def sendto(sock, *args):
        check(args[-1], "sendto")
        return originals["sendto"](sock, *args)

    def create(address, *args, **kwargs):
        check(address, "create_connection")
        return original_create(address, *args, **kwargs)

    def lookup(host, *args, **kwargs):
        check(host, "getaddrinfo")
        return original_lookup(host, *args, **kwargs)

    with contextlib.ExitStack() as guards:
        for owner, name, replacement in ((socket.socket, "connect", connect),
                (socket.socket, "connect_ex", connect_ex), (socket.socket, "sendto", sendto),
                (socket, "create_connection", create), (socket, "getaddrinfo", lookup)):
            guards.enter_context(patch.object(owner, name, replacement))
        yield


class WaveCapture:
    """Bounded WAV-backed replacement; no PortAudio imports or device opens."""
    def __init__(self, on_chunk, on_status, waveforms, *, generation=1, realtime=True,
                 queue_size=8, notify=lambda: None):
        self.on_chunk, self.on_status = on_chunk, on_status
        self.waveforms, self.generation = waveforms, generation
        self.realtime, self.queue_size, self.notify = realtime, queue_size, notify
        self.stop_requested, self.finished = threading.Event(), threading.Event()
        self.threads = []
        self.error = None
        self.summary = {"generation": generation, "queue_capacity_blocks_per_source": queue_size,
                        "realtime": realtime, "channels": {}, "errors": []}

    def start(self, enabled_channels=None):
        channels = set(enabled_channels or ())
        if not self.waveforms:
            self.finished.set()
            self.notify()
            return
        epoch = time.monotonic()
        channels = sorted(channels)
        queues = {channel: queue.Queue(maxsize=self.queue_size) for channel in channels}
        for channel in channels:
            self.summary["channels"][channel] = {"source_samples": int(self.waveforms[channel].size),
                "source_float32_sha256": waveform_digest(self.waveforms[channel]), "delivered_samples": 0,
                "delivered_blocks": 0, "queue_peak_blocks": 0, "backpressure_waits": 0}

        def consume(channel):
            try:
                while not self.stop_requested.is_set():
                    try:
                        item = queues[channel].get(timeout=.05)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    if self.stop_requested.is_set():
                        break
                    self.on_chunk(item)
                    self.summary["channels"][channel]["delivered_samples"] += item.samples.size
                    self.summary["channels"][channel]["delivered_blocks"] += 1
            except Exception as exc:
                self._fail(channel, exc)

        def produce(channel):
            try:
                audio = self.waveforms[channel]
                for offset in range(0, audio.size, BLOCK):
                    block = audio[offset:offset + BLOCK]
                    at = epoch + (offset + block.size) / RATE
                    if self.realtime and self.stop_requested.wait(max(0., at - time.monotonic())):
                        break
                    item = AudioChunk(channel, block.copy(), offset, offset + block.size,
                                      at if self.realtime else time.monotonic(), self.generation)
                    while not self.stop_requested.is_set():
                        try:
                            queues[channel].put(item, timeout=.01)
                            row = self.summary["channels"][channel]
                            row["queue_peak_blocks"] = max(row["queue_peak_blocks"], queues[channel].qsize())
                            break
                        except queue.Full:
                            self.summary["channels"][channel]["backpressure_waits"] += 1
                while not self.stop_requested.is_set():
                    try:
                        queues[channel].put(None, timeout=.05)
                        break
                    except queue.Full:
                        pass
            except Exception as exc:
                self._fail(channel, exc)

        for channel in channels:
            for label, target in (("consumer", consume), ("producer", produce)):
                thread = threading.Thread(target=target, args=(channel,),
                    name=f"wav-{label}-{self.generation}-{channel}", daemon=True)
                self.threads.append(thread)
                thread.start()

        def finish():
            for thread in tuple(self.threads):
                thread.join()
            self.finished.set()
            self.notify()

        threading.Thread(target=finish, name=f"wav-finish-{self.generation}", daemon=True).start()

    def _fail(self, channel, exc):
        self.error = f"{type(exc).__name__}: {exc}"
        self.summary["errors"].append(self.error)
        self.stop_requested.set()
        self.on_status(CaptureStatus(channel, "error", self.error, self.generation))
        self.notify()

    def stop(self):
        self.stop_requested.set()
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=.5)


def verify(models: Path, output: Path, timeout: float = 90) -> int:
    started = time.monotonic()
    deadline = started + timeout
    report = {"status": "running", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "audio_capture": False, "audio_backend": "bounded_wav_32ms_clock",
        "network_guard_scope": "Python socket entrypoints only; permits loopback helper; not an OS traffic audit",
        "measurement_scope": "Official short WAV replay and synthetic text controls; no live hardware or accuracy-rate claim",
        "sample_rate_hz": RATE, "block_samples": BLOCK, "tail_silence_seconds": TAIL_SECONDS,
        "wait_budget_seconds": timeout, "network_operations": [], "events": [], "segments": [],
        "filtered_segments": [], "level_event_counts": {}, "phases": [], "exact_join_controls": [],
        "memory_before": _memory_snapshot(), "concurrent_build_scope": "Root instructed builds to wait for replay"}
    condition = threading.Condition()
    release = threading.Event()
    captures, runs, processes = [], [], {}
    pipeline = None
    watched = [models / "asr/backend.json", models / "translation/backend.json", models.parent / "settings.json"]
    watched_before = {str(path): digest(path) for path in watched if path.is_file()}

    def save_progress():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".progress.json").write_text(json.dumps({
            "status": report["status"], "phase": report.get("active_phase"),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "event_counts": dict(Counter(item["kind"] for item in report["events"]))}, indent=2), encoding="utf-8")

    def notify():
        with condition:
            condition.notify_all()

    def capture_process():
        if pipeline is not None and pipeline._translator is not None:
            process = getattr(pipeline._translator, "_process", None)
            if process is not None:
                processes[process.pid] = process

    def event_received(event):
        with condition:
            if event["kind"] == "level":
                key = f"{event['session']}:{event['channel']}"
                report["level_event_counts"][key] = report["level_event_counts"].get(key, 0) + 1
            else:
                report["events"].append({**event, "observed_seconds": time.monotonic() - started})
                capture_process()
            condition.notify_all()

    def wait_until(predicate, description):
        with condition:
            while not predicate():
                errors = [event for event in report["events"] if event["kind"] == "error"]
                if errors:
                    raise RuntimeError(errors[-1]["text"])
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(description)
                condition.wait(min(left, .1))

    def phase(name):
        report["active_phase"] = name
        report["phases"].append({"name": name, "started_seconds": time.monotonic() - started})
        save_progress()

    def drain(run):
        run.asr_queue.close()
        workers = [thread for thread in run.threads if thread.name in ("recognition", "translation")]
        wait_until(lambda: all(not thread.is_alive() for thread in workers), "Pipeline drain timed out")

    try:
        samples, descriptors = {}, {}
        for channel, language in (("mic", "zh"), ("system", "ja")):
            path = models / "self_test" / f"{language}.wav"
            if not path.is_file():
                path = ROOT / "models/self_test" / f"{language}.wav"
            samples[channel], descriptors[channel] = _read_sample(path)
        report["sources"] = descriptors
        from fanyiji.speech_filter import SpeechOnlyFilter, MODEL_FILENAME, LABELS_FILENAME
        candidate = SpeechOnlyFilter(models)
        filtering = all((candidate.models_dir / name).is_file() for name in (MODEL_FILENAME, LABELS_FILENAME))
        candidate.close()
        report["speech_filter_enabled"] = filtering
        report["speech_filter_asset_policy"] = "Real filter enabled only when both explicitly supplied local assets exist; invalid assets fail preload"
        both = AppConfig(speech_only=filtering)
        system_only = AppConfig(microphone=SourceConfig(enabled=False), speech_only=filtering)
        mode = {"feed": True, "repeat": 2}

        def factory(on_chunk, on_status, **kwargs):
            waveforms = {}
            if mode["feed"]:
                silence = np.zeros(round(TAIL_SECONDS * RATE), dtype=np.float32)
                for channel, audio in samples.items():
                    pieces = []
                    for _ in range(mode["repeat"]):
                        pieces.extend((audio, silence))
                    waveforms[channel] = np.concatenate(pieces)
            capture = WaveCapture(on_chunk, on_status, waveforms, generation=len(captures) + 1, notify=notify)
            captures.append(capture)
            notify()
            return capture

        with loopback_guard(report), patch("fanyiji.audio.AudioCaptureManager", factory):
            pipeline = SubtitlePipeline(models, event_received)
            pipeline.on_segment = lambda job, accepted: report["segments"].append({
                "session": int(job.segment_id.split(":", 1)[0]), "segment_id": job.segment_id,
                "channel": job.channel, "accepted": accepted, "start_sample": job.start_sample,
                "end_sample": job.end_sample, "ended_by": job.ended_by,
                "source_float32_sha256": waveform_digest(job.samples),
                "duration_seconds": job.samples.size / RATE, "ended_at": job.ended_at})
            pipeline.on_filtered = lambda job, decision: report["filtered_segments"].append({
                "segment_id": job.segment_id, "channel": job.channel,
                "filtered_float32_sha256": waveform_digest(decision.samples), "metadata": decision.metadata})
            phase("dual_channel_real_vad_filter_quality_translation_context")
            pipeline.start(both, session=101)
            first = pipeline._run
            runs.append(first)
            wait_until(lambda: bool(captures), "Initial models/capture were not ready")
            wait_until(captures[0].finished.is_set, "Initial two-channel WAV feed timed out")
            if captures[0].error:
                raise RuntimeError(captures[0].error)
            drain(first)
            dual = [item for item in report["events"] if item["session"] == 101
                    and item["kind"] in ("original", "translation")]
            for channel in ("mic", "system"):
                for kind in ("original", "translation"):
                    if not any(item.get("channel") == channel and item["kind"] == kind
                               and item.get("text", "").strip() and item["text"] != "[此片段翻译失败]" for item in dual):
                        raise RuntimeError(f"No valid {kind} output for {channel}")
            report["dual_channel_outputs"] = dual
            report["context_exercised_channels"] = sorted({item["channel"] for item in dual
                if item["kind"] == "translation" and item.get("context")})
            if pipeline._translator.backend_info.get("supports_source_context") and not report["context_exercised_channels"]:
                raise RuntimeError("Repeated WAV replay did not exercise the source context path")
            if first.overloaded:
                raise RuntimeError("Initial pipeline overloaded")
            capture_process()
            pipeline.stop()

            phase("pause_cancels_delayed_real_asr_and_translation_results")
            mode["feed"] = False
            asr_ready, mt_ready = threading.Event(), threading.Event()
            computed = {}
            real_recognize = pipeline._recognizer.recognize
            real_translate = pipeline._translator.translate

            def delayed_recognize(audio, language):
                result = real_recognize(audio, language)
                computed["original"] = result
                asr_ready.set()
                notify()
                if not release.wait(max(0., deadline - time.monotonic())):
                    raise TimeoutError("Canceled ASR return gate timed out")
                return result

            def delayed_translate(text, source, target, *, context=""):
                result = real_translate(text, source, target, context=context)
                computed["translation"] = result
                mt_ready.set()
                notify()
                if not release.wait(max(0., deadline - time.monotonic())):
                    raise TimeoutError("Canceled translation return gate timed out")
                return result

            with patch.object(pipeline._recognizer, "recognize", delayed_recognize), patch.object(
                    pipeline._translator, "translate", delayed_translate):
                pipeline.start(both, session=202)
                old = pipeline._run
                runs.append(old)
                wait_until(lambda: len(captures) >= 2, "Controlled cancellation session did not initialize")
                text = next(item["text"] for item in dual if item["kind"] == "original" and item["channel"] == "mic")
                duration = samples["mic"].size / RATE
                if not old.asr_queue.put("mic", RecognitionJob("mic", "202:mic:delayed-asr", samples["mic"],
                        time.monotonic(), ("zh", "ja")), duration):
                    raise RuntimeError("Cancellation ASR job was not accepted")
                if not old.mt_queue.put("mic", TranslationJob("mic", "202:mic:delayed-mt", text,
                        time.monotonic(), ("zh", "ja"), duration), duration):
                    raise RuntimeError("Cancellation MT job was not accepted")
                wait_until(lambda: asr_ready.is_set() and mt_ready.is_set(), "Actual cancellation probe native results did not finish")
                pipeline.stop()
                canceled_at = time.monotonic() - started
                release.set()
                wait_until(lambda: all(not thread.is_alive() for thread in old.threads), "Canceled workers did not terminate")
                late = [item for item in report["events"] if item["session"] == 202 and item["observed_seconds"] >= canceled_at]
                report["pause_cancellation"] = {"session": 202, "cancel_observed_seconds": canceled_at,
                    "method": "Hold already computed real ASR/MT returns, stop session, release returns",
                    "computed_results": computed, "late_old_events": late, "old_workers_finished": True,
                    "resident_asr_after_pause": pipeline._recognizer.backend_info,
                    "resident_translation_after_pause": pipeline._translator.backend_info}
                if late:
                    raise RuntimeError("Canceled session published late events")

            phase("resume_system_only_real_wav")
            mode.update(feed=True, repeat=1)
            pipeline.start(system_only, session=303)
            resumed = pipeline._run
            runs.append(resumed)
            wait_until(lambda: len(captures) >= 3, "System-only recovery did not initialize")
            wait_until(captures[2].finished.is_set, "System-only recovery WAV feed timed out")
            if captures[2].error:
                raise RuntimeError(captures[2].error)
            drain(resumed)
            recovered = [item for item in report["events"] if item["session"] == 303 and item["kind"] == "translation"]
            if not any(item.get("channel") == "system" and item.get("text", "").strip()
                       and item["text"] != "[此片段翻译失败]" for item in recovered):
                raise RuntimeError("System-only resumed session did not translate")
            if any(item.get("channel") == "mic" for item in report["events"] if item["session"] == 303):
                raise RuntimeError("Disabled microphone produced resumed session events")
            report["system_only_recovery"] = {"session": 303, "outputs": recovered,
                "microphone_events": 0, "overloaded": resumed.overloaded}
            pipeline.stop()
            if resumed.overloaded:
                raise RuntimeError("Resumed session overloaded")

            phase("exact_newline_text_join_controls")
            for source, target, text, label in EXACT_JOIN_CONTROLS:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Text control wait budget exhausted")
                tick = time.monotonic()
                translated = pipeline._translator.translate(text, source, target, context="")
                report["exact_join_controls"].append({"source_language": source, "target_language": target,
                    "text": text, "translation": translated, "elapsed_seconds": time.monotonic() - tick,
                    "control": label, "source_role": "Synthetic exact newline text assembly; not human audio ground truth",
                    "semantic_review": "pending_root_review; nonempty output is not an accuracy claim"})
                capture_process()
            report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        release.set()
        if pipeline is not None:
            capture_process()
            try:
                pipeline.shutdown()
            except Exception as exc:
                report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
                report["status"] = "failed"
            report["asr_after_shutdown"] = pipeline._recognizer.backend_info if pipeline._recognizer is not None else None
            report["translation_after_shutdown"] = pipeline._translator.backend_info if pipeline._translator is not None else None
            report["filter_after_shutdown"] = pipeline._speech_filter.backend_info if pipeline._speech_filter is not None else None
        report["owned_helper_processes"] = [{"pid": pid, "returncode": process.poll(), "exited": process.poll() is not None}
                                              for pid, process in processes.items()]
        report["captures"] = [capture.summary for capture in captures]
        report["overloaded_sessions"] = [run.session for run in runs if run.overloaded]
        report["blocked_external_python_network_operations"] = sum(not item["allowed"] for item in report["network_operations"])
        report["profile_sha256_before"] = watched_before
        report["profile_sha256_after"] = {str(path): digest(path) for path in watched if path.is_file()}
        report["profiles_unchanged"] = report["profile_sha256_before"] == report["profile_sha256_after"]
        report["source_sha256_after"] = {channel: digest(Path(item["sample_path"])) for channel, item in report.get("sources", {}).items()}
        report["sources_unchanged"] = all(report["source_sha256_after"].get(channel) == item["sample_sha256"]
                                          for channel, item in report.get("sources", {}).items())
        report["total_seconds"] = time.monotonic() - started
        report["memory_after"] = _memory_snapshot()
        checks = {"profiles_unchanged": report["profiles_unchanged"], "sources_unchanged": report["sources_unchanged"],
            "owned_helpers_exited": all(item["exited"] for item in report["owned_helper_processes"]),
            "asr_instances_zero": not report.get("asr_after_shutdown") or report["asr_after_shutdown"].get("recognizer_instances") == 0,
            "no_external_python_network": report["blocked_external_python_network_operations"] == 0,
            "no_overload": not report["overloaded_sessions"],
            "capture_threads_exited": all(not thread.is_alive() for capture in captures for thread in capture.threads)}
        report["final_checks"] = checks
        if not all(checks.values()):
            report["status"] = "failed"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        save_progress()
        print(json.dumps({"status": report["status"], "report": str(output), "total_seconds": round(report["total_seconds"], 3),
                          "final_checks": checks, **({"error": report["error"]} if "error" in report else {})}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


def mock_check():
    """Three small harness checks; no weights, native inference, devices or network."""
    report = {"network_operations": []}
    calls = []
    with patch.object(socket.socket, "connect", lambda sock, address: calls.append(address)), loopback_guard(report):
        sock = socket.socket()
        try:
            sock.connect(("127.0.0.1", 12345))
            try:
                sock.connect(("192.0.2.1", 443))
            except RuntimeError:
                pass
            else:
                raise AssertionError("External connection was not blocked")
        finally:
            sock.close()
    assert calls == [("127.0.0.1", 12345)] and report["network_operations"][-1]["allowed"] is False
    delivered = []
    audio = {"mic": np.linspace(-.2, .2, 8193, dtype=np.float32),
             "system": np.linspace(.2, -.2, 4097, dtype=np.float32)}

    def callback(chunk):
        time.sleep(.001)
        delivered.append(chunk)

    capture = WaveCapture(callback, lambda _: None, audio, realtime=False, queue_size=2)
    capture.start({"mic", "system"})
    assert capture.finished.wait(3) and capture.error is None
    capture.stop()
    for channel in audio:
        rows = [item for item in delivered if item.channel == channel]
        assert rows[0].start_sample == 0 and rows[-1].end_sample == audio[channel].size
        assert all(left.end_sample == right.start_sample for left, right in zip(rows, rows[1:]))
        np.testing.assert_array_equal(np.concatenate([item.samples for item in rows]), audio[channel])
        assert capture.summary["channels"][channel]["queue_peak_blocks"] <= 2
    events = []
    pipeline = SubtitlePipeline(ROOT / "models", events.append)
    run = _Run(session=7, config=AppConfig(speech_only=False))
    run.cancel.set()
    pipeline._emit(run, "translation", text="old")
    assert events == []
    assert all("\n" in text for _, _, text, _ in EXACT_JOIN_CONTROLS)
    print(json.dumps({"status": "passed", "mock_checks": 3,
                      "scope": "Harness guards, bounded sample clocks, canceled event gate; no native inference"}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/accuracy_pipeline_replay.json")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--mock-check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 90:
        parser.error("--timeout must be between 1 and 90 seconds")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return mock_check() if args.mock_check else verify(args.models.resolve(), args.output.resolve(), args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
