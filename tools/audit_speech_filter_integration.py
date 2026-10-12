"""Read-only production-source audit; only the small classifier runs natively."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fanyiji.config import AppConfig
from fanyiji.pipeline import SubtitlePipeline
from fanyiji.self_test import _read_sample
from fanyiji.speech_filter import SpeechOnlyFilter


def audit(models: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"started_at": datetime.now().astimezone().isoformat(), "audio_capture_performed": False,
              "asr_or_translation_inference_performed": False, "network_attempts": 0, "results": []}
    source_files = [ROOT / "src" / "fanyiji" / name for name in
                    ("speech_filter.py", "pipeline.py", "diagnostics.py", "self_test.py")]
    report["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}

    def prohibit_network(event, args):
        if event == "socket.connect":
            report["network_attempts"] += 1
            raise RuntimeError("Network access prohibited in local integration audit")
        if event == "subprocess.Popen":
            raise RuntimeError("Native child process launch prohibited in integration audit")

    sys.addaudithook(prohibit_network)
    gate = SpeechOnlyFilter(models, num_threads=1)
    started = time.perf_counter()
    gate.preload()
    report["preload_seconds"] = time.perf_counter() - started
    try:
        for language in ("zh", "ja"):
            path = models / "self_test" / f"{language}.wav"
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            samples, details = _read_sample(path)
            saved = samples.copy()
            decision = gate.process(samples)
            result = {"language": language, **details, "decision": decision.metadata,
                      "output_equals_original_exactly": np.array_equal(decision.samples, samples),
                      "input_array_unchanged": np.array_equal(samples, saved),
                      "original_wav_sha256_unchanged": hashlib.sha256(path.read_bytes()).hexdigest() == before}
            report["results"].append(result)
            if decision.skip or not result["output_equals_original_exactly"] or not result["input_array_unchanged"]:
                raise RuntimeError(f"{language} self-test speech was altered by the filter")
        report["native_audio_validation"] = "passed"
    finally:
        gate.close()
        gc.collect()
        report["backend_after_close"] = gate.backend_info

    # A deliberately delayed constructor makes the otherwise short startup
    # publication interval reproducible without real ASR/MT/audio work.
    fixture = output.parent / "lifecycle-fixture-models"
    for relative in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = fixture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"mock fixture only")
    constructing, released = threading.Event(), threading.Event()
    classifier = Mock(backend_info={})
    translator = Mock(backend_info={})
    recognizer = Mock(backend_info={})

    def constructor(*args, **kwargs):
        constructing.set()
        if not released.wait(3):
            raise RuntimeError("Delayed classifier constructor fixture timed out")
        return classifier

    pipeline = SubtitlePipeline(fixture, Mock())
    with patch("fanyiji.engines.RecognizerEngine", return_value=recognizer), patch(
            "fanyiji.engines.create_translation_engine", return_value=translator), patch(
            "fanyiji.speech_filter.SpeechOnlyFilter", side_effect=constructor), patch(
            "fanyiji.audio.AudioCaptureManager") as capture:
        pipeline.start(AppConfig(speech_only=True), 1001)
        run = pipeline._run
        if not constructing.wait(2):
            raise RuntimeError("Classifier factory was not reached by the real loader")
        pipeline.shutdown()
        report["shutdown_constructor_publication_audit"] = {
            "shutdown_flag_before_factory_return": pipeline._shutdown,
            "resident_classifier_before_factory_return": pipeline._speech_filter is not None,
        }
        released.set()
        for worker in run.threads:
            worker.join(timeout=2)
        finding = report["shutdown_constructor_publication_audit"]
        finding.update(factory_finished=True, all_loader_threads_finished=all(not worker.is_alive() for worker in run.threads),
                       preload_calls_after_shutdown=classifier.preload.call_count,
                       classifier_close_calls=classifier.close.call_count,
                       classifier_published_after_shutdown=pipeline._speech_filter is classifier,
                       audio_capture_constructed=capture.called,
                       scope="Deterministic mocked delayed factory; no native ASR, MT or capture used")
    report["source_sha256_unchanged"] = all(
        hashlib.sha256(path.read_bytes()).hexdigest() == report["source_sha256"][path.name] for path in source_files)
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["status"] = "completed"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"native_audio_validation": report["native_audio_validation"],
                      "audio_actions": [(row["language"], row["decision"]["action"]) for row in report["results"]],
                      "shutdown_audit": report["shutdown_constructor_publication_audit"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.models.resolve(), args.output.resolve())
