"""Read-only frozen-audio ASR thread comparison; never opens audio devices."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fanyiji.asr_engine import ProfileRecognizerEngine


def audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Frozen input must be 16 kHz mono PCM16")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "artifacts/diagnostics/all-model-evaluation-20261009/corpus/audio")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/e2e-latency-20261011/asr-benchmark.json")
    args = parser.parse_args()
    inputs = []
    for name in ("recent-ja-0001", "recent-ja-0199", "recent-ja-1001"):
        path = args.corpus / (name + ".wav")
        inputs.append((name, path, audio(path)))
    profile_path = args.models / "backend.json"
    original_profile = profile_path.read_bytes()
    profile = json.loads(original_profile.decode("utf-8-sig"))
    if profile.get("backend") != "qwen3_asr" or profile.get("model_id") != "qwen3-asr-1.7b-int8":
        raise ValueError("This comparison requires the currently selected Qwen 1.7B INT8 profile")
    report = {"started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "status": "running", "scope": "Local frozen-audio native ASR only; no capture/VAD/filter/MT/UI/network",
              "audio_devices_opened": False, "downloaded": False, "profile_modified": False,
              "language_policy": "auto", "production_max_threads": 4,
              "evaluation_only_thread_override": True, "repetitions_per_audio": 2,
              "model_directory": str(args.models.resolve()), "profile_sha256": hashlib.sha256(original_profile).hexdigest(),
              "inputs": [{"id": name, "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "audio_seconds": len(samples) / 16000} for name, path, samples in inputs], "variants": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for threads in (4, 6, 8):
        began = time.perf_counter()
        engine = ProfileRecognizerEngine(args.models, num_threads=4, config=profile)
        # The production config rejects >4. Change only this isolated instance
        # before native construction to measure a possible future option.
        engine.num_threads = threads
        variant = {"threads": threads, "rows": []}
        try:
            engine._recognizer("auto")
            variant["cold_load_seconds"] = time.perf_counter() - began
            warm_started = time.perf_counter()
            warmup = engine.recognize_detailed(inputs[0][2], "auto")
            variant["first_decode_seconds"] = time.perf_counter() - warm_started
            variant["first_decode_text"] = warmup.text
            for repetition in (1, 2):
                for name, _, samples in inputs:
                    started = time.perf_counter()
                    result = engine.recognize_detailed(samples, "auto")
                    variant["rows"].append({"id": name, "repetition": repetition,
                        "seconds": time.perf_counter() - started, "text": result.text,
                        "detected_language": result.detected_language, "detection_source": result.detection_source})
            variant["warm_median_seconds"] = statistics.median(row["seconds"] for row in variant["rows"])
            variant["warm_total_seconds"] = sum(row["seconds"] for row in variant["rows"])
            variant["per_audio_median_seconds"] = {
                name: statistics.median(row["seconds"] for row in variant["rows"] if row["id"] == name)
                for name, _, _ in inputs}
        finally:
            engine.close()
            variant["closed_instances"] = len(engine._recognizers)
            report["variants"].append(variant)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: variant[key] for key in ("threads", "cold_load_seconds", "first_decode_seconds", "warm_median_seconds", "warm_total_seconds", "per_audio_median_seconds")}, ensure_ascii=False), flush=True)
    baseline = report["variants"][0]
    baseline_text = {(row["id"], row["repetition"]): row["text"] for row in baseline["rows"]}
    for variant in report["variants"]:
        variant["all_text_identical_to_cpu4"] = all(row["text"] == baseline_text[(row["id"], row["repetition"])] for row in variant["rows"])
        variant["warm_total_ratio_to_cpu4"] = variant["warm_total_seconds"] / baseline["warm_total_seconds"]
    if profile_path.read_bytes() != original_profile:
        raise RuntimeError("Live profile changed concurrently; comparison is not valid")
    report.update(status="completed", completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  interpretation="Three saved utterances, no human transcript reference, sequential order; thread speed is not accuracy or live end-to-end proof")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
