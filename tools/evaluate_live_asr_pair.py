"""Compare frozen Japanese livestream clips using one real ASR per process.

Automatic language selection matches the live application. Model transcripts
are never references; no audio devices or translation services are opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from evaluate_all_asr import load_corpus, model_spec, sha256
from evaluate_asr_candidates import process_memory
from prepare_asr_candidates import ROOT, verify_candidate

sys.path.insert(0, str(ROOT / "src"))
from fanyiji.asr_engine import ProfileRecognizerEngine
from fanyiji.paths import data_root
from reference_whisper import read_wave


def stamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("sensevoice-int8", "qwen3-asr-1.7b-int8"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--cache", type=Path, default=ROOT / "model-cache/asr-evaluation")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Existing report preserved; choose a fresh output path")
    manifest_digest = sha256(args.manifest)
    _, all_samples = load_corpus(args.manifest.resolve(), manifest_digest)
    samples = [item for item in all_samples if item["source_lang"] == "ja"
               and item.get("source_kind") == "saved_application_audio" and not item.get("synthetic")]
    if not samples:
        raise ValueError("No real Japanese livestream clips")
    candidate, directory, profile = model_spec(args.model, args.models.resolve(), args.cache.resolve())
    assets = verify_candidate(candidate, directory)
    settings_path = data_root() / "settings.json"
    settings_digest = sha256(settings_path)
    paths = [ROOT / "src/fanyiji" / name for name in ("asr_engine.py", "engines.py")]
    source_digests = {str(path): sha256(path) for path in paths}
    report = {"status": "running", "started_at": stamp(), "model_id": args.model,
              "model_directory": str(directory), "profile_in_memory_only": profile,
              "verified_assets": assets, "manifest_sha256": manifest_digest,
              "language_policy": "auto", "num_threads": 4, "provider": "cpu",
              "sample_count": len(samples), "ground_truth": False,
              "accuracy_status": "not_scored_no_independent_reference",
              "timing_scope": "Native ASR only, no VAD/filter/MT/UI/audio-device latency",
              "audio_capture_performed": False, "audio_uploaded": False,
              "translation_inference_performed": False, "user_settings_modified": False,
              "source_sha256": source_digests, "samples": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    engine = None
    try:
        load_started = time.perf_counter()
        engine = ProfileRecognizerEngine(directory, num_threads=4, config=profile)
        engine.preload(("auto",))
        report["model_load_seconds"] = time.perf_counter() - load_started
        for index, item in enumerate(samples, 1):
            pcm = np.asarray(read_wave(Path(item["audio_path"])), dtype=np.float32)
            before = hashlib.sha256(pcm.tobytes()).hexdigest()
            began = time.perf_counter()
            result, error = None, None
            try:
                result = engine.recognize_detailed(pcm, "auto")
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
            elapsed = time.perf_counter() - began
            text = result.text if result else None
            row = {"id": item["id"], "source_lang": "ja", "sha256": item["sha256"],
                   "audio_path": item["audio_path"], "source_kind": item["source_kind"],
                   "audio_seconds": len(pcm) / 16000, "processing_seconds": elapsed,
                   "production_text": text, "detected_language": result.detected_language if result else None,
                   "detection_source": result.detection_source if result else None,
                   "error": error, "human_truth": None,
                   "input_array_unchanged": hashlib.sha256(pcm.tobytes()).hexdigest() == before}
            report["samples"].append(row)
            save()
            print(json.dumps({"model": args.model, "sample": index, "total": len(samples),
                              "id": item["id"], "seconds": round(elapsed, 3),
                              "empty": not bool(text), "error": error}, ensure_ascii=False), flush=True)
        times = [row["processing_seconds"] for row in report["samples"]]
        duration = sum(row["audio_seconds"] for row in report["samples"])
        report.update(status="completed", total_audio_seconds=duration,
                      total_processing_seconds=sum(times), rtf=sum(times) / duration,
                      p50_seconds=statistics.median(times), p95_seconds=float(np.percentile(times, 95)),
                      max_seconds=max(times), empty_outputs=sum(not bool(row["production_text"]) for row in report["samples"]),
                      errors=sum(row["error"] is not None for row in report["samples"]),
                      process_memory=process_memory(), actual_backend=engine.backend_info)
    finally:
        if engine is not None:
            engine.close()
            report["closed_instances"] = engine.backend_info["recognizer_instances"]
        report.update(finished_at=stamp(), settings_unchanged=sha256(settings_path) == settings_digest,
                      manifest_unchanged=sha256(args.manifest) == manifest_digest,
                      source_unchanged=all(sha256(Path(path)) == digest for path, digest in source_digests.items()))
        save()
    return int(report["status"] != "completed" or not report["settings_unchanged"]
               or not report["source_unchanged"] or not report["manifest_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
