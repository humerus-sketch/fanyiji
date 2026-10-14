"""Replay saved audio through the real production ASR factory in an isolated root.

Only verified weights are hardlinked into artifacts/diagnostics. User profiles,
audio devices, network downloads and GPU inference are never accessed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import statistics
import sys
import time

from evaluate_asr_candidates import process_memory, write_report
from prepare_asr_candidates import CANDIDATES, ROOT, verify_candidate
from reference_whisper import read_wave

sys.path.insert(0, str(ROOT / "src"))


def prepare_root(candidate: dict, cache: Path, model_root: Path) -> tuple[dict, dict]:
    from fanyiji.asr_catalog import AsrCatalog, profile_for_entry

    allowed = (ROOT / "artifacts" / "diagnostics").resolve()
    if not model_root.is_relative_to(allowed) or model_root == allowed:
        raise ValueError("Production replay model root must be inside artifacts/diagnostics")
    source_directory = cache / candidate["directory"]
    verified = verify_candidate(candidate, source_directory)
    entry = AsrCatalog.default().get(candidate["directory"])
    profile = profile_for_entry(entry)
    for file in entry.files:
        relative = PurePosixPath(file.relative_path)
        prefix = PurePosixPath("asr") / candidate["directory"]
        name = relative.relative_to(prefix).as_posix()
        expected = candidate["files"][name]
        if (file.size, file.sha256) != expected:
            raise ValueError("Production catalog differs from the evaluated pinned model")
        source = source_directory / name
        target = model_root / file.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(source, target):
                raise ValueError(f"Existing artifact asset is not this verified hardlink: {target}")
        else:
            os.link(source, target)
        verified[name]["artifact_path"] = str(target)
        verified[name]["hardlinked_to_verified_cache"] = os.path.samefile(source, target)
    config = model_root / "asr" / "backend.json"
    contents = json.dumps(profile, ensure_ascii=False, indent=2)
    if config.exists():
        if json.loads(config.read_text(encoding="utf-8")) != profile:
            raise ValueError("Conflicting artifact backend.json is preserved")
    else:
        config.write_text(contents, encoding="utf-8")
    return profile, verified


def replay(candidate_id: str, cache: Path, model_root: Path, input_path: Path,
           native_path: Path, output: Path) -> None:
    from fanyiji.asr_engine import create_recognizer_engine

    candidate = CANDIDATES[candidate_id]
    profile, verified = prepare_root(candidate, cache, model_root)
    saved = json.loads(input_path.read_text(encoding="utf-8"))
    native = json.loads(native_path.read_text(encoding="utf-8"))
    if native.get("status") != "completed" or native.get("candidate") != candidate_id:
        raise ValueError("Completed native reference for this same candidate is required")
    native_by_id = {item["id"]: item for item in native["samples"]}
    samples = saved["samples"]
    if {item["id"] for item in samples} != set(native_by_id):
        raise ValueError("Native reference and saved input must contain the same sample IDs")
    if candidate.get("kind") == "qwen3_asr" and (
        native.get("generation", {}).get("max_new_tokens") != 128
        or native.get("language_selection") != "per-stream source language (Japanese/Chinese)"
    ):
        raise ValueError("Qwen production reference must use per-stream source language and 128 tokens")
    if "prefix" in candidate and not native.get("whisper_output_recovery"):
        raise ValueError("Whisper production reference must use byte-safe token output")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "starting", "started_at": datetime.now().astimezone().isoformat(),
        "candidate": candidate_id, "profile": profile, "asset_verification": verified,
        "factory": "fanyiji.asr_engine.create_recognizer_engine",
        "production_source_sha256": {
            name: hashlib.sha256((ROOT / "src" / "fanyiji" / name).read_bytes()).hexdigest()
            for name in ("asr_engine.py", "asr_catalog.py", "whisper_tokens.py")
        },
        "model_root": str(model_root), "saved_input": str(input_path), "native_reference": str(native_path),
        "sherpa_onnx_version": importlib.metadata.version("sherpa-onnx"), "num_threads": 4,
        "ground_truth": False, "user_profile_modified": False,
        "audio_devices_opened": False, "audio_uploaded": False,
        "timing_scope": "Same saved WAVs through production ASR; excludes capture/VAD/translation",
        "samples": [],
    }
    write_report(output, report)
    engine = None
    try:
        factory_started = time.perf_counter()
        engine = create_recognizer_engine(model_root / "asr", num_threads=4)
        report["factory_seconds"] = time.perf_counter() - factory_started
        load_started = time.perf_counter()
        engine.preload(sorted({item["source_lang"] for item in samples}))
        report["model_load_seconds"] = time.perf_counter() - load_started
        report["backend_after_preload"] = engine.backend_info
        report["memory_after_preload"] = process_memory()
        report["status"] = "running"
        write_report(output, report)
        print(json.dumps({"candidate": candidate_id, "status": "production_loaded",
                          "seconds": report["model_load_seconds"]}), flush=True)
        for index, item in enumerate(samples, 1):
            audio_path = Path(item["audio_path"])
            if not audio_path.is_absolute():
                audio_path = input_path.parent / audio_path
            audio = read_wave(audio_path)
            started = time.perf_counter()
            text = engine.recognize(audio, item["source_lang"])
            elapsed = time.perf_counter() - started
            expected = native_by_id[item["id"]]["candidate_text"]
            report["samples"].append({
                "id": item["id"], "source_lang": item["source_lang"], "audio_path": str(audio_path.resolve()),
                "audio_duration_seconds": audio.size / 16000, "production_text": text,
                "native_text": expected, "identical_to_native": text == expected,
                "processing_seconds": elapsed, "rtf": elapsed / (audio.size / 16000),
                "backend_after_sample": engine.backend_info,
            })
            write_report(output, report)
            print(json.dumps({"candidate": candidate_id, "sample": index, "total": len(samples),
                              "seconds": round(elapsed, 3), "identical": text == expected}), flush=True)
        timings = [item["processing_seconds"] for item in report["samples"]]
        audio_seconds = sum(item["audio_duration_seconds"] for item in report["samples"])
        report.update(
            status="completed", total_audio_seconds=audio_seconds,
            total_processing_seconds=sum(timings), aggregate_rtf=sum(timings) / audio_seconds,
            processing_seconds_summary={"median": statistics.median(timings), "max": max(timings)},
            all_samples_identical=all(item["identical_to_native"] for item in report["samples"]),
            backend_before_close=engine.backend_info, memory_before_close=process_memory(),
        )
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if engine is not None:
            started = time.perf_counter()
            engine.close()
            gc.collect()
            report["close_seconds"] = time.perf_counter() - started
            report["backend_after_close"] = engine.backend_info
            report["memory_after_close"] = process_memory()
            report["recognizer_instances_after_close"] = engine.backend_info["recognizer_instances"]
        report["finished_at"] = datetime.now().astimezone().isoformat()
        write_report(output, report)
    print(json.dumps({"candidate": candidate_id, "status": report["status"],
                      "all_identical": report["all_samples_identical"],
                      "closed_instances": report["recognizer_instances_after_close"]}), flush=True)
    if not report["all_samples_identical"]:
        raise RuntimeError("Production and native output differ; inspect the saved report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("qwen3-asr-1.7b-int8", "turbo-int8"), required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "model-cache" / "asr-evaluation")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.candidate, args.cache.resolve(), args.model_root.resolve(), args.input.resolve(),
           args.native_reference.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
