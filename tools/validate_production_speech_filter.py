"""Run the real speech filter on saved WAVs and known local overlap controls."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fanyiji.speech_filter import LABELS_FILENAME, MODEL_FILENAME, SpeechOnlyFilter
from evaluate_speech_filter_candidates import read_wave
from evaluate_asr_candidates import process_memory


def validate(cache: Path, evidence: Path) -> None:
    evidence.mkdir(parents=True, exist_ok=True)
    original_report = json.loads((evidence / "zipformer-small-cpu1.json").read_text(encoding="utf-8"))
    private = [record for record in original_report["input_snapshot"] if record["kind"] == "private_segment"][:30]
    controls = [record for record in original_report["input_snapshot"] if record["kind"] == "public_control"]
    report = {"status": "starting", "started_at": datetime.now().astimezone().isoformat(),
              "native_filter_used": "fanyiji.speech_filter.SpeechOnlyFilter",
              "private_sample_count": len(private), "no_human_ground_truth": True,
              "accuracy_claim": "None: runtime and waveform-integrity checks, not measured classification accuracy",
              "network_attempts": 0, "recording_uploaded": False, "audio_capture_performed": False,
              "rows": []}
    output = evidence / "production-filter-native-cpu1.json"
    source = ROOT / "src" / "fanyiji" / "speech_filter.py"
    report["production_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = cache / "zipformer-small-int8"
    asset_hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in (MODEL_FILENAME, LABELS_FILENAME)}
    report["asset_sha256"] = asset_hashes

    def prohibit_network(event, args):
        if event == "socket.connect":
            report["network_attempts"] += 1
            raise RuntimeError("Private audio network access is prohibited")
        if event == "subprocess.Popen":
            raise RuntimeError("Native child launch is prohibited in filter validation")

    sys.addaudithook(prohibit_network)
    gate = SpeechOnlyFilter(directory, num_threads=1)
    report["backend_before_preload"] = gate.backend_info
    started = time.perf_counter()
    gate.preload()
    report["preload_seconds"] = time.perf_counter() - started
    report["backend_after_preload"] = gate.backend_info
    report["memory_after_preload"] = process_memory()

    def process(record: dict, audio: np.ndarray):
        original_audio = audio.copy()
        result = gate.process(audio)
        if (result.samples.shape != audio.shape or result.samples.dtype != np.float32
                or not np.isfinite(result.samples).all() or not np.array_equal(original_audio, audio)):
            raise RuntimeError("Production filter changed the input or returned invalid output")
        report["rows"].append({"source_id": record["id"], "source_metadata": record,
                               "input_array_unchanged": True, "output_length_and_dtype_valid": True,
                               "output_audio_sha256": hashlib.sha256(result.samples.tobytes()).hexdigest(),
                               "output_matches_input_exactly": np.array_equal(result.samples, audio),
                               "decision": result.metadata})

    try:
        if len(private) != 30:
            raise RuntimeError("The fixed validation subset is not 30 private samples")
        hashes = {}
        for record in private + controls:
            path = Path(record["audio_path"])
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            audio, rate = read_wave(path)
            if rate != 16000:
                raise RuntimeError("Expected already-saved mono16k validation audio")
            process(record, audio)
        report["all_original_wav_hashes_unchanged"] = all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha256 for path, sha256 in hashes.items())
        report["original_wav_sha256"] = hashes
        chinese = next(record for record in controls if record["id"] == "official-chinese-speech")
        singing = next(record for record in controls if record["id"] == "official-singing-demo")
        spoken, rate = read_wave(Path(chinese["audio_path"]))
        sung, _ = read_wave(Path(singing["audio_path"]))
        spoken = spoken[3 * rate:][:sung.size]
        sung = sung[:spoken.size]
        sung = sung * np.sqrt(np.mean(spoken ** 2)) / np.sqrt(np.mean(sung ** 2))
        for decibels in (-6, 0, 6):
            mixture = np.ascontiguousarray(spoken * 10 ** (decibels / 20) + sung, dtype=np.float32)
            mixture /= max(1.0, float(np.max(np.abs(mixture))))
            process({"id": f"synthetic-speech-over-singing-{decibels:+d}dB", "kind": "synthetic_overlap_control",
                     "speech_source": chinese["audio_path"], "singing_source": singing["audio_path"],
                     "speech_start_seconds": 3, "speech_to_singing_rms_db": decibels,
                     "description": "Constructed two known source signals, not natural mixture or separation"}, mixture)
        report["memory_after_processing"] = process_memory()
        private_rows = report["rows"][:30]
        report["private_summary"] = {
            "actions": dict(Counter(row["decision"]["action"] for row in private_rows)),
            "total_input_seconds": sum(row["decision"]["input_duration_seconds"] for row in private_rows),
            "total_processing_seconds": sum(row["decision"]["processing_seconds"] for row in private_rows),
            "total_masked_seconds": sum(row["decision"]["masked_duration_seconds"] for row in private_rows),
            "median_processing_seconds": statistics.median(row["decision"]["processing_seconds"] for row in private_rows),
            "all_input_arrays_unchanged": all(row["input_array_unchanged"] for row in private_rows),
        }
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        started = time.perf_counter()
        gate.close()
        gc.collect()
        report["close_seconds"] = time.perf_counter() - started
        report["backend_after_close"] = gate.backend_info
        report["memory_after_close"] = process_memory()
        report["source_sha256_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == report["production_source_sha256"]
        report["all_model_asset_hashes_unchanged"] = all(
            hashlib.sha256((directory / name).read_bytes()).hexdigest() == sha256 for name, sha256 in asset_hashes.items())
        report["finished_at"] = datetime.now().astimezone().isoformat()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "private_summary": report["private_summary"],
                      "control_actions": [(row["source_id"], row["decision"]["action"], row["decision"]["retained_fraction"])
                                          for row in report["rows"][30:]], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "model-cache" / "speech-filter")
    parser.add_argument("--evidence", type=Path, default=ROOT / "artifacts" / "diagnostics" / "speech-filter-20261009")
    args = parser.parse_args()
    validate(args.cache.resolve(), args.evidence.resolve())
