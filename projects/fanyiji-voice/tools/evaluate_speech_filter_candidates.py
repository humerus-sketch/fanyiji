"""Measure pinned local AudioTagging models against saved private WAVs.

Scores are multi-label sound-event estimates, not an accuracy measurement.
No capture, upload, dependency changes, ASR/translation inference, or GUI edits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import wave

import numpy as np
import sherpa_onnx

from evaluate_asr_candidates import process_memory
from prepare_speech_filter_candidates import CANDIDATES, ROOT
from prepare_asr_candidates import verify_candidate

SPEECH_IDS = (0, 1, 2, 3, 4, 5, 7, 15)
SINGING_IDS = tuple(range(27, 38)) + (254, 255)
MUSIC_IDS = (137, 267, 270, 272)


def read_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        if stream.getsampwidth() != 2:
            raise RuntimeError(f"Expected local PCM16 WAV: {path}")
        rate, channels = stream.getframerate(), stream.getnchannels()
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio / 32768), rate


def inputs(cache: Path) -> list[dict]:
    records = []
    for label, name in (("asr-fixed8", "live-20261009-101554"),
                        ("live-qwen17-hy7", "live-20261009-123159-qwen17-hy7")):
        manifest = ROOT / "artifacts" / "diagnostics" / name / "results.json"
        content = json.loads(manifest.read_text(encoding="utf-8-sig"))
        for sample in content["samples"]:
            path = Path(sample["audio_path"])
            if not path.is_absolute():
                path = manifest.parent / path
            if not path.is_file():
                continue
            records.append({"id": label + ":" + sample["id"], "kind": "private_segment",
                            "audio_path": str(path.resolve()), "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "asr_text_for_context_not_ground_truth": sample.get("production_text") or sample.get("sensevoice_text"),
                            "human_annotation": None})
    singing = cache / "zipformer-small-int8" / "test_wavs" / "3.wav"
    records.append({"id": "official-singing-demo", "kind": "public_control",
                    "audio_path": str(singing), "known_description": "Official AudioTagging music / A capella / Singing example"})
    chinese = ROOT / "artifacts" / "diagnostics" / "model-evaluation-20261009" / "official-asr-self-test" / "noise2.wav"
    if chinese.is_file():
        records.append({"id": "official-chinese-speech", "kind": "public_control", "audio_path": str(chinese),
                        "known_description": "Official ASR demo, spoken Chinese with dialled-number correction"})
    continuous = ROOT / "artifacts" / "diagnostics" / "live-20261009-101554" / "system-continuous.wav"
    records.append({"id": "private-continuous", "kind": "private_continuous", "audio_path": str(continuous),
                    "human_annotation": None})
    return records


def evaluate(key: str, cache: Path, output: Path, threads: int, overlap_only: bool = False) -> None:
    candidate = CANDIDATES[key]
    directory = cache / candidate["directory"]
    verified = verify_candidate(candidate, directory)
    labels_path = directory / "class_labels_indices.csv"
    if hashlib.sha256(labels_path.read_bytes()).hexdigest() != "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429":
        raise RuntimeError("Fixed label table SHA-256 mismatch")
    labels = {int(row["index"]): row["display_name"] for row in csv.DictReader(labels_path.open(encoding="utf-8"))}
    if set(labels) != set(range(527)):
        raise RuntimeError("Expected all 527 AudioSet labels")
    report = {"status": "starting", "started_at": datetime.now().astimezone().isoformat(),
              "candidate": candidate, "verified_weight_files": verified, "sherpa_onnx_version": sherpa_onnx.__version__,
              "provider": "cpu", "num_threads": threads, "all_class_count": len(labels), "labels": labels,
              "network_attempts": 0, "recording_uploaded": False, "audio_capture_performed": False,
              "accuracy_claim": "None: private audio has no manual speech/singing/music annotations; classifier scores are not accuracy",
              "group_max_score_indices": {"speech": SPEECH_IDS, "singing": SINGING_IDS, "music": MUSIC_IDS},
              "rows": []}
    output.parent.mkdir(parents=True, exist_ok=True)

    def write():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def prohibit_network(event, args):
        if event == "socket.connect":
            report["network_attempts"] += 1
            raise RuntimeError("Network is prohibited for private audio evaluation")
        if event == "subprocess.Popen":
            raise RuntimeError("Native child launch is prohibited in local AudioTagging evaluation")

    sys.addaudithook(prohibit_network)
    model_args = {"ced": str(directory / "model.int8.onnx")} if candidate["kind"] == "ced" else {
        "zipformer": sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(model=str(directory / "model.int8.onnx"))}
    config = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(**model_args, num_threads=threads, provider="cpu", debug=False),
        labels=str(labels_path), top_k=len(labels))
    if not config.validate():
        raise RuntimeError("AudioTaggingConfig did not validate")
    started = time.perf_counter()
    tagger = sherpa_onnx.AudioTagging(config)
    report["load_seconds"] = time.perf_counter() - started
    report["memory_after_load"] = process_memory()
    source_records = inputs(cache)
    if overlap_only:
        speech = next(record for record in source_records if record["id"] == "official-chinese-speech")
        singing = next(record for record in source_records if record["id"] == "official-singing-demo")
        source_records = [{"id": f"synthetic-speech-over-singing-{decibels:+d}dB", "kind": "synthetic_overlap_control",
                           "speech_source": speech["audio_path"], "singing_source": singing["audio_path"],
                           "speech_start_seconds": 3.0, "speech_to_singing_rms_db": decibels,
                           "known_description": "Deterministic additive speech and singing test, not a naturally recorded mixture"}
                          for decibels in (-6, 0, 6)]
    report["input_snapshot"] = source_records

    def run(record: dict, samples: np.ndarray, rate: int, start: float, kind: str, window: float | None = None):
        started = time.perf_counter()
        stream = tagger.create_stream()
        stream.accept_waveform(sample_rate=rate, waveform=np.ascontiguousarray(samples, dtype=np.float32))
        events = tagger.compute(stream, top_k=len(labels))
        elapsed = time.perf_counter() - started
        by_id = {int(event.index): event for event in events}
        if len(events) != len(labels) or set(by_id) != set(labels):
            raise RuntimeError("Native AudioTagging did not return every class score")
        scores = [float(by_id[index].prob) for index in range(len(labels))]
        if not all(np.isfinite(score) and 0 <= score <= 1 for score in scores):
            raise RuntimeError("Invalid native classifier score")
        groups = {name: max(scores[index] for index in indices) for name, indices in
                  (("speech", SPEECH_IDS), ("singing", SINGING_IDS), ("music", MUSIC_IDS))}
        report["rows"].append({"source_id": record["id"], "kind": kind, "window_start_seconds": start,
                               "window_end_seconds": start + samples.size / rate, "requested_window_seconds": window,
                               "audio_duration_seconds": samples.size / rate, "processing_seconds": elapsed,
                               "scores": groups, "speech_minus_singing": groups["speech"] - groups["singing"],
                               "key_class_scores": {labels[index]: scores[index] for index in (*SPEECH_IDS, *SINGING_IDS, *MUSIC_IDS)},
                               "top12": [{"index": int(event.index), "name": event.name, "score": float(event.prob)}
                                         for event in events[:12]], "all_scores_index_order": scores})
        if len(report["rows"]) % 100 == 0:
            print(json.dumps({"candidate": key, "status": "progress", "rows": len(report["rows"])}, ensure_ascii=False), flush=True)

    try:
        # Native short zero-wave control exercises the runtime without relying on a soundtrack label.
        run({"id": "generated-silence"}, np.zeros(32000, dtype=np.float32), 16000, 0, "silence_control", 2)
        for record in source_records:
            if record["kind"] == "synthetic_overlap_control":
                spoken, rate = read_wave(Path(record["speech_source"]))
                sung, singing_rate = read_wave(Path(record["singing_source"]))
                if singing_rate != rate:
                    raise RuntimeError("Synthetic overlap inputs have different sample rates")
                spoken = spoken[round(record["speech_start_seconds"] * rate):][:sung.size]
                sung = sung[:spoken.size]
                gain = 10 ** (record["speech_to_singing_rms_db"] / 20)
                sung = sung * (np.sqrt(np.mean(spoken ** 2)) / np.sqrt(np.mean(sung ** 2)))
                audio = np.ascontiguousarray(spoken * gain + sung, dtype=np.float32)
                # Avoid clipping; preserve the requested relative RMS ratio.
                peak = max(1.0, float(np.max(np.abs(audio))))
                audio = audio / peak
                record["common_peak_scaling"] = 1 / peak
            else:
                audio, rate = read_wave(Path(record["audio_path"]))
            if record["kind"] == "private_continuous":
                for start in range(0, audio.size - rate * 2 + 1, rate * 3):
                    run(record, audio[start:start + rate * 2], rate, start / rate, "continuous_window_2s", 2)
                continue
            run(record, audio, rate, 0, "whole_segment")
            # 2 s context with 1 s hop keeps within-segment music/singing/speech variation visible.
            for start in range(0, audio.size, rate):
                part = audio[start:start + rate * 2]
                if part.size < rate * 0.5:
                    continue
                run(record, part, rate, start / rate, "sliding_window_2s", 2)
            if record["kind"] == "public_control" or record["id"].endswith(("g1-u2", "g1-u11", "g1-u16")):
                for seconds in (1.0, 1.5, 3.0):
                    part = audio[:round(seconds * rate)]
                    if part.size >= rate * 0.5:
                        run(record, part, rate, 0, "window_length_probe", seconds)
        timings = [row["processing_seconds"] for row in report["rows"]]
        report["summary"] = {"classification_count": len(timings), "total_processing_seconds": sum(timings),
                             "median_processing_seconds": statistics.median(timings), "max_processing_seconds": max(timings),
                             "processed_audio_seconds_including_repeated_windows": sum(row["audio_duration_seconds"] for row in report["rows"])}
        report["memory_after_evaluation"] = process_memory()
        report["status"] = "completed"
    except Exception as error:
        report["error"] = str(error)
        report["status"] = "failed"
        raise
    finally:
        del tagger
        gc.collect()
        report["memory_after_release"] = process_memory()
        report["finished_at"] = datetime.now().astimezone().isoformat()
        write()
    print(json.dumps({"status": report["status"], "candidate": key, "summary": report["summary"],
                      "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "model-cache" / "speech-filter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overlap-only", action="store_true")
    args = parser.parse_args()
    evaluate(args.candidate, args.cache.resolve(), args.output.resolve(), args.threads, args.overlap_only)
