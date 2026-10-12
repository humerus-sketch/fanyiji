"""Replay one frozen corpus through one real production ASR adapter, CPU4.

Run models sequentially in separate processes. A transparent recorder observes
the same native decode call, including raw results rejected by production.
No user profiles, waveform files, capture, downloads, GPU or MT are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
import wave

import numpy as np

from evaluate_asr_candidates import EVALUATION_CANDIDATES, process_memory, write_report
from prepare_asr_candidates import ROOT, verify_candidate

sys.path.insert(0, str(ROOT / "src"))
MODEL_KEYS = {
    "sensevoice-int8": "sensevoice-int8", "whisper-small-int8": "small-int8",
    "qwen3-asr-0.6b-int8": "qwen3-asr-0.6b-int8", "qwen3-asr-1.7b-int8": "qwen3-asr-1.7b-int8",
    "whisper-turbo-int8": "turbo-int8", "whisper-large-v3-int8": "large-v3-int8",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_corpus(path: Path, expected_sha256: str) -> tuple[dict, list[dict]]:
    if sha256(path) != expected_sha256:
        raise ValueError("Frozen corpus manifest SHA-256 mismatch")
    corpus = json.loads(path.read_text(encoding="utf-8-sig"))
    if corpus.get("status") != "frozen" or not isinstance(corpus.get("samples"), list) or not corpus["samples"]:
        raise ValueError("Expected a nonempty frozen corpus")
    seen, result = set(), []
    for item in corpus["samples"]:
        if not isinstance(item.get("id"), str) or not item["id"] or item["id"] in seen:
            raise ValueError("Frozen corpus sample IDs must be unique strings")
        seen.add(item["id"])
        if item.get("source_lang") not in ("zh", "ja"):
            raise ValueError("Frozen corpus source language must be Chinese or Japanese")
        audio_path = Path(item["audio_path"])
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_relative_to(path.parent.resolve()):
            raise ValueError("Primary frozen WAV must stay within the frozen corpus directory")
        actual = sha256(audio_path)
        if actual != item.get("sha256"):
            raise ValueError(f"Frozen WAV SHA-256 mismatch: {item['id']}")
        with wave.open(str(audio_path), "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (16000, 1, 2, "NONE"):
                raise ValueError("Frozen WAV must be mono16k PCM16")
            frames = audio.getnframes()
            if frames <= 0 or len(audio.readframes(frames)) != frames * 2:
                raise ValueError("Frozen WAV has incomplete or empty PCM")
        result.append({**item, "audio_path": str(audio_path), "verified_sha256": actual,
                       "verified_frames": frames, "verified_audio_seconds": frames / 16000})
    return corpus, result


def model_spec(model_id: str, models: Path, cache: Path) -> tuple[dict, Path, dict]:
    candidate = EVALUATION_CANDIDATES[MODEL_KEYS[model_id]]
    if model_id == "sensevoice-int8":
        directory = models / "asr"
        profile = {"backend": "sense_voice", "model_id": model_id}
    elif model_id == "whisper-small-int8":
        directory = cache.parent / candidate["directory"]
        profile = {"backend": "whisper", "model_id": model_id}
    else:
        directory = cache / candidate["directory"]
        profile = {"backend": candidate.get("kind", "whisper"), "model_id": model_id}
    if profile["backend"] == "whisper":
        for field in ("encoder", "decoder", "tokens"):
            suffix = f"{field}.int8.onnx" if field != "tokens" else "tokens.txt"
            profile[field] = f"{candidate['prefix']}-{suffix}"
    elif profile["backend"] == "qwen3_asr":
        profile.update(encoder="encoder.int8.onnx", decoder="decoder.int8.onnx",
                       conv_frontend="conv_frontend.onnx", tokenizer="tokenizer")
    return candidate, directory.resolve(), profile


class RecordingRecognizer:
    """Delegate the existing native object; never perform an extra decode."""

    def __init__(self, delegate, capture: dict):
        self.delegate = delegate
        self.capture = capture

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def decode_stream(self, stream):
        result = self.delegate.decode_stream(stream)
        native = stream.result
        tokens = list(native.tokens)
        self.capture.update(native_text=native.text, exposed_token_count=len(tokens),
                            exposed_tokens=[item if isinstance(item, (str, int, float, bool)) or item is None
                                            else str(item) for item in tokens], native_decode_calls=1)
        return result


def instrument(engine, capture: dict) -> None:
    for key, recognizer in list(engine._recognizers.items()):
        engine._recognizers[key] = RecordingRecognizer(recognizer, capture)


def acceptance(text: str | None, error: Exception | None, raw: dict, backend: dict) -> str:
    if error is None:
        return "accepted_nonempty" if text else "empty_output"
    guard = backend.get("generation", {}).get("cleaned_token_guard") if backend.get("generation") else None
    if guard is not None and raw.get("exposed_token_count", -1) >= guard:
        return "rejected_generation_guard"
    return "error"


def run(args) -> None:
    from fanyiji.asr_engine import ProfileRecognizerEngine
    from fanyiji.whisper_tokens import restore_whisper_bytes

    args.input, args.output = args.input.resolve(), args.output.resolve()
    corpus, samples = load_corpus(args.input, args.expected_input_sha256)
    candidate, directory, profile = model_spec(args.model, args.models.resolve(), args.cache.resolve())
    assets = verify_candidate(candidate, directory)
    if args.output.exists():
        raise ValueError("Existing evaluation report is preserved; choose a new output path")
    if args.output.is_relative_to(args.input.parent):
        raise ValueError("Evaluation reports must not modify the frozen corpus directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    control = args.output.with_name(args.output.stem + ".control.json")
    source_paths = [ROOT / "src" / "fanyiji" / filename for filename in
                    ("asr_engine.py", "engines.py", "whisper_tokens.py")]
    report = {
        "status": "prepared", "started_at": datetime.now().astimezone().isoformat(), "pid": os.getpid(),
        "model_id": args.model, "model_source": candidate, "verified_assets": assets,
        "model_directory": str(directory), "profile_in_memory_only": profile, "user_profile_modified": False,
        "adapter": "fanyiji.asr_engine.ProfileRecognizerEngine", "num_threads": 4, "provider": "cpu",
        "sherpa_onnx_version": importlib.metadata.version("sherpa-onnx"), "network_attempts": 0,
        "audio_capture_performed": False, "audio_uploaded": False, "translation_inference_performed": False,
        "frozen_corpus": str(args.input), "frozen_corpus_sha256": args.expected_input_sha256,
        "sample_count": len(samples), "total_audio_seconds": sum(item["verified_audio_seconds"] for item in samples),
        "languages": dict(Counter(item["source_lang"] for item in samples)), "ground_truth": False,
        "comparison_rule": "Text differences and model agreement are not human transcription accuracy",
        "timing_scope": "Preload and one ASR call per original frozen PCM; no VAD/filter/denoising/MT",
        "raw_capture_scope": "Transparent wrapper observes the production adapter's single native decode; no repeat inference",
        "production_source_sha256": {path.name: sha256(path) for path in source_paths}, "samples": [],
    }

    def write(stage: str, **extra):
        write_report(args.output, report)
        write_report(control, {"pid": os.getpid(), "model_id": args.model, "stage": stage,
                               "completed_samples": len(report["samples"]), "total_samples": len(samples),
                               "updated_at": datetime.now().astimezone().isoformat(), **extra})

    def prohibit_network(event, event_args):
        if event == "socket.connect":
            report["network_attempts"] += 1
            raise RuntimeError("Private frozen audio evaluation prohibits network connections")
        if event == "subprocess.Popen":
            raise RuntimeError("ASR benchmark prohibits native child processes")

    sys.addaudithook(prohibit_network)
    write("prepared")
    print(json.dumps({"stage": "prepared", "model_id": args.model, "pid": os.getpid(),
                      "report": str(args.output), "control": str(control), "clips": len(samples)}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    engine = None
    capture = {}
    try:
        report["status"] = "loading"
        write("loading")
        started = time.perf_counter()
        engine = ProfileRecognizerEngine(directory, num_threads=4, config=profile)
        engine.preload(sorted(report["languages"]))
        report["preload_seconds"] = time.perf_counter() - started
        report["backend_after_preload"] = engine.backend_info
        report["memory_after_preload"] = process_memory()
        instrument(engine, capture)
        print(json.dumps({"stage": "loaded", "model_id": args.model, "pid": os.getpid(),
                          "seconds": report["preload_seconds"], "memory": report["memory_after_preload"]}), flush=True)
        report["status"] = "running"
        for index, item in enumerate(samples, 1):
            path = Path(item["audio_path"])
            if sha256(path) != item["verified_sha256"]:
                raise ValueError("Frozen WAV changed after validation")
            with wave.open(str(path), "rb") as stream:
                audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
            before = hashlib.sha256(audio.tobytes()).hexdigest()
            capture.clear()
            write("recognizing", current_sample_id=item["id"], current_sample=index)
            started = time.perf_counter()
            text, error = None, None
            try:
                text = engine.recognize(audio, item["source_lang"])
            except Exception as exception:
                error = exception
            elapsed = time.perf_counter() - started
            backend = engine.backend_info
            state = acceptance(text, error, capture, backend)
            raw_text = capture.get("native_text")
            restored = None
            recovery_error = None
            if profile["backend"] == "whisper" and raw_text is not None:
                try:
                    restored = restore_whisper_bytes(raw_text)
                except ValueError as exception:
                    recovery_error = str(exception)
            context = item.get("production_text")
            row = {"id": item["id"], "source_lang": item["source_lang"], "sha256": item["verified_sha256"],
                   "audio_path": str(path), "audio_duration_seconds": item["verified_audio_seconds"],
                   "processing_seconds": elapsed, "rtf": elapsed / item["verified_audio_seconds"],
                   "production_acceptance": state, "production_text": text, "native_raw": dict(capture),
                   "whisper_restored_text_before_sanitization": restored, "whisper_recovery_error": recovery_error,
                   "error": {"type": type(error).__name__, "message": str(error)} if error else None,
                   "backend_after_sample": backend, "memory_after_sample": process_memory(),
                   "input_array_sha256": before, "input_array_unchanged": hashlib.sha256(audio.tobytes()).hexdigest() == before,
                   "context_production_text_not_ground_truth": context,
                   "matches_context_model_output": text == context if isinstance(context, str) and text is not None else None,
                   "similarity_to_context_not_accuracy": SequenceMatcher(None, text, context).ratio()
                    if isinstance(context, str) and text is not None else None,
                   "stratum": item.get("stratum"), "stratum_label_status": item.get("stratum_label_status"),
                   "source_kind": item.get("source_kind"), "synthetic": item.get("synthetic"),
                   "human_truth": item.get("human_truth")}
            report["samples"].append(row)
            write("clip_completed", latest_sample_id=item["id"])
            print(json.dumps({"model_id": args.model, "sample": index, "total": len(samples), "id": item["id"],
                              "seconds": round(elapsed, 3), "acceptance": state}, ensure_ascii=False), flush=True)
        timings = [row["processing_seconds"] for row in report["samples"]]
        report.update(status="completed", total_processing_seconds=sum(timings),
                      aggregate_rtf=sum(timings) / report["total_audio_seconds"],
                      timing_summary={"median_seconds": statistics.median(timings), "max_seconds": max(timings)},
                      production_acceptance_counts=dict(Counter(row["production_acceptance"] for row in report["samples"])),
                      all_inputs_unchanged=all(row["input_array_unchanged"] for row in report["samples"]),
                      backend_before_close=engine.backend_info, memory_before_close=process_memory())
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if engine is not None:
            started = time.perf_counter()
            engine.close()
            gc.collect()
            report["close_seconds"] = time.perf_counter() - started
            report["backend_after_close"] = engine.backend_info
            report["memory_after_close"] = process_memory()
        report["source_sha256_unchanged"] = all(sha256(path) == report["production_source_sha256"][path.name] for path in source_paths)
        report["corpus_and_wav_sha256_unchanged"] = (sha256(args.input) == args.expected_input_sha256 and
            all(sha256(Path(item["audio_path"])) == item["verified_sha256"] for item in samples))
        report["finished_at"] = datetime.now().astimezone().isoformat()
        write(report["status"])
    print(json.dumps({"model_id": args.model, "status": report["status"],
                      "total_processing_seconds": report.get("total_processing_seconds"),
                      "aggregate_rtf": report.get("aggregate_rtf"),
                      "acceptance_counts": report.get("production_acceptance_counts"),
                      "instances_after_close": report.get("backend_after_close", {}).get("recognizer_instances")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--cache", type=Path, default=ROOT / "model-cache" / "asr-evaluation")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
