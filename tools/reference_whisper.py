"""Compare saved diagnostic segments with an independent, local Whisper ASR.

Only --download-only accesses the network, to fetch pinned public model files.
Inference reads local WAV files and never opens audio devices or uploads content.
The reference transcript is a second model's output, not human ground truth.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import statistics
import sys
import time
import unicodedata
import urllib.request
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REPO = "csukuangfj/sherpa-onnx-whisper-small"
REVISION = "8f3c18b358db4d1f2fc1eae49d75cd20989e4309"
MODEL_FILES = {
    "small-encoder.int8.onnx": (112442483, "4cbe7b22fa9026b843b60a68640c747de05bafb1a11b57edc0e66c232d9f33a9"),
    "small-decoder.int8.onnx": (262226114, "acad50b5c782696e91b55914cc5ab4f756f1532f76e22aa6fc615f39fb69a8ee"),
    "small-tokens.txt": (816730, "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"),
}
TURBO_FILES = {
    "turbo-encoder.int8.onnx": (674716297, "b02dcdf54f348741e93fe732b67d933c8dcb6735655f710640143081db38878b"),
    "turbo-decoder.int8.onnx": (361080764, "20accd02388482eb3a46bd615631adfdc85e1eb2c7db9ea3f02a40ffe6b81547"),
    "turbo-tokens.txt": (816730, "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"),
}
LICENSE_URL = "https://raw.githubusercontent.com/openai/whisper/main/LICENSE"
DOCUMENTATION_URL = "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/index.html"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_models(directory: Path, variant: str = "small") -> dict[str, dict[str, Any]]:
    files = {}
    definitions = MODEL_FILES if variant == "small" else TURBO_FILES
    repo, revision = (REPO, REVISION) if variant == "small" else (
        "csukuangfj/sherpa-onnx-whisper-turbo", "2ca6ff69fc878651b770880507669577ac41c2ff")
    for name, (size, expected_hash) in definitions.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"Missing/incomplete reference model: {path}")
        digest = sha256(path)
        if expected_hash is not None and digest != expected_hash:
            raise RuntimeError(f"Reference model SHA256 mismatch: {path}")
        files[name] = {"size": size, "sha256": digest,
                       "source": f"https://huggingface.co/{repo}/resolve/{revision}/{name}"}
    return files


def download_models(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, expected_hash) in MODEL_FILES.items():
        path = directory / name
        if path.is_file() and path.stat().st_size == size:
            if expected_hash is None or sha256(path) == expected_hash:
                print(f"Already verified: {name}", flush=True)
                continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        temporary = path.with_suffix(path.suffix + ".download")
        print(f"Downloading {name}: {size / 1024 ** 2:.1f} MiB", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "fanyiji-reference/1.0"})
        received = 0
        next_progress = 20 * 1024 ** 2
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while data := response.read(1024 ** 2):
                output.write(data)
                received += len(data)
                if received >= next_progress:
                    print(f"  {received / 1024 ** 2:.1f}/{size / 1024 ** 2:.1f} MiB", flush=True)
                    next_progress += 20 * 1024 ** 2
        if received != size or (expected_hash is not None and sha256(temporary) != expected_hash):
            raise RuntimeError(f"Downloaded model verification failed: {temporary}")
        temporary.replace(path)
    license_path = directory / "LICENSE.whisper"
    if not license_path.is_file():
        with urllib.request.urlopen(LICENSE_URL, timeout=30) as response:
            license_path.write_bytes(response.read())
    manifest = {"repo": REPO, "revision": REVISION, "files": verify_models(directory),
                "license": {"name": "MIT", "source": LICENSE_URL},
                "documentation": DOCUMENTATION_URL}
    (directory / "SOURCE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Reference models verified; production models were not changed.", flush=True)


def read_wave(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getframerate(), source.getsampwidth(), source.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError(f"Expected uncompressed mono 16 kHz PCM16 WAV: {path}")
        frames = source.getnframes()
        audio = np.frombuffer(source.readframes(frames), dtype="<i2").astype(np.float32) / 32768.0
    if audio.size != frames or audio.size == 0:
        raise ValueError(f"Empty/incomplete WAV: {path}")
    return audio


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def differences(production: str, reference: str) -> dict[str, Any]:
    edits = []
    for operation, a, b, c, d in difflib.SequenceMatcher(None, production, reference, autojunk=False).get_opcodes():
        if operation != "equal":
            edits.append({"operation": operation, "production": production[a:b], "reference": reference[c:d]})
    return {"identical": production == reference,
            "normalized_similarity": difflib.SequenceMatcher(None, normalized(production), normalized(reference), autojunk=False).ratio(),
            "character_edits": edits,
            "interpretation": "Model disagreement only; similarity is not an accuracy score or CER against human truth."}


def compare(results_path: Path, model_directory: Path, output: Path, threads: int, variant: str = "small") -> None:
    import sherpa_onnx
    from fanyiji.whisper_tokens import METHOD, WHISPER_TOKEN_SHA256, prepare_whisper_tokens, restore_whisper_bytes

    production = json.loads(results_path.read_text(encoding="utf-8"))
    samples = production.get("samples", [])
    if not samples or any(item.get("source_lang") != "ja" for item in samples):
        raise ValueError("This reference run requires saved Japanese diagnostic samples")
    skipped_ids = [item["id"] for item in samples if item.get("speech_filter", {}).get("action") == "skip"]
    samples = [item for item in samples if item.get("speech_filter", {}).get("action") != "skip"]
    if not samples:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "completed", "ground_truth": False, "audio_uploaded": False,
            "samples": [], "production_filter_skipped_ids": skipped_ids,
            "comparison_scope": "No production ASR input: all segments were skipped by speech filtering",
            "total_audio_seconds": 0, "total_processing_seconds": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    files = verify_models(model_directory, variant)
    load_started = time.perf_counter()
    table = prepare_whisper_tokens(model_directory / f"{variant}-tokens.txt", expected_sha256=WHISPER_TOKEN_SHA256)
    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(model_directory / f"{variant}-encoder.int8.onnx"),
        decoder=str(model_directory / f"{variant}-decoder.int8.onnx"),
        tokens=str(table.path), language="ja", task="transcribe",
        num_threads=threads, provider="cpu", decoding_method="greedy_search", debug=False,
    )
    report: dict[str, Any] = {
        "status": "running", "ground_truth": False, "audio_uploaded": False,
        "input_results": str(results_path.resolve()), "model": f"Whisper {variant} multilingual ONNX INT8",
        "repo": REPO if variant == "small" else "csukuangfj/sherpa-onnx-whisper-turbo",
        "revision": REVISION if variant == "small" else "2ca6ff69fc878651b770880507669577ac41c2ff",
        "model_files": files, "output_recovery": {"method": METHOD, "source_token_sha256": table.source_sha256,
                                                  "derived_token_sha256": table.derived_sha256},
        "provider": "cpu", "num_threads": threads, "language": "ja", "task": "transcribe",
        "decoding_method": "greedy_search", "model_load_seconds": time.perf_counter() - load_started,
        "timing_scope": "Sequential saved-segment inference; excludes capture, VAD endpointing and translation.",
        "samples": [],
        "production_filter_skipped_ids": skipped_ids,
        "comparison_scope": "Same retained/masked waveform path as production ASR; raw skipped segments excluded",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(samples, 1):
        audio_path = Path(item["audio_path"])
        if not audio_path.is_absolute():
            audio_path = results_path.parent / audio_path
        audio = read_wave(audio_path)
        started = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        recognizer.decode_stream(stream)
        reference = restore_whisper_bytes(stream.result.text).strip()
        elapsed = time.perf_counter() - started
        production_text = item.get("production_text", "")
        result = {"id": item["id"], "channel": item["channel"], "audio_path": str(audio_path.resolve()),
                  "audio_duration_seconds": audio.size / 16000, "production_text": production_text,
                  "production_translation": item.get("production_translation", ""),
                  "reference_text": reference, "reference_processing_seconds": elapsed,
                  "reference_rtf": elapsed / (audio.size / 16000),
                  "production_pipeline_latency_seconds": item.get("pipeline_latency_seconds"),
                  "differences": differences(production_text, reference)}
        report["samples"].append(result)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{index}/{len(samples)} duration={audio.size / 16000:.3f}s processing={elapsed:.3f}s", flush=True)
    timings = [item["reference_processing_seconds"] for item in report["samples"]]
    total_audio = sum(item["audio_duration_seconds"] for item in report["samples"])
    report.update(status="completed", finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds"),
                  total_audio_seconds=total_audio, total_processing_seconds=sum(timings),
                  aggregate_rtf=sum(timings) / total_audio,
                  processing_seconds_summary={"min": min(timings), "median": statistics.median(timings), "max": max(timings)})
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reference comparison complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", help="Saved local diagnostics results.json")
    parser.add_argument("--models", type=Path, default=ROOT / "model-cache" / "reference-whisper-small")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--variant", choices=("small", "turbo"), default="small")
    parser.add_argument("--download-only", action="store_true", help="Download public model files; does not read audio")
    args = parser.parse_args()
    if args.download_only:
        if args.variant != "small":
            parser.error("turbo models are prepared by tools/prepare_asr_candidates.py")
        download_models(args.models.resolve())
    elif args.results is None:
        parser.error("results is required unless --download-only is set")
    elif not 1 <= args.num_threads <= 4:
        parser.error("--num-threads must be 1 through 4")
    else:
        compare(args.results.resolve(), args.models.resolve(),
                (args.output or args.results.parent / "reference-whisper.json").resolve(), args.num_threads, args.variant)


if __name__ == "__main__":
    main()
