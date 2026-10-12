"""Compare saved PCM with a pinned local ASR candidate on CPU.

No download, audio device, network request, or GPU inference occurs. Previous
SenseVoice/small transcripts are model outputs, never human ground truth.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
from datetime import datetime, timedelta, timezone
import importlib.metadata
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from prepare_asr_candidates import CANDIDATES, ROOT, verify_candidate
from reference_whisper import differences, read_wave

TZ = timezone(timedelta(hours=8))
EVALUATION_CANDIDATES = {
    **CANDIDATES,
    "sensevoice-int8": {
        "id": "sensevoice-int8", "directory": "asr", "kind": "sensevoice",
        "repo": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        "revision": "2365baeacb507f821a0c8120fcee3d484dba7a07",
        "name": "SenseVoiceSmall multilingual ONNX INT8",
        "files": {
            "model.int8.onnx": (239233841, "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"),
            "tokens.txt": (315894, "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"),
        },
    },
    "small-int8": {
        "id": "small-int8", "directory": "reference-whisper-small", "prefix": "small",
        "repo": "csukuangfj/sherpa-onnx-whisper-small",
        "revision": "8f3c18b358db4d1f2fc1eae49d75cd20989e4309",
        "name": "Whisper small multilingual ONNX INT8",
        "files": {
            "small-encoder.int8.onnx": (112442483, "4cbe7b22fa9026b843b60a68640c747de05bafb1a11b57edc0e66c232d9f33a9"),
            "small-decoder.int8.onnx": (262226114, "acad50b5c782696e91b55914cc5ab4f756f1532f76e22aa6fc615f39fb69a8ee"),
            "small-tokens.txt": (816730, "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"),
        },
    },
}
SENSEVOICE_TAG = re.compile(r"<\|[^<>\r\n]*\|>")
WHISPER_BYTE_MARKER = re.compile(r"__WBYTE_([0-9a-f]*)__")


def byte_safe_token_table(source: Path) -> tuple[Path, str]:
    """Preserve BPE bytes across sherpa's per-token UTF-8 normalization.

    IDs and decoder weights stay unchanged. The output-only symbol table
    carries ASCII hex until all generated pieces can be joined in Python.
    """
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        encoded, token_id = line.split()
        # The official table represents the empty token 50256 with '='.
        original = b"" if encoded == "=" else base64.b64decode(encoded, validate=True)
        marker = "__WBYTE_" + original.hex() + "__"
        lines.append(base64.b64encode(marker.encode("ascii")).decode("ascii") + " " + token_id)
    contents = ("\n".join(lines) + "\n").encode("ascii")
    output = source.with_name(source.stem + ".byte-safe.txt")
    if output.exists():
        if output.read_bytes() != contents:
            raise RuntimeError(f"Existing derived token table differs; preserved: {output}")
    else:
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_bytes(contents)
        temporary.replace(output)
    return output, hashlib.sha256(contents).hexdigest()


def restore_whisper_bytes(text: str) -> str:
    if not text:
        return ""
    matches = list(WHISPER_BYTE_MARKER.finditer(text))
    if "".join(item.group() for item in matches) != text:
        raise ValueError("Whisper output contains data outside byte-safe token markers")
    # Whisper tokens are byte fragments. Decode only after concatenating them.
    # 'replace' matches the reference tokenizer's handling of malformed output.
    return b"".join(bytes.fromhex(item[1]) for item in matches).decode("utf-8", "replace")


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_memory() -> dict:
    """Read this evaluation process only; no child/helper memory is included."""
    if not hasattr(ctypes, "WinDLL"):
        return {"available": False}

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return {"available": False, "win32_error": ctypes.get_last_error()}
    return {"available": True, "working_set_bytes": counters.WorkingSetSize,
            "peak_working_set_bytes": counters.PeakWorkingSetSize,
            "scope": "This CPU evaluation process only"}


def compare(candidate_id: str, models_root: Path, input_path: Path, output: Path,
            small_reference: Path | None, threads: int,
            qwen_language: str = "auto", qwen_max_new_tokens: int = 512,
            whisper_byte_safe_tokens: bool = False) -> None:
    import sherpa_onnx

    candidate = EVALUATION_CANDIDATES[candidate_id]
    if whisper_byte_safe_tokens and "prefix" not in candidate:
        raise ValueError("Byte-safe token output is available only for Whisper")
    directory = models_root / candidate["directory"]
    files = verify_candidate(candidate, directory)
    saved = json.loads(input_path.read_text(encoding="utf-8"))
    samples = saved.get("samples", [])
    if not samples or any(item.get("source_lang") not in ("zh", "ja") for item in samples):
        raise ValueError("Saved diagnostic samples must contain Japanese or Chinese speech")
    small_by_id = {}
    if small_reference is not None:
        previous = json.loads(small_reference.read_text(encoding="utf-8"))
        small_by_id = {item["id"]: item for item in previous.get("samples", [])}
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "loading", "started_at": datetime.now(TZ).isoformat(timespec="milliseconds"),
        "candidate": candidate_id, "model": candidate["name"], "repo": candidate["repo"],
        "revision": candidate["revision"], "model_files": files,
        "conversion_provider": candidate.get("conversion_provider", "sherpa-onnx maintainer"),
        "input_results": str(input_path.resolve()), "small_reference": str(small_reference.resolve()) if small_reference else None,
        "provider": "cpu", "num_threads": threads, "task": "transcribe", "decoding_method": "greedy_search",
        "sherpa_onnx_version": importlib.metadata.version("sherpa-onnx"),
        "ground_truth": False, "audio_uploaded": False, "audio_devices_opened": False,
        "timing_scope": "Saved-segment CPU inference; excludes capture/VAD endpointing/translation. Concurrent local work may affect timings.",
        "samples": [],
    }
    write_report(output, report)
    recognizers = {}
    load_started = time.perf_counter()
    try:
        for language in sorted({item["source_lang"] for item in samples}):
            if candidate.get("kind") == "sensevoice":
                recognizers[language] = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(directory / "model.int8.onnx"), tokens=str(directory / "tokens.txt"),
                    language=language, use_itn=True, sample_rate=16000,
                    num_threads=threads, provider="cpu", debug=False,
                )
            elif candidate.get("kind") == "qwen3_asr":
                recognizers[language] = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                    conv_frontend=str(directory / "conv_frontend.onnx"),
                    encoder=str(directory / "encoder.int8.onnx"), decoder=str(directory / "decoder.int8.onnx"),
                    tokenizer=str(directory / "tokenizer"), sample_rate=16000, feature_dim=128,
                    num_threads=threads, provider="cpu", decoding_method="greedy_search", debug=False,
                    max_total_len=512, max_new_tokens=qwen_max_new_tokens,
                    temperature=0.000001, top_p=0.8, seed=42,
                )
                # Qwen has no constructor language argument; v1.13.8 supports
                # a per-stream language option. Keep automatic as the baseline.
                report["language_selection"] = (
                    "automatic (source labels retained only for comparison)" if qwen_language == "auto"
                    else "per-stream source language (Japanese/Chinese)"
                )
                report["generation"] = {"max_total_len": 512, "max_new_tokens": qwen_max_new_tokens,
                                         "temperature": 0.000001, "top_p": 0.8, "seed": 42}
                report["qwen_token_count_scope"] = (
                    "Native result.tokens follows protocol-prefix cleanup. A count below max_new_tokens "
                    "does not prove no truncation; use a conservative 16-token margin."
                )
            else:
                prefix = candidate["prefix"]
                token_path = directory / f"{prefix}-tokens.txt"
                if whisper_byte_safe_tokens:
                    token_path, derived_sha = byte_safe_token_table(token_path)
                    report["whisper_output_recovery"] = {
                        "method": "ASCII hex token table, concatenate bytes before UTF-8 decode",
                        "derived_token_table": str(token_path.resolve()), "sha256": derived_sha,
                        "original_token_ids_preserved": True, "original_model_files_modified": False,
                    }
                recognizers[language] = sherpa_onnx.OfflineRecognizer.from_whisper(
                    encoder=str(directory / f"{prefix}-encoder.int8.onnx"),
                    decoder=str(directory / f"{prefix}-decoder.int8.onnx"),
                    tokens=str(token_path), language=language, task="transcribe",
                    num_threads=threads, provider="cpu", decoding_method="greedy_search", debug=False,
                )
        report.update(status="running", model_load_seconds=time.perf_counter() - load_started)
        report["memory_after_model_load"] = process_memory()
        write_report(output, report)
        print(json.dumps({"candidate": candidate_id, "status": "loaded",
                          "model_load_seconds": report["model_load_seconds"]}), flush=True)
        for index, item in enumerate(samples, 1):
            audio_path = Path(item["audio_path"])
            if not audio_path.is_absolute():
                audio_path = input_path.parent / audio_path
            audio = read_wave(audio_path)
            recognizer = recognizers[item["source_lang"]]
            started = time.perf_counter()
            stream = recognizer.create_stream()
            if candidate.get("kind") == "qwen3_asr" and qwen_language == "source":
                stream.set_option("language", {"ja": "Japanese", "zh": "Chinese"}[item["source_lang"]])
            stream.accept_waveform(16000, audio)
            recognizer.decode_stream(stream)
            raw_text = stream.result.text
            restored_text = restore_whisper_bytes(raw_text) if whisper_byte_safe_tokens else raw_text
            text = SENSEVOICE_TAG.sub("", restored_text).strip()
            elapsed = time.perf_counter() - started
            production = item.get("production_text", "")
            small = small_by_id.get(item["id"], {}).get("reference_text")
            result = {
                "id": item["id"], "channel": item["channel"], "source_lang": item["source_lang"],
                "audio_path": str(audio_path.resolve()), "audio_duration_seconds": audio.size / 16000,
                "sensevoice_text": production, "whisper_small_text": small,
                "raw_candidate_text": raw_text, "candidate_text": text,
                "processing_seconds": elapsed, "rtf": elapsed / (audio.size / 16000),
                "decoded_token_count": len(stream.result.tokens),
                "generation_limit_reached": (
                    True if candidate.get("kind") == "qwen3_asr" and len(stream.result.tokens) >= qwen_max_new_tokens
                    else None
                ),
                "possibly_truncated": (
                    candidate.get("kind") == "qwen3_asr" and len(stream.result.tokens) >= qwen_max_new_tokens - 16
                ),
                "restored_utf8_replacement_characters": restored_text.count("\ufffd"),
                "historical_small_processing_seconds": small_by_id.get(item["id"], {}).get("reference_processing_seconds"),
                "production_pipeline_latency_seconds": item.get("pipeline_latency_seconds"),
                "difference_from_sensevoice": differences(production, text),
                "difference_from_small": differences(small, text) if small is not None else None,
            }
            report["samples"].append(result)
            write_report(output, report)
            print(json.dumps({"candidate": candidate_id, "sample": index, "total": len(samples),
                              "audio_seconds": round(audio.size / 16000, 3),
                              "processing_seconds": round(elapsed, 3)}), flush=True)
        timings = [item["processing_seconds"] for item in report["samples"]]
        total_audio = sum(item["audio_duration_seconds"] for item in report["samples"])
        report.update(status="completed", total_audio_seconds=total_audio,
                      total_processing_seconds=sum(timings), aggregate_rtf=sum(timings) / total_audio,
                      processing_seconds_summary={"min": min(timings), "median": statistics.median(timings), "max": max(timings)},
                      finished_at=datetime.now(TZ).isoformat(timespec="milliseconds"))
        report["memory_after_inference"] = process_memory()
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      finished_at=datetime.now(TZ).isoformat(timespec="milliseconds"))
        write_report(output, report)
        raise
    finally:
        # Dropping local recognizer references releases their CPU ONNX sessions.
        recognizers.clear()
    write_report(output, report)
    print(json.dumps({"candidate": candidate_id, "status": "completed", "output": str(output.resolve()),
                      "aggregate_rtf": report["aggregate_rtf"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=EVALUATION_CANDIDATES, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "model-cache" / "asr-evaluation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-reference", type=Path)
    parser.add_argument("--num-threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--qwen-language", choices=("auto", "source"), default="auto",
                        help="Qwen only: optionally set the per-stream source language")
    parser.add_argument("--qwen-max-new-tokens", type=int, choices=(128, 512), default=512)
    parser.add_argument("--whisper-byte-safe-tokens", action="store_true",
                        help="Preserve byte-fragment BPE output through sherpa UTF-8 token normalization")
    args = parser.parse_args()
    compare(args.candidate, args.models.resolve(), args.input.resolve(), args.output.resolve(),
            args.small_reference.resolve() if args.small_reference else None, args.num_threads,
            args.qwen_language, args.qwen_max_new_tokens, args.whisper_byte_safe_tokens)


if __name__ == "__main__":
    main()
