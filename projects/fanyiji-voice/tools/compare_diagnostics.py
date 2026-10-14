"""Compare saved local translations without capture, uploads, or model changes.

Run with .model-venv when --fp32-source is supplied. The FP32 comparison uses
the original local PyTorch checkpoint, not dequantized CT2 INT8 weights.
Different backend/search results alone do not prove an INT8 quality regression.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
TZ = timezone(timedelta(hours=8))


@contextmanager
def offline_guard():
    """Block Python socket entry points in this process, not system networking."""
    attempts: list[str] = []

    def deny(*args, **kwargs):
        attempts.append("blocked_python_socket_call")
        raise RuntimeError("离线对照禁止任何 Python socket 网络调用")

    with ExitStack() as stack:
        for target in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, target, deny))
        for target in ("create_connection", "getaddrinfo"):
            stack.enter_context(patch.object(socket, target, deny))
        yield attempts


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # During collection only the last line may still be incomplete.
            if index == len(lines) - 1:
                break
            raise
        if isinstance(record, dict):
            records.append(record)
    return records


def read_diagnostics(path: Path) -> tuple[dict, list[dict]]:
    results = path / "results.json" if path.is_dir() else path
    if results.exists() and results.suffix.lower() == ".json":
        report = json.loads(results.read_text(encoding="utf-8-sig"))
        if not isinstance(report.get("samples"), list):
            raise ValueError("results.json 缺少 samples 列表")
        return {
            "path": str(results.resolve()), "status": report.get("status"),
            "collection_scope": report.get("collection_scope"),
            "settings_snapshot": report.get("settings_snapshot"),
            "complete": report.get("status") == "completed",
        }, report["samples"]
    directory = path if path.is_dir() else path.parent
    samples = {item["id"]: item for item in read_jsonl(directory / "segments.jsonl")}
    for event in read_jsonl(directory / "events.jsonl"):
        sample = samples.get(event.get("segment_id"))
        if sample is None:
            continue
        if event.get("kind") == "original":
            sample["production_text"] = event.get("text", "")
        elif event.get("kind") == "translation":
            sample["production_translation"] = event.get("text", "")
            sample["pipeline_latency_seconds"] = event.get("latency")
    return {
        "path": str(directory.resolve()), "status": "collection_in_progress",
        "collection_scope": "Local sidecar source pipeline; not an EXE UI export",
        "complete": False,
    }, list(samples.values())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions() -> dict:
    result = {"python": sys.version.split()[0]}
    for name in ("ctranslate2", "sentencepiece", "torch", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def official_tokenizer(source: Path):
    import sentencepiece
    from transformers.models.m2m_100 import tokenization_m2m_100

    def load_spm(path, kwargs):
        processor = sentencepiece.SentencePieceProcessor(**kwargs)
        if not processor.LoadFromSerializedProto(Path(path).read_bytes()):
            raise RuntimeError("无法加载本地官方 SentencePiece 模型")
        return processor

    # SentencePiece's native file loader cannot open this Unicode Windows path.
    tokenization_m2m_100.load_spm = load_spm
    return tokenization_m2m_100.M2M100Tokenizer.from_pretrained(
        str(source), local_files_only=True,
    )


def run_comparison(args, report: dict, samples: list[dict]) -> None:
    import ctranslate2
    from fanyiji.engines import _M2M100Tokenizer

    translation_dir = args.models / "translation"
    started = time.perf_counter()
    tokenizer = _M2M100Tokenizer.from_directory(translation_dir)
    translator = ctranslate2.Translator(
        str(translation_dir), device="cpu", compute_type="int8",
        inter_threads=1, intra_threads=args.threads,
    )
    report["load_seconds"]["ct2_int8"] = round(time.perf_counter() - started, 6)

    reference = None
    model = None
    torch = None
    if args.fp32_source is not None:
        import torch as torch_module
        from transformers import AutoModelForSeq2SeqLM

        torch = torch_module
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        source = args.fp32_source.resolve()
        required = ("config.json", "pytorch_model.bin", "vocab.json", "sentencepiece.bpe.model")
        missing = [str(source / name) for name in required if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError("本地 FP32 资产缺失；不会下载：" + ", ".join(missing))
        report["fp32_source"] = {
            "path": str(source), "checkpoint_size": (source / "pytorch_model.bin").stat().st_size,
            "tokenizer_files_match_production": {
                name: sha256(source / name) == sha256(translation_dir / name)
                for name in ("vocab.json", "sentencepiece.bpe.model")
            },
        }
        started = time.perf_counter()
        reference = official_tokenizer(source)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(source), local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        report["load_seconds"]["original_pytorch_fp32"] = round(time.perf_counter() - started, 6)

    selected_ids = set(args.fp32_ids.split(",")) if args.fp32_ids else None
    eligible = 0
    for sample in samples:
        item = dict(sample)
        item["ct2_int8"] = {}
        report["samples"].append(item)
        text = sample.get("production_text", "")
        if not sample.get("accepted", True) or not isinstance(text, str) or not text.strip():
            item["comparison_status"] = "skipped_missing_or_rejected_source_text"
            continue
        if args.limit is not None and eligible >= args.limit:
            item["comparison_status"] = "skipped_by_limit"
            continue
        eligible += 1
        source_lang, target_lang = sample.get("source_lang"), sample.get("target_lang")
        if source_lang not in ("zh", "ja") or target_lang not in ("zh", "ja"):
            item["comparison_status"] = "skipped_unsupported_language"
            continue
        tokens = tokenizer.encode(text.strip(), source_lang)
        item["source_token_count"] = len(tokens)
        if len(tokens) > 512:
            item["comparison_status"] = "skipped_source_over_512_tokens"
            continue
        if reference is not None:
            reference.src_lang = source_lang
            official_ids = reference(text.strip(), add_special_tokens=True)["input_ids"]
            official_tokens = reference.convert_ids_to_tokens(official_ids)
            item["tokenizer_parity"] = {
                "same_source_tokens": tokens == official_tokens,
                "local_token_count": len(tokens), "official_token_count": len(official_tokens),
                "source_prefix": tokens[0], "source_suffix": tokens[-1],
                "target_prefix": f"__{target_lang}__",
                "target_prefix_id": reference.get_lang_id(target_lang),
            }
            if tokens != official_tokens:
                item["tokenizer_parity"]["local_tokens"] = tokens
                item["tokenizer_parity"]["official_tokens"] = official_tokens
        for beam in (1, 2, 4):
            started = time.perf_counter()
            result = translator.translate_batch(
                [tokens], target_prefix=[[f"__{target_lang}__"]], beam_size=beam,
                max_input_length=512, max_decoding_length=256, num_hypotheses=1,
            )[0]
            output = tokenizer.decode(result.hypotheses[0])
            item["ct2_int8"][f"beam{beam}"] = {
                "text": output, "seconds": round(time.perf_counter() - started, 6),
                "output_token_count": len(result.hypotheses[0]),
                "matches_production_translation": output == sample.get("production_translation"),
            }
        item["ct2_beam_variants_identical"] = len({v["text"] for v in item["ct2_int8"].values()}) == 1
        if model is not None and (selected_ids is None or sample["id"] in selected_ids):
            inputs = reference(text.strip(), return_tensors="pt")
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, forced_bos_token_id=reference.get_lang_id(target_lang),
                    num_beams=2, do_sample=False, max_new_tokens=256,
                    length_penalty=1.0, use_cache=True,
                )[0].tolist()
            elapsed = time.perf_counter() - started
            language_ids = set(reference.lang_code_to_id.values())
            generated = [index for index in generated if index not in language_ids]
            output = reference.decode(generated, skip_special_tokens=True).strip()
            item["original_pytorch_fp32_beam2"] = {
                "text": output, "seconds": round(elapsed, 6),
                "matches_ct2_int8_beam2": output == item["ct2_int8"]["beam2"]["text"],
            }
        item["comparison_status"] = "compared"
        print(json.dumps({"sample_id": sample["id"], "status": "compared",
                          "tokenizer_parity": item.get("tokenizer_parity", {}).get("same_source_tokens"),
                          "fp32_compared": "original_pytorch_fp32_beam2" in item}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="诊断目录或 results.json")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4, choices=range(2, 5))
    parser.add_argument("--limit", type=int, help="最多比较多少条有效转写，默认全部")
    parser.add_argument("--fp32-source", type=Path, help="已在本机的原始 M2M100 PyTorch 目录")
    parser.add_argument("--fp32-ids", help="只对这些完整 segment id 跑 FP32，逗号分隔")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于零")
    input_metadata, samples = read_diagnostics(args.input.resolve())
    output = args.output or (
        (args.input if args.input.is_dir() else args.input.parent) / "translation_comparison.json"
    )
    manifest = args.models / "manifest.json"
    report = {
        "status": "running", "created_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "input": input_metadata, "versions": versions(), "threads": args.threads,
        "model_manifest_sources": json.loads(manifest.read_text(encoding="utf-8")).get("sources") if manifest.exists() else None,
        "scope": "Saved source text translation comparison only; no audio capture or uploads; not an EXE UI export",
        "independent_asr": {"status": "not_performed_by_this_tool", "reason": "A second local ASR model or human transcript is needed; production ASR text is not ground truth"},
        "network_guard": {"scope": "Python socket connect/connect_ex/sendto/create_connection/getaddrinfo in this process; not an OS-level audit"},
        "limitations": [
            "No human reference transcript or translation; output differences are not accuracy scores",
            "PyTorch FP32 and CT2 INT8 use different inference/search implementations; differences do not isolate quantization",
            "Timing is a single local run with the production app possibly running; not a latency benchmark",
        ],
        "parameters": {"ct2_beams": [1, 2, 4], "target_prefix": "__target_lang__", "max_input_length": 512,
                       "max_decoding_length": 256, "fp32_num_beams": 2, "fp32_max_new_tokens": 256,
                       "fp32_generation_early_stopping": "inherited from original local model config"},
        "load_seconds": {}, "samples": [],
    }
    started = time.perf_counter()
    error = None
    with offline_guard() as attempts:
        try:
            run_comparison(args, report, samples)
            report["status"] = "completed"
        except Exception as exc:
            error = exc
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
        report["network_guard"]["blocked_attempts"] = len(attempts)
    compared = [item for item in report["samples"] if item.get("comparison_status") == "compared"]
    report["summary"] = {
        "input_samples": len(samples), "compared_samples": len(compared),
        "tokenizer_checked": sum("tokenizer_parity" in item for item in compared),
        "tokenizer_mismatches": sum(not item.get("tokenizer_parity", {}).get("same_source_tokens", True) for item in compared),
        "ct2_beam2_matches_production": sum(item["ct2_int8"]["beam2"]["matches_production_translation"] for item in compared),
        "fp32_compared": sum("original_pytorch_fp32_beam2" in item for item in compared),
        "fp32_matches_ct2_int8_beam2": sum(item.get("original_pytorch_fp32_beam2", {}).get("matches_ct2_int8_beam2", False) for item in compared),
    }
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output.resolve()), "summary": report["summary"]}, ensure_ascii=False), flush=True)
    if error is not None:
        print(report["error"], file=sys.stderr)
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
