"""Serial subscription-only low/medium comparison on frozen public text."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fanyiji.codex_engine import CodexTranslationEngine
from fanyiji.paths import data_root


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate_limited(exc: Exception) -> bool:
    return bool(re.search(r"rate[ _-]?limit|too many requests|429|quota|usage[ _-]?limit|limit reached|限流|限额",
                          str(exc), re.IGNORECASE))


def summarize(rows: list[dict]) -> dict:
    success = [row for row in rows if row["status"] == "success"]
    warm = [row for row in success if not row["first_request_after_preload"]]
    summary = {"attempted": len(rows), "successful": len(success), "warm_successful": len(warm)}
    for label, selected in (("all", success), ("warm", warm)):
        for key in ("seconds", "first_output_seconds", "receive_seconds", "output_completion_seconds"):
            values = [row[key] for row in selected if isinstance(row.get(key), (int, float))]
            summary[f"{label}_{key}_median"] = statistics.median(values) if values else None
            summary[f"{label}_{key}_maximum"] = max(values) if values else None
    return summary


def run(corpus: Path, output: Path, *, execute: bool = False) -> int:
    original = corpus.read_bytes()
    value = json.loads(original.decode("utf-8-sig"))
    samples = value.get("samples") if isinstance(value, dict) else None
    if not isinstance(samples, list) or len(samples) != 10:
        raise ValueError("The frozen livestream comparison must contain exactly ten samples")
    identifiers = set()
    for row in samples:
        if (not isinstance(row, dict) or row.get("source_lang") not in ("ja", "mixed")
                or row.get("target_lang") != "zh" or not isinstance(row.get("source_text"), str)
                or not row["source_text"].strip() or not isinstance(row.get("source_context", ""), str)
                or not isinstance(row.get("id"), str) or row["id"] in identifiers):
            raise ValueError("Invalid or duplicate frozen source sample")
        identifiers.add(row["id"])
    manifest = {"model": "gpt-6-luna", "efforts": ["low", "medium"], "speed": "fast",
                "corpus_sha256": hashlib.sha256(original).hexdigest(), "sample_count_per_effort": 10,
                "order": ["low:01-05", "medium:01-05", "low:06-10", "medium:06-10"],
                "subscription_only": True, "api_key_allowed": False, "implicit_retry": False,
                "independent_context_generation_per_sample": True, "references_sent_to_model": False,
                "settings_modified": False, "new_audio_captured": False, "training_exported": False,
                "public_synthetic_text_only": True}
    if not execute:
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
        return 0
    output.mkdir(parents=True, exist_ok=False)
    settings_path = data_root() / "settings.json"
    settings_digest = sha256(settings_path) if settings_path.is_file() else None
    source_digest = sha256(ROOT / "src/fanyiji/codex_engine.py")
    report = {"manifest": manifest, "started_at": timestamp(), "status": "running", "preload": {},
              "settings_sha256_before": settings_digest, "codex_engine_sha256": source_digest,
              "rows": [], "limitations": [
                  "Given text only; excludes capture, segmentation, ASR and UI display.",
                  "The corpus is synthetic regression material, not a live audio ground truth.",
                  "Twenty serial requests in four chunks do not establish service-wide latency.",
                  "Fresh context generations isolate samples; this does not measure thread reuse.",
                  "First output observes streaming message arrival; receive time also contains server, network and client work."]}
    engines = {effort: CodexTranslationEngine("gpt-6-luna", effort, "fast") for effort in ("low", "medium")}
    (output / "corpus.json").write_bytes(original)
    def save() -> None:
        (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    completed = False
    try:
        for effort, engine in engines.items():
            started = time.perf_counter()
            engine.preload()
            info = engine.backend_info
            if (info.get("auth_mode") != "chatgpt" or info.get("subscription_only") is not True
                    or info.get("model_id") != "gpt-6-luna" or info.get("reasoning_effort") != effort
                    or info.get("speed") != "fast" or info.get("service_tier") not in ("fast", "priority")):
                raise RuntimeError("Actual model/auth/effort/tier differs; comparison stopped")
            report["preload"][effort] = {"seconds": time.perf_counter() - started,
                "backend": {key: info.get(key) for key in ("model_id", "reasoning_effort", "speed", "service_tier", "auth_mode", "subscription_only")}}
            save()
        attempted = {"low": 0, "medium": 0}
        for effort, start, stop in (("low", 0, 5), ("medium", 0, 5), ("low", 5, 10), ("medium", 5, 10)):
            engine = engines[effort]
            for index in range(start, stop):
                sample = samples[index]
                row = {"id": sample["id"], "effort": effort, "source_text": sample["source_text"],
                       "source_context": sample.get("source_context", ""), "source_lang": sample["source_lang"],
                       "target_lang": "zh", "reference_translation": sample.get("reference_translation"),
                       "review_points": sample.get("review_points", []), "started_at": timestamp(),
                       "first_request_after_preload": attempted[effort] == 0, "context_generation": index}
                attempted[effort] += 1
                began = time.perf_counter()
                try:
                    result = engine.translate(sample["source_text"], sample["source_lang"], "zh",
                        context=sample.get("source_context", ""), conversation_key="effort-evaluation:system",
                        context_generation=index)
                    row.update(status="success", translation=result)
                except Exception as exc:
                    row.update(status="error", error_type=type(exc).__name__, rate_limited=rate_limited(exc))
                row["seconds"] = time.perf_counter() - began
                row["completed_at"] = timestamp()
                metrics = engine.last_request_metrics
                row["rpc_metrics"] = metrics
                for key in ("first_output_seconds", "receive_seconds", "output_completion_seconds"):
                    row[key] = metrics.get(key)
                report["rows"].append(row)
                with (output / "results.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                save()
                print(json.dumps({"completed_requests": len(report["rows"]), "total": 20,
                    "effort": effort, "id": row["id"], "status": row["status"],
                    "seconds": round(row["seconds"], 4), "first_output_seconds": row["first_output_seconds"]}, ensure_ascii=False), flush=True)
                if row.get("rate_limited"):
                    report["stop_reason"] = "rate_limit_no_retry"
                    return 2
        completed = True
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, rate_limited=rate_limited(exc))
    finally:
        for engine in engines.values():
            engine.close()
        report["finished_at"] = timestamp()
        report["settings_hash_unchanged"] = (sha256(settings_path) if settings_path.is_file() else None) == settings_digest
        report["corpus_hash_unchanged"] = corpus.read_bytes() == original
        report["summary"] = {effort: summarize([row for row in report["rows"] if row["effort"] == effort]) for effort in engines}
        if not completed and report["status"] == "running":
            report["status"] = "stopped"
        save()
    return 0 if completed and all(row["status"] == "success" for row in report["rows"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "artifacts/livestream-optimization-20261010/corpus.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/e2e-latency-20261011/codex-effort")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return run(args.corpus, args.output, execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
