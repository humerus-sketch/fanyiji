"""Serial, subscription-only model comparison using the production translation path.

Public frozen text only. No capture, profile changes, training export or implicit
retry. References are written to review artifacts and never sent to the engine.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fanyiji.codex_engine import CodexTranslationEngine

WARMUP = {"source_text": "今日はいい天気ですね。", "source_lang": "ja", "target_lang": "zh"}
BACKEND_KEYS = ("backend", "model_id", "model_name", "auth_mode", "subscription_only",
                "requires_network", "reasoning_effort", "speed", "service_tier")


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_corpus(path: Path) -> tuple[list[dict[str, Any]], str]:
    checksum = digest(path)
    sidecar = path.with_suffix(".sha256")
    if sidecar.exists() and sidecar.read_text(encoding="ascii").strip() != checksum:
        raise ValueError("Frozen corpus checksum mismatch")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    samples = value.get("samples") if isinstance(value, dict) else value
    if not isinstance(samples, list) or not samples:
        raise ValueError("Corpus must contain a nonempty samples list")
    seen: set[str] = set()
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Every sample must be an object")
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
            raise ValueError("Sample ids must be unique nonempty strings")
        seen.add(sample_id)
        if (row.get("source_lang"), row.get("target_lang")) not in (("ja", "zh"), ("zh", "ja")):
            raise ValueError("Only ja-to-zh and zh-to-ja are supported")
        if not isinstance(row.get("source_text"), str) or not row["source_text"].strip():
            raise ValueError("Source text must be nonempty")
        if len(row["source_text"].strip()) > CodexTranslationEngine.MAX_TEXT_CHARACTERS:
            raise ValueError("Source text exceeds production engine limit")
        if not isinstance(row.get("source_context", ""), str):
            raise ValueError("source_context must be a string")
    return samples, checksum


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Incomplete result log at {path.name}:{number}; preserving it") from exc
        if not isinstance(row, dict):
            raise ValueError("Invalid result log object")
        rows.append(row)
    return rows


def safe_catalog(rows: Any) -> list[dict[str, Any]]:
    result = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            continue
        result.append({"model": row["model"], "displayName": row.get("displayName"),
                       "isDefault": row.get("isDefault"),
                       "defaultReasoningEffort": row.get("defaultReasoningEffort"),
                       "supportedReasoningEfforts": [option["reasoningEffort"]
                           for option in row.get("supportedReasoningEfforts", [])
                           if isinstance(option, dict) and isinstance(option.get("reasoningEffort"), str)]})
    return result


def check_backend(info: dict[str, Any], model: str, effort: str, speed: str) -> dict[str, Any]:
    if info.get("auth_mode") != "chatgpt" or info.get("subscription_only") is not True:
        raise ValueError("Subscription authentication was not verified; API Key is forbidden")
    if info.get("model_id") != model:
        raise ValueError("Actual model differs from requested model; comparison stopped")
    if info.get("reasoning_effort") != effort or info.get("speed") != speed:
        raise ValueError("Actual effort/speed differs from requested settings")
    catalog = safe_catalog(info.get("available_models"))
    selected = next((row for row in catalog if row["model"] == model), None)
    if selected is None or effort not in selected["supportedReasoningEfforts"]:
        raise ValueError("Requested model/effort is absent from current account catalog")
    return {key: info.get(key) for key in BACKEND_KEYS}


def error_kind(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)) or any(
        marker in message for marker in ("json", "空译文", "译文过长", "译文格式", "无效响应", "无效翻译", "无法识别的响应")
    ):
        return "response_protocol"
    if isinstance(exc, (ValueError, TypeError)) or any(marker in message for marker in (
        "模型不同", "配置不同", "不支持该思考强度", "未采用标准速度", "不支持 fast", "无法启用 fast",
        "模型在当前账号不可用", "没有可用的 codex 翻译模型",
    )):
        return "configuration"
    # Generic failure guidance mentions network/account quota as possibilities;
    # this does not prove quota exhaustion. Only classify explicit quota errors.
    if any(marker in message for marker in ("额度已用完", "额度用完", "quota exceeded", "rate limit",
                                             "usagelimitexceeded", "sessionbudgetexceeded")):
        return "quota"
    if any(marker in message for marker in ("登录已失效", "api key", "仅支持 chatgpt 订阅登录", "unauthorized",
                                             "authentication failed", "authentication was not verified")):
        return "authentication"
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(
        marker in message for marker in ("网络", "超时", "timeout", "连接", "transport", "翻译失败", "请求失败")
    ):
        return "transport_or_remote"
    return "model_or_unknown"


def safe_error(exc: Exception) -> dict[str, str]:
    # The production engine emits controlled guidance. Unknown third-party errors
    # are represented by class only, so account/transport payloads cannot leak.
    kind = error_kind(exc)
    return {"error_type": type(exc).__name__, "error_kind": kind}


def request(engine: Any, row: dict[str, Any], model: str, effort: str, speed: str,
            clock: Callable[[], float]) -> dict[str, Any]:
    started = clock()
    result: dict[str, Any] = {"id": row.get("id", "warmup"), "started_at": timestamp(),
                              "source_lang": row["source_lang"], "target_lang": row["target_lang"]}
    try:
        translation = engine.translate(row["source_text"], row["source_lang"], row["target_lang"],
                                       context=row.get("source_context", ""))
        backend = check_backend(engine.backend_info, model, effort, speed)
        if not isinstance(translation, str) or not translation.strip():
            raise RuntimeError("Codex 返回了空译文")
        result.update(status="success", translation=translation, model_actual=backend["model_id"])
    except Exception as exc:
        result.update(status="error", **safe_error(exc))
    result["seconds"] = round(clock() - started, 6)
    result["rpc_metrics"] = {key: round(value, 6) for key, value in engine.last_request_metrics.items()
                             if isinstance(key, str) and isinstance(value, (int, float))
                             and not isinstance(value, bool) and math.isfinite(value)}
    return result


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(row["seconds"] for row in rows if row.get("status") == "success")
    return {"attempted": len(rows), "successful": len(latencies), "failed": len(rows) - len(latencies),
            "success_seconds_total": round(sum(latencies), 6),
            "attempt_seconds_total_including_failures": round(sum(row["seconds"] for row in rows), 6),
            "median_seconds": statistics.median(latencies) if latencies else None,
            "p90_seconds_nearest_rank": latencies[max(0, math.ceil(len(latencies) * .9) - 1)] if latencies else None,
            "min_seconds": min(latencies) if latencies else None,
            "max_seconds": max(latencies) if latencies else None}


def write_summary(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    runs = []
    for model in manifest["models"]:
        all_rows = []
        startup = []
        warmup = []
        run_wall = []
        for round_index in range(1, manifest["rounds"] + 1):
            folder = directory / f"round-{round_index:02d}" / model
            rows = read_rows(folder / "results.jsonl")
            all_rows.extend(rows)
            startup.extend(read_rows(folder / "startup.jsonl"))
            warmup.extend(read_rows(folder / "warmup.jsonl"))
            run_wall.extend(read_rows(folder / "run-wall.jsonl"))
        run = {"model": model, **latency_summary(all_rows), "startup": startup, "warmup": warmup,
               "expected_samples": manifest["sample_count"] * manifest["rounds"],
               "status": "completed" if len(all_rows) == manifest["sample_count"] * manifest["rounds"] else "partial",
               "accuracy_score": None, "accuracy_status": "awaiting_blind_review",
               "startup_seconds_total": round(sum(row["seconds"] for row in startup), 6),
               "warmup_seconds_total": round(sum(row["seconds"] for row in warmup), 6),
               "run_wall_seconds_total_including_startup_warmup_close": round(sum(row["seconds"] for row in run_wall), 6),
               "directions": {f"{source}_to_{target}": latency_summary([
                   row for row in all_rows if (row["source_lang"], row["target_lang"]) == (source, target)])
                   for source, target in (("ja", "zh"), ("zh", "ja"))}}
        runs.append(run)
    summary = {"generated_at": timestamp(), "corpus_sha256": manifest["corpus_sha256"],
               "scope": "Serial production text translation; excludes ASR/audio and subtitle batching",
               "latency_excludes_startup_and_warmup": True, "automatic_accuracy_score": False,
               "models": runs}
    save(directory / "summary.json", summary)
    return summary


def export_blind(directory: Path, samples: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    target = directory / "blind-review.json"
    key_path = directory / "blind-key.json"
    if target.exists() or key_path.exists():
        raise FileExistsError("Preserving existing blind review/key")
    models = manifest["models"].copy()
    random.Random(manifest["seed"] + 19).shuffle(models)
    mapping = {model: f"Model-{index + 1:02d}" for index, model in enumerate(models)}
    cases = {row["id"]: row for row in samples}
    entries = []
    for model in models:
        for index in range(1, manifest["rounds"] + 1):
            for result in read_rows(directory / f"round-{index:02d}" / model / "results.jsonl"):
                case = cases[result["id"]]
                entries.append({"alias": mapping[model], "round": index, "id": result["id"],
                                "source_lang": case["source_lang"], "target_lang": case["target_lang"],
                                "source_text": case["source_text"], "source_context": case.get("source_context", ""),
                                "reference_translation": case.get("reference_translation"),
                                "expected_assertions": case.get("expected_assertions", []),
                                "translation": result.get("translation"), "status": result["status"],
                                "meaning_score": None, "naturalness_score": None,
                                "critical_error": None, "review_notes": ""})
    random.Random(manifest["seed"] + 23).shuffle(entries)
    save(target, {"corpus_sha256": manifest["corpus_sha256"], "samples": entries,
                  "reference_is_not_necessarily_human_ground_truth": True})
    save(key_path, {alias: model for model, alias in mapping.items()})


def run(corpus: Path, models: list[str], output: Path, *, effort: str = "low",
        speed: str = "standard", seed: int = 20261009, rounds: int = 1,
        execute: bool = False, blind_export: bool = False,
        engine_factory: Callable[..., Any] = CodexTranslationEngine,
        clock: Callable[[], float] = time.monotonic) -> int:
    samples, checksum = read_corpus(corpus)
    if not models or len(set(models)) != len(models) or any(not re.fullmatch(r"[A-Za-z0-9._-]+", model) for model in models):
        raise ValueError("Provide unique safe model ids")
    if rounds < 1 or rounds > 2:
        raise ValueError("rounds must be 1 or 2 (second round reverses model order)")
    if not effort or speed not in ("standard", "fast"):
        raise ValueError("Explicit effort and supported speed are required")
    manifest = {"schema_version": 1, "corpus_sha256": checksum, "sample_count": len(samples),
                "models": models, "effort": effort, "speed": speed, "seed": seed, "rounds": rounds,
                "production_prompt": True, "references_sent_to_model": False,
                "audio_capture": False, "api_key_allowed": False,
                "training_export_allowed": False, "warmup": WARMUP}
    if not execute:
        print(json.dumps({"plan": manifest, "output": str(output), "execute": False}, ensure_ascii=False), flush=True)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing run manifest differs; choose a new output directory")
    elif any(output.iterdir()):
        raise ValueError("Nonempty output has no manifest; preserving existing files")
    else:
        save(manifest_path, manifest)
        (output / "corpus.json").write_bytes(corpus.read_bytes())
    ordered = samples.copy()
    random.Random(seed).shuffle(ordered)
    transport_failures = 0
    stopped = False
    try:
        for round_index in range(1, rounds + 1):
            for model in (models if round_index == 1 else models[::-1]):
                folder = output / f"round-{round_index:02d}" / model
                folder.mkdir(parents=True, exist_ok=True)
                rows_path = folder / "results.jsonl"
                prior = read_rows(rows_path)
                completed = {row["id"] for row in prior}
                if len(completed) != len(prior) or not completed <= {row["id"] for row in samples}:
                    raise ValueError("Duplicate/unknown sample in prior result log; preserving it")
                pending = [row for row in ordered if row["id"] not in completed]
                if not pending:
                    continue
                engine = engine_factory(model=model, effort=effort, speed=speed)
                began = clock()
                try:
                    engine.preload()
                    info = engine.backend_info
                    backend = check_backend(info, model, effort, speed)
                    append(folder / "startup.jsonl", {"status": "success", "started_at": timestamp(),
                            "seconds": round(clock() - began, 6), "backend": backend})
                    if not (output / "account-model-catalog.json").exists():
                        save(output / "account-model-catalog.json", safe_catalog(info.get("available_models")))
                    transport_failures = 0
                    if not read_rows(folder / "warmup.jsonl"):
                        warmup = request(engine, WARMUP, model, effort, speed, clock)
                        append(folder / "warmup.jsonl", warmup)
                        if warmup.get("error_kind") == "configuration":
                            stopped = True
                            break
                        transport_failures = int(warmup.get("error_kind") in ("quota", "authentication", "transport_or_remote"))
                    for row in pending:
                        result = request(engine, row, model, effort, speed, clock)
                        append(rows_path, result)
                        print(json.dumps({"model": model, "round": round_index, "id": row["id"],
                                          "status": result["status"], "seconds": result["seconds"],
                                          "error_kind": result.get("error_kind")}, ensure_ascii=False), flush=True)
                        if result.get("error_kind") in ("quota", "authentication", "transport_or_remote"):
                            transport_failures += 1
                        else:
                            transport_failures = 0
                        if transport_failures >= 2 or result.get("error_kind") == "configuration":
                            stopped = True
                            break
                except Exception as exc:
                    append(folder / "startup.jsonl", {"status": "error", "started_at": timestamp(),
                            "seconds": round(clock() - began, 6), **safe_error(exc)})
                    kind = error_kind(exc)
                    transport_failures = transport_failures + 1 if kind in ("quota", "authentication", "transport_or_remote") else 0
                    stopped = kind == "configuration" or transport_failures >= 2
                finally:
                    engine.close()
                    append(folder / "run-wall.jsonl", {"seconds": round(clock() - began, 6),
                                                        "ended_at": timestamp()})
                    write_summary(output, manifest)
                if stopped:
                    break
            if stopped:
                break
    finally:
        summary = write_summary(output, manifest)
    if blind_export:
        export_blind(output, samples, manifest)
    print(json.dumps({"output": str(output), "stopped_early": stopped,
                      "completed": all(row["status"] == "completed" for row in summary["models"])}, ensure_ascii=False), flush=True)
    return int(stopped or any(row["status"] != "completed" or row["failed"] for row in summary["models"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="分别串行测试 ChatGPT 订阅模型；默认只显示执行计划")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--models", required=True, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--speed", choices=("standard", "fast"), default="standard")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--blind-export", action="store_true")
    args = parser.parse_args()
    return run(args.corpus, args.models, args.output, effort=args.effort, speed=args.speed,
               seed=args.seed, rounds=args.rounds, execute=args.execute, blind_export=args.blind_export)


if __name__ == "__main__":
    raise SystemExit(main())
