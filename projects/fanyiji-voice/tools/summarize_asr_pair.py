"""Summarize measured ASR/replay results without inventing accuracy scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import unicodedata


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text or "")
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def distribution(values):
    values = sorted(value for value in values if isinstance(value, (int, float)) and math.isfinite(value))
    return {"count": len(values), "p50": statistics.median(values) if values else None,
            "p95": values[max(0, math.ceil(.95 * len(values)) - 1)] if values else None,
            "max": max(values) if values else None}


def identity(row):
    return (row.get("channel"), row.get("segment_id"), row.get("revision", 1), row.get("context_generation"))


def finals(document):
    rows = [row for row in document["latency_records"] if row.get("valid") and row["stage"] == "final"]
    latest = {}
    for row in rows:
        key = (row["channel"], row["segment_id"], row["context_generation"])
        if key not in latest or row["revision"] >= latest[key]["revision"]:
            latest[key] = row
    return list(latest.values())


def summarize_run(document):
    rows = finals(document)
    sources = source_finals(document)
    return {"status": document["status"], "model_id": document["actual_models"]["asr"]["model_id"],
            "actual_models": document["actual_models"], "coverage": document["coverage"],
            "input_hash": document["replay_float32_sha256"], "settings_unchanged": document["settings_unchanged"],
            "configuration_unchanged": document["configuration_unchanged"],
            "input_audio_unchanged": document["input_audio_unchanged"], "errors": document["errors"],
            "model_load_seconds": document["model_load_seconds"],
            "all_final_count": len([r for r in document["latency_records"] if r["stage"] == "final"]),
            "latest_final_count": len(rows),
            "source_segment_count": len(sources),
            "unpainted_source_segment_ids": [segment["segment_id"] for segment in document["segments"]
                if segment["accepted"] and segment["segment_id"] not in {row["segment_id"] for row in sources}],
            "per_source_audio_start_to_final": distribution([row["start_to_final"] for row in sources]),
            "per_source_audio_end_to_final": distribution([row["end_to_final"] for row in sources]),
            "per_source_rows": sources,
            "recovered_asr_generation_limit_segments": [row["segment_id"] for row in document["recognition"]
                if any(attempt.get("error") and "128 token" in attempt["error"].get("message", "")
                       for attempt in row.get("quality", {}).get("attempts", []))],
            "audio_start_to_final": distribution([row["end_to_end_seconds"] for row in rows]),
            "audio_end_to_final": distribution([row["post_audio_seconds"] for row in rows]),
            "stages": {key: distribution([row["breakdown"].get(key) for row in rows]) for key in
                       ("audio_seconds", "segmentation_seconds", "asr_queue_seconds", "speech_filter_seconds",
                        "asr_seconds", "assembly_seconds", "translation_queue_seconds", "translation_seconds",
                        "source_confirmation_seconds", "ui_paint_seconds")}}


def source_finals(document):
    """Assign the actual latest final paint to each original VAD audio segment.

    Joined sentences may produce one subtitle from two source segments. Measuring
    both original sources independently keeps the same audio population, without
    dropping the slow joined case or counting that single paint as two requests.
    """
    originals = {identity(row): row for row in document["originals"]}
    painted_at = {}
    for row in finals(document):
        original = originals.get(identity(row))
        if original is None:
            continue
        for identifier in original.get("source_segment_ids", (row["segment_id"],)):
            painted_at[identifier] = max(painted_at.get(identifier, 0), row["timing"]["ui_painted_at"])
    origin = document["capture_started_monotonic"]
    return [{"segment_id": row["segment_id"], "start_sample": row["start_sample"], "end_sample": row["end_sample"],
             "start_to_final": painted_at[row["segment_id"]] - (origin + row["start_sample"] / 16000),
             "end_to_final": painted_at[row["segment_id"]] - (origin + row["end_sample"] / 16000)}
            for row in document["segments"] if row["segment_id"] in painted_at]


def interval_records(document):
    segments = {row["segment_id"]: (row["start_sample"], row["end_sample"]) for row in document["segments"]}
    originals = {identity(row): row for row in document["originals"]}
    result = {}
    for row in finals(document):
        original = originals.get(identity(row))
        if original is None:
            continue
        ids = original.get("source_segment_ids", (row["segment_id"],))
        if not all(identifier in segments for identifier in ids):
            continue
        result[tuple(segments[identifier] for identifier in ids)] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    sense, qwen = [read(directory / f"native-{name}.json") for name in ("sensevoice", "qwen")]
    paired = list(zip(sense["samples"], qwen["samples"]))
    same = len(sense["samples"]) == len(qwen["samples"]) and all(
        (a["id"], a["sha256"], a["audio_seconds"]) == (b["id"], b["sha256"], b["audio_seconds"])
        for a, b in paired)
    if not same:
        raise ValueError("Native ASR inputs differ")
    fields = ("model_id", "sample_count", "total_audio_seconds", "total_processing_seconds", "rtf",
              "p50_seconds", "p95_seconds", "max_seconds", "model_load_seconds", "empty_outputs", "errors",
              "process_memory", "settings_unchanged", "source_unchanged", "manifest_unchanged")
    native = {"same_inputs": same, "language_policy": "auto", "num_threads": 4,
              "sensevoice": {key: sense[key] for key in fields}, "qwen": {key: qwen[key] for key in fields},
              "qwen_to_sensevoice_compute_ratio": qwen["total_processing_seconds"] / sense["total_processing_seconds"],
              "normalized_exact_agreement_not_accuracy": sum(normalized(a["production_text"]) == normalized(b["production_text"]) for a, b in paired),
              "language_metadata": {name: dict(Counter(row["detected_language"] or "unknown" for row in document["samples"]))
                                    for name, document in (("sensevoice", sense), ("qwen", qwen))}}
    documents = {path.parent.name: read(path) for path in sorted(directory.glob("e2e-*/results.json"))}
    runs = {name: summarize_run(document) for name, document in documents.items()}
    grouped = defaultdict(list)
    source_grouped = defaultdict(list)
    for document in documents.values():
        if document["status"] == "completed":
            grouped[document["actual_models"]["asr"]["model_id"]].extend(finals(document))
            source_grouped[document["actual_models"]["asr"]["model_id"]].extend(source_finals(document))
    aggregate = {name: {"final_count": len(rows),
        "audio_start_to_final": distribution([row["end_to_end_seconds"] for row in rows]),
        "audio_end_to_final": distribution([row["post_audio_seconds"] for row in rows]),
        "stages": {key: distribution([row["breakdown"].get(key) for row in rows]) for key in
            ("segmentation_seconds", "asr_seconds", "translation_queue_seconds", "translation_seconds")}}
        for name, rows in grouped.items()}
    source_aggregate = {name: {"source_segment_count": len(rows),
        "audio_start_to_final": distribution([row["start_to_final"] for row in rows]),
        "audio_end_to_final": distribution([row["end_to_final"] for row in rows])}
        for name, rows in source_grouped.items()}
    comparisons = []
    for suffix in ("1", "2"):
        names = (f"e2e-sensevoice-{suffix}", f"e2e-qwen-{suffix}")
        if not all(name in documents for name in names):
            continue
        a, b = (documents[name] for name in names)
        if a["status"] != "completed" or b["status"] != "completed":
            continue
        boundaries = lambda document: [(row["start_sample"], row["end_sample"], row["float32_sha256"], row["accepted"]) for row in document["segments"]]
        interval_a, interval_b = interval_records(a), interval_records(b)
        common = sorted(set(interval_a) & set(interval_b))
        delta = [interval_b[key]["end_to_end_seconds"] - interval_a[key]["end_to_end_seconds"] for key in common]
        comparisons.append({"runs": list(names), "same_replayed_pcm": a["replay_float32_sha256"] == b["replay_float32_sha256"],
            "same_vad_segments": boundaries(a) == boundaries(b), "matched_final_audio_intervals": len(common),
            "qwen_minus_sensevoice_paired_seconds": distribution(delta),
            "paired_rows": [{"source_audio_intervals": key,
                "sensevoice_start_to_final": interval_a[key]["end_to_end_seconds"],
                "qwen_start_to_final": interval_b[key]["end_to_end_seconds"],
                "sensevoice_end_to_final": interval_a[key]["post_audio_seconds"],
                "qwen_end_to_final": interval_b[key]["post_audio_seconds"]} for key in common]})
    source_pairs = []
    for suffix in ("1", "2"):
        names = (f"e2e-sensevoice-{suffix}", f"e2e-qwen-{suffix}")
        if not all(name in documents and documents[name]["status"] == "completed" for name in names):
            continue
        a, b = ({(r["start_sample"], r["end_sample"]): r for r in source_finals(documents[name])} for name in names)
        common = sorted(set(a) & set(b))
        source_pairs.append({"runs": list(names), "matched_original_source_segments": len(common),
            "qwen_minus_sensevoice_seconds": distribution([b[key]["start_to_final"] - a[key]["start_to_final"] for key in common]),
            "paired_rows": [{"audio_interval": key, "sensevoice": a[key], "qwen": b[key]} for key in common]})
    report = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "native": native,
              "accuracy": read(directory / "accuracy-status.json"), "runs": runs,
              "end_to_end_aggregate": aggregate, "paired_comparisons": comparisons,
              "per_source_end_to_end_aggregate": source_aggregate, "per_source_paired_comparisons": source_pairs,
              "p95_definition": "nearest rank ceil(0.95*n); native P95 is original interpolated percentile",
              "limitations": ["No independent Japanese transcripts; no accuracy ranking is possible",
                  "Only ASR text is sent to Codex; audio remains local",
                  "Realtime WAV injection measures production VAD/filter/ASR/MT/visible Qt paint, not physical device latency or panel scan-out",
                  "Audio start means first sample of the production VAD segment including its pre-roll",
                  "Per-source measurements reuse the final merged paint for every original VAD source it covers; this is source waiting time, not additional translation requests",
                  "Continuous replay is a fixed 90-second excerpt; repetitions control input but not cloud/network timing",
                  "Final fragments can differ due to recognition-dependent sentence assembly; matched-interval comparisons are reported separately"]}
    (directory / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = ["# SenseVoice 与 Qwen 日语直播评测", "", "本轮实测仅支持速度结论；没有独立听写，准确率尚未评分。", "",
            "## 纯识别：同一批 28 段直播音频", "", "自动多语、CPU 4 线程；共 177.210 秒真实日语直播录音。加载单列，不包含断句、过滤、翻译或显示。", "",
            "| 指标 | SenseVoice INT8 | Qwen3-ASR 1.7B INT8 |", "|---|---:|---:|"]
    for label, key in (("单段中位耗时 / 秒", "p50_seconds"), ("单段 P95 / 秒", "p95_seconds"),
                       ("累计耗时 / 秒", "total_processing_seconds"), ("空输出", "empty_outputs"), ("异常", "errors")):
        rows.append(f"| {label} | {sense[key]:.3f} | {qwen[key]:.3f} |")
    rows += ["", "Qwen 出现的一段空输出只有 0.52 秒，未核听，不能直接判为漏识别。非空输出也不代表正确。", "",
             "## 连续录音到终稿实际显示", "", "同一 90 秒未掩蔽连续录音、原生产 VAD/讲话筛选/句段合并、同一 Codex 翻译参数。", "",
             "| 测试 | 终稿样本 | 声音首样本→终稿 P50/P95 | 该段音频结束→终稿 P50/P95 |", "|---|---:|---:|---:|"]
    for name, run in runs.items():
        a, b = run["audio_start_to_final"], run["audio_end_to_final"]
        fmt = lambda value: f"{value:.3f}" if value is not None else "未测得"
        rows.append(f"| {name} ({run['status']}) | {a['count']} | {fmt(a['p50'])} / {fmt(a['p95'])} 秒 | {fmt(b['p50'])} / {fmt(b['p95'])} 秒 |")
    rows += ["", "### 按同一批原始 VAD 音频段统计", "",
             "为避免两段合并成一条字幕改变样本分布，以下每个原始音频段都计到包含它的最终字幕真正绘制时刻。一条合并字幕可对应两个原始音频段，这不代表发送两次翻译请求。", "",
             "| 模型 | 原始音频段数 | 声音首样本→终稿 P50/P95 | 该段音频结束→终稿 P50/P95 |", "|---|---:|---:|---:|"]
    for name, stats in source_aggregate.items():
        a, b = stats["audio_start_to_final"], stats["audio_end_to_final"]
        rows.append(f"| {name} | {stats['source_segment_count']} | {a['p50']:.3f} / {a['p95']:.3f} 秒 | {b['p50']:.3f} / {b['p95']:.3f} 秒 |")
    rows += ["", "总延迟含说话时长，另列音频结束后的等待。未真实绘制、空识别、过滤、失败和被替换结果均保留在 JSON 覆盖率中，不当作更快成功。", "",
             "### 已确认的限制来源", ""]
    if runs:
        first = next(iter(runs.values()))["actual_models"]["translation"]
        rows.append(f"本轮固定翻译参数：{first['model_id']} / {first['reasoning_effort']} / {first['speed']} / service_tier={first['service_tier']}。")
    guards = [(name, run["recovered_asr_generation_limit_segments"]) for name, run in runs.items()
              if run["recovered_asr_generation_limit_segments"]]
    if guards:
        rows += ["", "Qwen 在两轮连续录音重放中均出现生成上限守卫，原始识别达到上限后进入同模型切段复核，后续音频因此排队。恢复状态标注 uncertain，恢复有文本不等于识别准确。", ""]
        rows += [f"- {name}：{', '.join(identifiers)}" for name, identifiers in guards]
    rows += ["", "逐个音频段可能因云端时延、生成内容和上下文变化出现反向结果；不能称 SenseVoice 每一句终稿都更快。每模型仅同一段 90 秒录音重复两轮，长尾与日语内容覆盖范围有限。", "",
             "## 决策边界", "", "SenseVoice 的计算速度明显占优；综合准确率最优仍未验证。不能用模型共识、译文流畅或 Codex 文字纠错替代原音参考。", "",
             "本机实时速度首选是 SenseVoice + 现有 Silero VAD + Codex 终稿，Qwen 保留可切换。由于准确率优先且缺独立参考，本轮不将速度结论升级为综合准确率验收，也未启用双识别自动复核。", "",
             "若后续补齐独立人工/已核音字幕，可通过 review.html 导出 references JSON 后重算 CER，并单独核查否定、人名、数字和游戏名。", "",
             "本轮未修改生产代码、模型文件、用户设置或默认模型。所有配置与音频完整性状态见 comparison.json。", "",
             "复现工具：tools/evaluate_live_asr_pair.py、tools/evaluate_e2e_replay.py、tools/evaluate_asr_accuracy.py、tools/summarize_asr_pair.py。", ""]
    (directory / "评测报告.md").write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps({"native": native, "end_to_end_aggregate": aggregate,
                      "per_source_end_to_end_aggregate": source_aggregate,
                      "paired_comparisons": [{k: v for k, v in item.items() if k != "paired_rows"} for item in comparisons]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
