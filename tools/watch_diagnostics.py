"""Watch saved capture snapshots and run sequential local comparison batches.

This tool uses only the standard library. It does not capture audio, download
models, upload content, or change the application. Failed batches retain their
inputs/logs and are not automatically retried, including after a restart.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import time
import unicodedata
import wave
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))


def timestamp() -> str:
    return datetime.now(TZ).isoformat(timespec="milliseconds")


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ProcessWatch:
    """Retain a Windows process handle so PID reuse cannot extend monitoring."""

    def __init__(self, pid: int) -> None:
        if os.name != "nt":
            raise RuntimeError("capture_pid monitoring requires Windows")
        from ctypes import wintypes

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x00100000, False, pid)
        if not self.handle and ctypes.get_last_error() != 87:
            raise OSError(ctypes.get_last_error(), "Cannot monitor capture process")

    def running(self) -> bool:
        if not self.handle:
            return False
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result == 0xFFFFFFFF:
            raise OSError(ctypes.get_last_error(), "Capture process wait failed")
        return result == 258

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


@contextmanager
def single_watcher(directory: Path):
    import msvcrt

    # OS releases the byte lock after a crash; the file need not be deleted.
    with (directory / "watch.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("This capture directory already has an active watcher") from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def normalized(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def repetition(text: str) -> list[dict[str, Any]]:
    clean = normalized(text)
    flags = []
    for match in re.finditer(r"(.{1,12}?)\1{2,}", clean):
        if len(match[0]) >= 6 and len(match[0]) >= len(clean) * 0.35:
            flags.append({"phrase": match[1], "count": len(match[0]) // len(match[1]),
                          "label": "疑似连续重复，需人工复核；不是错误率"})
    return flags


def eligible_samples(report: dict[str, Any], source: Path, handled: set[str], limit: int) -> list[dict[str, Any]]:
    selected = []
    seen = set(handled)
    for item in report.get("samples", []):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] in seen:
            continue
        if item.get("accepted") is not True or not all(isinstance(item.get(key), str) and item[key].strip()
                                                     for key in ("production_text", "production_translation")):
            continue
        if item.get("source_lang") not in ("ja", "zh") or item.get("target_lang") not in ("ja", "zh"):
            continue
        path = Path(item.get("audio_path", ""))
        path = (path if path.is_absolute() else source / path).resolve()
        # Never infer a completed segment from the still-open continuous WAV.
        if not path.is_relative_to(source / "segments") or not path.is_file():
            continue
        try:
            with wave.open(str(path), "rb") as audio:
                valid = (audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getcomptype()) == (1, 16000, 2, "NONE")
                frames = audio.getnframes()
            if not valid or frames <= 0 or path.stat().st_size < 44 + frames * 2:
                continue
        except (OSError, EOFError, wave.Error):
            continue
        selected.append({**item, "audio_path": str(path)})
        seen.add(item["id"])
        if len(selected) == limit:
            break
    return selected


def load_batches(directory: Path) -> tuple[list[dict[str, Any]], set[str]]:
    records, handled = [], set()
    for batch in sorted((directory / "batches").glob("*")):
        if not batch.is_dir() or not (batch / "results.json").is_file():
            continue
        snapshot = read_json(batch / "results.json")
        ids = [item["id"] for item in snapshot.get("samples", [])]
        handled.update(ids)
        if (batch / "summary.json").is_file():
            record = read_json(batch / "summary.json")
        else:
            record = {"batch": batch.name, "sample_ids": ids, "status": "interrupted",
                      "error": "上次运行未完成；输入保留，不自动重跑", "tools": {}}
        if record.get("status") == "running":
            record = {**record, "status": "interrupted", "error": "上次监测运行中断；证据保留，不自动重跑"}
        records.append(record)
    return records, handled


def run_tool(command: list[str], batch: Path, name: str, output: Path, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    info: dict[str, Any] = {"command": command, "started_at": timestamp(), "output": str(output),
                            "stdout": str(batch / f"{name}.stdout.log"), "stderr": str(batch / f"{name}.stderr.log")}
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                   "TOKENIZERS_PARALLELISM": "false", "PYTHONUTF8": "1"}
    with Path(info["stdout"]).open("wb") as stdout, Path(info["stderr"]).open("wb") as stderr:
        process = None
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, env=environment,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            info["returncode"] = process.wait(timeout=timeout)
            report_status = read_json(output).get("status") if output.is_file() else "missing"
            info["report_status"] = report_status
            info["status"] = "completed" if info["returncode"] == 0 and report_status == "completed" else "failed"
        except subprocess.TimeoutExpired:
            info.update(status="failed", error="Local comparison exceeded its timeout; partial evidence retained")
        except Exception as exc:
            info.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    info.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
                    # Do not launch the next stage alongside an unreaped child.
                    atomic_json(batch / f"{name}.cleanup.json", info)
                    raise RuntimeError("Cannot stop/reap the comparison subprocess") from exc
            info["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return info


def optional_report(path: Path) -> dict[str, Any]:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def batch_statistics(batch: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    reference = optional_report(batch / "reference-whisper.json")
    translation = optional_report(batch / "translation_comparison.json")
    originals = {item["id"]: item for item in samples}
    compared = [item for item in reference.get("samples", []) if item.get("id") in originals]
    disagreements = [item["id"] for item in compared
                     if normalized(item.get("reference_text", "")) != normalized(originals[item["id"]]["production_text"])]
    flags = []
    for item in samples:
        detected = repetition(item["production_translation"])
        if detected:
            flags.append({"id": item["id"], "variant": "production", "flags": detected})
    for item in translation.get("samples", []):
        for variant, result in item.get("ct2_int8", {}).items():
            detected = repetition(result.get("text", ""))
            if detected:
                flags.append({"id": item["id"], "variant": variant, "flags": detected})
    return {"asr_compared_segments": len(compared), "asr_disagreement_ids": disagreements,
            "asr_disagreement_definition": "NFKC并去除标点空白后文本不同；模型分歧，非准确率或人工真值",
            "repetition_flags": flags, "repetition_flagged_ids": sorted({flag["id"] for flag in flags}),
            "ct2_compared_segments": sum(item.get("comparison_status") == "compared" for item in translation.get("samples", []))}


def run_batch(args, directory: Path, report: dict[str, Any], samples: list[dict[str, Any]], final: bool) -> dict[str, Any]:
    batch = directory / "batches" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S-%f")
    batch.mkdir(parents=True, exist_ok=False)
    snapshot = {**report, "samples": samples, "immutable_snapshot": True,
                "snapshot_created_at": timestamp(), "source_results": str(directory / "results.json")}
    atomic_json(batch / "results.json", snapshot)
    summary = {"batch": batch.name, "directory": str(batch), "created_at": timestamp(), "status": "running",
               "sample_ids": [item["id"] for item in samples], "sample_count": len(samples), "final_drain": final,
               "capture_status_at_snapshot": report.get("status"), "capture_raw_audio_seconds": report.get("raw_audio_seconds", {}),
               "tools": {}, "limitations": ["模型分歧不是准确率，Whisper不是人工真值", "批量耗时不是实时字幕延迟", "仅对已关闭的完整分段WAV对照；连续录音尾部不是终稿"]}
    atomic_json(batch / "summary.json", summary)
    japanese = [item for item in samples if item["source_lang"] == "ja"]
    whisper_input = batch / "results.json"
    if japanese and len(japanese) != len(samples):
        whisper_input = batch / "whisper-input.json"
        atomic_json(whisper_input, {**snapshot, "samples": japanese})
    try:
        if japanese:
            summary["tools"]["whisper"] = run_tool(
                [str(args.python), "-B", str(ROOT / "tools" / "reference_whisper.py"), str(whisper_input),
                 "--models", str(args.reference_models), "--variant", args.reference_variant,
                 "--num-threads", "2", "--output", str(batch / "reference-whisper.json")],
                batch, "whisper", batch / "reference-whisper.json", args.tool_timeout)
        else:
            summary["tools"]["whisper"] = {"status": "not_applicable", "reason": "现有参考工具仅支持日语片段"}
        atomic_json(batch / "summary.json", summary)
        summary["tools"]["translation"] = run_tool(
            [str(args.python), "-B", str(ROOT / "tools" / "compare_diagnostics.py"), "--input", str(batch / "results.json"),
             "--models", str(args.models), "--threads", "2", "--output", str(batch / "translation_comparison.json")],
            batch, "ct2", batch / "translation_comparison.json", args.tool_timeout)
        summary["status"] = "completed" if all(stage["status"] in ("completed", "not_applicable") for stage in summary["tools"].values()) else "failed"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        raise
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary.update(batch_statistics(batch, samples), finished_at=timestamp())
        atomic_json(batch / "summary.json", summary)
    return summary


def escaped(text: str) -> str:
    return str(text).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def publish(directory: Path, capture: dict[str, Any], records: list[dict[str, Any]], state: str, error: str | None = None) -> None:
    overview = {"status": state, "updated_at": timestamp(), "capture_status": capture.get("status"),
                "capture_raw_audio_seconds": capture.get("raw_audio_seconds", {}), "batches": records,
                "successful_segments": sum(record.get("sample_count", 0) for record in records if record.get("status") == "completed"),
                "failed_batches": sum(record.get("status") in ("failed", "interrupted") for record in records),
                "asr_compared_segments": sum(record.get("asr_compared_segments", 0) for record in records),
                "asr_disagreement_segments": sum(len(record.get("asr_disagreement_ids", [])) for record in records),
                "repetition_flagged_segments": sum(len(record.get("repetition_flagged_ids", [])) for record in records),
                "accuracy_evaluated": False, "content_uploaded": False, "error": error}
    atomic_json(directory / "watch-summary.json", overview)
    lines = ["# 本地字幕持续对照", "", f"更新时间：{overview['updated_at']}；监测状态：{state}；采样状态：{capture.get('status', '未知')}",
             f"采集总秒数：{json.dumps(overview['capture_raw_audio_seconds'], ensure_ascii=False)}", "",
             f"批次 {len(records)}；成功处理片段 {overview['successful_segments']}；失败或中断批次 {overview['failed_batches']}。",
             f"已对照 ASR 片段 {overview['asr_compared_segments']}；模型分歧片段 {overview['asr_disagreement_segments']}；疑似重复译文片段 {overview['repetition_flagged_segments']}。", "",
             "分歧和重复标记均需人工复核，不能换算为准确率。Whisper不是人工真值；本地批量耗时不是实时字幕延迟。仅使用已关闭的分段WAV，连续录音未完成尾部不算字幕终稿。失败批次保留输入和日志，不自动重跑。", "",
             "## 最近批次", "", "| 批次 | 状态 | 片段数 | ASR分歧 | 重复标记 |", "|---|---|---:|---:|---:|"]
    if error:
        lines[2:2] = [f"监测错误：{escaped(error)}", ""]
    for record in records[-20:]:
        path = (directory / "batches" / record["batch"] / "summary.json").as_posix()
        lines.append(f"| [{record['batch']}]({path}) | {record['status']} | {record.get('sample_count', len(record.get('sample_ids', [])))} | {len(record.get('asr_disagreement_ids', []))} | {len(record.get('repetition_flagged_ids', []))} |")
    for record in records[-3:]:
        batch = directory / "batches" / record["batch"]
        snapshot = optional_report(batch / "results.json")
        refs = {item["id"]: item for item in optional_report(batch / "reference-whisper.json").get("samples", [])}
        translations = {item["id"]: item for item in optional_report(batch / "translation_comparison.json").get("samples", [])}
        lines.extend(["", f"## {record['batch']}（{record['status']}）", ""])
        for item in snapshot.get("samples", []):
            sid = item["id"]
            lines.extend([f"### {escaped(sid)}", "", f"原文：{escaped(item.get('production_text', ''))}", "",
                          f"Whisper：{escaped(refs.get(sid, {}).get('reference_text', '[未得到参考结果]'))}", "",
                          f"现有译文：{escaped(item.get('production_translation', ''))}", ""])
            for variant, value in translations.get(sid, {}).get("ct2_int8", {}).items():
                lines.extend([f"CT2 {variant}：{escaped(value.get('text', ''))}", ""])
            if sid in record.get("repetition_flagged_ids", []):
                lines.extend(["标记：疑似重复译文，需人工复核。", ""])
    temporary = directory / "review.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, directory / "review.md")


def watch_loop(args, directory: Path, watch: ProcessWatch) -> int:
    """The caller retains the exclusive lock through every publish/error path."""
    records, capture = [], {}
    try:
        records, handled = load_batches(directory)
        next_batch, next_publish, read_failures = 0.0, 0.0, 0
        while True:
            running = watch.running()
            try:
                capture = read_json(directory / "results.json")
                read_failures = 0
            except (OSError, ValueError):
                read_failures += 1
                if not running and read_failures >= 3:
                    raise RuntimeError("Capture exited but its final results.json cannot be read")
                time.sleep(1)
                continue
            # The app stays open after settings/model changes, but its opt-in
            # recorder has ended. Drain that session instead of watching idle
            # snapshots until the whole GUI process exits.
            running = running and capture.get("status") in {"collecting", "running"}
            now = time.monotonic()
            if not running or now >= next_batch:
                samples = eligible_samples(capture, directory, handled, args.batch_size)
                if samples:
                    handled.update(item["id"] for item in samples)
                    next_batch = now + args.interval
                    result = run_batch(args, directory, capture, samples, final=not running)
                    records.append(result)
                    publish(directory, capture, records, "watching" if running and watch.running() else "draining")
                    print(json.dumps({"batch": result["batch"], "status": result["status"], "segments": result["sample_count"]}), flush=True)
                    continue
                next_batch = now + args.interval
                if not running:
                    capture_failed = capture.get("status") not in {
                        "completed", "stopped_by_request", "application_closed", "settings_changed", "model_changed",
                    }
                    failed = capture_failed or any(record.get("status") in ("failed", "interrupted") for record in records)
                    message = f"采样进程退出时最后状态为 {capture.get('status')}，未确认正常终结；完整片段已处理。" if capture_failed else None
                    publish(directory, capture, records, "completed_with_failures" if failed else "completed", message)
                    return 1 if failed else 0
            if now >= next_publish:
                publish(directory, capture, records, "watching")
                next_publish = now + args.interval
            time.sleep(1)
    except KeyboardInterrupt:
        records, _ = load_batches(directory)
        publish(directory, capture, records, "interrupted", "监测被中断；采样进程未被停止")
        return 130
    except Exception as exc:
        records, _ = load_batches(directory)
        publish(directory, capture, records, "failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=ROOT / "artifacts" / "diagnostics" / "continuous-current.json")
    parser.add_argument("--interval", type=float, default=90, help="Batch start interval, 60 to 90 seconds")
    parser.add_argument("--batch-size", type=int, default=16, choices=range(1, 17))
    parser.add_argument("--tool-timeout", type=float, default=600)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv" / "Scripts" / "python.exe")
    parser.add_argument("--reference-models", type=Path, default=ROOT / "model-cache" / "reference-whisper-small")
    parser.add_argument("--reference-variant", choices=("small", "turbo"), default="small")
    parser.add_argument("--baseline-models", type=Path, default=ROOT / "models",
                        help="Original M2M100 assets for comparison, independent of the selected production GGUF")
    args = parser.parse_args()
    if not 60 <= args.interval <= 90 or args.tool_timeout <= 0:
        parser.error("--interval must be 60 through 90 and --tool-timeout must be positive")
    control = read_json(args.control.resolve())
    pid = control.get("capture_pid")
    if type(pid) is not int or pid <= 0:
        parser.error("control.capture_pid must be a positive integer")
    directory = Path(control["output"]).resolve()
    args.models = args.baseline_models.resolve()
    args.python, args.reference_models = args.python.resolve(), args.reference_models.resolve()
    if not directory.is_dir() or not args.python.is_file():
        parser.error("Capture directory or local Python executable is missing")
    watch = ProcessWatch(pid)
    try:
        with single_watcher(directory):
            return watch_loop(args, directory, watch)
    finally:
        watch.close()


if __name__ == "__main__":
    raise SystemExit(main())
