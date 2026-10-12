"""Observe one owned local evaluation helper, without audio or remote requests."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import time

TZ = timezone(timedelta(hours=8))


def timestamp() -> str:
    return datetime.now(TZ).isoformat(timespec="milliseconds")


def save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


class SystemMemory(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in (
            "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
            "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


def memory(pid: int) -> dict:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_ulong]
    system = SystemMemory()
    system.dwLength = ctypes.sizeof(system)
    kernel.GlobalMemoryStatusEx(ctypes.byref(system))
    result = {"system_total_physical_bytes": system.ullTotalPhys,
              "system_available_physical_bytes": system.ullAvailPhys}
    handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)
    if not handle:
        return {**result, "process_exists": False, "windows_error": ctypes.get_last_error()}
    try:
        values = ProcessMemory()
        values.cb = ctypes.sizeof(values)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(values), values.cb):
            return {**result, "process_exists": True, "windows_error": ctypes.get_last_error()}
        return {**result, "process_exists": True, "working_set_bytes": values.WorkingSetSize,
                "peak_working_set_bytes": values.PeakWorkingSetSize,
                "private_bytes": values.PrivateUsage}
    finally:
        kernel.CloseHandle(handle)


def gpu_query(query: str, pid: int | None = None) -> dict:
    # Process names can contain ANSI-encoded Unicode Windows paths. They are
    # unnecessary for owned-helper attribution; keep PID and memory only.
    query = query.replace("pid,process_name,used_memory", "pid,used_memory")
    query = query.replace("pid,process_name,used_gpu_memory", "pid,used_gpu_memory")
    completed = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                               capture_output=True, timeout=10,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    output = completed.stdout.decode("utf-8", errors="replace").strip()
    if pid is not None:
        output = "\n".join(line for line in output.splitlines() if line.startswith(f"{pid},"))
    return {"query": query, "returncode": completed.returncode,
            "stdout": output, "stderr": completed.stderr.decode("utf-8", errors="replace").strip()}


def template_check(report: dict, dataset: dict, destination: Path) -> dict:
    command = report["server_command"]
    port = int(command[command.index("--port") + 1])
    item = dataset["samples"][0]
    target = "简体中文" if item["target_lang"] == "zh" else "日语"
    user_prompt = f"将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{item['source_text']}"
    payload = {"messages": [{"role": "user", "content": user_prompt}],
               "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
               "reasoning_effort": "none"}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("POST", "/apply-template", json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        if response.status != 200:
            raise RuntimeError(f"Local /apply-template HTTP {response.status}: {body[:200]}")
        content = json.loads(body)
    finally:
        connection.close()
    expanded = content.get("prompt", "")
    expected_start = "<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think<｜hy_User｜>"
    expected_end = "<｜hy_Assistant｜><think></think>"
    result = {"checked_at": timestamp(), "evaluation_helper_pid": report["server_pid"],
              "model": report["model"], "endpoint": f"http://127.0.0.1:{port}/apply-template",
              "request": payload, "response": content, "expanded_prompt_sha256": hashlib.sha256(expanded.encode()).hexdigest(),
              "expected_default_no_think_start": expected_start, "expected_no_think_end": expected_end,
              "start_matches": expanded.startswith(expected_start), "end_matches": expanded.endswith(expected_end),
              "contains_exact_source_text": item["source_text"] in expanded}
    result["verified_no_think"] = all(result[key] for key in ("start_matches", "end_matches", "contains_exact_source_text"))
    save(destination, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Windows process memory sampling is required for this tool")
    started_at = datetime.now(TZ)
    deadline = time.monotonic() + args.timeout
    output = args.report.with_suffix(".resource-samples.json")
    template_path = args.report.with_suffix(".template-check.json")
    result = {"status": "waiting", "started_at": started_at.isoformat(),
              "report": str(args.report.resolve()), "samples": [],
              "limitations": ["GPU memory values are whole-device usage, not per-process usage",
                              "WDDM may report per-process GPU memory as N/A",
                              "WorkingSet is resident process memory; PrivateUsage is private commit",
                              "Concurrent applications can affect memory and single-run latency"]}
    template_done = False
    while time.monotonic() < deadline:
        if not args.report.is_file():
            time.sleep(0.2)
            continue
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if datetime.fromisoformat(report["started_at"]) < started_at - timedelta(seconds=2):
            time.sleep(0.2)
            continue
        if "server_pid" not in report:
            if report.get("status") == "failed":
                result.update(status="evaluation_failed", error=report.get("error"))
                save(output, result)
                return 1
            time.sleep(0.2)
            continue
        pid = report["server_pid"]
        if not template_done and report.get("status") == "running":
            try:
                dataset = json.loads(Path(report["dataset"]).read_text(encoding="utf-8-sig"))
                check = template_check(report, dataset, template_path)
                result["template_verified_no_think"] = check["verified_no_think"]
                result["template_check"] = str(template_path.resolve())
                template_done = True
                print(json.dumps({"status": "template_checked", "pid": pid,
                                  "verified_no_think": check["verified_no_think"]}), flush=True)
            except (OSError, RuntimeError, ValueError) as exc:
                result["last_template_error"] = f"{type(exc).__name__}: {exc}"
        sample = {"sampled_at": timestamp(), "evaluation_helper_pid": pid,
                  "completed_samples": len(report.get("samples", [])), "evaluation_status": report["status"],
                  "memory": memory(pid),
                  "gpu": gpu_query("--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu"),
                  "compute_processes": gpu_query("--query-compute-apps=pid,process_name,used_gpu_memory", pid)}
        result["samples"].append(sample)
        result["status"] = "monitoring"
        save(output, result)
        if report.get("status") in ("completed", "failed") and not sample["memory"]["process_exists"]:
            result.update(status="completed", finished_at=timestamp(),
                          evaluation_status=report["status"], owned_helper_exited=True)
            save(output, result)
            print(json.dumps({"status": "monitor_completed", "pid": pid,
                              "samples": len(result["samples"]), "owned_helper_exited": True}), flush=True)
            return 0
        time.sleep(args.interval)
    result.update(status="monitor_timeout", finished_at=timestamp())
    save(output, result)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
