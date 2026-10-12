"""Compare fixed source text locally; never capture audio or fetch a model.

One llama.cpp candidate runs at a time. The server binds only to 127.0.0.1,
uses a fixed local GGUF and is terminated after evaluation. Results are not
accuracy scores; live ASR text is not a human transcript.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TZ = timezone(timedelta(hours=8))


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def request(port, route, payload=None, timeout=120):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection.request("GET" if body is None else "POST", route, body,
                           {"Content-Type": "application/json"} if body is not None else {})
        response = connection.getresponse()
        data = response.read()
        if response.status != 200:
            raise RuntimeError(f"Local inference HTTP {response.status}: {data[:500].decode('utf-8', errors='replace')}")
        return json.loads(data)
    finally:
        connection.close()


def prompt(text, source, target, kind):
    language = "简体中文" if target == "zh" else "日语"
    if kind == "hy":
        return f"将以下文本翻译为{language}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
    original = "中文" if source == "zh" else "日语"
    return (f"将以下{original}文本翻译为{language}。只输出译文，保留人名、数字与否定关系，"
            f"不要补充原文没有的内容。\n\n{text}\n/no_think")


def evaluation(args, dataset, report):
    engine = None
    process = None
    output_stream = None
    try:
        if args.kind == "m2m":
            from fanyiji.engines import TranslationEngine
            started = time.perf_counter()
            engine = TranslationEngine(args.models / "translation", num_threads=4)
            engine.preload()
            report["load_seconds"] = time.perf_counter() - started
        else:
            if not args.server or not args.gguf or not args.server.is_file() or not args.gguf.is_file():
                raise FileNotFoundError("Local llama-server or GGUF is missing; no download attempted")
            with socket.socket() as reserve:
                reserve.bind(("127.0.0.1", 0))
                port = reserve.getsockname()[1]
            alias = f"local-eval-{os.getpid()}-{time.time_ns()}"
            command = [str(args.server.resolve()), "-m", str(args.gguf.resolve()), "--host", "127.0.0.1",
                       "--port", str(port), "--alias", alias, "--parallel", "1", "-c", "2048",
                       "-t", "4", "-tb", "4", "-ngl", args.gpu_layers if args.device == "cuda" else "0",
                       "--fit", "on", "--fit-target", "1024",
                       "--no-webui", "--jinja"]
            environment = os.environ.copy()
            windows = environment.get("SystemRoot", r"C:\Windows")
            environment["PATH"] = os.pathsep.join([str(args.server.parent.resolve()), str(Path(windows) / "System32"), windows])
            server_log = args.output.with_suffix(".server.log")
            output_stream = server_log.open("wb")
            started = time.perf_counter()
            process = subprocess.Popen(command, cwd=args.server.parent.resolve(), env=environment,
                                       stdout=output_stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            report.update(server_pid=process.pid, server_log=str(server_log.resolve()),
                          server_command=command, local_http_only=True)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited {process.returncode}; see {server_log}")
                try:
                    request(port, "/health", timeout=1)
                    ids = {item["id"] for item in request(port, "/v1/models", timeout=2).get("data", [])}
                    if alias in ids:
                        break
                except (OSError, RuntimeError, ValueError):
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("Local model did not become ready in 120 seconds")
            report["load_seconds"] = time.perf_counter() - started
            report["local_model_ids_verified"] = True
            write_report(args.output, report)

        for item in dataset["samples"][:args.limit] if args.limit else dataset["samples"]:
            started = time.perf_counter()
            text = item["source_text"]
            if engine is not None:
                output = engine.translate(text, item["source_lang"], item["target_lang"])
                finish_reason, usage = "backend_default", None
            else:
                payload = {"model": alias, "messages": [{"role": "user", "content": prompt(text, item["source_lang"], item["target_lang"], args.kind)}],
                           "temperature": args.temperature, "top_p": args.top_p,
                           "top_k": args.top_k, "repeat_penalty": args.repeat_penalty,
                           "max_tokens": 512, "seed": 42, "stream": False,
                           "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
                response = request(port, "/v1/chat/completions", payload)
                choice = response["choices"][0]
                output = (choice["message"].get("content") or "").strip()
                finish_reason, usage = choice.get("finish_reason"), response.get("usage")
            elapsed = time.perf_counter() - started
            result = {**item, "translation": output, "seconds": elapsed,
                      "finish_reason": finish_reason, "usage": usage,
                      "empty": not bool(output), "has_thinking_markup": "<think>" in output,
                      "suspicious_repetition": bool(re.search(r"(.{1,12}?)\1{3,}", output.replace(" ", "")))}
            report["samples"].append(result)
            write_report(args.output, report)
            print(json.dumps({"id": item["id"], "model": args.label,
                              "seconds": round(elapsed, 3), "status": "translated"}), flush=True)
        times = [item["seconds"] for item in report["samples"]]
        report["summary"] = {"samples": len(times), "median_seconds": statistics.median(times),
                             "max_seconds": max(times), "total_seconds": sum(times),
                             "empty_outputs": sum(item["empty"] for item in report["samples"]),
                             "length_limit_outputs": sum(item["finish_reason"] == "length" for item in report["samples"]),
                             "suspicious_repetition_outputs": sum(item["suspicious_repetition"] for item in report["samples"])}
        report["status"] = "completed"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if output_stream is not None:
            output_stream.close()
        if engine is not None and hasattr(engine, "close"):
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", choices=("m2m", "hy", "qwen"), required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--server", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repeat-penalty", type=float, default=1.05)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    dataset = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not dataset.get("samples"):
        parser.error("The fixed evaluation dataset is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started_at": datetime.now(TZ).isoformat(), "model": args.label,
              "kind": args.kind, "device": "cpu" if args.kind == "m2m" else args.device,
              "gguf": str(args.gguf.resolve()) if args.gguf else None, "dataset": str(args.input.resolve()),
              "parameters": {"temperature": args.temperature, "seed": 42, "max_tokens": 512,
                             "top_p": args.top_p, "top_k": args.top_k,
                             "repeat_penalty": args.repeat_penalty, "threads": 4, "thinking_enabled": False},
              "limitations": ["No human audio transcript; live source text may contain ASR errors",
                              "Hand-written text checks have expected meanings, not a general benchmark",
                              "Single local run; continuous capture/reference work may affect timing",
                              "Outputs and repetition flags are not accuracy percentages"], "samples": []}
    try:
        evaluation(args, dataset, report)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = datetime.now(TZ).isoformat()
        write_report(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output), "summary": report.get("summary")}), flush=True)


if __name__ == "__main__":
    main()
