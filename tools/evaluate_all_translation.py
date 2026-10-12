"""Prepare, or explicitly execute, serial local MT evaluation with production engines.

Default mode only validates assets/corpus and writes a plan. --execute creates
one fresh subprocess per model; no profile is activated, models fetched, or
audio accessed. A separate --blind-export mode prepares anonymous human review.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CACHE = ROOT / "model-cache" / "translation-evaluation"
DEFAULT_DIR = ROOT / "artifacts/diagnostics/all-model-evaluation-20261009"
TZ = timezone(timedelta(hours=8))
MODELS = {
    "m2m100": {"kind": "m2m", "directory": ROOT / "models/translation"},
    "hy18": {"kind": "hy", "directory": CACHE / "hy-mt2-1.8b-q6_k"},
    "hy7": {"kind": "hy", "directory": CACHE / "hy-mt2-7b-q6_k"},
    "hy30": {"kind": "hy", "directory": CACHE / "hy-mt2-30b-a3b-q4_k_m"},
    "qwen17": {"kind": "qwen", "directory": CACHE / "qwen3-1.7b-q8_0"},
    "m2m100_fp32": {"kind": "fp32", "directory": ROOT / "model-cache/m2m100-source"},
}
CONTROL_PROVENANCE = {
    "authored_by": "assistant", "semantic_review_by": "assistant/root",
    "user_human_translation_ground_truth": False,
    "human_fields_meaning": "human_expected/expected_assertions are designed semantic control guidance, not user annotations.",
    "acceptance_notes": {"mt-ja-13": "Accept rain stopping before/by six, including the endpoint.",
                         "mt-ja-08": "王さん has unspecified gender; do not require female.",
                         "mt-ja-20": "できません may express cannot/not permitted here; accept either natural wording."},
}


def timestamp():
    return datetime.now(TZ).isoformat(timespec="milliseconds")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_corpus(path):
    corpus = json.loads(path.read_text(encoding="utf-8-sig"))
    samples = corpus.get("samples", [])
    if len(samples) != 60 or len({x.get("id") for x in samples}) != 60:
        raise ValueError("Expected exactly 60 unique frozen text controls")
    for source, target in (("zh", "ja"), ("ja", "zh")):
        if sum(x.get("source_lang") == source and x.get("target_lang") == target for x in samples) != 30:
            raise ValueError("Expected 30 controls per direction")
    if any(not x.get("source_text", "").strip() or not x.get("human_expected") for x in samples):
        raise ValueError("Every control requires source text and human fact assertions")
    checksum = path.with_suffix(".sha256")
    if not checksum.is_file() or checksum.read_text(encoding="ascii").split()[0] != digest(path):
        raise ValueError("Frozen corpus checksum does not match")
    return corpus


def inventory(model_id):
    model = MODELS[model_id]
    directory = model["directory"]
    if model["kind"] in ("hy", "qwen"):
        manifest_path = directory / "source_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = [manifest["filename"]]
    else:
        manifest_path = ROOT / "models/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = (["model.bin", "shared_vocabulary.json", "vocab.json", "sentencepiece.bpe.model"]
                 if model["kind"] == "m2m" else
                 ["pytorch_model.bin", "config.json", "vocab.json", "sentencepiece.bpe.model",
                  "tokenizer_config.json", "special_tokens_map.json"])
    files = []
    for name in names:
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Required local asset missing; no download attempted: {path}")
        stat = path.stat()
        files.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if model["kind"] in ("hy", "qwen") and files[0]["size"] != manifest["size"]:
        raise ValueError("GGUF size differs from verified pinned manifest")
    return {"model_id": model_id, "kind": model["kind"], "directory": str(directory.resolve()),
            "source_manifest": str(manifest_path.resolve()), "source_manifest_sha256": digest(manifest_path),
            "pinned_source": manifest, "files": files,
            "prepare_verification": "Previously pinned assets; current size/mtime only. Worker rehashes before load."}


@contextmanager
def loopback_only():
    """Restrict Python connections; this is not an OS-level packet audit."""
    attempts = []
    original = socket.socket.connect
    original_ex = socket.socket.connect_ex

    def check(address):
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
            attempts.append("non_loopback_connection_blocked")
            raise RuntimeError("Evaluation only permits owned local helper connections")

    def connect(sock, address):
        check(address)
        return original(sock, address)

    def connect_ex(sock, address):
        check(address)
        return original_ex(sock, address)

    with patch.object(socket.socket, "connect", connect), patch.object(socket.socket, "connect_ex", connect_ex):
        yield attempts


def cpu_times(pid):
    if os.name != "nt":
        return {"available": False}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return {"available": False, "windows_error": ctypes.get_last_error()}
    try:
        values = [ctypes.c_uint64() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            return {"available": False, "windows_error": ctypes.get_last_error()}
        return {"available": True, "kernel_seconds": values[2].value / 10_000_000,
                "user_seconds": values[3].value / 10_000_000}
    finally:
        kernel.CloseHandle(handle)


class ResourceSampler:
    def __init__(self, engine):
        self.engine = engine
        self.samples = []
        self.phase = "load"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name="mt-resource-sampler", daemon=True)

    def sample(self, gpu=False):
        from monitor_translation_evaluation import memory, gpu_query
        info = getattr(self.engine, "backend_info", {})
        pid = info.get("helper_pid")
        value = {"at": timestamp(), "phase": self.phase, "python_pid": os.getpid(),
                 "python": {**memory(os.getpid()), "cpu_times": cpu_times(os.getpid())},
                 "helper_pid": pid, "helper": ({**memory(pid), "cpu_times": cpu_times(pid)} if pid else None)}
        if gpu:
            for key, query, filter_pid in (
                ("whole_gpu", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", None),
                ("owned_compute_app", "--query-compute-apps=pid,process_name,used_memory", pid),
            ):
                try:
                    # Never include unrelated process names in artifacts.
                    value[key] = gpu_query(query, filter_pid) if key == "whole_gpu" or pid else {"not_loaded": True}
                except (OSError, subprocess.TimeoutExpired) as exc:
                    value[key] = {"unavailable": type(exc).__name__}
        self.samples.append(value)

    def run(self):
        next_gpu = 0.0
        while not self.stop.is_set():
            try:
                now = time.monotonic()
                self.sample(gpu=now >= next_gpu)
                if now >= next_gpu:
                    next_gpu = now + 2
            except Exception as exc:
                self.samples.append({"at": timestamp(), "sampler_error": type(exc).__name__})
            self.stop.wait(0.25)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=25)
        self.sample(gpu=True)
        return self.samples


class CaptureLogOnClose:
    """Read the temporary native log only after production disposal stops helper."""
    def __init__(self, stream, path, secrets_to_redact=()):
        self.stream, self.path = stream, path
        self.secrets_to_redact = secrets_to_redact

    def close(self):
        try:
            self.stream.flush()
            self.stream.seek(0)
            # Command/header secrets are not dumped. Native log has no request headers.
            raw = self.stream.read().decode("utf-8", errors="replace")
            raw = re.sub(r"fanyiji-[a-f0-9]+", "<owned-alias>", raw)
            for value in self.secrets_to_redact:
                raw = raw.replace(value, "<redacted>")
            self.path.write_text(raw, encoding="utf-8")
        finally:
            self.stream.close()


def audit_gguf(engine, report, directory):
    original_start, original_request = engine._start_runtime, engine._request

    def start(device, executable):
        original_start(device, executable)
        command = list(engine._process.args)
        command[command.index("--alias") + 1] = "<owned-alias>"
        report.setdefault("helper_launches", []).append({"at": timestamp(), "device": device,
                                                       "pid": engine._process.pid, "command": command,
                                                       "server_sha256": digest(executable)})
        engine._log = CaptureLogOnClose(engine._log, directory / f"{report['model_id']}-{device}-helper.log",
                                       (engine._api_key, engine._alias))

    def request(method, path, body=None, **kwargs):
        response = original_request(method, path, body, **kwargs)
        if path == "/v1/chat/completions":
            recorded = dict(body)
            recorded["model"] = "<owned-alias>"
            report["last_actual_request"] = recorded
            report["last_response_metadata"] = {
                "usage": response.get("usage"),
                "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
            }
        return response

    engine._start_runtime, engine._request = start, request


class Fp32Engine:
    def __init__(self, directory):
        self.directory = directory
        self.model = self.tokenizer = self.torch = None
        self.backend_info = {"backend": "pytorch", "model_kind": "m2m100", "device": "cpu",
                             "compute_type": "float32", "threads": 4, "num_beams": 2,
                             "do_sample": False, "max_new_tokens": 256,
                             "length_penalty": 1.0, "use_cache": True,
                             "attn_implementation": "eager", "local_files_only": True,
                             "generation_early_stopping": "inherited from original local model config"}

    def preload(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM
        from compare_diagnostics import official_tokenizer
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        self.torch = torch
        self.tokenizer = official_tokenizer(self.directory)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.directory), local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="eager").eval()
        self.backend_info["effective_generation_config"] = self.model.generation_config.to_dict()
        parameter = next(self.model.parameters())
        self.backend_info.update(actual_parameter_device=str(parameter.device), actual_parameter_dtype=str(parameter.dtype),
                                 actual_torch_threads=torch.get_num_threads(), actual_interop_threads=torch.get_num_interop_threads())

    def translate(self, text, source, target):
        self.tokenizer.src_lang = source
        inputs = self.tokenizer(text, return_tensors="pt")
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, forced_bos_token_id=self.tokenizer.get_lang_id(target), num_beams=2,
                do_sample=False, max_new_tokens=256, length_penalty=1.0, use_cache=True)[0].tolist()
        language_ids = set(self.tokenizer.lang_code_to_id.values())
        return self.tokenizer.decode([x for x in generated if x not in language_ids], skip_special_tokens=True).strip()

    def close(self):
        self.model = self.tokenizer = self.torch = None


def make_engine(model_id, device, report, directory):
    model = MODELS[model_id]
    if model["kind"] == "fp32":
        return Fp32Engine(model["directory"])
    if model["kind"] == "m2m":
        from fanyiji.engines import TranslationEngine
        return TranslationEngine(model["directory"], num_threads=4)
    from fanyiji.gguf_engine import GgufTranslationEngine
    manifest = json.loads((model["directory"] / "source_manifest.json").read_text(encoding="utf-8"))
    engine = GgufTranslationEngine(model["directory"], num_threads=4, config={
        "backend": "llama_cpp", "model_file": manifest["filename"], "model_kind": model["kind"],
        "runtime_preference": device, "gpu_layers": None,
    })
    audit_gguf(engine, report, directory)
    return engine


def resource_summary(samples):
    result = {}
    for key in ("python", "helper"):
        values = [x[key] for x in samples if x.get(key) and "working_set_bytes" in x[key]]
        result[key] = {"max_observed_working_set_bytes": max((x["working_set_bytes"] for x in values), default=None),
                       "os_peak_working_set_bytes": max((x["peak_working_set_bytes"] for x in values), default=None),
                       "max_observed_private_bytes": max((x["private_bytes"] for x in values), default=None),
                       "last_process_cpu_times": values[-1].get("cpu_times") if values else None}
    available = [x["python"]["system_available_physical_bytes"] for x in samples if x.get("python")]
    result["min_system_available_physical_bytes"] = min(available, default=None)
    result["sample_count"] = len(samples)
    result["limitations"] = ["250 ms requested CPU sampling; GPU query overhead can increase gaps.",
                              "GPU memory is whole-card, WDDM per-process memory can be N/A.",
                              "WS is resident memory, private bytes are commit, not interchangeable.",
                              "Single isolated run; no end-to-end capture latency or P95 guarantee."]
    return result


def profile_hashes():
    from fanyiji.paths import data_root
    paths = [ROOT / "models/translation/backend.json",
             data_root() / "models/translation/backend.json"]
    return {str(path): digest(path) if path.is_file() else None for path in paths}


def worker(args, model_id):
    output = args.output_dir / f"{model_id}.json"
    if output.exists():
        raise FileExistsError(f"Preserving prior report; use a fresh output directory: {output}")
    corpus = read_corpus(args.input)
    assets = inventory(model_id)
    report = {"status": "verifying", "model_id": model_id, "started_at": timestamp(),
              "input": str(args.input.resolve()), "input_sha256": digest(args.input), "assets": assets,
              "profile_hashes_before": profile_hashes(), "samples": [],
              "scope": "Supplied text MT, not audio accuracy or EXE UI results",
              "control_provenance": CONTROL_PROVENANCE,
              "automatic_accuracy_score": False, "python": sys.version,
              "source_sha256": {name: digest(ROOT / name) for name in
                                ("src/fanyiji/engines.py", "src/fanyiji/gguf_engine.py")}}
    for name in ("ctranslate2", "sentencepiece", "torch", "transformers"):
        try:
            report.setdefault("versions", {})[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    engine = sampler = None
    try:
        started = time.perf_counter()
        for item in assets["files"]:
            item["actual_sha256"] = digest(Path(item["path"]))
        if assets["kind"] in ("hy", "qwen") and assets["files"][0]["actual_sha256"] != assets["pinned_source"]["sha256"]:
            raise ValueError("Pinned GGUF SHA256 mismatch; inference refused")
        report["verification_seconds"] = time.perf_counter() - started
        report["load_cache_boundary"] = "Full SHA verification ran before load and may warm OS file cache. Not cold-start latency."
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", FANYIJI_RUNTIME=str(CACHE / "llama-b11429"))
        with loopback_only() as blocked:
            engine = make_engine(model_id, args.device, report, args.output_dir)
            sampler = ResourceSampler(engine)
            sampler.thread.start()
            started = time.perf_counter()
            report["status"] = "loading"
            save(output, report)
            engine.preload()
            report["load_seconds"] = time.perf_counter() - started
            report["backend_after_load"] = engine.backend_info
            report["status"] = "running"
            sampler.phase = "translate"
            for sample in corpus["samples"]:
                started = time.perf_counter()
                text = engine.translate(sample["source_text"], sample["source_lang"], sample["target_lang"])
                elapsed = time.perf_counter() - started
                item = {**sample, "translation": text, "seconds": elapsed,
                        "empty": not text.strip(), "has_thinking_markup": "<think>" in text,
                        "suspicious_repetition": bool(re.search(r"(.{1,12}?)\1{3,}", text.replace(" ", "")))}
                if "last_actual_request" in report:
                    item["actual_request"] = report.pop("last_actual_request")
                    item["response_metadata"] = report.pop("last_response_metadata")
                    if "applied_template" not in report:
                        body = item["actual_request"]
                        template = engine._request("POST", "/apply-template", {
                            "messages": body["messages"], "add_generation_prompt": True,
                            "chat_template_kwargs": body["chat_template_kwargs"],
                            "reasoning_effort": body["reasoning_effort"],
                        })
                        expanded = template.get("prompt", "")
                        report["applied_template"] = {"response": template,
                                                      "prompt_sha256": hashlib.sha256(expanded.encode()).hexdigest(),
                                                      "exact_source_present": sample["source_text"] in expanded,
                                                      "thinking_disabled_in_actual_request": True}
                report["samples"].append(item)
                save(output, report)
                print(json.dumps({"model_id": model_id, "sample": sample["id"], "seconds": round(elapsed, 3)}), flush=True)
            report["network_guard"] = {"python_non_loopback_blocked_attempts": len(blocked),
                                       "helper_offline_cli": assets["kind"] in ("hy", "qwen"),
                                       "scope": "Python socket connect methods; no OS packet audit"}
            report["backend_before_close"] = engine.backend_info
            times = [x["seconds"] for x in report["samples"]]
            report["summary"] = {"samples": len(times), "median_seconds": statistics.median(times),
                                 "max_seconds": max(times), "total_seconds": sum(times),
                                 "empty_outputs": sum(x["empty"] for x in report["samples"]),
                                 "suspicious_repetition_outputs": sum(x["suspicious_repetition"] for x in report["samples"])}
            report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if sampler is not None:
                report["resource_samples"] = sampler.finish()
                report["resources"] = resource_summary(report["resource_samples"])
        except Exception as exc:
            report["resource_collection_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if engine is not None and hasattr(engine, "close"):
                    engine.close()
            except Exception as exc:
                report.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                native_lines = []
                for log_path in args.output_dir.glob(f"{model_id}-*-helper.log"):
                    native_lines.extend(line for line in log_path.read_text(encoding="utf-8").splitlines()
                                        if re.search(r"offload|buffer size|CUDA.*device|model type", line, re.I))
                report["native_offload_evidence"] = native_lines
                report["device_interpretation"] = "backend device names selected binary; native offload logs and GPU samples show actual placement. CUDA does not imply full GPU offload."
                report["profile_hashes_after"] = profile_hashes()
                report["profile_hashes_unchanged"] = report["profile_hashes_before"] == report["profile_hashes_after"]
                report["finished_at"] = timestamp()
                save(output, report)


def export_blind(output_dir, corpus):
    if (output_dir / "mt-blind-review.json").exists() or (output_dir / "mt-blind-key.json").exists():
        raise FileExistsError("Preserving prior blind review; choose a fresh output directory")
    reports = []
    for path in output_dir.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "completed" and value.get("model_id") in MODELS:
            if value.get("input_sha256") != digest(corpus):
                raise ValueError("Refuse to mix model reports from different frozen corpus")
            reports.append(value)
    if not reports:
        raise ValueError("No completed model reports to export")
    rng = random.Random(20261009)
    records, key = [], {}
    by_id = {x["model_id"]: {s["id"]: s for s in x["samples"]} for x in reports}
    for sample in read_corpus(corpus)["samples"]:
        order = list(by_id)
        rng.shuffle(order)
        for model_id in order:
            item = by_id[model_id][sample["id"]]
            anonymous_id = f"review-{len(records) + 1:03d}"
            key[anonymous_id] = {"model_id": model_id, "sample_id": sample["id"]}
            records.append({"review_id": anonymous_id, "source_lang": sample["source_lang"],
                            "target_lang": sample["target_lang"], "source_text": sample["source_text"],
                            "translation": item["translation"], "human_expected": sample["human_expected"],
                            "fact_reviews": [{"assertion": x, "verdict": None, "evidence": None}
                                             for x in sample["human_expected"]],
                            "overall": None, "reviewer_notes": None})
    save(output_dir / "mt-blind-review.json", {"status": "awaiting_human_review", "samples": records,
                                              "control_provenance": CONTROL_PROVENANCE,
                                              "rubric": "Each fact: preserved / wrong / omitted / unclear. Accept alternate wording; distinguish fluency from factual meaning. No automatic accuracy score."})
    save(output_dir / "mt-blind-key.json", key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DIR / "mt-fixed-corpus.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR / "mt")
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--execute", action="store_true", help="Explicitly run owned local inference workers serially")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--blind-export", action="store_true")
    args = parser.parse_args()
    args.input, args.output_dir = args.input.resolve(), args.output_dir.resolve()
    corpus = read_corpus(args.input)
    ids = args.model or list(MODELS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.blind_export:
        export_blind(args.output_dir, args.input)
    elif args.worker:
        if not args.execute or len(ids) != 1:
            parser.error("Worker requires --execute and exactly one --model")
        worker(args, ids[0])
    elif args.execute:
        for model_id in ids:
            python = ROOT / (".model-venv" if model_id == "m2m100_fp32" else ".venv") / "Scripts/python.exe"
            subprocess.run([str(python), str(Path(__file__).resolve()), "--execute", "--worker", "--model", model_id,
                            "--input", str(args.input), "--output-dir", str(args.output_dir), "--device", args.device], check=True)
    else:
        plan = {"status": "prepared_not_executed", "at": timestamp(), "input_sha256": digest(args.input),
                "control_provenance": CONTROL_PROVENANCE,
                "samples": len(corpus["samples"]), "model_ids": ids, "device_requested": args.device,
                "execution_order": "One fresh process per model, parent waits for exit before next.",
                "profiles_modified": False, "inventory": [inventory(model_id) for model_id in ids]}
        save(args.output_dir / "plan.json", plan)
        print(json.dumps({"status": plan["status"], "models": len(ids), "samples_each": 60, "path": str(args.output_dir)}))


if __name__ == "__main__":
    main()
