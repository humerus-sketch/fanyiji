"""Eight owned-helper controls comparing actual continuation assembly vs newline.

Pure synthetic text controls, not audio transcripts or an accuracy-rate test.
No microphone/loopback capture and no settings or model profile writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fanyiji.engines import create_translation_engine
from verify_accuracy_pipeline import EXACT_JOIN_CONTROLS, loopback_guard


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_controls():
    # Import at execution time so the tool can be prepared before the shared
    # continuation helper is ready; do not reproduce its implementation here.
    from fanyiji.text_context import join_continuation_text, looks_incomplete
    controls = []
    for source, target, original, label in EXACT_JOIN_CONTROLS:
        previous, current = original.split("\n", 1)
        candidate = looks_incomplete(previous, source)
        joined = join_continuation_text(previous, current, source)
        controls.append({"control": label, "source_language": source, "target_language": target,
            "previous": previous, "current": current, "looks_incomplete": candidate,
            "legacy_text": original, "new_text": joined, "new_text_changed": joined != original,
            "assembly_function": "fanyiji.text_context.join_continuation_text",
            "source_role": "Synthetic text fragments authored for comparison; not human audio ground truth",
            "eligibility_scope": "Only incomplete-text trigger checked; not audio gap/time/generation join gates"})
    return controls


def verify(models, output):
    started = time.monotonic()
    report = {"status": "running", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "audio_capture": False, "native_mt_calls_expected": 8, "native_mt_calls_actual": 0,
        "measurement_scope": "Four paired synthetic controls; nonempty output is not proof of accuracy",
        "network_guard_scope": "Python socket entrypoints allow loopback; not an OS traffic audit",
        "network_operations": [], "controls": [], "source_code_sha256": {}, "helper_processes": []}
    profiles = [models / "asr/backend.json", models / "translation/backend.json", models.parent / "settings.json"]
    before = {str(path): digest(path) for path in profiles if path.is_file()}
    context_source = ROOT / "src/fanyiji/text_context.py"
    source_sha = digest(context_source)
    engine, processes = None, {}
    try:
        controls = prepare_controls()
        report["controls"] = controls
        with loopback_guard(report):
            engine = create_translation_engine(models / "translation", num_threads=4)
            engine.preload()
            report["translation_backend_loaded"] = engine.backend_info
            process = getattr(engine, "_process", None)
            if process is not None:
                processes[process.pid] = process
            for control in controls:
                control["results"] = {}
                for policy, key in (("legacy_newline", "legacy_text"), ("new_actual_helper", "new_text")):
                    tick = time.monotonic()
                    translated = engine.translate(control[key], control["source_language"],
                                                   control["target_language"], context="")
                    report["native_mt_calls_actual"] += 1
                    control["results"][policy] = {"text": control[key], "translation": translated,
                        "elapsed_seconds": time.monotonic() - tick,
                        "semantic_review": "pending_root_review"}
                    process = getattr(engine, "_process", None)
                    if process is not None:
                        processes[process.pid] = process
            report["status"] = "completed_pending_semantic_review"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if engine is not None:
            try:
                engine.close()
                report["translation_backend_after_close"] = engine.backend_info
            except Exception as exc:
                report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
                report["status"] = "failed"
        report["helper_processes"] = [{"pid": pid, "returncode": process.poll(), "exited": process.poll() is not None}
                                        for pid, process in processes.items()]
        report["profile_sha256_before"] = before
        report["profile_sha256_after"] = {str(path): digest(path) for path in profiles if path.is_file()}
        report["profiles_unchanged"] = before == report["profile_sha256_after"]
        report["source_code_sha256"] = {"before": source_sha, "after": digest(context_source)}
        report["source_code_unchanged"] = source_sha == report["source_code_sha256"]["after"]
        report["external_python_connections_blocked"] = sum(not item["allowed"] for item in report["network_operations"])
        report["checks"] = {"calls_complete": report["native_mt_calls_actual"] == 8,
            "profiles_unchanged": report["profiles_unchanged"], "source_code_unchanged": report["source_code_unchanged"],
            "owned_helpers_exited": all(item["exited"] for item in report["helper_processes"]),
            "no_external_python_network": report["external_python_connections_blocked"] == 0}
        if not all(report["checks"].values()):
            report["status"] = "failed"
        report["total_seconds"] = time.monotonic() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(output), "calls": report["native_mt_calls_actual"],
                          "total_seconds": round(report["total_seconds"], 3), "checks": report["checks"]}, ensure_ascii=False))
    return 1 if report["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/continuation_translation.json")
    parser.add_argument("--prepare-only", action="store_true", help="Print actual helper assembly without native inference")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.prepare_only:
        print(json.dumps(prepare_controls(), ensure_ascii=False, indent=2))
        return 0
    return verify(args.models.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
