"""Fetch pinned sherpa-compatible Whisper/Qwen exports into an evaluation cache.

Downloads weights only when explicitly invoked. Range retries retain .part
files and validate the full size/SHA-256 before publishing any model file.
This script never opens audio devices, loads a model, or uploads local data.
Qwen ONNX weights come from the third-party converter linked by sherpa docs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))
RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
REQUEST_BYTES = 64 * 1024 * 1024
PRINT_LOCK = threading.Lock()
LICENSE_URL = "https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/LICENSE"
DOCUMENTATION_URL = "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/export-onnx.html"
TOKEN_SHA256 = "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"
CANDIDATES = {
    "large-v3-int8": {
        "id": "large-v3-int8", "directory": "whisper-large-v3-int8",
        "repo": "csukuangfj/sherpa-onnx-whisper-large-v3",
        "revision": "2a6507094dd6020d939d78e3f1834a1d06267fca",
        "prefix": "large-v3", "name": "Whisper large-v3 multilingual ONNX INT8",
        "files": {
            "large-v3-encoder.int8.onnx": (766671985, "d531cf17248acc43e8c09b472a0877055e770877857a5332fc1304b36534ec85"),
            "large-v3-decoder.int8.onnx": (1008265203, "ebc6bfd88e162a46cb3edee8a7e727e1dcbc65cabecb19e2573695e4d495e1af"),
            "large-v3-tokens.txt": (816730, TOKEN_SHA256),
        },
    },
    "turbo-int8": {
        "id": "turbo-int8", "directory": "whisper-turbo-int8",
        "repo": "csukuangfj/sherpa-onnx-whisper-turbo",
        "revision": "2ca6ff69fc878651b770880507669577ac41c2ff",
        "prefix": "turbo", "name": "Whisper large-v3-turbo multilingual ONNX INT8",
        "files": {
            "turbo-encoder.int8.onnx": (674716297, "b02dcdf54f348741e93fe732b67d933c8dcb6735655f710640143081db38878b"),
            "turbo-decoder.int8.onnx": (361080764, "20accd02388482eb3a46bd615631adfdc85e1eb2c7db9ea3f02a40ffe6b81547"),
            "turbo-tokens.txt": (816730, TOKEN_SHA256),
        },
    },
    "qwen3-asr-0.6b-int8": {
        "id": "qwen3-asr-0.6b-int8", "directory": "qwen3-asr-0.6b-int8", "kind": "qwen3_asr",
        "repo": "zengshuishui/Qwen3-ASR-onnx", "source_provider": "ModelScope",
        "revision": "9c182309f7bb075f241424441add9e16c5086dfb", "remote_model_directory": "model_0.6B",
        "name": "Qwen3-ASR 0.6B ONNX INT8 (third-party conversion linked by sherpa)",
        "conversion_provider": "zengshuishui / Wasser1462; not a Qwen-official ONNX export",
        "documentation": "https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/export.html",
        "license_name": "Apache-2.0", "license_filename": "LICENSE.qwen-asr",
        "license_url": "https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/7c6daf77a2421100f5fb066495372c00129d39ff/LICENSE",
        "files": {
            "conv_frontend.onnx": (44148281, "d22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e"),
            "encoder.int8.onnx": (182491662, "60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9"),
            "decoder.int8.onnx": (755914231, "4f6885be5959ae26af3089d38ee7972c5fafbeeb1cf8d5e76eab6d8b61ca5771"),
            "tokenizer/merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
            "tokenizer/tokenizer_config.json": (12487, "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c"),
            "tokenizer/vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
        },
    },
    "qwen3-asr-1.7b-int8": {
        "id": "qwen3-asr-1.7b-int8", "directory": "qwen3-asr-1.7b-int8", "kind": "qwen3_asr",
        "repo": "zengshuishui/Qwen3-ASR-onnx", "source_provider": "ModelScope",
        "revision": "9c182309f7bb075f241424441add9e16c5086dfb", "remote_model_directory": "model_1.7B",
        "name": "Qwen3-ASR 1.7B ONNX INT8 (third-party conversion linked by sherpa)",
        "conversion_provider": "zengshuishui / Wasser1462; not a Qwen-official ONNX export",
        "documentation": "https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/export.html",
        "license_name": "Apache-2.0", "license_filename": "LICENSE.qwen-asr",
        "license_url": "https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/7c6daf77a2421100f5fb066495372c00129d39ff/LICENSE",
        "files": {
            "conv_frontend.onnx": (48080441, "fa894a4ba53da6a4238f2a6ca0b09362e505d39cecbd646051b033e2e8d7e2fb"),
            "encoder.int8.onnx": (314222162, "436fbd910a0c8914851e5ac1354e807be9f283d08a5da728adaa609731c41469"),
            "decoder.int8.onnx": (2037458645, "c43c853fa6e97d08365cb8a5502b360b595cd43c00dc60e4d8ca7cc18cad460b"),
            "tokenizer/merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
            "tokenizer/tokenizer_config.json": (12487, "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c"),
            "tokenizer/vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
        },
    },
}


def emit(record: dict) -> None:
    with PRINT_LOCK:
        print(json.dumps({"observed_at": datetime.now(TZ).isoformat(timespec="seconds"),
                          **record}, ensure_ascii=False), flush=True)


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def resolve_url(candidate: dict, filename: str) -> str:
    if candidate.get("source_provider") == "ModelScope":
        remote = filename if filename.startswith("tokenizer/") else candidate["remote_model_directory"] + "/" + filename
        return f"https://modelscope.cn/models/{candidate['repo']}/resolve/{candidate['revision']}/{remote}"
    return f"https://huggingface.co/{candidate['repo']}/resolve/{candidate['revision']}/{filename}"


def verify_candidate(candidate: dict, directory: Path) -> dict:
    files = {}
    for name, (size, expected) in candidate["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"Missing or incomplete model: {path}")
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"Model SHA-256 mismatch: {path}")
        files[name] = {"path": str(path.resolve()), "size": size, "sha256": actual,
                       "source": resolve_url(candidate, name)}
    return files


def download_file(candidate: dict, directory: Path, name: str, *, opener=urlopen) -> None:
    size, expected = candidate["files"][name]
    final, part = directory / name, directory / (name + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if final.is_file() and final.stat().st_size == size and digest(final) == expected:
            emit({"candidate": candidate["id"], "file": name, "status": "already_verified", "bytes": size})
            return
        raise RuntimeError(f"Existing final file failed verification; preserved: {final}")
    if part.exists() and (not part.is_file() or part.stat().st_size > size):
        raise RuntimeError(f"Invalid partial file preserved for inspection: {part}")
    initial = part.stat().st_size if part.exists() else 0
    started, last_progress = time.monotonic(), time.monotonic()
    emit({"candidate": candidate["id"], "file": name, "status": "download_started",
          "resume_bytes": initial, "expected_bytes": size, "revision": candidate["revision"]})
    failures = 0
    while (part.stat().st_size if part.exists() else 0) < size:
        offset = part.stat().st_size if part.exists() else 0
        end = min(size - 1, offset + REQUEST_BYTES - 1)
        # A different query per range avoids cached signed links returning a
        # response for a previous range. The content revision remains pinned.
        url = resolve_url(candidate, name) + f"?download=true&range_start={offset}"
        request = Request(url, headers={"User-Agent": "fanyiji-asr-evaluation/1.0",
                                       "Accept-Encoding": "identity", "Range": f"bytes={offset}-{end}"})
        try:
            with opener(request, timeout=60) as response:
                status = response.status
                if status == 206:
                    matched = RANGE.fullmatch(response.headers.get("Content-Range", ""))
                    if (matched is None or int(matched[1]) != offset
                            or int(matched[2]) != end or int(matched[3]) != size):
                        raise RuntimeError("HTTP Content-Range differs from the pinned range")
                    expected_body = end - offset + 1
                elif status == 200 and offset == 0:
                    expected_body = size
                else:
                    raise RuntimeError("Server ignored Range; existing partial file was preserved")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != expected_body:
                    raise RuntimeError("HTTP Content-Length differs from the pinned range")
                received = 0
                with part.open("ab") as target:
                    while received < expected_body:
                        block = response.read(min(1024 * 1024, expected_body - received))
                        if not block:
                            raise OSError("HTTP body ended before the requested range was complete")
                        target.write(block)
                        received += len(block)
                        current = time.monotonic()
                        if current - last_progress >= 30:
                            complete = offset + received
                            emit({"candidate": candidate["id"], "file": name, "status": "downloading",
                                  "bytes": complete, "expected_bytes": size,
                                  "percent": round(complete / size * 100, 2),
                                  "mib_per_second": round((complete - initial) / max(current - started, .001) / 1024 ** 2, 3)})
                            last_progress = current
            failures = 0
        except (OSError, HTTPError, URLError) as exc:
            failures += 1
            emit({"candidate": candidate["id"], "file": name, "status": "retrying",
                  "retry": failures, "error_type": type(exc).__name__,
                  "resume_bytes": part.stat().st_size if part.exists() else 0})
            if failures >= 5:
                raise RuntimeError(f"Network retries exhausted; partial file retained: {part}") from exc
            time.sleep(min(failures * 2, 8))
    if part.stat().st_size != size or digest(part) != expected:
        raise RuntimeError(f"Downloaded model failed verification; partial file retained: {part}")
    part.replace(final)
    emit({"candidate": candidate["id"], "file": name, "status": "verified",
          "bytes": size, "sha256": expected})


def prepare(candidate_id: str, root: Path, *, verify_only: bool = False) -> dict:
    candidate = CANDIDATES[candidate_id]
    directory = root / candidate["directory"]
    if not verify_only:
        directory.mkdir(parents=True, exist_ok=True)
        for name in candidate["files"]:
            download_file(candidate, directory, name)
        license_url = candidate.get("license_url", LICENSE_URL)
        license_path = directory / candidate.get("license_filename", "LICENSE.whisper")
        if not license_path.exists():
            with urlopen(Request(license_url, headers={"User-Agent": "fanyiji-asr-evaluation/1.0"}), timeout=30) as response:
                license_bytes = response.read(1024 * 1024)
            if not license_bytes or len(license_bytes) == 1024 * 1024:
                raise RuntimeError("Unexpected upstream license size")
            temporary = license_path.with_name(license_path.name + ".part")
            temporary.write_bytes(license_bytes)
            temporary.replace(license_path)
    files = verify_candidate(candidate, directory)
    manifest = {"id": candidate_id, "model": candidate["name"], "repo": candidate["repo"],
                "revision": candidate["revision"], "files": files, "verified": True,
                "license": {"name": candidate.get("license_name", "MIT"), "source": candidate.get("license_url", LICENSE_URL)},
                "documentation": candidate.get("documentation", DOCUMENTATION_URL), "audio_uploaded": False,
                "source_provider": candidate.get("source_provider", "HuggingFace/sherpa maintainer"),
                "conversion_provider": candidate.get("conversion_provider", "sherpa-onnx maintainer")}
    if not verify_only:
        temporary = directory / "SOURCE.json.part"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "SOURCE.json")
    emit({"candidate": candidate_id, "status": "prepared", "directory": str(directory.resolve()),
          "total_bytes": sum(data["size"] for data in files.values())})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["all", *CANDIDATES], nargs="+", default=["all"])
    parser.add_argument("--output", type=Path, default=ROOT / "model-cache" / "asr-evaluation")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="At most one download connection for each of two candidates")
    args = parser.parse_args()
    selected = list(CANDIDATES) if "all" in args.candidate else list(dict.fromkeys(args.candidate))
    if args.parallel and len(selected) > 1:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(prepare, candidate, args.output.resolve(), verify_only=args.verify_only)
                       for candidate in selected]
            for future in futures:
                future.result()
    else:
        for candidate in selected:
            prepare(candidate, args.output.resolve(), verify_only=args.verify_only)


if __name__ == "__main__":
    main()
