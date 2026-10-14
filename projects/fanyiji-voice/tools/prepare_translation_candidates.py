"""Download pinned public GGUF candidates to an isolated evaluation cache.

Only the evaluation directory is written. Standard-library HTTP downloads use
validated Range resume, retain partial files after errors, and verify complete
weight size plus SHA-256 before publishing the final filename.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
PRINT_LOCK = threading.Lock()
LOG_PATH: Path | None = None
USER_AGENT = "fanyiji-candidate-evaluation/1.0"
RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
REQUEST_CHUNK_BYTES = 64 * 1024 * 1024
CANDIDATES = (
    {
        "id": "hy-mt2-1.8b-q6_k", "repo": "tencent/Hy-MT2-1.8B-GGUF",
        "revision": "a0c709d9fac510f2c807aa3af52872340dc37a4a",
        "filename": "Hy-MT2-1.8B-Q6_K.gguf", "size": 1_474_785_120,
        "sha256": "d98fe604dec1f28f58f80d7d560f7177e584d3b8e5835862687660e5ff97cb40",
        "license_file": "LICENSE.txt", "license": "apache-2.0",
    },
    {
        "id": "qwen3-1.7b-q8_0", "repo": "Qwen/Qwen3-1.7B-GGUF",
        "revision": "90862c4b9d2787eaed51d12237eafdfe7c5f6077",
        "filename": "Qwen3-1.7B-Q8_0.gguf", "size": 1_834_426_016,
        "sha256": "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
        "license_file": "LICENSE", "license": "apache-2.0",
    },
    {
        "id": "hy-mt2-7b-q6_k", "repo": "tencent/Hy-MT2-7B-GGUF",
        "revision": "ab8472660ac61fac25f1af43fac2599d52a8a775",
        "filename": "HY-MT2-7B-Q6_K.gguf", "size": 6_164_482_720,
        "sha256": "88ef0aba59952a4cfe4be36cb5baf797dbb370bc60e9dcbd7297036021e52831",
        "license_file": "LICENSE.txt", "license": "apache-2.0", "connections": 4,
    },
    {
        "id": "hy-mt2-30b-a3b-q4_k_m", "repo": "tencent/Hy-MT2-30B-A3B-GGUF",
        "revision": "fd3dcbb6b31e9e03923ef4f9f42500f74c3851c3",
        "filename": "Hy-MT2-30B-A3B-Q4_K_M.gguf", "size": 18_236_702_880,
        "sha256": "bb44b11bb0f7cd3d1321645b41e911cd3de2e473731227fc6bf37aa18b543f88",
        "license_file": "LICENSE.txt", "license": "apache-2.0", "connections": 8,
    },
)


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def emit(record: dict) -> None:
    with PRINT_LOCK:
        line = json.dumps({"observed_at": now(), **record}, ensure_ascii=False)
        print(line, flush=True)
        if LOG_PATH is not None:
            with LOG_PATH.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify(path: Path, size: int, sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == size and digest(path) == sha256


def resolve_url(candidate: dict, filename: str) -> str:
    return "https://huggingface.co/{repo}/resolve/{revision}/{filename}".format(
        repo=candidate["repo"], revision=candidate["revision"], filename=filename,
    )


def request(url: str, headers: dict | None = None):
    return urlopen(Request(url, headers={
        "User-Agent": USER_AGENT, "Accept-Encoding": "identity", **(headers or {}),
    }), timeout=60)


def preserve_invalid(path: Path, reason: str) -> None:
    # Keep bad assets available for inspection instead of deleting them.
    destination = path.with_name(path.name + ".invalid-" + str(time.time_ns()))
    path.rename(destination)
    emit({"status": "preserved_invalid_file", "path": str(destination), "reason": reason})


def download_weight(candidate: dict, directory: Path) -> dict:
    final = directory / candidate["filename"]
    partial = directory / (candidate["filename"] + ".part")
    expected_size, expected_sha = candidate["size"], candidate["sha256"]
    if verify(final, expected_size, expected_sha):
        emit({"candidate": candidate["id"], "status": "already_verified", "bytes": expected_size})
        return {"path": str(final.resolve()), "size": expected_size, "sha256": expected_sha,
                "verified": True, "downloaded_this_run": False}
    if final.exists():
        preserve_invalid(final, "Final file failed pinned size/SHA-256 verification")
    if partial.exists() and partial.stat().st_size > expected_size:
        preserve_invalid(partial, "Partial file exceeds the pinned size")
    started, last_progress = time.monotonic(), time.monotonic()
    initial_size = partial.stat().st_size if partial.exists() else 0
    url = resolve_url(candidate, candidate["filename"])
    emit({"candidate": candidate["id"], "status": "download_started", "resume_bytes": initial_size,
          "expected_bytes": expected_size, "revision": candidate["revision"]})
    for attempt in range(1, 5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            break
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with request(url, headers) as response:
                status = response.status
                if offset and status == 200:
                    # A server that ignores Range cannot be appended safely.
                    preserve_invalid(partial, "Server ignored Range; restart from byte zero")
                    offset = 0
                elif status == 206:
                    matched = RANGE.fullmatch(response.headers.get("Content-Range", ""))
                    if matched is None or int(matched[1]) != offset or int(matched[3]) != expected_size:
                        raise RuntimeError("HTTP Content-Range does not match the pinned resume offset/size")
                elif status != 200:
                    raise RuntimeError(f"Unexpected HTTP status {status}")
                content_length = response.headers.get("Content-Length")
                if content_length and offset + int(content_length) != expected_size:
                    raise RuntimeError("HTTP Content-Length does not match the pinned weight size")
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        if offset + len(block) > expected_size:
                            raise RuntimeError("HTTP body exceeds the pinned weight size")
                        stream.write(block)
                        offset += len(block)
                        current = time.monotonic()
                        if current - last_progress >= 30:
                            elapsed = current - started
                            emit({"candidate": candidate["id"], "status": "downloading", "bytes": offset,
                                  "expected_bytes": expected_size, "percent": round(100 * offset / expected_size, 2),
                                  "download_mib_per_second": round((offset - initial_size) / elapsed / 1024**2, 2)})
                            last_progress = current
            if offset != expected_size:
                raise RuntimeError(f"HTTP response ended early at {offset}/{expected_size} bytes")
            break
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            emit({"candidate": candidate["id"], "status": "download_attempt_failed", "attempt": attempt,
                  "partial_bytes": partial.stat().st_size if partial.exists() else 0, "error": str(exc)})
            if attempt == 4:
                raise
    actual_size = partial.stat().st_size
    actual_sha = digest(partial)
    if actual_size != expected_size or actual_sha != expected_sha:
        preserve_invalid(partial, f"Pinned verification failed: size={actual_size}, sha256={actual_sha}")
        raise RuntimeError(f"{candidate['id']} weight size/SHA-256 mismatch")
    partial.rename(final)
    emit({"candidate": candidate["id"], "status": "weight_verified", "bytes": actual_size,
          "sha256": actual_sha, "elapsed_seconds": round(time.monotonic() - started, 3)})
    return {"path": str(final.resolve()), "size": actual_size, "sha256": actual_sha,
            "verified": True, "downloaded_this_run": True}


def download_segmented_weight(candidate: dict, directory: Path) -> dict:
    """Resume bounded independent ranges; preserve all range files after merge."""
    final = directory / candidate["filename"]
    expected_size, expected_sha = candidate["size"], candidate["sha256"]
    if verify(final, expected_size, expected_sha):
        emit({"candidate": candidate["id"], "status": "already_verified", "bytes": expected_size})
        return {"path": str(final.resolve()), "size": expected_size, "sha256": expected_sha,
                "verified": True, "downloaded_this_run": False}
    if final.exists():
        preserve_invalid(final, "Final file failed pinned size/SHA-256 verification")
    connections = candidate["connections"]
    url = resolve_url(candidate, candidate["filename"])
    ranges = []
    for index in range(connections):
        start = expected_size * index // connections
        end = expected_size * (index + 1) // connections - 1
        ranges.append((index, start, end, directory / (candidate["filename"] + f".part.{index:02d}")))
    started = time.monotonic()
    emit({"candidate": candidate["id"], "status": "segmented_download_started", "connections": connections,
          "expected_bytes": expected_size, "revision": candidate["revision"],
          "resume_bytes": sum(path.stat().st_size for _, _, _, path in ranges if path.exists()),
          "max_http_range_bytes": REQUEST_CHUNK_BYTES})

    def fetch(section):
        index, start, end, path = section
        size = end - start + 1
        if path.exists() and path.stat().st_size > size:
            preserve_invalid(path, "Range file exceeds its bounded size")
        last_progress = time.monotonic()
        while (path.stat().st_size if path.exists() else 0) < size:
            # Multi-GiB responses were cut short by the current HTTP route.
            # Short bounded subrequests keep the same persistent range file.
            for attempt in range(1, 5):
                offset = path.stat().st_size if path.exists() else 0
                request_end = min(start + offset + REQUEST_CHUNK_BYTES - 1, end)
                expected_offset = request_end - start + 1
                try:
                    with request(url, {"Range": f"bytes={start + offset}-{request_end}"}) as response:
                        matched = RANGE.fullmatch(response.headers.get("Content-Range", ""))
                        if response.status != 206 or matched is None or (
                            int(matched[1]), int(matched[2]), int(matched[3])
                        ) != (start + offset, request_end, expected_size):
                            raise RuntimeError("Server did not honor the exact bounded HTTP Range")
                        length = response.headers.get("Content-Length")
                        if length and int(length) != expected_offset - offset:
                            raise RuntimeError("Range Content-Length does not match the requested bytes")
                        with path.open("ab" if offset else "wb") as stream:
                            while block := response.read(1024 * 1024):
                                if offset + len(block) > expected_offset:
                                    raise RuntimeError("HTTP range body exceeds its bounded size")
                                stream.write(block)
                                offset += len(block)
                                current = time.monotonic()
                                if current - last_progress >= 30:
                                    total = sum(part.stat().st_size for _, _, _, part in ranges if part.exists())
                                    emit({"candidate": candidate["id"], "status": "downloading_ranges", "range": index,
                                          "bytes": total, "expected_bytes": expected_size,
                                          "percent": round(100 * total / expected_size, 2)})
                                    last_progress = current
                    if offset != expected_offset:
                        raise RuntimeError("Bounded HTTP range ended before all requested bytes arrived")
                    break
                except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
                    emit({"candidate": candidate["id"], "status": "range_attempt_failed", "range": index,
                          "attempt": attempt, "partial_bytes": path.stat().st_size if path.exists() else 0,
                          "error": str(exc)})
                    if attempt == 4:
                        raise

    with ThreadPoolExecutor(max_workers=connections) as executor:
        list(executor.map(fetch, ranges))
    assembled = directory / (candidate["filename"] + ".part")
    if assembled.exists():
        if verify(assembled, expected_size, expected_sha):
            assembled.rename(final)
            return {"path": str(final.resolve()), "size": expected_size, "sha256": expected_sha,
                    "verified": True, "downloaded_this_run": True, "connections": connections}
        preserve_invalid(assembled, "Preserve interrupted/invalid merged file before reassembly")
    with assembled.open("wb") as destination:
        for _, _, _, path in ranges:
            with path.open("rb") as source:
                while block := source.read(4 * 1024 * 1024):
                    destination.write(block)
    actual_sha = digest(assembled)
    if assembled.stat().st_size != expected_size or actual_sha != expected_sha:
        preserve_invalid(assembled, f"Merged range SHA-256 mismatch: {actual_sha}")
        # No per-range authoritative hashes exist. Retain suspect parts under
        # new names so a rerun fetches fresh ranges instead of reusing them.
        for _, _, _, path in ranges:
            preserve_invalid(path, "Preserve suspect range after final SHA-256 failure")
        raise RuntimeError("Merged range weight failed pinned size/SHA-256 verification")
    assembled.rename(final)
    emit({"candidate": candidate["id"], "status": "weight_verified", "bytes": expected_size,
          "sha256": actual_sha, "elapsed_seconds": round(time.monotonic() - started, 3)})
    return {"path": str(final.resolve()), "size": expected_size, "sha256": actual_sha,
            "verified": True, "downloaded_this_run": True, "connections": connections,
            "retained_range_files": [str(path.resolve()) for _, _, _, path in ranges]}


def prepare(candidate: dict, cache: Path) -> dict:
    directory = cache / candidate["id"]
    directory.mkdir(parents=True, exist_ok=True)
    record = {**candidate, "status": "preparing", "prepared_at": now(), "assets": {}}
    # Small public source documents stay at the same immutable revision.
    for filename in ("README.md", candidate["license_file"]):
        url = resolve_url(candidate, filename)
        with request(url) as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise RuntimeError("Source document unexpectedly exceeds 1 MiB")
        text = data.decode("utf-8")
        if filename == candidate["license_file"] and not (
            "Apache License" in text and "Version 2.0" in text
        ):
            raise RuntimeError("Pinned license file does not contain Apache License Version 2.0")
        path = directory / filename
        path.write_bytes(data)
        record["assets"][filename] = {"path": str(path.resolve()), "url": url,
                                     "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    downloader = download_segmented_weight if candidate.get("connections", 1) > 1 else download_weight
    record["assets"][candidate["filename"]] = {"url": resolve_url(candidate, candidate["filename"]),
                                              **downloader(candidate, directory)}
    record["status"] = "verified"
    (directory / "source_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> int:
    global LOG_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "model-cache/translation-evaluation")
    parser.add_argument("--candidate", choices=[candidate["id"] for candidate in CANDIDATES], action="append")
    parser.add_argument("--log", type=Path, help="单独保存本次下载 JSONL 日志，追加写入")
    args = parser.parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if args.log is not None:
        LOG_PATH = args.log.resolve()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates = [item for item in CANDIDATES if args.candidate is None or item["id"] in args.candidate]
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = {executor.submit(prepare, item, cache): item for item in candidates}
        for future in as_completed(tasks):
            candidate = tasks[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"id": candidate["id"], "error": f"{type(exc).__name__}: {exc}"})
                emit({"candidate": candidate["id"], "status": "failed", "error": errors[-1]["error"]})
    # Separate candidate processes may finish at different times. Preserve
    # already verified candidates in the aggregate catalog instead of replacing
    # it with just the latest --candidate selection.
    completed = {item["id"] for item in results}
    requested = {item["id"] for item in candidates}
    for candidate in CANDIDATES:
        if candidate["id"] in completed or candidate["id"] in requested:
            continue
        saved = cache / candidate["id"] / "source_manifest.json"
        weight = cache / candidate["id"] / candidate["filename"]
        if saved.is_file() and verify(weight, candidate["size"], candidate["sha256"]):
            record = json.loads(saved.read_text(encoding="utf-8"))
            if record.get("status") == "verified" and record.get("revision") == candidate["revision"]:
                results.append(record)
    report = {"status": "verified" if not errors else "failed", "finished_at": now(),
              "cache": str(cache), "production_models_modified": False,
              "requested_candidates": sorted(requested),
              "candidates": sorted(results, key=lambda item: item["id"]), "errors": errors}
    (cache / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit({"status": report["status"], "cache": str(cache), "verified_candidates": len(results), "errors": errors})
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
