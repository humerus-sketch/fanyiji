"""Prepare pinned official Windows llama.cpp packages in an isolated cache.

No system PATH, driver, virtual environment, production model, or application
is changed. Verification runs --version, --help and --list-devices without a
model; it does not start a listening server or perform model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TAG = "b11429"
SOURCE = f"https://github.com/ggml-org/llama.cpp/releases/tag/{TAG}"
LICENSE_URL = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{TAG}/LICENSE"
CUDA_LICENSE_URL = "https://docs.nvidia.com/cuda/archive/12.4.0/eula/index.html"
ASSETS = {
    "llama-b11429-bin-win-cpu-x64.zip": (19398918, "1283323272b04cd07905816a597a0da810918102de958f4ff6f7bbaa70ed2efe"),
    "llama-b11429-bin-win-cuda-12.4-x64.zip": (264478423, "dd6df685c1024e6aa55ca55ad35038692d5e284d4dc895e7a646c3245d3d14ff"),
    "cudart-llama-bin-win-cuda-12.4-x64.zip": (391443627, "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fanyiji-runtime-preparation"}), timeout=60)


def release_assets() -> dict[str, dict[str, Any]]:
    with request(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{TAG}") as response:
        release = json.load(response)
    if release["tag_name"] != TAG:
        raise RuntimeError("Unexpected llama.cpp release tag")
    result = {}
    for asset in release["assets"]:
        name = asset["name"]
        if name not in ASSETS:
            continue
        size, digest = ASSETS[name]
        expected_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{name}"
        if (asset["size"], asset.get("digest"), asset["browser_download_url"]) != (size, f"sha256:{digest}", expected_url):
            raise RuntimeError(f"Official metadata differs from the pinned runtime asset: {name}")
        result[name] = {"size": size, "sha256": digest, "url": expected_url}
    if set(result) != set(ASSETS):
        raise RuntimeError("The pinned release is missing required Windows runtime packages")
    return result


def download(asset: dict[str, Any], path: Path) -> None:
    if path.is_file() and path.stat().st_size == asset["size"] and sha256(path) == asset["sha256"]:
        print(f"Verified cached package: {path.name}", flush=True)
        return
    temporary = path.with_suffix(path.suffix + ".download")
    print(f"Downloading {path.name}: {asset['size'] / 1024 ** 2:.1f} MiB", flush=True)
    count, progress = 0, 64 * 1024 ** 2
    with request(asset["url"]) as response, temporary.open("wb") as output:
        while data := response.read(1024 ** 2):
            output.write(data)
            count += len(data)
            if count >= progress:
                print(f"  {count / 1024 ** 2:.0f}/{asset['size'] / 1024 ** 2:.1f} MiB", flush=True)
                progress += 64 * 1024 ** 2
    if count != asset["size"] or sha256(temporary) != asset["sha256"]:
        raise RuntimeError(f"Downloaded runtime verification failed: {temporary}")
    temporary.replace(path)


def archive_target(info: zipfile.ZipInfo, directory: Path) -> Path:
    member = PurePosixPath(info.filename.replace("\\", "/"))
    mode = info.external_attr >> 16
    forbidden = {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)], *[f"lpt{i}" for i in range(1, 10)]}
    if member.is_absolute() or not member.parts or ".." in member.parts or stat.S_ISLNK(mode):
        raise ValueError(f"Unsafe runtime archive entry: {info.filename}")
    if any(":" in part or part.endswith((" ", ".")) or part.split(".")[0].lower() in forbidden for part in member.parts):
        raise ValueError(f"Unsafe Windows runtime archive name: {info.filename}")
    target = directory.joinpath(*member.parts).resolve()
    if not target.is_relative_to(directory.resolve()):
        raise ValueError(f"Runtime archive entry escapes the destination: {info.filename}")
    return target


def extract(archive: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = [(info, archive_target(info, directory)) for info in source.infolist()]
        if sum(info.file_size for info, _ in members) > 4 * 1024 ** 3:
            raise ValueError("Unexpectedly large runtime archive")
        for info, target in members:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 ** 2)
            if target.stat().st_size != info.file_size:
                raise RuntimeError(f"Incomplete runtime extraction: {target}")


def find_server(directory: Path) -> Path:
    candidates = list(directory.rglob("llama-server.exe"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one llama-server.exe in {directory}, found {len(candidates)}")
    return candidates[0]


def check(executable: Path, arguments: list[str], output: Path) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run([str(executable), *arguments], cwd=executable.parent,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    output.with_suffix(".stdout.log").write_text(stdout, encoding="utf-8")
    output.with_suffix(".stderr.log").write_text(stderr, encoding="utf-8")
    print(f"Checked {executable.parent.name} {arguments}: exit={result.returncode}", flush=True)
    return {"command": [str(executable), *arguments], "returncode": result.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3), "stdout": stdout, "stderr": stderr}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "model-cache" / "translation-evaluation" / f"llama-{TAG}")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("These fixed runtime packages support Windows x64 only")
    directory = args.output.resolve()
    cache_root = (ROOT / "model-cache" / "translation-evaluation").resolve()
    if not directory.is_relative_to(cache_root) or directory == cache_root:
        parser.error("--output must stay inside the isolated translation-evaluation cache")
    downloads = directory / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    assets = release_assets()
    for name, asset in assets.items():
        download(asset, downloads / name)
    for profile, name in (("cpu", "llama-b11429-bin-win-cpu-x64.zip"), ("cuda", "llama-b11429-bin-win-cuda-12.4-x64.zip")):
        extract(downloads / name, directory / profile)
    cpu = find_server(directory / "cpu")
    cuda = find_server(directory / "cuda")
    extract(downloads / "cudart-llama-bin-win-cuda-12.4-x64.zip", directory / "cuda" / "cuda-runtime")
    for library in (directory / "cuda" / "cuda-runtime").rglob("*.dll"):
        shutil.copy2(library, cuda.parent / library.name)
    with request(LICENSE_URL) as response:
        license_text = response.read()
    for location in (directory, cpu.parent, cuda.parent):
        (location / "LICENSE.llama.cpp").write_bytes(license_text)
    manifest: dict[str, Any] = {"tag": TAG, "source": SOURCE, "packages": assets,
                              "license": {"llama_cpp": {"name": "MIT", "source": LICENSE_URL}, "cuda_runtime": {"source": CUDA_LICENSE_URL}},
                              "cpu_server": str(cpu), "cuda_server": str(cuda),
                              "server_sha256": {"cpu": sha256(cpu), "cuda": sha256(cuda)},
                              "system_path_changed": False, "driver_changed": False, "venv_changed": False,
                              "model_inference_performed": False, "checks": {}}
    (directory / "SOURCE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log_directory = directory / "verification"
    log_directory.mkdir(exist_ok=True)
    for profile, executable in (("cpu", cpu), ("cuda", cuda)):
        manifest["checks"][profile] = {
            flag[2:]: check(executable, [flag], log_directory / f"{profile}-{flag[2:]}")
            for flag in ("--version", "--help", "--list-devices")
        }
    manifest["checked_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")
    manifest["basic_checks_passed"] = all(result["returncode"] == 0 for profile in manifest["checks"].values() for result in profile.values())
    (directory / "SOURCE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cpu_server": str(cpu), "cuda_server": str(cuda), "basic_checks_passed": manifest["basic_checks_passed"]}, ensure_ascii=False), flush=True)
    return 0 if manifest["basic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
