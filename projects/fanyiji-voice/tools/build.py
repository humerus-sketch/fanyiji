"""Standalone portable build; runtime libraries included, weights excluded."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fanyiji.identity import APP_ID, APP_NAME, INSTALL_DIRECTORY, EDITION

def _write_public_runtime_source(source: Path, destination: Path) -> None:
    """Publish provenance only; cached execution paths and device probes stay local."""
    value = json.loads(source.read_text(encoding="utf-8"))
    public = {
        "tag": value["tag"],
        "source": value["source"],
        "packages": {
            name: {field: asset[field] for field in ("size", "sha256", "url")}
            for name, asset in value["packages"].items()
        },
        "license": {
            "llama_cpp": {field: value["license"]["llama_cpp"][field]
                          for field in ("name", "source")},
            "cuda_runtime": {"source": value["license"]["cuda_runtime"]["source"]},
        },
        "server_sha256": {device: value["server_sha256"][device] for device in ("cpu", "cuda")},
    }
    destination.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def _copy_distribution_licenses(distribution, package: str, destination: Path) -> None:
    for item in distribution.files or []:
        relative = PurePosixPath(str(item).replace("\\", "/"))
        if (len(relative.parts) < 2 or not relative.parts[0].endswith(".dist-info")
                or not any(term in relative.name.lower() for term in ("license", "copying", "notice"))):
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("安装依赖的许可证路径无效")
        source = Path(distribution.locate_file(item))
        if source.is_file():
            # Nested notices often share a basename; retain their original paths.
            target = destination / package / Path(*relative.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

def _copy_supplemental_licenses(source: Path, destination: Path) -> None:
    manifest = source / "MANIFEST.json"
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise RuntimeError("缺少或无法读取发行包补充许可证清单") from exc
    files = value.get("files") if isinstance(value, dict) else None
    if not isinstance(files, list) or not files or any(not isinstance(item, str) for item in files):
        raise RuntimeError("发行包补充许可证清单无效")
    for item in files:
        relative = PurePosixPath(item.replace("\\", "/"))
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                or any(":" in part for part in relative.parts)):
            raise RuntimeError("发行包补充许可证路径无效")
        path = source.joinpath(*relative.parts)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("发行包补充许可证缺失或为空：" + item)
    shutil.copytree(source, destination, dirs_exist_ok=True)
def main():
    parser = argparse.ArgumentParser(description="Build this edition without model weights or user data.")
    parser.add_argument("--llama-runtime", type=Path, default=ROOT / "model-cache/translation-evaluation/llama-b11429")
    args = parser.parse_args()
    destination = ROOT / "dist" / INSTALL_DIRECTORY
    if EDITION == "voice" and not all((args.llama_runtime / device / "llama-server.exe").is_file() for device in ("cpu", "cuda")):
        raise SystemExit("Prepare tools/prepare_llama_runtime.py or pass --llama-runtime DIRECTORY.")
    environment = os.environ.copy()
    windows = Path(environment.get("SystemRoot", "C:/Windows"))
    environment["PATH"] = os.pathsep.join((sys.base_prefix, str(Path(sys.executable).parent), str(windows / "System32"), str(windows)))
    environment.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(ROOT / "dist"),
                    "--workpath", str(ROOT / "build/analysis"), str(ROOT / "FanyiJi.spec")], cwd=ROOT, env=environment, check=True)
    if EDITION == "voice":
        for device in ("cpu", "cuda"):
            target = destination / "runtime" / device
            target.mkdir(parents=True, exist_ok=True)
            for source in (args.llama_runtime / device).iterdir():
                if source.is_file():
                    shutil.copy2(source, target / source.name)
        _write_public_runtime_source(args.llama_runtime / "SOURCE.json", destination / "runtime/SOURCE.json")
        training = destination / "training_tools"
        training.mkdir(exist_ok=True)
        for source in (ROOT / "tools/train_specialized_model.py", ROOT / "src/fanyiji/specialized_training.py", ROOT / "src/fanyiji/livestream_prompt.py"):
            shutil.copy2(source, training / source.name)
    shutil.copy2(ROOT / "README.md", destination / "使用说明.md")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", destination / "THIRD_PARTY_NOTICES.md")
    import importlib.metadata
    licenses = destination / "licenses"
    packages = ("PySide6-Essentials", "shiboken6", "numpy", "sherpa-onnx", "sherpa-onnx-core", "websocket-client", "pyinstaller")
    if EDITION == "voice":
        packages += ("ctranslate2", "sentencepiece", "PyAudioWPatch", "soxr", "pywebview", "pythonnet", "clr-loader", "cffi", "proxy-tools", "bottle", "typing-extensions", "pycparser")
    for package in packages:
        _copy_distribution_licenses(importlib.metadata.distribution(package), package, licenses)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        raise RuntimeError("缺少所用 Python 运行时许可证")
    (licenses / "Python").mkdir(exist_ok=True)
    shutil.copy2(python_license, licenses / "Python/LICENSE.txt")
    _copy_supplemental_licenses(ROOT / "third_party_licenses", licenses / "supplemental")
    (destination / "build-info.json").write_text(json.dumps({"application": APP_ID, "display_name": APP_NAME,
        "edition": EDITION, "schema_version": 1, "models_bundled": False,
        "built_at": datetime.now(timezone(timedelta(hours=8))).isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")
    forbidden = {".gguf", ".onnx", ".safetensors", ".pt", ".pth", ".key", ".pfx", ".p12"}
    private = {"model.bin", "cookies", "login data", "credentials.json", "auth.json", "secrets.json", "settings.json",
               "local-features.json", "browser-preferences.json", "chrome-caption-settings.json", "danmaku-queue.sqlite3",
               "delivery.jsonl", "danmaku-content-filter.json", "danmaku-wordlist.json"}
    files = [file for file in destination.rglob("*") if file.is_file()]
    if any(file.suffix.lower() in forbidden or file.name.casefold() in private for file in files):
        raise RuntimeError("Package contains model weights or user data.")
    print(json.dumps({"portable_folder": str(destination), "application": APP_ID, "edition": EDITION,
                      "total_bytes": sum(file.stat().st_size for file in files), "models_bundled": False}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
