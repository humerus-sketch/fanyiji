"""Build a portable application with native runtime, excluding model weights."""
from __future__ import annotations

import json
import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fanyiji.identity import APP_ID, APP_NAME, INSTALL_DIRECTORY


def _build_output(install_local: bool) -> tuple[Path, Path]:
    parent = ROOT / ("build" if install_local else "dist")
    if install_local:
        parent /= "portable"
    return parent, parent / INSTALL_DIRECTORY


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


def _copy_speech_filter_notices(source: Path, destination: Path) -> None:
    """Only upstream notices are public; local evaluation SOURCE metadata is not."""
    for name in ("LICENSE", "LICENSE.txt", "README.md"):
        notice = source / name
        if notice.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(notice, destination / name)


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


def _finish_local_install(destination: Path, installer) -> dict:
    installer.verify_local_install(destination)
    try:
        checked = subprocess.run([str(destination / (APP_NAME + ".exe")), "--smoke-test"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=45, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("新版界面验证执行失败；旧版备份和程序包保留。") from exc
    try:
        smoke = json.loads(checked.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("新版界面验证结果无效；旧版备份和程序包保留。") from exc
    if not isinstance(smoke, dict) or smoke.get("ui_smoke") != "passed" or smoke.get("audio_capture") is not False or smoke.get("settings_written") is not False:
        raise RuntimeError("新版界面验证未通过；旧版备份和程序包保留。")
    return installer.cleanup_local_versions(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-local", action="store_true", help="构建完成后更新项目根固定翻译姬目录")
    args = parser.parse_args()
    runtime = ROOT / "model-cache" / "translation-evaluation" / "llama-b11429"
    if not all((runtime / device / "llama-server.exe").is_file() for device in ("cpu", "cuda")):
        raise SystemExit("请先运行 tools/prepare_llama_runtime.py 准备原生推理组件；打包不需要模型权重。")
    build_parent, destination = _build_output(args.install_local)
    work_directory = ROOT / "build" / "analysis"
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(build_parent),
               "--workpath", str(work_directory), str(ROOT / "FanyiJi.spec")]
    # Other desktop tools may put incompatible DLLs (for example Poppler ICU)
    # on PATH. Dependency analysis must use this Python and Windows libraries.
    environment = os.environ.copy()
    windows = Path(environment.get("SystemRoot", "C:/Windows"))
    environment["PATH"] = os.pathsep.join((sys.base_prefix, str(Path(sys.executable).parent),
                                           str(windows / "System32"), str(windows)))
    environment.pop("PYTHONPATH", None)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    # Weights are deliberately excluded. Runtime libraries belong to the app;
    # downloaded ASR/VAD/GGUF assets live in a separate per-user directory.
    for device in ("cpu", "cuda"):
        target = destination / "runtime" / device
        target.mkdir(parents=True, exist_ok=True)
        for source in (runtime / device).iterdir():
            if source.is_file():
                shutil.copy2(source, target / source.name)
    _write_public_runtime_source(runtime / "SOURCE.json", destination / "runtime" / "SOURCE.json")
    shutil.copy2(ROOT / "README.md", destination / "使用说明.md")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", destination / "THIRD_PARTY_NOTICES.md")
    (destination / "build-info.json").write_text(
        json.dumps({"application": APP_ID, "display_name": APP_NAME, "schema_version": 1,
                    "built_at": datetime.now().astimezone().isoformat(), "models_bundled": False}, indent=2)
        + "\n", encoding="utf-8")
    training_tools = destination / "training_tools"
    training_tools.mkdir(exist_ok=True)
    for source in (ROOT / "tools/train_specialized_model.py", ROOT / "src/fanyiji/specialized_training.py",
                   ROOT / "src/fanyiji/livestream_prompt.py"):
        shutil.copy2(source, training_tools / source.name)
    licenses = destination / "licenses"
    licenses.mkdir(exist_ok=True)
    model_notices = licenses / "models"
    for name in ("asr", "vad", "audio_processing"):
        for source in (ROOT / "models" / name).glob("*LICENSE*"):
            if source.is_file():
                target = model_notices / name / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    for folder in (ROOT / "model-cache/translation-evaluation").iterdir():
        if (folder / "LICENSE.txt").is_file():
            target = model_notices / folder.name / "LICENSE.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / "LICENSE.txt", target)
    speech_filter = ROOT / "model-cache/speech-filter/zipformer-small-int8"
    _copy_speech_filter_notices(speech_filter, model_notices / "speech-filter-zipformer")
    import importlib.metadata
    for package in ("PySide6-Essentials", "shiboken6", "numpy", "sherpa-onnx", "sherpa-onnx-core",
                    "ctranslate2", "sentencepiece", "PyAudioWPatch", "soxr", "pywebview", "pythonnet",
                    "clr-loader", "cffi", "proxy-tools", "bottle", "typing-extensions", "pycparser"):
        distribution = importlib.metadata.distribution(package)
        _copy_distribution_licenses(distribution, package, licenses)
    _copy_supplemental_licenses(ROOT / "third_party_licenses", licenses / "supplemental")
    size = sum(file.stat().st_size for file in destination.rglob("*") if file.is_file())
    forbidden = {".gguf", ".onnx", ".safetensors", ".pt", ".pth"}
    weights = [str(file.relative_to(destination)) for file in destination.rglob("*")
               if file.is_file() and (file.suffix.lower() in forbidden or file.name == "model.bin")]
    if weights:
        raise RuntimeError("打包目录意外包含模型权重：" + ", ".join(weights))
    private_files = [file for file in destination.rglob("*") if file.is_file()
                     and file.name.casefold() in {"signing-key.bin", "cookies", "login data",
                                                  "credentials.json", "auth.json", "secrets.json", "settings.json",
                                                  "local-features.json", "browser-preferences.json",
                                                  "live-source-settings.json",
                                                  "speaker-settings.json", "speaker-profiles.json",
                                                  "danmaku-queue.sqlite3", "delivery.jsonl", "danmaku-content-filter.json",
                                                  "danmaku-wordlist.json"}]
    if private_files:
        raise RuntimeError("打包目录意外包含凭证或用户资料")
    built_destination = destination
    installation = {"installed_locally": False, "backup_folder": None}
    cleanup = None
    if args.install_local:
        if __package__:
            from . import install_local as installer
        else:
            import install_local as installer
        installation = installer.install_local(destination)
        destination = Path(installation["portable_folder"])
        from fanyiji.local_features import enable_local_danmaku
        enable_local_danmaku()
        cleanup = _finish_local_install(destination, installer)
        if installation["backup_folder"] in cleanup["removed_folders"]:
            installation["backup_folder"] = None
    print(json.dumps({"portable_folder": str(destination), "build_folder": str(built_destination),
                      "installed_locally": installation["installed_locally"], "backup_folder": installation["backup_folder"], "total_bytes": size,
                      "cleanup": cleanup,
                      "models_bundled": False, "first_model_download_requires_network": True,
                      "local_translation_internet_required_after_download": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
