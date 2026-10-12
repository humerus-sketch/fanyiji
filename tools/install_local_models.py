"""Install already downloaded, pinned weights outside the application package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fanyiji.models_download import ModelCatalog, ModelDownloader
from fanyiji.asr_catalog import AsrCatalog
from fanyiji.paths import data_root


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(source, target, size=None, sha=None):
    if size is not None and (source.stat().st_size != size or digest(source) != sha):
        raise RuntimeError(f"Source did not pass pinned verification: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.stat().st_size != target.stat().st_size or digest(source) != digest(target):
            raise RuntimeError(f"Existing target differs; preserved without replacement: {target}")
        return "already_present"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=data_root() / "models")
    parser.add_argument("--active-model")
    parser.add_argument("--active-asr")
    args = parser.parse_args()
    directory = args.destination.resolve()
    catalog = ModelCatalog.default()
    asr_catalog = AsrCatalog.default(catalog)
    records = []
    for entry in catalog.entries:
        sources = []
        for file in entry.files:
            source = ROOT / "model-cache/speech-filter/zipformer-small-int8" / Path(file.relative_path).name if file.relative_path.startswith("speech_filter/") else (
                ROOT / "models" / file.relative_path if entry.required else (
                ROOT / "model-cache/translation-evaluation" / entry.id / Path(file.relative_path).name)
            )
            sources.append((source, file))
        if not all(source.is_file() for source, _ in sources):
            records.append({"id": entry.id, "status": "not_downloaded"})
            continue
        methods = []
        for source, file in sources:
            methods.append(install(source, directory / file.relative_path, file.size, file.sha256))
        records.append({"id": entry.id, "status": "verified_and_installed", "methods": methods})
        if entry.id == "speech-filter-zipformer":
            candidate = sources[0][0].parent
            for name in ("LICENSE", "LICENSE.txt", "SOURCE.json", "README.md"):
                if (candidate / name).is_file():
                    install(candidate / name, directory / "licenses" / entry.id / name)
        if not entry.required:
            candidate = sources[0][0].parent
            for name in ("LICENSE.txt", "README.md", "source_manifest.json"):
                if (candidate / name).is_file():
                    install(candidate / name, directory / "licenses" / entry.id / name)
    for entry in asr_catalog.entries:
        if entry.model_kind == "sense_voice":
            continue
        candidate = ROOT / "model-cache/asr-evaluation" / entry.id
        sources = [(candidate / Path(file.relative_path).relative_to("asr").relative_to(entry.id), file)
                   for file in entry.files]
        if not all(source.is_file() for source, _ in sources):
            records.append({"id": entry.id, "status": "not_downloaded"})
            continue
        methods = [install(source, directory / file.relative_path, file.size, file.sha256)
                   for source, file in sources]
        records.append({"id": entry.id, "status": "verified_and_installed", "methods": methods})
        for name in ("SOURCE.json", "LICENSE.whisper", "LICENSE.qwen-asr"):
            if (candidate / name).is_file():
                install(candidate / name, directory / "licenses" / entry.id / name)
    for folder, names in (("asr", ("LICENSE", "MODEL_LICENSE")), ("vad", ("LICENSE",))):
        for name in names:
            source = ROOT / "models" / folder / name
            if source.is_file():
                install(source, directory / folder / name)
    for source in (ROOT / "models/self_test").glob("*.wav"):
        install(source, directory / "self_test" / source.name)
    if args.active_model:
        downloader = ModelDownloader(directory)
        if args.active_asr:
            downloader.activate_models(catalog, args.active_model, asr_catalog, args.active_asr)
        else:
            downloader.activate_model(catalog, args.active_model)
    report = {"models_directory": str(directory), "models_bundled": False,
              "active_model": args.active_model, "active_asr": args.active_asr, "entries": records}
    path = ROOT / "artifacts/diagnostics/local-model-installation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
