"""Copy only the relevant legacy user data, once, without changing the source.

Weights are referenced in place. Each edition owns its profiles and queue. The
old pending queue is deliberately not imported; historical filter rules remain.
"""
from __future__ import annotations
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
from .identity import EDITION
from .paths import data_root, path_override

_COMMON = ("danmaku-content-filter.json", "danmaku-wordlist.json", "local-features.json")
_CACHE_NAMES = {"Cache", "Code Cache", "GPUCache", "ShaderCache", "GrShaderCache", "Crashpad", "BrowserMetrics"}

def legacy_data_root() -> Path | None:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local")))
    for name in ("FanyiJi", "LocalJaZhSubtitles"):
        candidate = base / name
        if (candidate / "settings.json").is_file() or (candidate / "models").is_dir():
            return candidate
    return None

def _no_links(folder: Path) -> None:
    root_info = folder.lstat()
    if stat.S_ISLNK(root_info.st_mode) or getattr(root_info, "st_file_attributes", 0) & 0x400:
        raise OSError("Legacy profile root contains a link.")
    for parent, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs + files:
            item = Path(parent) / name
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise OSError("Legacy profile contains a link; preserving it without migration.")

def _copy_directory(source: Path, target: Path) -> bool:
    if not source.is_dir() or target.exists():
        return False
    _no_links(source)
    staging = target.with_name(target.name + ".migration-staging")
    if staging.exists():
        raise OSError("Interrupted profile migration exists; no existing data was overwritten.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns(*_CACHE_NAMES, "Singleton*", "DevToolsActivePort"))
    os.replace(staging, target)
    return True

def _copy_failure_history(source: Path, target: Path) -> None:
    original, destination = source / "danmaku-queue.sqlite3", target / "danmaku-queue.sqlite3"
    if not original.is_file() or destination.exists():
        return
    # Use SQLite's snapshot API, not a file copy that can miss committed WAL.
    staging = destination.with_suffix(".migration-staging")
    if staging.exists():
        raise OSError("Interrupted queue migration exists; no data was overwritten.")
    with closing(sqlite3.connect(original.as_uri() + "?mode=ro", uri=True)) as reader:
        with closing(sqlite3.connect(staging)) as writer:
            reader.backup(writer)
    os.replace(staging, destination)
    from .danmaku_store import DanmakuStore
    store = DanmakuStore(destination)
    try:
        store.reset_for_restart()
    finally:
        store.close()

def ensure_initial_migration() -> dict:
    target = data_root()
    marker = target / "edition-migration.json"
    # Explicit test/user locations must never acquire another account's data.
    if path_override("DATA") is not None:
        return {"migrated": False}
    target.mkdir(parents=True, exist_ok=True)
    source = legacy_data_root()
    if marker.exists():
        if source is not None:
            _copy_failure_history(source, target)
        return {"migrated": False}
    result = {"schema_version": 1, "edition": EDITION, "migrated": source is not None,
              "copied_files": [], "copied_directories": [], "models_copied": False,
              "pending_queue_imported": False}
    if source is not None and source.resolve() != target.resolve():
        files = (*_COMMON, "settings.json")
        files += (("browser-preferences.json", "live-source-settings.json") if EDITION == "voice"
                  else ("chrome-caption-settings.json",))
        for name in files:
            original, destination = source / name, target / name
            if original.is_file() and not destination.exists():
                if name == "settings.json":
                    value = json.loads(original.read_text(encoding="utf-8-sig"))
                    if not isinstance(value, dict):
                        raise ValueError("Legacy settings must be an object.")
                    value["processing_mode"] = "standard" if EDITION == "voice" else "chrome_caption"
                    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    shutil.copy2(original, destination)
                result["copied_files"].append(name)
        folders = ("browser", "specialized_model", "cli-runtime") if EDITION == "voice" else ("chrome-caption",)
        for name in folders:
            if _copy_directory(source / name, target / name):
                result["copied_directories"].append(name)
        _copy_failure_history(source, target)
        model_directory = source / "models"
        location = target / "models-location.json"
        if model_directory.is_dir() and not location.exists():
            location.write_text(json.dumps({"directory": str(model_directory.resolve())}, indent=2), encoding="utf-8")
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, marker)
    return result
