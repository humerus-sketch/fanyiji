"""Independent preferences and profiles; downloaded weights may be reused."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from .identity import APP_ID, ENV_PREFIX

def path_override(name: str) -> Path | None:
    value = (os.environ.get(f"{ENV_PREFIX}_{name}") or os.environ.get(f"FANYIJI_{name}")
             or os.environ.get(f"LOCAL_SUBTITLES_{name}"))
    return Path(value).resolve() if value else None

def app_root() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]

def data_root() -> Path:
    override = path_override("DATA")
    if override is not None:
        return override
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local"))) / APP_ID

def models_root() -> Path:
    override = path_override("MODELS")
    if override is not None:
        return override
    try:
        value = json.loads((data_root() / "models-location.json").read_text(encoding="utf-8"))
        stored = value.get("directory") if isinstance(value, dict) else None
        if isinstance(stored, str) and Path(stored).is_absolute() and Path(stored).is_dir():
            return Path(stored)
    except (OSError, ValueError, UnicodeError):
        pass
    local = app_root() / "models"
    if not getattr(sys, "frozen", False) and local.is_dir():
        return local
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local"))) / "FanyiJiModels" / "models"
