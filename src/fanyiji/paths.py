"""Resolve external model storage separately from application/runtime files."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .identity import APP_ID, LEGACY_APP_ID


def path_override(name: str) -> Path | None:
    """The renamed variable wins; explicit legacy overrides remain supported."""
    value = os.environ.get(f"FANYIJI_{name}") or os.environ.get(f"LOCAL_SUBTITLES_{name}")
    return Path(value).resolve() if value else None


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def models_root() -> Path:
    override = path_override("MODELS")
    if override is not None:
        return override
    if getattr(sys, "frozen", False):
        return data_root() / "models"
    return app_root() / "models"


def data_root() -> Path:
    override = path_override("DATA")
    if override is not None:
        return override
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local")))
    current, legacy = base / APP_ID, base / LEGACY_APP_ID
    # Reuse populated legacy storage in place: models may be many GiB. Once the
    # new directory exists it always wins, even if it is currently empty.
    if current.exists() or current.is_symlink():
        return current
    if (legacy / "settings.json").is_file() or any(
        (legacy / name).is_dir() for name in ("models", "specialized_model")
    ):
        return legacy
    return current
