"""Host-local feature grants stay outside distributable application packages."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

from .paths import data_root


def _host_key() -> str:
    return hashlib.sha256(platform.node().casefold().encode("utf-8")).hexdigest()


def danmaku_available() -> bool:
    if not getattr(sys, "frozen", False):
        return True
    try:
        value = json.loads((data_root() / "local-features.json").read_text(encoding="utf-8"))
        return (isinstance(value, dict) and value.get("schema_version") == 1
                and value.get("host") == _host_key() and value.get("danmaku") is True)
    except (OSError, ValueError, UnicodeError):
        return False


def enable_local_danmaku() -> None:
    """Only the explicit local installer grants this machine the feature."""
    path = data_root() / "local-features.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": 1, "host": _host_key(), "danmaku": True}
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
