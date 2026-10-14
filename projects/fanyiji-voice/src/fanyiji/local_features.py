"""Public feature availability and compatibility with legacy local installers."""
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
    """All editions include sending; preferences and browser login control its use."""
    return True


def enable_local_danmaku() -> None:
    """Keep old installer calls compatible; this file no longer gates the feature."""
    path = data_root() / "local-features.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": 1, "host": _host_key(), "danmaku": True}
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
