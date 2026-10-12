"""Read-only WebView2 runtime discovery and official stable catalog checks.

Evergreen updates are managed by Microsoft Edge Update. The public download
catalog is a comparison source, not a promise that a rollout is available on
this device. No browser profile or account credentials are inspected here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
from urllib.request import Request, urlopen

try:
    import winreg
except ImportError:  # Keeps version checking testable without Windows or Qt.
    winreg = None


CATALOG_URL = "https://developer.microsoft.com/microsoft-edge/api/webview2"
CATALOG_MAX_BYTES = 1024 * 1024
# Microsoft's documented Evergreen client ID, also used by the host SDK.
RUNTIME_CLIENT_ID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){0,3}\Z")


def version_key(value: str) -> tuple[int, ...] | None:
    """Return numeric version components; malformed/channel strings are unknown."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if len(value) > 48 or not _VERSION.fullmatch(value):
        return None
    parts = tuple(int(part) for part in value.split("."))
    return parts if all(part <= 0xFFFFFFFF for part in parts) else None


def _runtime_version(value: object) -> str | None:
    parts = version_key(value)
    if parts is None or len(parts) != 4 or parts[0] == 0:
        return None
    return ".".join(str(part) for part in parts)


def _registry_versions() -> list[str]:
    if winreg is None:
        return []
    versions = []
    key_path = rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{RUNTIME_CLIENT_ID}"
    # Both per-user and machine installations exist. Explicit registry views
    # avoid relying on the bitness of the Python process or installer.
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
            try:
                with winreg.OpenKey(hive, key_path, 0, winreg.KEY_READ | view) as key:
                    version = _runtime_version(winreg.QueryValueEx(key, "pv")[0])
                if version:
                    versions.append(version)
            except OSError:
                continue
    return versions


def _application_directories() -> list[Path]:
    directories = []
    for variable in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(variable)
        if base:
            path = Path(base) / "Microsoft" / "EdgeWebView" / "Application"
            if path not in directories:
                directories.append(path)
    return directories


def _file_versions() -> list[str]:
    versions = []
    for directory in _application_directories():
        try:
            for child in directory.iterdir():
                version = _runtime_version(child.name)
                if version and (child / "msedgewebview2.exe").is_file():
                    versions.append(version)
        except OSError:
            continue
    return versions


def installed_runtime_version() -> str | None:
    """Find the newest registered/installed Evergreen version, without running it."""
    versions = _registry_versions() + _file_versions()
    return max(versions, key=version_key) if versions else None


def _native_architecture() -> str:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if machine in ("amd64", "x86_64", "x64"):
        return "x64"
    if machine in ("x86", "i386", "i686"):
        return "x86"
    raise RuntimeError("无法确定本机浏览器运行时架构")


def latest_catalog_version(timeout: float = 8) -> str:
    """Read Microsoft's bounded stable download metadata; never download a runtime."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 60:
        raise ValueError("版本检查 timeout 必须大于 0 且不超过 60 秒")
    architecture = _native_architecture()
    request = Request(CATALOG_URL, headers={"Accept": "application/json", "User-Agent": "FanyiJi-RuntimeCheck"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(CATALOG_MAX_BYTES + 1)
    except Exception:
        # Network exceptions may contain proxy credentials or machine details.
        raise RuntimeError("无法读取官方浏览器版本目录，请检查网络后重试") from None
    if len(raw) > CATALOG_MAX_BYTES:
        raise RuntimeError("官方浏览器版本目录超过大小限制")
    try:
        catalog = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise RuntimeError("官方浏览器版本目录格式异常") from None
    if not isinstance(catalog, list):
        raise RuntimeError("官方浏览器版本目录格式异常")
    versions = []
    for entry in catalog:
        if not isinstance(entry, dict):
            continue
        version = _runtime_version(entry.get("version"))
        builds = entry.get("builds")
        if not version or not isinstance(builds, list):
            continue
        if any(isinstance(build, dict) and build.get("architecture") == architecture for build in builds):
            versions.append(version)
    if not versions:
        raise RuntimeError("官方浏览器版本目录没有本机架构的有效稳定版本")
    return max(versions, key=version_key)


def runtime_snapshot(active_version: str | None = None, remote: bool = True, *,
                     catalog_version: str | None = None, check_error: str | None = None) -> dict:
    """Describe installed/active/catalog versions, without installing or restarting.

    States: missing, unknown, check_unavailable, new_available,
    restart_required, current. ``current`` means the detected installed version
    is at least as new as the catalog; it does not verify Edge Update policy.
    """
    installed = installed_runtime_version()
    active = _runtime_version(active_version)
    catalog = _runtime_version(catalog_version)
    error = check_error or ""
    if remote:
        try:
            catalog = latest_catalog_version()
        except RuntimeError as exc:
            error = str(exc)
    detected = installed or active
    if detected is None:
        state = "missing"
        message = "未检测到 Microsoft WebView2 Runtime"
    elif installed and active and version_key(installed) > version_key(active):
        state = "restart_required"
        message = "浏览器运行时已更新，重新打开专用浏览器后使用新版本"
    elif error:
        state = "check_unavailable"
        message = error
    elif catalog is None:
        state = "unknown"
        message = "尚未联网检查官方浏览器稳定版本目录"
    elif version_key(catalog) > version_key(detected):
        state = "new_available"
        message = "官方目录有更新版本；Evergreen 将按 Microsoft 更新策略自动部署"
    else:
        state = "current"
        message = "检测到的运行时版本不低于官方稳定版本目录"
    return {
        "installed_version": installed,
        "active_version": active,
        "catalog_version": catalog,
        "state": state,
        "message": message,
        "catalog_latest_not_evergreen_guarantee": True,
        "check_error": error or None,
    }
