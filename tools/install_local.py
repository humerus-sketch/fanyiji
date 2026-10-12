"""Copy a verified portable build into the fixed local application folder."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import marshal
import os
import re
import shutil
import stat
import sys
import types
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
APP_ID = "FanyiJi"
DISPLAY_NAME = "翻译姬"
APPLICATION = DISPLAY_NAME + ".exe"
LEGACY_APP_ID = "LocalJaZhSubtitles"
LEGACY_APPLICATION = LEGACY_APP_ID + ".exe"
MANIFEST = "local-install-manifest.json"
_WEIGHTS = {".gguf", ".onnx", ".safetensors", ".pt", ".pth"}
_PRIVATE = {"signing-key.bin",
            "cookies", "login data", "credentials.json", "auth.json", "secrets.json", "settings.json",
            "speaker-settings.json", "speaker-profiles.json",
            "live-source-settings.json",
            "danmaku-content-filter.json", "danmaku-wordlist.json"}
_CACHE_SOURCES = {
    f"training_tools/__pycache__/{name}.{sys.implementation.cache_tag}.pyc": f"training_tools/{name}.py"
    for name in ("livestream_prompt", "specialized_training")
}


class InstallError(RuntimeError):
    pass


def _plain_path(value: Path) -> Path:
    path = Path(os.path.abspath(value))
    for ancestor in (path, *path.parents):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise InstallError("安装路径含 symlink/junction，已拒绝：" + str(ancestor))
    return path.resolve()


def _path_key(path: Path) -> str:
    return str(path).replace("\\", "/").rstrip("/").casefold()


def _disjoint(first: Path, second: Path) -> None:
    left, right = _path_key(first), _path_key(second)
    if left == right or left.startswith(right + "/") or right.startswith(left + "/"):
        raise InstallError("源、安装目录和备份目录不能相同或相互嵌套。")


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise InstallError("安装清单包含无效相对路径。")
    parts = value.split("/")
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    if any(part in {"", ".", ".."} or part.endswith((" ", ".")) or part.split(".")[0].casefold() in reserved for part in parts):
        raise InstallError("安装清单包含无效相对路径。")
    if PurePosixPath(value).is_absolute():
        raise InstallError("安装清单必须使用相对路径。")
    return value


def _files(folder: Path, *, directories: set[str] | None = None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    folded: set[str] = set()
    if not folder.exists():
        return result
    if not folder.is_dir():
        raise InstallError("安装路径必须是目录：" + str(folder))
    stack = [folder]
    while stack:
        for entry in stack.pop().iterdir():
            info = entry.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise InstallError("包目录含 symlink/junction，已拒绝：" + str(entry))
            if stat.S_ISDIR(info.st_mode):
                if directories is not None:
                    directories.add(_relative(entry.relative_to(folder).as_posix()))
                stack.append(entry)
            elif stat.S_ISREG(info.st_mode):
                relative = _relative(entry.relative_to(folder).as_posix())
                if relative.casefold() in folded:
                    raise InstallError("包目录包含大小写冲突路径。")
                folded.add(relative.casefold())
                result[relative] = entry
            else:
                raise InstallError("包目录包含非常规文件。")
    return result


def _read_json(path: Path) -> dict:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise InstallError("构建或安装清单过大。")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise InstallError("构建或安装清单不可读。") from exc
    if not isinstance(value, dict):
        raise InstallError("构建或安装清单无效。")
    return value


def _identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def _source_files(source: Path) -> dict[str, Path]:
    directories: set[str] = set()
    files = _files(source, directories=directories)
    marker = files.get("build-info.json")
    if marker is None or APPLICATION not in files:
        raise InstallError("仅可安装带构建信息和完整 EXE 的程序包。")
    info = _read_json(marker)
    if info.get("application") != APP_ID or info.get("display_name") != DISPLAY_NAME or type(info.get("schema_version")) is not int or info["schema_version"] != 1 or info.get("models_bundled") is not False:
        raise InstallError("源包构建信息无效。")
    if LEGACY_APPLICATION in files:
        raise InstallError("源包仍包含旧名称 EXE，已拒绝安装。")
    if any(relative.casefold().startswith(MANIFEST.casefold()) for relative in files):
        raise InstallError("源包包含安装器保留的清单文件。")
    for relative in (*files, *directories):
        path = Path(relative)
        parts = [part.casefold() for part in relative.split("/")]
        name = parts[-1]
        forbidden = path.suffix.casefold() in _WEIGHTS or name == "model.bin" or name in _PRIVATE
        forbidden |= name == ".env" or name.startswith(".env.") or path.suffix.casefold() in {".key", ".pfx", ".p12"}
        forbidden |= "node_modules" in parts or parts[0] in {"models", "model-cache", "user data", "diagnostics", "specialized-training"}
        forbidden |= "__pycache__" in parts or path.suffix.casefold() == ".pyc"
        if forbidden:
            raise InstallError("源包包含模型、凭证、用户资料或开发依赖，已拒绝：" + relative)
    return files


def _verified_installation(folder: Path, *, current: bool = False) -> dict:
    """Validate all owned files before retiring any known package directory."""
    directories: set[str] = set()
    files = _files(folder, directories=directories)
    if MANIFEST not in files:
        raise InstallError("目录缺少安装管理清单，所有文件保留。")
    managed, preserved = _managed_files(folder, files, files, directories)
    manifest = _read_json(files[MANIFEST])
    if current and manifest.get("application") != APP_ID:
        raise InstallError("最新安装应用身份无效，已停止清理。")
    marker = files.get("build-info.json")
    if marker is None:
        raise InstallError("目录缺少构建信息，所有文件保留。")
    info = _read_json(marker)
    if info.get("application") not in ((APP_ID,) if current else (APP_ID, LEGACY_APP_ID)) or info.get("models_bundled") is not False:
        raise InstallError("目录构建信息无效，所有文件保留。")
    # Manifest ownership alone must not authorize deleting credentials or models.
    for relative in (*managed, *directories):
        parts = [part.casefold() for part in relative.split("/")]
        path = Path(relative)
        if (path.suffix.casefold() in _WEIGHTS | {".log"} or parts[-1] == "model.bin" or parts[-1] in _PRIVATE
                or parts[-1] == ".env" or parts[-1].startswith(".env.")
                or path.suffix.casefold() in {".key", ".pfx", ".p12"}
                or parts[-1] in {"preferences", "secure preferences", "history"}
                or "node_modules" in parts or parts[0] in {"models", "model-cache", "user data", "diagnostics", "specialized-training", "browser", "profiles", "profile", "logs", "log"}):
            raise InstallError("目录包含模型、凭证或用户资料，所有文件保留。")
    if any(path.stat().st_nlink > 1 for path in files.values()):
        raise InstallError("目录包含 hardlink，所有文件保留。")
    return {"files": managed, "preserved_python_caches": preserved,
            "all_files": {relative: _identity(path) for relative, path in files.items()},
            "directories": directories}


def verify_local_install(destination: Path | None = None) -> dict:
    destination = _plain_path(destination or ROOT / DISPLAY_NAME)
    verified = _verified_installation(destination, current=True)
    return {"installed_folder": str(destination), "managed_files": len(verified["files"]),
            "exe_sha256": verified["files"][APPLICATION]["sha256"]}


def cleanup_local_versions(destination: Path | None = None) -> dict:
    """Retire only verified installs and exact duplicate packages in fixed paths."""
    root = _plain_path(ROOT)
    destination = _plain_path(destination or root / DISPLAY_NAME)
    if destination != root / DISPLAY_NAME:
        raise InstallError("自动清理仅支持项目根固定安装目录。")
    current = _verified_installation(destination, current=True)
    versions = [current["files"]]
    candidates: list[tuple[Path, dict]] = []
    skipped = []
    backup_root = root / "build" / "local-install-backups"
    try:
        backup_root = _plain_path(backup_root)
        if backup_root.exists():
            for folder in sorted(backup_root.iterdir()):
                if not re.fullmatch(r"install-\d{8}-\d{6}-\d{6}", folder.name):
                    skipped.append({"folder": str(folder), "reason": "未知备份名称，保留"})
                    continue
                try:
                    folder = _plain_path(folder)
                    verified = _verified_installation(folder)
                    _check_available(folder, _files(folder))
                    versions.append(verified["files"])
                    candidates.append((folder, verified))
                except (OSError, InstallError) as exc:
                    skipped.append({"folder": str(folder), "reason": str(exc)})
    except (OSError, InstallError) as exc:
        skipped.append({"folder": str(backup_root), "reason": str(exc)})
    for relative in ("build/portable/" + DISPLAY_NAME, "dist/" + DISPLAY_NAME):
        folder = root / relative
        try:
            folder = _plain_path(folder)
            if not folder.exists():
                continue
            directories: set[str] = set()
            files = _source_files(folder)
            _files(folder, directories=directories)
            identities = {name: _identity(path) for name, path in files.items()}
            if identities not in versions:
                raise InstallError("程序包不匹配已验证安装版本，所有文件保留。")
            allowed_directories = {parent.as_posix() for name in files for parent in PurePosixPath(name).parents if parent.as_posix() != "."}
            if not directories.issubset(allowed_directories):
                raise InstallError("程序包含未知空目录，所有文件保留。")
            _check_available(folder, files)
            candidates.append((folder, {"all_files": identities, "directories": directories}))
        except (OSError, InstallError) as exc:
            skipped.append({"folder": str(folder), "reason": str(exc)})
    removed = []
    removed_bytes = 0
    for folder, verified in candidates:
        try:
            # Resolve and recheck immediately before a recursive delete. Only the
            # fixed folders inventoried above can enter this removal list.
            folder = _plain_path(folder)
            if not folder.is_relative_to(root) or folder == root or folder == destination:
                raise InstallError("清理路径超出允许范围，所有文件保留。")
            _disjoint(folder, destination)
            directories = set()
            files = _files(folder, directories=directories)
            if (directories != verified["directories"] or
                    {name: _identity(path) for name, path in files.items()} != verified["all_files"]):
                raise InstallError("清理前目录内容已变化，所有文件保留。")
            _check_available(folder, files)
            size = sum(identity["size"] for identity in verified["all_files"].values())
            shutil.rmtree(folder)
            removed.append(str(folder))
            removed_bytes += size
        except (OSError, InstallError) as exc:
            skipped.append({"folder": str(folder), "reason": str(exc)})
    return {"removed_folders": removed, "removed_bytes": removed_bytes, "skipped": skipped}


def _normalized_code(code):
    if isinstance(code, types.CodeType):
        return code.replace(co_filename="", co_consts=tuple(_normalized_code(item) for item in code.co_consts))
    return code


def _preserved_caches(existing: dict[str, Path], managed: dict, new: dict[str, Path], manifest: dict) -> dict:
    """Retain only verified training imports; never delete caches or ship them."""
    recorded = manifest.get("preserved_python_caches", {})
    if not isinstance(recorded, dict) or not set(recorded).issubset(_CACHE_SOURCES):
        raise InstallError("保留的 Python 缓存清单无效。")
    result = {}
    for relative in set(existing) - set(managed) - {MANIFEST}:
        source_name = _CACHE_SOURCES.get(relative)
        path = existing[relative]
        if source_name not in managed or source_name not in new or path.stat().st_size > 2 * 1024 * 1024:
            raise InstallError("安装目录含未知文件，已拒绝覆盖；所有文件保留。")
        try:
            raw = path.read_bytes()
            if len(raw) < 16 or raw[:4] != importlib.util.MAGIC_NUMBER:
                raise ValueError("Invalid Python cache")
            flags = int.from_bytes(raw[4:8], "little")
            if flags not in (0, 3):
                raise ValueError("Unchecked Python cache")
            code = marshal.loads(raw[16:])
            if not isinstance(code, types.CodeType):
                raise ValueError("Invalid Python code")
            identity = _identity(path)
            old = existing[source_name]
            expected = compile(old.read_bytes(), str(old), "exec", dont_inherit=True, optimize=0)
            if _normalized_code(code) != _normalized_code(expected) and recorded.get(relative) != identity:
                raise ValueError("Unverified Python cache")
            upcoming = new[source_name]
            source_bytes = upcoming.read_bytes()
            if flags == 0:
                applies = raw[8:12] == (int(upcoming.stat().st_mtime) & 0xffffffff).to_bytes(4, "little")
                applies &= raw[12:16] == (len(source_bytes) & 0xffffffff).to_bytes(4, "little")
            else:
                applies = raw[8:16] == importlib.util.source_hash(source_bytes)
            # A retained old cache may never override updated source. If its
            # header still matches, prove its code matches the incoming source.
            if applies:
                expected = compile(source_bytes, str(upcoming), "exec", dont_inherit=True, optimize=0)
                if _normalized_code(code) != _normalized_code(expected):
                    raise ValueError("Python cache shadows incoming source")
            result[relative] = identity
        except (ValueError, TypeError, EOFError, OSError, SyntaxError) as exc:
            raise InstallError("安装目录含未验证的 Python 缓存，已拒绝覆盖：" + relative) from exc
    return result


def _managed_files(destination: Path, existing: dict[str, Path], new: dict[str, Path], directories: set[str]) -> tuple[dict, dict]:
    if not existing:
        # Empty directories with unknown names also make an initial target non-empty.
        if destination.exists() and any(destination.iterdir()):
            raise InstallError("首次安装目录必须为空；现有文件和目录均保留。")
        return {}, {}
    if MANIFEST not in existing:
        raise InstallError("安装目录非空且没有管理清单，已拒绝覆盖；现有文件均保留。")
    manifest = _read_json(existing[MANIFEST])
    managed = manifest.get("files")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or not isinstance(managed, dict):
        raise InstallError("现有安装管理清单无效，已拒绝覆盖。")
    folded: set[str] = set()
    for relative, identity in managed.items():
        _relative(relative)
        if relative == MANIFEST or relative.casefold() in folded or not isinstance(identity, dict):
            raise InstallError("现有安装清单包含冲突路径。")
        folded.add(relative.casefold())
        if type(identity.get("size")) is not int or identity["size"] < 0 or not isinstance(identity.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]):
            raise InstallError("现有安装文件校验信息无效。")
    application = manifest.get("application")
    if application not in (None, APP_ID, LEGACY_APP_ID):
        raise InstallError("现有安装管理清单的应用身份无效，已拒绝覆盖。")
    executable = LEGACY_APPLICATION if application == LEGACY_APP_ID else APPLICATION
    if application is None:
        executable = APPLICATION if APPLICATION in managed else LEGACY_APPLICATION
    if executable not in managed:
        raise InstallError("现有安装清单缺少必要构建文件。")
    if not set(managed).issubset(existing):
        raise InstallError("安装目录含未知或缺失文件，已拒绝覆盖；所有文件保留。")
    for relative, identity in managed.items():
        if _identity(existing[relative]) != identity:
            raise InstallError("已安装文件被更改，已拒绝覆盖：" + relative)
    preserved = _preserved_caches(existing, managed, new, manifest)
    allowed_directories = {parent.as_posix() for relative in (*managed, *preserved)
                           for parent in PurePosixPath(relative).parents if parent.as_posix() != "."}
    if not directories.issubset(allowed_directories):
        raise InstallError("安装目录含未知目录，已拒绝覆盖；所有文件和目录保留。")
    return managed, preserved


def _running_executables() -> list[Path]:
    if os.name != "nt":
        raise InstallError("本机安装只支持 Windows；无法确认程序运行状态。")
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise InstallError("无法检查程序运行状态，已停止安装。")
    result = []
    try:
        entry = Entry(dwSize=ctypes.sizeof(Entry))
        has_entry = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        if not has_entry:
            raise InstallError("无法读取程序运行状态，已停止安装。")
        while has_entry:
            if entry.szExeFile.casefold() in {APPLICATION.casefold(), LEGACY_APPLICATION.casefold()}:
                handle = kernel.OpenProcess(0x1000, False, entry.th32ProcessID)
                if not handle:
                    raise InstallError("无法确认字幕程序的位置，请关闭字幕程序后安装。")
                try:
                    size = wintypes.DWORD(32768)
                    buffer = ctypes.create_unicode_buffer(size.value)
                    if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                        raise InstallError("无法确认字幕程序的位置，请关闭字幕程序后安装。")
                    result.append(Path(buffer.value))
                finally:
                    kernel.CloseHandle(handle)
            has_entry = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() not in (0, 18):
            raise InstallError("程序运行状态检查未完成，已停止安装。")
    finally:
        kernel.CloseHandle(snapshot)
    return result


def _check_available(destination: Path, existing: dict[str, Path]) -> None:
    expected = {_path_key(destination / name) for name in (APPLICATION, LEGACY_APPLICATION)}
    if any(_path_key(path) in expected for path in _running_executables()):
        raise InstallError("固定目录中的字幕程序正在运行，请自行关闭后再安装；旧文件未修改。")
    for path in existing.values():
        if path.stat().st_nlink > 1:
            raise InstallError("安装文件存在 hardlink，已拒绝覆盖：" + str(path))
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handles = []
    try:
        for path in existing.values():
            handle = kernel.CreateFileW(str(path), 0xC0000000, 0, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise InstallError("安装文件被占用或不可写，请关闭字幕程序后再安装；旧文件未修改：" + str(path))
            handles.append(handle)
    finally:
        for handle in handles:
            kernel.CloseHandle(handle)


def install_local(source: Path, destination: Path | None = None, *, backup_root: Path | None = None) -> dict:
    source = _plain_path(source)
    default_install = destination is None
    destination = _plain_path(destination or ROOT / DISPLAY_NAME)
    backup_root = _plain_path(backup_root or ROOT / "build" / "local-install-backups")
    for first, second in ((source, destination), (source, backup_root), (destination, backup_root)):
        _disjoint(first, second)
    files = _source_files(source)
    legacy = None
    if default_install:
        install_root = _plain_path(ROOT)
        legacy_candidate = _plain_path(install_root / LEGACY_APP_ID)
        # Automatic migration is restricted to the two fixed, sibling folders.
        # Explicit destinations never inspect or relocate an unrelated install.
        if destination.parent != install_root or legacy_candidate.parent != install_root:
            raise InstallError("名称迁移目录必须位于项目根目录，已停止安装。")
        if legacy_candidate.exists():
            if destination.exists():
                raise InstallError("新旧固定安装目录同时存在，已拒绝自动迁移；所有文件保留。")
            _disjoint(source, legacy_candidate)
            _disjoint(backup_root, legacy_candidate)
            legacy = legacy_candidate
    installed = legacy or destination
    directories: set[str] = set()
    existing = _files(installed, directories=directories)
    _check_available(installed, existing)
    if legacy is not None:
        _check_available(destination, {})
    managed, preserved = _managed_files(installed, existing, files, directories)
    if legacy is not None and not managed:
        raise InstallError("旧固定目录缺少有效安装管理清单，已拒绝迁移。")
    identities = {relative: _identity(path) for relative, path in files.items()}
    backup = None
    if managed:
        backup = backup_root / datetime.now().strftime("install-%Y%m%d-%H%M%S-%f")
        for relative, path in existing.items():
            target = backup / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    try:
        if legacy is not None:
            if destination.exists():
                raise InstallError("新安装目录在迁移前已出现，已拒绝覆盖。")
            legacy.rename(destination)
        destination.mkdir(parents=False, exist_ok=True)
        for relative, path in files.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if _identity(target) != identities[relative]:
                raise InstallError("复制后文件校验失败：" + relative)
        # Only unchanged, manifest-owned files may be retired during an upgrade.
        # New files are verified first; the complete previous installation is backed up.
        for relative in sorted(set(managed) - set(files)):
            target = destination / relative
            if _identity(target) != managed[relative]:
                raise InstallError("旧管理文件在更新期间被更改，已拒绝删除：" + relative)
            target.unlink()
        for relative in sorted(directories, key=lambda value: len(value.split("/")), reverse=True):
            folder = destination / relative
            if not any(folder.iterdir()):
                folder.rmdir()
        manifest = {"schema_version": 1, "application": APP_ID, "display_name": DISPLAY_NAME,
                    "files": identities}
        if preserved:
            manifest["preserved_python_caches"] = preserved
        pending_manifest = destination / (MANIFEST + ".pending")
        with pending_manifest.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending_manifest, destination / MANIFEST)
    except (OSError, InstallError) as exc:
        message = "复制安装失败，未完成安装；目标或管理清单可能部分更新。"
        if backup is not None:
            message += "旧版完整备份保留在：" + str(backup)
        raise InstallError(message) from exc
    return {"portable_folder": str(destination), "build_folder": str(source), "installed_locally": True,
            "installed_folder": str(destination), "source_folder": str(source),
            "manifest_path": str(destination / MANIFEST), "manifest_sha256": _identity(destination / MANIFEST)["sha256"],
            "exe_sha256": identities[APPLICATION]["sha256"],
            "backup_folder": str(backup) if backup is not None else None, "managed_files": len(files),
            "migrated_from": str(legacy) if legacy is not None else None,
            "preserved_python_cache_files": sorted(preserved)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="已验证的程序包")
    args = parser.parse_args()
    print(json.dumps(install_local(args.source), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
