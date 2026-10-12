"""Installer checks use synthetic packages and temporary directories only."""
import ctypes
import importlib.util
import json
import os
import py_compile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location("test_tool_" + name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load_tool("install_local")
builder = load_tool("build")
pytestmark = pytest.mark.skipif(os.name != "nt", reason="Local installer and exclusive file probes target Windows")


@pytest.fixture(autouse=True)
def no_live_process_inventory(monkeypatch):
    monkeypatch.setattr(installer, "_running_executables", lambda: [])


def package(path, version="v1"):
    path.mkdir()
    (path / "build-info.json").write_text(json.dumps({"application": "FanyiJi", "display_name": "翻译姬", "schema_version": 1,
                                                       "models_bundled": False}), encoding="utf-8")
    (path / installer.APPLICATION).write_bytes(("synthetic-exe-" + version).encode())
    (path / "_internal").mkdir()
    (path / "_internal/library.dll").write_bytes(("synthetic-library-" + version).encode())
    return path


def install(tmp_path, source):
    return installer.install_local(source, tmp_path / "翻译姬", backup_root=tmp_path / "backups")


def snapshot(folder):
    return {file.relative_to(folder).as_posix(): file.read_bytes() for file in folder.rglob("*") if file.is_file()}


def test_first_install_has_fixed_path_and_managed_hashes(tmp_path):
    source = package(tmp_path / "source")
    result = install(tmp_path, source)
    destination = tmp_path / "翻译姬"
    assert result["portable_folder"] == result["installed_folder"] == str(destination)
    assert result["build_folder"] == result["source_folder"] == str(source)
    assert result["installed_locally"] is True
    assert result["backup_folder"] is None
    assert (destination / "build-info.json").is_file()
    manifest = json.loads((destination / installer.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["application"] == "FanyiJi"
    assert manifest["display_name"] == "翻译姬"
    assert manifest["files"][installer.APPLICATION] == installer._identity(source / installer.APPLICATION)
    assert result["manifest_sha256"] == installer._identity(destination / installer.MANIFEST)["sha256"]
    assert not (destination / (installer.MANIFEST + ".pending")).exists()


@pytest.mark.parametrize("bundled", [True, "false", 0, None])
def test_invalid_build_marker_cannot_install(tmp_path, bundled):
    source = package(tmp_path / "source")
    marker = source / "build-info.json"
    marker.write_text(json.dumps({"application": "FanyiJi", "display_name": "翻译姬", "schema_version": 1,
                                 "models_bundled": bundled}), encoding="utf-8")
    with pytest.raises(installer.InstallError, match="构建信息无效"):
        install(tmp_path, source)
    assert not (tmp_path / "翻译姬").exists()


def test_missing_executable_cannot_install(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "build-info.json").write_text('{"application":"FanyiJi","display_name":"翻译姬","schema_version":1,"models_bundled":false}', encoding="utf-8")
    with pytest.raises(installer.InstallError, match="完整 EXE"):
        install(tmp_path, source)


@pytest.mark.parametrize("relative", ["models/weights.bin", "runtime/model.onnx", "model.bin",
                                     "auth.json", "credentials.json", "user data/Preferences", "node_modules/example/index.js",
                                     "danmaku-wordlist.json", "danmaku-content-filter.json",
                                     "local-install-manifest.json", "local-install-manifest.json.pending"])
def test_source_models_credentials_and_tests_are_rejected_before_target_creation(tmp_path, relative):
    source = package(tmp_path / "source")
    file = source / relative
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"synthetic-forbidden-data")
    with pytest.raises(installer.InstallError):
        install(tmp_path, source)
    assert not (tmp_path / "翻译姬").exists()


def test_empty_forbidden_source_directory_is_rejected(tmp_path):
    source = package(tmp_path / "source")
    (source / "node_modules").mkdir()
    with pytest.raises(installer.InstallError, match="开发依赖"):
        install(tmp_path, source)


@pytest.mark.parametrize("mode", ["same", "destination_nested", "source_nested"])
def test_source_and_destination_cannot_alias_or_nest(tmp_path, mode):
    source = package(tmp_path / "source")
    destination = source if mode == "same" else source / "nested" if mode == "destination_nested" else tmp_path
    with pytest.raises(installer.InstallError, match="相互嵌套"):
        installer.install_local(source, destination, backup_root=tmp_path / "backups")
    assert not (source / "nested").exists()


def test_symlink_source_is_rejected_before_resolving_alias(tmp_path):
    source = package(tmp_path / "source")
    link = tmp_path / "link"
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("Creating a symlink is unavailable on this test host")
    with pytest.raises(installer.InstallError, match="symlink/junction"):
        install(tmp_path, link)


def test_reparse_ancestor_is_rejected_before_target_creation(tmp_path, monkeypatch):
    source = package(tmp_path / "source")
    unsafe = tmp_path / "reparse-parent"
    unsafe.mkdir()
    native_lstat = Path.lstat
    def flagged(path):
        info = native_lstat(path)
        return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400) if path == unsafe else info
    monkeypatch.setattr(Path, "lstat", flagged)
    with pytest.raises(installer.InstallError, match="symlink/junction"):
        installer.install_local(source, unsafe / "destination", backup_root=tmp_path / "backups")
    assert not (unsafe / "destination").exists()


def test_managed_upgrade_copies_backup_and_keeps_fixed_path(tmp_path):
    install(tmp_path, package(tmp_path / "source-v1"))
    destination = tmp_path / "翻译姬"
    old = snapshot(destination)
    source = package(tmp_path / "source-v2", version="v2")
    result = install(tmp_path, source)
    assert snapshot(Path(result["backup_folder"])) == old
    assert (destination / installer.APPLICATION).read_bytes() == (source / installer.APPLICATION).read_bytes()
    assert result["installed_folder"] == str(destination)
    assert (destination / "build-info.json").is_file()


def test_running_target_is_rejected_before_any_backup_or_overwrite(tmp_path, monkeypatch):
    install(tmp_path, package(tmp_path / "source-v1"))
    destination = tmp_path / "翻译姬"
    old = snapshot(destination)
    monkeypatch.setattr(installer, "_running_executables", lambda: [destination / installer.APPLICATION])
    with pytest.raises(installer.InstallError, match="正在运行"):
        install(tmp_path, package(tmp_path / "source-v2", version="v2"))
    assert snapshot(destination) == old
    assert not (tmp_path / "backups").exists()


@pytest.mark.parametrize("relative", [installer.APPLICATION, installer.MANIFEST, "_internal/library.dll"])
def test_native_exclusive_lock_rejects_update_without_changing_any_file(tmp_path, relative):
    from ctypes import wintypes
    install(tmp_path, package(tmp_path / "source-v1"))
    destination = tmp_path / "翻译姬"
    old = snapshot(destination)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(destination / relative), 0x80000000, 0, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    try:
        with pytest.raises(installer.InstallError, match="被占用"):
            install(tmp_path, package(tmp_path / "source-v2", version="v2"))
    finally:
        kernel.CloseHandle(handle)
    assert snapshot(destination) == old
    assert not (tmp_path / "backups").exists()


@pytest.mark.parametrize("relative", [installer.MANIFEST, "_internal/library.dll"])
def test_hardlinked_target_cannot_modify_an_external_file(tmp_path, relative):
    install(tmp_path, package(tmp_path / "source-v1"))
    destination = tmp_path / "翻译姬"
    old = snapshot(destination)
    outside = tmp_path / "external-file"
    os.link(destination / relative, outside)
    with pytest.raises(installer.InstallError, match="hardlink"):
        install(tmp_path, package(tmp_path / "source-v2", version="v2"))
    assert snapshot(destination) == old
    assert outside.read_bytes() == old[relative]


@pytest.mark.parametrize("unknown_directory", [False, True])
def test_unknown_files_or_empty_directories_abort_upgrade_before_overwrite(tmp_path, unknown_directory):
    install(tmp_path, package(tmp_path / "source-v1"))
    destination = tmp_path / "翻译姬"
    source = package(tmp_path / "source-v2", version="v2")
    if unknown_directory:
        (destination / "new-module.dll").mkdir()
        (source / "new-module.dll").write_bytes(b"new-module")
    else:
        (destination / "unknown.dll").write_bytes(b"keep-unknown")
    old = snapshot(destination)
    with pytest.raises(installer.InstallError, match="未知"):
        install(tmp_path, source)
    assert snapshot(destination) == old
    assert not (tmp_path / "backups").exists()


def test_existing_unmanaged_folder_is_never_overwritten(tmp_path):
    destination = tmp_path / "翻译姬"
    destination.mkdir()
    (destination / "user-file.txt").write_bytes(b"preserve")
    with pytest.raises(installer.InstallError, match="没有管理清单"):
        install(tmp_path, package(tmp_path / "source"))
    assert (destination / "user-file.txt").read_bytes() == b"preserve"


def test_manual_edits_are_not_overwritten(tmp_path):
    install(tmp_path, package(tmp_path / "source-v1"))
    source = package(tmp_path / "source-v2", version="v2")
    destination = tmp_path / "翻译姬"
    manifest = json.loads((destination / installer.MANIFEST).read_text(encoding="utf-8"))
    (destination / "_internal/library.dll").write_bytes(b"manually-modified")
    with pytest.raises(installer.InstallError, match="被更改"):
        install(tmp_path, source)
    assert json.loads((destination / installer.MANIFEST).read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize("relative", ["../outside.dll", "/absolute.dll", "C:/outside.dll", "folder/../escape", "folder\\escape", "CON.txt", "name. "])
def test_manifest_paths_are_strictly_relative_and_unambiguous(relative):
    with pytest.raises(installer.InstallError):
        installer._relative(relative)


def test_atomic_manifest_failure_preserves_old_manifest_and_reports_backup(tmp_path, monkeypatch):
    install(tmp_path, package(tmp_path / "source-v1"))
    destination = tmp_path / "翻译姬"
    old_manifest = (destination / installer.MANIFEST).read_bytes()
    def fail_replace(*args):
        raise OSError("synthetic atomic-rename failure")
    monkeypatch.setattr(installer.os, "replace", fail_replace)
    with pytest.raises(installer.InstallError, match="完整备份"):
        install(tmp_path, package(tmp_path / "source-v2", version="v2"))
    assert (destination / installer.MANIFEST).read_bytes() == old_manifest
    assert (destination / (installer.MANIFEST + ".pending")).exists()
    backup = next((tmp_path / "backups").iterdir())
    assert (backup / installer.MANIFEST).read_bytes() == old_manifest


def test_local_staging_uses_one_fixed_build_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    parent, staged = builder._build_output(True)
    assert parent == tmp_path / "build/portable"
    assert staged == parent / "翻译姬"
    assert builder._build_output(False)[0] == tmp_path / "dist"
    (tmp_path / "dist/翻译姬").mkdir(parents=True)
    assert builder._build_output(False)[0] == tmp_path / "dist"


def test_upgrade_removes_unchanged_retired_files_and_empty_directories(tmp_path):
    source = package(tmp_path / "source-v1")
    retired = source / "retired_feature" / "nested"
    retired.mkdir(parents=True)
    (retired / "feature.js").write_bytes(b"synthetic-retired-code")
    install(tmp_path, source)
    destination = tmp_path / "翻译姬"
    old = snapshot(destination)
    result = install(tmp_path, package(tmp_path / "source-v2", version="v2"))
    assert snapshot(Path(result["backup_folder"])) == old
    assert not (destination / "retired_feature").exists()
    assert snapshot(destination).keys() == {installer.MANIFEST, installer.APPLICATION,
                                           "build-info.json", "_internal/library.dll"}


def test_legacy_manifest_can_upgrade_without_feature_permission(tmp_path):
    install(tmp_path, package(tmp_path / "source-v1"))
    destination = tmp_path / "翻译姬"
    path = destination / installer.MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("application")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = install(tmp_path, package(tmp_path / "source-v2", version="v2"))
    assert result["installed_locally"] is True
    assert json.loads(path.read_text(encoding="utf-8"))["application"] == "FanyiJi"


def test_changed_retired_file_prevents_deletion(tmp_path):
    source = package(tmp_path / "source-v1")
    (source / "retired.dll").write_bytes(b"retired-original")
    install(tmp_path, source)
    destination = tmp_path / "翻译姬"
    (destination / "retired.dll").write_bytes(b"preserve-user-edit")
    with pytest.raises(installer.InstallError, match="被更改"):
        install(tmp_path, package(tmp_path / "source-v2", version="v2"))
    assert (destination / "retired.dll").read_bytes() == b"preserve-user-edit"
    assert not (tmp_path / "backups").exists()


def legacy_install(path):
    """Create an old managed install without invoking the new source validator."""
    path.mkdir()
    (path / installer.LEGACY_APPLICATION).write_bytes(b"synthetic-legacy-exe")
    (path / "build-info.json").write_text(json.dumps({"application": installer.LEGACY_APP_ID,
                                                    "schema_version": 1, "models_bundled": False}),
                                         encoding="utf-8")
    (path / "_internal").mkdir()
    (path / "_internal/library.dll").write_bytes(b"synthetic-legacy-library")
    (path / "retired-build-marker.json").write_bytes(b"synthetic-legacy-marker")
    files = installer._files(path)
    (path / installer.MANIFEST).write_text(json.dumps({
        "schema_version": 1, "application": installer.LEGACY_APP_ID,
        "files": {name: installer._identity(file) for name, file in files.items()},
    }), encoding="utf-8")
    return path


def test_default_first_install_uses_new_brand_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    result = installer.install_local(package(tmp_path / "source"), backup_root=tmp_path / "backups")
    assert result["installed_folder"] == str(tmp_path / "翻译姬")
    assert result["migrated_from"] is None
    assert not (tmp_path / installer.LEGACY_APP_ID).exists()


def test_default_install_migrates_verified_legacy_identity_with_complete_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    legacy = legacy_install(tmp_path / installer.LEGACY_APP_ID)
    old = snapshot(legacy)
    source = package(tmp_path / "source")
    result = installer.install_local(source, backup_root=tmp_path / "backups")
    destination = tmp_path / "翻译姬"
    assert result["migrated_from"] == str(legacy)
    assert result["installed_folder"] == str(destination)
    assert snapshot(Path(result["backup_folder"])) == old
    assert not legacy.exists()
    assert not (destination / installer.LEGACY_APPLICATION).exists()
    assert not (destination / "retired-build-marker.json").exists()
    assert snapshot(destination).keys() == snapshot(source).keys() | {installer.MANIFEST}
    manifest = json.loads((destination / installer.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["application"] == "FanyiJi"
    assert manifest["display_name"] == "翻译姬"
    assert (destination / installer.APPLICATION).read_bytes() == (source / installer.APPLICATION).read_bytes()


def test_explicit_legacy_destination_upgrades_in_place_without_automatic_relocation(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    legacy = legacy_install(tmp_path / installer.LEGACY_APP_ID)
    result = installer.install_local(package(tmp_path / "source"), legacy, backup_root=tmp_path / "backups")
    assert result["installed_folder"] == str(legacy)
    assert result["migrated_from"] is None
    assert legacy.exists() and (legacy / installer.APPLICATION).exists()
    assert not (legacy / installer.LEGACY_APPLICATION).exists()
    assert not (tmp_path / "翻译姬").exists()


def test_explicit_destination_never_scans_or_changes_neighboring_legacy_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    legacy = tmp_path / installer.LEGACY_APP_ID
    legacy.mkdir()
    (legacy / "user-data.txt").write_bytes(b"unmanaged-preserve")
    old = snapshot(legacy)
    result = installer.install_local(package(tmp_path / "source"), tmp_path / "custom-install",
                                     backup_root=tmp_path / "backups")
    assert result["migrated_from"] is None
    assert snapshot(legacy) == old


@pytest.mark.parametrize("new_managed", [False, True])
def test_default_install_refuses_conflicting_old_and_new_folders(tmp_path, monkeypatch, new_managed):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    legacy = legacy_install(tmp_path / installer.LEGACY_APP_ID)
    destination = tmp_path / "翻译姬"
    if new_managed:
        installer.install_local(package(tmp_path / "prior-source"), destination,
                                backup_root=tmp_path / "backups")
    else:
        destination.mkdir()
        (destination / "user-data.txt").write_bytes(b"preserve-new")
    old, new = snapshot(legacy), snapshot(destination)
    with pytest.raises(installer.InstallError, match="同时存在"):
        installer.install_local(package(tmp_path / "source"), backup_root=tmp_path / "backups")
    assert snapshot(legacy) == old
    assert snapshot(destination) == new
    assert not (tmp_path / "backups").exists()


@pytest.mark.parametrize("fault", ["changed", "unknown_file", "unknown_directory", "missing_manifest", "wrong_identity"])
def test_legacy_validation_failure_never_moves_or_overwrites_any_files(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    legacy = legacy_install(tmp_path / installer.LEGACY_APP_ID)
    if fault == "changed":
        (legacy / installer.LEGACY_APPLICATION).write_bytes(b"preserve-user-edit")
    elif fault == "unknown_file":
        (legacy / "user-data.txt").write_bytes(b"preserve-user-data")
    elif fault == "unknown_directory":
        (legacy / "unmanaged-empty").mkdir()
    elif fault == "missing_manifest":
        (legacy / installer.MANIFEST).unlink()
    else:
        manifest_path = legacy / installer.MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["application"] = "UnrelatedApplication"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    old = snapshot(legacy)
    with pytest.raises(installer.InstallError):
        installer.install_local(package(tmp_path / "source"), backup_root=tmp_path / "backups")
    assert snapshot(legacy) == old
    assert not (tmp_path / "翻译姬").exists()
    assert not (tmp_path / "backups").exists()


@pytest.mark.parametrize("executable", [installer.LEGACY_APPLICATION, installer.APPLICATION])
def test_running_legacy_or_new_named_executable_prevents_migration(tmp_path, monkeypatch, executable):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    legacy = legacy_install(tmp_path / installer.LEGACY_APP_ID)
    old = snapshot(legacy)
    monkeypatch.setattr(installer, "_running_executables", lambda: [legacy / executable])
    with pytest.raises(installer.InstallError, match="正在运行"):
        installer.install_local(package(tmp_path / "source"), backup_root=tmp_path / "backups")
    assert snapshot(legacy) == old
    assert not (tmp_path / "翻译姬").exists()
    assert not (tmp_path / "backups").exists()


def test_legacy_hardlink_is_rejected_before_migration(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    legacy = legacy_install(tmp_path / installer.LEGACY_APP_ID)
    outside = tmp_path / "external-library"
    os.link(legacy / "_internal/library.dll", outside)
    old = snapshot(legacy)
    with pytest.raises(installer.InstallError, match="hardlink"):
        installer.install_local(package(tmp_path / "source"), backup_root=tmp_path / "backups")
    assert snapshot(legacy) == old
    assert outside.read_bytes() == old["_internal/library.dll"]
    assert not (tmp_path / "翻译姬").exists()
    assert not (tmp_path / "backups").exists()


@pytest.mark.parametrize("relative", [installer.LEGACY_APPLICATION, installer.MANIFEST])
def test_native_legacy_file_lock_is_rejected_before_migration(tmp_path, monkeypatch, relative):
    from ctypes import wintypes
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    legacy = legacy_install(tmp_path / installer.LEGACY_APP_ID)
    old = snapshot(legacy)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(legacy / relative), 0x80000000, 0, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    try:
        with pytest.raises(installer.InstallError, match="被占用"):
            installer.install_local(package(tmp_path / "source"), backup_root=tmp_path / "backups")
    finally:
        kernel.CloseHandle(handle)
    assert snapshot(legacy) == old
    assert not (tmp_path / "翻译姬").exists()
    assert not (tmp_path / "backups").exists()


def test_legacy_reparse_directory_is_rejected_before_migration(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    legacy = legacy_install(tmp_path / installer.LEGACY_APP_ID)
    old = snapshot(legacy)
    native_lstat = Path.lstat
    def flagged(path):
        info = native_lstat(path)
        return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400) if path == legacy else info
    monkeypatch.setattr(Path, "lstat", flagged)
    with pytest.raises(installer.InstallError, match="symlink/junction"):
        installer.install_local(package(tmp_path / "source"), backup_root=tmp_path / "backups")
    assert snapshot(legacy) == old
    assert not (tmp_path / "翻译姬").exists()
    assert not (tmp_path / "backups").exists()


def test_default_legacy_source_overlap_is_rejected_before_any_move(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    legacy = legacy_install(tmp_path / installer.LEGACY_APP_ID)
    source = package(legacy / "source")
    old = snapshot(legacy)
    with pytest.raises(installer.InstallError, match="相互嵌套"):
        installer.install_local(source, backup_root=tmp_path / "backups")
    assert snapshot(legacy) == old
    assert not (tmp_path / "翻译姬").exists()


def test_legacy_executable_cannot_be_bundled_in_new_package(tmp_path):
    source = package(tmp_path / "source")
    (source / installer.LEGACY_APPLICATION).write_bytes(b"unexpected-legacy-exe")
    with pytest.raises(installer.InstallError, match="旧名称 EXE"):
        install(tmp_path, source)
    assert not (tmp_path / "翻译姬").exists()


def training_package(path, text):
    source = package(path)
    folder = source / "training_tools"
    folder.mkdir()
    (folder / "livestream_prompt.py").write_text(text, encoding="utf-8")
    return source


def training_cache(destination):
    source = destination / "training_tools/livestream_prompt.py"
    return Path(py_compile.compile(str(source), doraise=True))


def test_verified_training_cache_is_backed_up_preserved_and_recorded(tmp_path):
    source = training_package(tmp_path / "source-v1", "VALUE = 'old'\n")
    install(tmp_path, source)
    destination = tmp_path / "翻译姬"
    cache = training_cache(destination)
    cached = cache.read_bytes()
    old = snapshot(destination)
    result = install(tmp_path, training_package(tmp_path / "source-v2", "VALUE = 'a new source value'\n"))
    assert cache.read_bytes() == cached
    assert snapshot(Path(result["backup_folder"])) == old
    relative = cache.relative_to(destination).as_posix()
    assert result["preserved_python_cache_files"] == [relative]
    manifest = json.loads((destination / installer.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["preserved_python_caches"][relative] == installer._identity(cache)
    assert relative not in manifest["files"]
    assert not list((tmp_path / "source-v2").rglob("*.pyc"))
    # The original artifact remains safe through subsequent source upgrades.
    result = install(tmp_path, training_package(tmp_path / "source-v3", "VALUE = 'another newer source'\n"))
    assert result["preserved_python_cache_files"] == [relative]
    assert cache.read_bytes() == cached


@pytest.mark.parametrize("mode", ["corrupt", "foreign-code", "unchecked-hash"])
def test_unverified_training_cache_aborts_before_overwrite(tmp_path, mode):
    install(tmp_path, training_package(tmp_path / "source-v1", "VALUE = 'old'\n"))
    destination = tmp_path / "翻译姬"
    cache = training_cache(destination)
    if mode == "corrupt":
        cache.write_bytes(b"not a python cache")
    elif mode == "foreign-code":
        foreign = tmp_path / "foreign.py"
        foreign.write_text("VALUE = 'foreign'\n", encoding="utf-8")
        py_compile.compile(str(foreign), cfile=str(cache), doraise=True)
    else:
        raw = bytearray(cache.read_bytes())
        raw[4:8] = (1).to_bytes(4, "little")
        cache.write_bytes(raw)
    old = snapshot(destination)
    with pytest.raises(installer.InstallError, match="未验证"):
        install(tmp_path, training_package(tmp_path / "source-v2", "VALUE = 'a new source value'\n"))
    assert snapshot(destination) == old
    assert not (tmp_path / "backups").exists()


def test_cache_cannot_shadow_different_source_with_same_timestamp_and_size(tmp_path):
    install(tmp_path, training_package(tmp_path / "source-v1", "VALUE = 'old'\n"))
    destination = tmp_path / "翻译姬"
    training_cache(destination)
    source = training_package(tmp_path / "source-v2", "VALUE = 'new'\n")
    old_source = destination / "training_tools/livestream_prompt.py"
    os.utime(source / "training_tools/livestream_prompt.py", (old_source.stat().st_atime, old_source.stat().st_mtime))
    old = snapshot(destination)
    with pytest.raises(installer.InstallError, match="未验证"):
        install(tmp_path, source)
    assert snapshot(destination) == old


def test_unrelated_cache_name_does_not_bypass_unknown_file_check(tmp_path):
    install(tmp_path, training_package(tmp_path / "source-v1", "VALUE = 'old'\n"))
    destination = tmp_path / "翻译姬"
    cache = training_cache(destination)
    (cache.parent / "unknown.cpython-312.pyc").write_bytes(cache.read_bytes())
    old = snapshot(destination)
    with pytest.raises(installer.InstallError, match="未知"):
        install(tmp_path, training_package(tmp_path / "source-v2", "VALUE = 'a new source value'\n"))
    assert snapshot(destination) == old


def test_incoming_cache_cannot_overwrite_preserved_import_cache(tmp_path):
    install(tmp_path, training_package(tmp_path / "source-v1", "VALUE = 'old'\n"))
    destination = tmp_path / "翻译姬"
    cache = training_cache(destination)
    source = training_package(tmp_path / "source-v2", "VALUE = 'a new source value'\n")
    incoming = source / cache.relative_to(destination)
    incoming.parent.mkdir()
    incoming.write_bytes(cache.read_bytes())
    old = snapshot(destination)
    with pytest.raises(installer.InstallError, match="开发依赖"):
        install(tmp_path, source)
    assert snapshot(destination) == old
    assert not (tmp_path / "backups").exists()


def cleanable_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    (tmp_path / "dist").mkdir()
    old = package(tmp_path / "dist/翻译姬")
    installer.install_local(old)
    (tmp_path / "build/portable").mkdir(parents=True)
    new = package(tmp_path / "build/portable/翻译姬", version="v2")
    result = installer.install_local(new)
    return old, new, Path(result["backup_folder"])


def test_verified_local_cleanup_keeps_only_current_install_and_retains_user_data(tmp_path, monkeypatch):
    old, new, backup = cleanable_versions(tmp_path, monkeypatch)
    destination = tmp_path / "翻译姬"
    expected = snapshot(destination)
    for relative in ("models/asr/model.onnx", "model-cache/model.gguf", "browser/default/Cookies",
                     "artifacts/diagnostic.log", "src/fanyiji/example.py", "settings.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve-user-data")
    result = installer.cleanup_local_versions()
    assert set(result["removed_folders"]) == {str(old), str(new), str(backup)}
    assert result["removed_bytes"] > 0 and result["skipped"] == []
    assert snapshot(destination) == expected
    assert installer.verify_local_install()["exe_sha256"] == installer._identity(destination / installer.APPLICATION)["sha256"]
    for relative in ("models/asr/model.onnx", "model-cache/model.gguf", "browser/default/Cookies",
                     "artifacts/diagnostic.log", "src/fanyiji/example.py", "settings.json"):
        assert (tmp_path / relative).read_bytes() == b"preserve-user-data"


@pytest.mark.parametrize("mode", ["changed", "unknown-file", "unknown-directory", "missing-manifest", "settings.json", "diagnostic.log", "browser/default/Preferences"])
def test_cleanup_preserves_unverified_backup(tmp_path, monkeypatch, mode):
    _, _, backup = cleanable_versions(tmp_path, monkeypatch)
    if mode == "changed":
        (backup / "_internal/library.dll").write_bytes(b"user-modified")
    elif mode == "unknown-file":
        (backup / "user-notes.txt").write_bytes(b"preserve")
    elif mode == "unknown-directory":
        (backup / "user-empty-folder").mkdir()
    elif mode == "missing-manifest":
        (backup / installer.MANIFEST).unlink()
    else:
        private = backup / mode
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_bytes(b"preserve")
        manifest_path = backup / installer.MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][mode] = installer._identity(private)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = snapshot(backup)
    result = installer.cleanup_local_versions()
    assert snapshot(backup) == before
    assert str(backup) not in result["removed_folders"]
    assert any(item["folder"] == str(backup) for item in result["skipped"])


@pytest.mark.parametrize("mode", ["changed", "unknown-file", "unknown-directory", "model"])
def test_cleanup_preserves_unverified_fixed_package(tmp_path, monkeypatch, mode):
    _, new, _ = cleanable_versions(tmp_path, monkeypatch)
    if mode == "changed":
        (new / "_internal/library.dll").write_bytes(b"modified")
    elif mode == "unknown-file":
        (new / "user-notes.txt").write_bytes(b"preserve")
    elif mode == "unknown-directory":
        (new / "user-empty-folder").mkdir()
    else:
        (new / "model.onnx").write_bytes(b"preserve-model")
    before = snapshot(new)
    result = installer.cleanup_local_versions()
    assert snapshot(new) == before
    assert str(new) not in result["removed_folders"]
    assert any(item["folder"] == str(new) for item in result["skipped"])


def test_cleanup_never_scans_unknown_backup_names_or_other_build_folders(tmp_path, monkeypatch):
    cleanable_versions(tmp_path, monkeypatch)
    unknown = tmp_path / "build/local-install-backups/manual-archive"
    unknown.mkdir()
    (unknown / "user-notes.txt").write_bytes(b"preserve")
    other = package(tmp_path / "dist/other-build")
    expected = snapshot(other)
    result = installer.cleanup_local_versions()
    assert (unknown / "user-notes.txt").read_bytes() == b"preserve"
    assert snapshot(other) == expected
    assert any(item["folder"] == str(unknown) for item in result["skipped"])


def test_cleanup_rejects_nonfixed_destination_without_touching_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    with pytest.raises(installer.InstallError, match="固定安装目录"):
        installer.cleanup_local_versions(tmp_path / "other")


def test_cleanup_validation_failure_keeps_all_old_versions(tmp_path, monkeypatch):
    old, new, backup = cleanable_versions(tmp_path, monkeypatch)
    (tmp_path / "翻译姬/_internal/library.dll").write_bytes(b"modified-current")
    before = [snapshot(folder) for folder in (old, new, backup)]
    with pytest.raises(installer.InstallError, match="被更改"):
        installer.cleanup_local_versions()
    assert [snapshot(folder) for folder in (old, new, backup)] == before


def test_cleanup_rechecks_contents_before_recursive_deletion(tmp_path, monkeypatch):
    old, new, backup = cleanable_versions(tmp_path, monkeypatch)
    native_disjoint = installer._disjoint
    def mutate_before_delete(first, second):
        native_disjoint(first, second)
        if first == backup:
            (first / "user-notes.txt").write_bytes(b"new-user-file")
    monkeypatch.setattr(installer, "_disjoint", mutate_before_delete)
    result = installer.cleanup_local_versions()
    assert (backup / "user-notes.txt").read_bytes() == b"new-user-file"
    assert str(backup) not in result["removed_folders"]


def test_cleanup_refuses_linked_build_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    installer.install_local(package(tmp_path / "source"))
    outside = tmp_path.parent / (tmp_path.name + "-external")
    outside.mkdir()
    source = package(outside / "翻译姬")
    try:
        (tmp_path / "dist").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating a symlink is unavailable on this test host")
    before = snapshot(source)
    result = installer.cleanup_local_versions()
    assert snapshot(source) == before
    assert result["removed_folders"] == []
    assert any("symlink/junction" in item["reason"] for item in result["skipped"])


@pytest.mark.parametrize("response", ["invalid-json", "[]", '{"ui_smoke":"failed"}',
                                     '{"ui_smoke":"passed","audio_capture":true,"settings_written":false}'])
def test_build_post_install_failed_smoke_never_cleans_versions(tmp_path, monkeypatch, response):
    events = []
    verified = SimpleNamespace(verify_local_install=lambda folder: events.append("verify"),
                               cleanup_local_versions=lambda folder: events.append("cleanup"))
    monkeypatch.setattr(builder.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=response))
    with pytest.raises(RuntimeError, match="保留"):
        builder._finish_local_install(tmp_path / "翻译姬", verified)
    assert events == ["verify"]


def test_build_post_install_verifies_and_checks_smoke_before_cleanup(tmp_path, monkeypatch):
    events = []
    verified = SimpleNamespace(verify_local_install=lambda folder: events.append("verify"),
                               cleanup_local_versions=lambda folder: events.append("cleanup") or {"removed_folders": []})
    def smoke(command, **kwargs):
        assert command == [str(tmp_path / "翻译姬/翻译姬.exe"), "--smoke-test"]
        assert kwargs["check"] is True and kwargs["timeout"] == 45
        events.append("smoke")
        return SimpleNamespace(stdout='{"ui_smoke":"passed","audio_capture":false,"settings_written":false}')
    monkeypatch.setattr(builder.subprocess, "run", smoke)
    assert builder._finish_local_install(tmp_path / "翻译姬", verified) == {"removed_folders": []}
    assert events == ["verify", "smoke", "cleanup"]


@pytest.mark.parametrize("failure", [OSError("cannot launch"), builder.subprocess.TimeoutExpired("synthetic", 45),
                                    builder.subprocess.CalledProcessError(1, "synthetic")])
def test_build_post_install_launch_failure_preserves_versions(tmp_path, monkeypatch, failure):
    events = []
    verified = SimpleNamespace(verify_local_install=lambda folder: events.append("verify"),
                               cleanup_local_versions=lambda folder: events.append("cleanup"))
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(builder.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="保留"):
        builder._finish_local_install(tmp_path / "翻译姬", verified)
    assert events == ["verify"]


def test_cleanup_running_backup_is_retained(tmp_path, monkeypatch):
    _, _, backup = cleanable_versions(tmp_path, monkeypatch)
    monkeypatch.setattr(installer, "_running_executables", lambda: [backup / installer.APPLICATION])
    before = snapshot(backup)
    result = installer.cleanup_local_versions()
    assert snapshot(backup) == before
    assert any(item["folder"] == str(backup) and "正在运行" in item["reason"] for item in result["skipped"])
