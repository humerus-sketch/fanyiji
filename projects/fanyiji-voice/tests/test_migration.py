"""Edition migration uses fake accounts, fake LOCALAPPDATA and temporary SQLite."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from fanyiji import migration, paths
from fanyiji.danmaku_store import DanmakuStore
from fanyiji.identity import APP_ID, EDITION, ENV_PREFIX


@pytest.fixture(autouse=True)
def isolated_user_data(monkeypatch, tmp_path):
    # Clear both editions' overrides as well as all legacy launch-time values.
    # All user data accessed by this test must live under pytest's tmp_path.
    for name in tuple(os.environ):
        if name.upper().startswith(("FANYIJI", "LOCAL_SUBTITLES")):
            monkeypatch.delenv(name, raising=False)
    base = tmp_path / "fake-local-app-data"
    base.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(base))
    monkeypatch.setattr(paths, "app_root", lambda: tmp_path / "fake-program")
    assert paths.data_root() == base / APP_ID
    return base


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files_hashes(root):
    # Opening a WAL database read-only can create its empty WAL/shared-memory
    # lock index. These are SQLite bookkeeping, not settings/login/DB payload.
    # The nonempty committed WAL is byte-checked by the dedicated live-WAL test.
    return {path.relative_to(root).as_posix(): _hash(path)
            for path in root.rglob("*") if path.is_file()
            and path.name not in {"danmaku-queue.sqlite3-shm", "danmaku-queue.sqlite3-wal"}}


def _assert_database_released(path):
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # Read only, OPEN_EXISTING, with sharing disabled: an outstanding SQLite
    # reader/writer handle would make this open fail with a sharing violation.
    handle = kernel.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)
    assert handle not in (None, 0, ctypes.c_void_p(-1).value), f"database_handle_retained:{ctypes.get_last_error()}"
    assert kernel.CloseHandle(handle)


def _legacy_fixture(base, *, with_queue=True):
    source = base / "LocalJaZhSubtitles"
    wrong_mode = "chrome_caption" if EDITION == "voice" else "standard"
    _write_json(source / "settings.json", {"processing_mode": wrong_mode,
                "translation_provider": "codex", "ui_language": "ja", "fixture_setting": "preserved"})
    _write_json(source / "danmaku-content-filter.json", {"fixture_rule": {"source": "评价", "replacement": "感想"}})
    _write_json(source / "danmaku-wordlist.json", {"fixture_words": ["词条"]})
    _write_json(source / "local-features.json", {"schema_version": 1, "host": "fixture-host", "danmaku": True})
    _write_json(source / "browser-preferences.json", {"room": "1698225", "fixture": "webview"})
    _write_json(source / "live-source-settings.json", {"enabled": True})
    _write_json(source / "chrome-caption-settings.json", {"room_url": "https://live.bilibili.com/21399689",
                "hide_caption": True, "audio_processing": "gtcrn", "push_preferences": {"enabled": False}})
    fixtures = {
        "browser/default/Default/Cookies": b"fake-webview-login",
        "chrome-caption/dedicated-profile/Default/Cookies": b"fake-chrome-login",
        "chrome-caption/dedicated-profile/Local State": b"fake-local-state",
        "specialized_model/collection.json": b'{"enabled":false}',
        "cli-runtime/kimi/fixture.txt": b"fake-cli-runtime",
        "models/audio_processing/gtcrn_simple.onnx": b"fake-external-gtcrn-weights",
        "models/asr/model.int8.onnx": b"fake-external-sensevoice-weights",
    }
    for name, content in fixtures.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for folder in ("browser/default", "chrome-caption/dedicated-profile"):
        for name in ("Cache/data", "Code Cache/data", "SingletonLock", "DevToolsActivePort"):
            path = source / folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture-transient-cache")
    if with_queue:
        _populate_queue(source / "danmaku-queue.sqlite3")
    return source


def _populate_queue(path, *, leave_open=False):
    store = DanmakuStore(path, wall_clock=lambda: 100.0)
    store.stage_failed_handoff("1698225", "【已移交的历史失败句】", reason="platform_f")
    assert store.enqueue("1698225", "system", "pending-fixture", 1,
                         "这是还没有发送的句子", ["【这是还没有发送的句子】"])
    assert store.enqueue("1698225", "system", "failed-fixture", 1,
                         "这是未保存的失败句", ["【这是未保存的失败句】"])
    row = dict(store.connection.execute("SELECT * FROM messages WHERE segment_key='failed-fixture'").fetchone())
    store.update(row["id"], "verify", reason="platform_f", platform_code=0)
    store.save_recovery_context(row["id"], row["part"], {"handoff_pending": True})
    store.increment("accepted")
    store.increment("failures")
    if leave_open:
        return store
    store.close()


def _profile_relative():
    return Path("browser/default") if EDITION == "voice" else Path("chrome-caption/dedicated-profile")


def _assert_runtime_empty(store):
    assert store.head() is None
    assert all(value == 0 for value in store.counts().values())
    for table in ("messages", "contents", "segments", "metrics"):
        assert store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_migration_copies_only_matching_edition_data_and_preserves_original(isolated_user_data):
    source = _legacy_fixture(isolated_user_data)
    before = _files_hashes(source)
    result = migration.ensure_initial_migration()
    target = paths.data_root()
    assert result["migrated"] and result["edition"] == EDITION
    assert result["models_copied"] is False and result["pending_queue_imported"] is False
    settings = json.loads((target / "settings.json").read_text(encoding="utf-8"))
    assert settings["processing_mode"] == ("standard" if EDITION == "voice" else "chrome_caption")
    assert settings["fixture_setting"] == "preserved" and settings["ui_language"] == "ja"
    for name in migration._COMMON:
        assert (target / name).read_bytes() == (source / name).read_bytes()
    relative = _profile_relative()
    assert (target / relative / "Default/Cookies").read_bytes() == (source / relative / "Default/Cookies").read_bytes()
    assert not (target / relative / "Cache").exists()
    assert not (target / relative / "Code Cache").exists()
    assert not (target / relative / "SingletonLock").exists()
    assert not (target / relative / "DevToolsActivePort").exists()
    if EDITION == "voice":
        assert (target / "browser-preferences.json").is_file()
        assert (target / "live-source-settings.json").is_file()
        assert (target / "specialized_model/collection.json").is_file()
        assert (target / "cli-runtime/kimi/fixture.txt").is_file()
        assert not (target / "chrome-caption").exists()
        assert not (target / "chrome-caption-settings.json").exists()
    else:
        assert (target / "chrome-caption-settings.json").is_file()
        assert not (target / "browser").exists()
        assert not (target / "browser-preferences.json").exists()
        assert not (target / "live-source-settings.json").exists()
        assert not (target / "specialized_model").exists()
        assert not (target / "cli-runtime").exists()
    assert _files_hashes(source) == before


def test_weights_are_referenced_in_place_and_never_copied(isolated_user_data):
    source = _legacy_fixture(isolated_user_data, with_queue=False)
    before = _files_hashes(source / "models")
    migration.ensure_initial_migration()
    target = paths.data_root()
    location = json.loads((target / "models-location.json").read_text(encoding="utf-8"))
    assert Path(location["directory"]) == (source / "models").resolve()
    assert paths.models_root() == (source / "models").resolve()
    assert not (target / "models").exists()
    assert _files_hashes(source / "models") == before
    assert not list(target.rglob("*.onnx"))


def test_queue_migration_preserves_handoffs_but_resets_head_and_metrics(isolated_user_data, monkeypatch):
    source = _legacy_fixture(isolated_user_data)
    before = _hash(source / "danmaku-queue.sqlite3")
    calls = []
    original_connect = sqlite3.connect

    def connect(database, *args, **kwargs):
        calls.append((database, dict(kwargs)))
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(migration.sqlite3, "connect", connect)
    migration.ensure_initial_migration()
    assert any(database == (source / "danmaku-queue.sqlite3").as_uri() + "?mode=ro"
               and kwargs.get("uri") is True for database, kwargs in calls)
    store = DanmakuStore(paths.data_root() / "danmaku-queue.sqlite3")
    try:
        _assert_runtime_empty(store)
        failures = store.failed_handoffs()
        assert {row["text"] for row in failures} == {"【已移交的历史失败句】", "【这是未保存的失败句】"}
        assert len({row["queue_id"] for row in failures}) == 2
        assert (paths.data_root() / "queue-before-last-restart.sqlite3").is_file()
    finally:
        store.close()
    assert _hash(source / "danmaku-queue.sqlite3") == before
    _assert_database_released(source / "danmaku-queue.sqlite3")
    _assert_database_released(paths.data_root() / "danmaku-queue.sqlite3")


def test_readonly_sqlite_backup_includes_committed_wal_without_changing_source(isolated_user_data):
    source = _legacy_fixture(isolated_user_data, with_queue=False)
    original = _populate_queue(source / "danmaku-queue.sqlite3", leave_open=True)
    try:
        wal = Path(str(original.path) + "-wal")
        assert wal.is_file() and wal.stat().st_size > 0
        db_hash, wal_hash = _hash(original.path), _hash(wal)
        source_handoffs = original.failed_handoffs()
        migration.ensure_initial_migration()
        migrated = DanmakuStore(paths.data_root() / "danmaku-queue.sqlite3")
        try:
            _assert_runtime_empty(migrated)
            texts = {row["text"] for row in migrated.failed_handoffs()}
            assert texts == {"【已移交的历史失败句】", "【这是未保存的失败句】"}
        finally:
            migrated.close()
        assert original.failed_handoffs() == source_handoffs
        assert original.counts()["queued"] == 2
        assert original.counts()["accepted"] == 1
        assert _hash(original.path) == db_hash
        assert _hash(wal) == wal_hash
    finally:
        original.close()


def test_profiles_and_rules_are_independent_copies(isolated_user_data):
    source = _legacy_fixture(isolated_user_data, with_queue=False)
    before = _files_hashes(source)
    migration.ensure_initial_migration()
    target = paths.data_root()
    cookie = target / _profile_relative() / "Default/Cookies"
    cookie.write_bytes(b"changed-fake-edition-login")
    rule = target / "danmaku-content-filter.json"
    rule.write_text('{"edition_rule":"new-rule"}', encoding="utf-8")
    assert cookie.read_bytes() != (source / _profile_relative() / "Default/Cookies").read_bytes()
    assert rule.read_bytes() != (source / "danmaku-content-filter.json").read_bytes()
    assert _files_hashes(source) == before


def test_repeated_migration_does_not_overwrite_new_preferences_profiles_rules_or_queue(isolated_user_data):
    source = _legacy_fixture(isolated_user_data)
    source_before = _files_hashes(source)
    migration.ensure_initial_migration()
    target = paths.data_root()
    _write_json(target / "settings.json", {"fixture_setting": "new-edition-choice"})
    (target / _profile_relative() / "Default/Cookies").write_bytes(b"new-edition-login")
    _write_json(target / "danmaku-content-filter.json", {"fixture_rule": "new-edition-rule"})
    store = DanmakuStore(target / "danmaku-queue.sqlite3")
    try:
        assert store.enqueue("1698225", "system", "new-session", 1, "这是新版本的等待句", ["【这是新版本的等待句】"])
        store.increment("accepted")
    finally:
        store.close()
    target_before = _files_hashes(target)
    result = migration.ensure_initial_migration()
    assert result == {"migrated": False}
    assert _files_hashes(target) == target_before
    assert _files_hashes(source) == source_before


def test_existing_edition_profiles_rules_and_settings_are_never_overwritten(isolated_user_data):
    source = _legacy_fixture(isolated_user_data, with_queue=False)
    target = paths.data_root()
    _write_json(target / "settings.json", {"fixture_setting": "already-present"})
    _write_json(target / "danmaku-content-filter.json", {"fixture_rule": "already-present"})
    cookie = target / _profile_relative() / "Default/Cookies"
    cookie.parent.mkdir(parents=True)
    cookie.write_bytes(b"already-present-edition-login")
    protected = {path: _hash(path) for path in (target / "settings.json", target / "danmaku-content-filter.json", cookie)}
    before = _files_hashes(source)
    migration.ensure_initial_migration()
    assert all(_hash(path) == digest for path, digest in protected.items())
    assert _files_hashes(source) == before


@pytest.mark.parametrize("override_kind", ["edition", "renamed", "legacy"])
def test_explicit_data_override_does_not_copy_legacy_accounts(isolated_user_data, tmp_path, monkeypatch, override_kind):
    source = _legacy_fixture(isolated_user_data)
    before = _files_hashes(source)
    custom = tmp_path / "explicit-user-data"
    name = {"edition": f"{ENV_PREFIX}_DATA", "renamed": "FANYIJI_DATA",
            "legacy": "LOCAL_SUBTITLES_DATA"}[override_kind]
    monkeypatch.setenv(name, str(custom))
    assert paths.data_root() == custom.resolve()
    assert migration.ensure_initial_migration() == {"migrated": False}
    assert not custom.exists()
    assert not (isolated_user_data / APP_ID).exists()
    assert _files_hashes(source) == before


def test_completed_marker_recovers_missing_failure_history_without_recopied_profiles(isolated_user_data):
    source = _legacy_fixture(isolated_user_data, with_queue=False)
    migration.ensure_initial_migration()
    target = paths.data_root()
    cookie = target / _profile_relative() / "Default/Cookies"
    cookie.write_bytes(b"already-migrated-new-login")
    _populate_queue(source / "danmaku-queue.sqlite3")
    assert migration.ensure_initial_migration() == {"migrated": False}
    assert cookie.read_bytes() == b"already-migrated-new-login"
    store = DanmakuStore(target / "danmaku-queue.sqlite3")
    try:
        _assert_runtime_empty(store)
        assert len(store.failed_handoffs()) == 2
    finally:
        store.close()
