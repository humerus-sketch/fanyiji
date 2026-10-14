"""Offline checks for browser version detection and honest update reporting."""
import json
from types import SimpleNamespace

import pytest

from fanyiji import browser_runtime as runtime


@pytest.mark.parametrize("value,expected", [
    ("155.0.4283.45", (155, 0, 4283, 45)),
    (" 154.10.2.03 ", (154, 10, 2, 3)),
    ("1.2", (1, 2)),
    ("", None), (None, None), (True, None),
    ("155.0.1.2 beta", None), ("155.0.1.2.3", None),
    ("155.-1.2.3", None), ("155.0.1.4294967296", None),
])
def test_version_key(value, expected):
    assert runtime.version_key(value) == expected


def test_registry_checks_both_hives_and_views_without_profile_access(monkeypatch):
    assert runtime.RUNTIME_CLIENT_ID == "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    visits = []

    class Key:
        def __init__(self, hive, view):
            self.hive, self.view = hive, view

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def open_key(hive, path, reserved, access):
        visits.append((hive, path, access))
        return Key(hive, access & 3)

    def value(key, name):
        assert name == "pv"
        return ({(10, 1): "154.0.2.3", (10, 2): "bad", (20, 1): "155.0.2.3", (20, 2): "0.0.0.0"}[(key.hive, key.view)], 1)

    registry = SimpleNamespace(HKEY_CURRENT_USER=10, HKEY_LOCAL_MACHINE=20,
                               KEY_WOW64_32KEY=1, KEY_WOW64_64KEY=2, KEY_READ=4,
                               OpenKey=open_key, QueryValueEx=value)
    monkeypatch.setattr(runtime, "winreg", registry)
    monkeypatch.setattr(runtime, "_file_versions", lambda: [])
    assert runtime.installed_runtime_version() == "155.0.2.3"
    assert len(visits) == 4
    assert all(path == rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{runtime.RUNTIME_CLIENT_ID}" for _, path, _ in visits)


def test_registry_permissions_or_missing_keys_do_not_abort_detection(monkeypatch):
    def denied(*args):
        raise PermissionError("no access")

    monkeypatch.setattr(runtime, "winreg", SimpleNamespace(
        HKEY_CURRENT_USER=10, HKEY_LOCAL_MACHINE=20, KEY_WOW64_32KEY=1,
        KEY_WOW64_64KEY=2, KEY_READ=4, OpenKey=denied))
    monkeypatch.setattr(runtime, "_file_versions", lambda: ["155.0.1.2"])
    assert runtime.installed_runtime_version() == "155.0.1.2"


def test_file_fallback_requires_runtime_executable_and_numeric_order(monkeypatch, tmp_path):
    application = tmp_path / "Application"
    for version in ("99.0.1.1", "155.0.1.1", "999.0.1.1", "155.0.2.2 beta"):
        directory = application / version
        directory.mkdir(parents=True)
        if version != "999.0.1.1":
            (directory / "msedgewebview2.exe").write_bytes(b"test")
    monkeypatch.setattr(runtime, "_application_directories", lambda: [application, tmp_path / "missing"])
    monkeypatch.setattr(runtime, "_registry_versions", lambda: [])
    assert runtime.installed_runtime_version() == "155.0.1.1"


class Response:
    def __init__(self, data):
        self.data = data
        self.read_limits = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        self.read_limits.append(limit)
        return self.data[:limit]


def mock_catalog(monkeypatch, data, architecture="x64"):
    response = Response(data if isinstance(data, bytes) else json.dumps(data).encode())
    requests = []

    def open_url(request, timeout):
        requests.append((request, timeout))
        return response

    monkeypatch.setattr(runtime, "urlopen", open_url)
    monkeypatch.setattr(runtime, "_native_architecture", lambda: architecture)
    return response, requests


def test_catalog_is_bounded_official_https_and_architecture_specific(monkeypatch):
    response, requests = mock_catalog(monkeypatch, [
        {"version": "99.0.9.9", "builds": [{"architecture": "x64"}]},
        {"version": "155.0.1.2", "builds": [{"architecture": "x64"}]},
        {"version": "156.0.1.2", "builds": [{"architecture": "arm64"}]},
        {"version": "157.0.1.2 beta", "builds": [{"architecture": "x64"}]},
    ])
    assert runtime.latest_catalog_version(timeout=3) == "155.0.1.2"
    assert requests[0][0].full_url == runtime.CATALOG_URL
    assert requests[0][0].headers["Accept"] == "application/json"
    assert "Cookie" not in requests[0][0].headers
    assert requests[0][1] == 3
    assert response.read_limits == [runtime.CATALOG_MAX_BYTES + 1]


@pytest.mark.parametrize("catalog", [{}, [], [None], [{"version": "155.0.1.2", "builds": "x64"}],
                                    [{"version": "0.0.0.0", "builds": [{"architecture": "x64"}]}],
                                    [{"version": "155.0.1.2", "builds": [{"architecture": "arm64"}]}]])
def test_bad_catalog_never_reports_current(monkeypatch, catalog):
    mock_catalog(monkeypatch, catalog)
    with pytest.raises(RuntimeError):
        runtime.latest_catalog_version()


def test_oversized_catalog_is_not_parsed(monkeypatch):
    mock_catalog(monkeypatch, b"[" + b" " * runtime.CATALOG_MAX_BYTES + b"]")
    with pytest.raises(RuntimeError, match="大小限制"):
        runtime.latest_catalog_version()


@pytest.mark.parametrize("exception", [OSError, RuntimeError])
def test_failed_check_does_not_leak_network_exception(monkeypatch, exception):
    def failed(*args, **kwargs):
        raise exception("proxy-user:proxy-secret@127.0.0.1")

    monkeypatch.setattr(runtime, "urlopen", failed)
    monkeypatch.setattr(runtime, "_native_architecture", lambda: "x64")
    with pytest.raises(RuntimeError) as caught:
        runtime.latest_catalog_version()
    assert "proxy-secret" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("timeout", [0, -1, 61, True, "8", None])
def test_invalid_timeouts_do_not_open_network(monkeypatch, timeout):
    def unexpected(*args, **kwargs):
        pytest.fail("network was accessed")

    monkeypatch.setattr(runtime, "urlopen", unexpected)
    with pytest.raises(ValueError):
        runtime.latest_catalog_version(timeout)


@pytest.mark.parametrize("machine,expected", [("AMD64", "x64"), ("x86_64", "x64"),
                                            ("ARM64", "arm64"), ("aarch64", "arm64"),
                                            ("i686", "x86")])
def test_native_architecture_maps_supported_runtime_builds(monkeypatch, machine, expected):
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)
    assert runtime._native_architecture() == expected


def test_unknown_architecture_fails_before_network(monkeypatch):
    monkeypatch.setattr(runtime.platform, "machine", lambda: "unknown-cpu")
    monkeypatch.setattr(runtime, "urlopen", lambda *args, **kwargs: pytest.fail("network accessed"))
    with pytest.raises(RuntimeError, match="架构"):
        runtime.latest_catalog_version()


@pytest.mark.parametrize("installed,active,catalog,expected", [
    (None, None, "155.0.1.2", "missing"),
    ("154.0.1.2", "154.0.1.2", "155.0.1.2", "new_available"),
    ("155.0.1.2", "154.0.1.2", "155.0.1.2", "restart_required"),
    ("155.0.1.2", "155.0.1.2", "155.0.1.2", "current"),
    ("156.0.1.2", None, "155.0.1.2", "current"),
    (None, "155.0.1.2", "155.0.1.2", "current"),
])
def test_snapshot_compares_installed_active_and_catalog(monkeypatch, installed, active, catalog, expected):
    monkeypatch.setattr(runtime, "installed_runtime_version", lambda: installed)
    monkeypatch.setattr(runtime, "latest_catalog_version", lambda: catalog)
    result = runtime.runtime_snapshot(active)
    assert result["state"] == expected
    assert result["installed_version"] == installed
    assert result["active_version"] == active
    assert result["catalog_version"] == catalog
    assert result["catalog_latest_not_evergreen_guarantee"] is True


def test_remote_opt_out_is_unknown_and_never_requests_metadata(monkeypatch):
    monkeypatch.setattr(runtime, "installed_runtime_version", lambda: "155.0.1.2")
    monkeypatch.setattr(runtime, "latest_catalog_version", lambda: pytest.fail("network check requested"))
    result = runtime.runtime_snapshot(remote=False)
    assert result["state"] == "unknown"
    assert result["catalog_version"] is None


def test_unavailable_network_is_not_misreported_as_latest(monkeypatch):
    def failed():
        raise RuntimeError("无法读取官方浏览器版本目录，请检查网络后重试")

    monkeypatch.setattr(runtime, "installed_runtime_version", lambda: "155.0.1.2")
    monkeypatch.setattr(runtime, "latest_catalog_version", failed)
    result = runtime.runtime_snapshot()
    assert result["state"] == "check_unavailable"
    assert result["catalog_version"] is None
    assert result["check_error"]


def test_installed_update_still_requires_restart_when_network_is_unavailable(monkeypatch):
    monkeypatch.setattr(runtime, "installed_runtime_version", lambda: "155.0.1.2")
    monkeypatch.setattr(runtime, "latest_catalog_version", lambda: (_ for _ in ()).throw(RuntimeError("network unavailable")))
    assert runtime.runtime_snapshot("154.0.1.2")["state"] == "restart_required"
