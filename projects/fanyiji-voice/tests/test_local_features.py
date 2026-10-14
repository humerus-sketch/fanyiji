"""Public packages expose sending without inspecting or altering legacy grants."""
import json

import pytest

from fanyiji import local_features


@pytest.fixture
def grant_root(tmp_path, monkeypatch):
    monkeypatch.setattr(local_features, "data_root", lambda: tmp_path)
    monkeypatch.setattr(local_features.sys, "frozen", True, raising=False)
    monkeypatch.setattr(local_features.platform, "node", lambda: "test-host")
    return tmp_path


def test_distribution_without_external_grant_is_available(grant_root):
    assert local_features.danmaku_available()
    assert not list(grant_root.iterdir())


def test_legacy_local_installer_grant_remains_atomic_and_compatible(grant_root, monkeypatch):
    local_features.enable_local_danmaku()
    assert local_features.danmaku_available()
    assert not (grant_root / "local-features.tmp").exists()
    original = (grant_root / "local-features.json").read_bytes()
    monkeypatch.setattr(local_features.platform, "node", lambda: "different-host")
    assert local_features.danmaku_available()
    assert (grant_root / "local-features.json").read_bytes() == original


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": 99},
    {"schema_version": 1, "host": "test-host", "danmaku": True}])
def test_invalid_external_grant_does_not_gate_or_modify_feature(grant_root, value):
    (grant_root / "local-features.json").write_text(json.dumps(value), encoding="utf-8")
    original = (grant_root / "local-features.json").read_bytes()
    assert local_features.danmaku_available()
    assert (grant_root / "local-features.json").read_bytes() == original


def test_malformed_external_grant_does_not_gate_or_modify_feature(grant_root):
    (grant_root / "local-features.json").write_bytes(b"\xffnot-json")
    assert local_features.danmaku_available()
    assert (grant_root / "local-features.json").read_bytes() == b"\xffnot-json"


@pytest.mark.parametrize("frozen", [False, True])
def test_feature_availability_does_not_resolve_user_files_or_host(grant_root, monkeypatch, frozen):
    monkeypatch.setattr(local_features.sys, "frozen", frozen)
    monkeypatch.setattr(local_features, "data_root", lambda: pytest.fail("must not access user data"))
    monkeypatch.setattr(local_features, "_host_key", lambda: pytest.fail("must not identify the host"))
    assert local_features.danmaku_available()
