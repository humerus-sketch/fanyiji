"""A distribution has no sender grant; local installation grants this host only."""
import json

import pytest

from fanyiji import local_features


@pytest.fixture
def grant_root(tmp_path, monkeypatch):
    monkeypatch.setattr(local_features, "data_root", lambda: tmp_path)
    monkeypatch.setattr(local_features.sys, "frozen", True, raising=False)
    monkeypatch.setattr(local_features.platform, "node", lambda: "test-host")
    return tmp_path


def test_distribution_without_external_grant_is_disabled(grant_root):
    assert not local_features.danmaku_available()
    assert not list(grant_root.iterdir())


def test_local_installer_grant_is_atomic_and_host_bound(grant_root, monkeypatch):
    local_features.enable_local_danmaku()
    assert local_features.danmaku_available()
    assert not (grant_root / "local-features.tmp").exists()
    monkeypatch.setattr(local_features.platform, "node", lambda: "different-host")
    assert not local_features.danmaku_available()


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": 99},
    {"schema_version": 1, "host": "test-host", "danmaku": True}])
def test_invalid_external_grant_stays_disabled(grant_root, value):
    (grant_root / "local-features.json").write_text(json.dumps(value), encoding="utf-8")
    assert not local_features.danmaku_available()


def test_malformed_external_grant_stays_disabled(grant_root):
    (grant_root / "local-features.json").write_bytes(b"\xffnot-json")
    assert not local_features.danmaku_available()
