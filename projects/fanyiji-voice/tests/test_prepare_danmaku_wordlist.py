"""Pinned vocabulary preparation is deterministic and preserves existing files."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def preparer():
    path = Path(__file__).resolve().parents[1] / "tools/prepare_danmaku_wordlist.py"
    spec = importlib.util.spec_from_file_location("wordlist_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_keeps_license_deduplicates_and_excludes_single_chars(preparer, tmp_path, monkeypatch):
    def download(path):
        if path == "LICENSE":
            return b"MIT License\nCopyright fixture\n"
        return "测试词\n测试词\n词\n正常参考\n# comment\n".encode("utf-8")
    monkeypatch.setattr(preparer, "_download", download)
    output, evidence = tmp_path / "dictionary.json", tmp_path / "evidence"
    report = preparer.prepare(output, evidence)
    value = json.loads(output.read_text(encoding="utf-8"))
    assert report["entry_count"] == 2
    assert report["official_bilibili_list"] is False
    assert report["automatic_replacement"] is False
    assert value["sources"][0]["license_text"] == "MIT License\nCopyright fixture\n"
    assert value["allowlist"] == []
    assert {row["term"] for row in value["entries"]} == {"测试词", "正常参考"}
    assert len(report["files"]) == 6


def test_existing_dictionary_is_preserved_without_any_download(preparer, tmp_path, monkeypatch):
    output = tmp_path / "dictionary.json"
    output.write_bytes(b"existing user terms and allowlist")
    monkeypatch.setattr(preparer, "_download", lambda path: pytest.fail("Must not download"))
    with pytest.raises(FileExistsError):
        preparer.prepare(output, tmp_path / "evidence")
    assert output.read_bytes() == b"existing user terms and allowlist"


def test_unverified_license_does_not_create_dictionary(preparer, tmp_path, monkeypatch):
    monkeypatch.setattr(preparer, "_download", lambda path: b"Unknown permission")
    output = tmp_path / "dictionary.json"
    with pytest.raises(ValueError):
        preparer.prepare(output, tmp_path / "evidence")
    assert not output.exists()
