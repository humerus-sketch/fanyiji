"""Distribution metadata tests use synthetic cache files, never user data."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("test_public_builder", ROOT / "tools" / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_runtime_provenance_excludes_local_execution_and_preserves_cache(tmp_path):
    public = {
        "tag": "b11429",
        "source": "https://github.com/ggml-org/llama.cpp/releases/tag/b11429",
        "packages": {"cpu.zip": {
            "size": 123,
            "sha256": "a" * 64,
            "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11429/cpu.zip",
        }},
        "license": {
            "llama_cpp": {"name": "MIT", "source": "https://example.test/llama/LICENSE"},
            "cuda_runtime": {"source": "https://example.test/cuda/eula"},
        },
        "server_sha256": {"cpu": "b" * 64, "cuda": "c" * 64},
    }
    cached = json.loads(json.dumps(public))
    cached.update({
        "cpu_server": "C:/Users/private/runtime/cpu/llama-server.exe",
        "cuda_server": "C:/Users/private/runtime/cuda/llama-server.exe",
        "checks": {"cuda": {"list-devices": {"stdout": "private-hardware-name"}}},
        "credentials": "private-extra-field",
    })
    cached["packages"]["cpu.zip"]["cache_path"] = "private-package-path"
    cached["license"]["llama_cpp"]["local_path"] = "private-license-path"
    cached["server_sha256"]["extra"] = "private-extra-hash"
    source = tmp_path / "cached-SOURCE.json"
    source.write_text(json.dumps(cached), encoding="utf-8")
    original_bytes = source.read_bytes()
    destination = tmp_path / "public-SOURCE.json"

    builder._write_public_runtime_source(source, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == public
    assert "private" not in destination.read_text(encoding="utf-8")
    assert source.read_bytes() == original_bytes


def test_distribution_licenses_preserve_nested_paths_with_duplicate_names(tmp_path):
    incoming = tmp_path / "site-packages"
    files = [Path("sample.dist-info/licenses/LICENSE.txt"),
             Path("sample.dist-info/licenses/vendor/LICENSE.txt")]
    for item, content in zip(files, ("main copyright", "vendor copyright")):
        path = incoming / item
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    distribution = SimpleNamespace(files=files, locate_file=lambda item: incoming / item)
    outgoing = tmp_path / "licenses"

    builder._copy_distribution_licenses(distribution, "sample", outgoing)

    assert (outgoing / "sample/licenses/LICENSE.txt").read_text(encoding="utf-8") == "main copyright"
    assert (outgoing / "sample/licenses/vendor/LICENSE.txt").read_text(encoding="utf-8") == "vendor copyright"


def test_speech_filter_notices_exclude_local_metadata_and_preserve_cache(tmp_path):
    source = tmp_path / "cache"
    source.mkdir()
    notices = {"LICENSE": b"upstream license\r\n",
               "LICENSE.txt": b"additional license\n",
               "README.md": "# 上游模型说明\n".encode("utf-8")}
    cached = {**notices,
              "SOURCE.json": json.dumps({"model_path": "C:/Users/private/model.onnx",
                                         "evaluation": "private-local-results"}).encode("utf-8"),
              "model.int8.onnx": b"synthetic weights"}
    for name, content in cached.items():
        (source / name).write_bytes(content)
    destination = tmp_path / "package/licenses/models/speech-filter-zipformer"

    builder._copy_speech_filter_notices(source, destination)

    assert {path.name: path.read_bytes() for path in destination.iterdir()} == notices
    assert not (destination / "SOURCE.json").exists()
    assert {path.name: path.read_bytes() for path in source.iterdir()} == cached


def test_supplemental_licenses_copy_manifest_and_required_files(tmp_path):
    source = tmp_path / "source"
    (source / "Qt").mkdir(parents=True)
    (source / "Qt/LGPL-3.0.txt").write_text("synthetic license text", encoding="utf-8")
    (source / "MANIFEST.json").write_text(json.dumps({"files": ["Qt/LGPL-3.0.txt"]}), encoding="utf-8")
    destination = tmp_path / "package/licenses/supplemental"

    builder._copy_supplemental_licenses(source, destination)

    assert (destination / "Qt/LGPL-3.0.txt").read_text(encoding="utf-8") == "synthetic license text"
    assert json.loads((destination / "MANIFEST.json").read_text(encoding="utf-8"))["files"] == ["Qt/LGPL-3.0.txt"]


@pytest.mark.parametrize("files", [[], ["missing-license.txt"], ["../outside-license.txt"],
                                  ["C:/outside-license.txt"], ["/outside-license.txt"], [42]])
def test_supplemental_licenses_fail_before_copy_if_required_files_invalid(tmp_path, files):
    source = tmp_path / "source"
    source.mkdir()
    (source / "MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    destination = tmp_path / "destination"

    with pytest.raises(RuntimeError, match="许可证"):
        builder._copy_supplemental_licenses(source, destination)

    assert not destination.exists()
