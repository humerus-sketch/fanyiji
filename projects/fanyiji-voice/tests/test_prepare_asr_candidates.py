"""Download integrity checks use in-memory HTTP responses, never public weights."""
import hashlib
import importlib.util
import io
from pathlib import Path
import re

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_asr_candidates_test", Path(__file__).resolve().parents[1] / "tools" / "prepare_asr_candidates.py")
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class Response:
    def __init__(self, data, headers, status=206):
        self.body = io.BytesIO(data)
        self.headers = headers
        self.status = status

    def read(self, count):
        return self.body.read(count)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.body.close()


def fixture(data=b"valid-model-data"):
    return {"id": "test", "repo": "maintainer/model", "revision": "a" * 40,
            "files": {"encoder.onnx": (len(data), hashlib.sha256(data).hexdigest())}}, data


def ranged_opener(data, observed):
    def open_response(request, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        observed.append((start, end))
        return Response(data[start:end + 1], {
            "Content-Range": f"bytes {start}-{end}/{len(data)}", "Content-Length": str(end - start + 1)})
    return open_response


def test_range_chunks_publish_only_verified_final(tmp_path, monkeypatch):
    candidate, data = fixture()
    monkeypatch.setattr(downloader, "REQUEST_BYTES", 5)
    observed = []
    downloader.download_file(candidate, tmp_path, "encoder.onnx", opener=ranged_opener(data, observed))
    assert (tmp_path / "encoder.onnx").read_bytes() == data
    assert observed == [(0, 4), (5, 9), (10, 14), (15, 15)]
    assert not (tmp_path / "encoder.onnx.part").exists()


def test_resume_keeps_previous_bytes(tmp_path):
    candidate, data = fixture()
    (tmp_path / "encoder.onnx.part").write_bytes(data[:7])
    observed = []
    downloader.download_file(candidate, tmp_path, "encoder.onnx", opener=ranged_opener(data, observed))
    assert observed[0][0] == 7
    assert (tmp_path / "encoder.onnx").read_bytes() == data


def test_ignored_resume_preserves_part(tmp_path):
    candidate, data = fixture()
    partial = tmp_path / "encoder.onnx.part"
    partial.write_bytes(data[:7])
    with pytest.raises(RuntimeError, match="ignored Range"):
        downloader.download_file(candidate, tmp_path, "encoder.onnx",
                                 opener=lambda *a, **kw: Response(data, {"Content-Length": str(len(data))}, status=200))
    assert partial.read_bytes() == data[:7]
    assert not (tmp_path / "encoder.onnx").exists()


def test_wrong_range_preserves_part_without_append(tmp_path):
    candidate, data = fixture()
    partial = tmp_path / "encoder.onnx.part"
    partial.write_bytes(data[:7])
    with pytest.raises(RuntimeError, match="Content-Range"):
        downloader.download_file(candidate, tmp_path, "encoder.onnx",
                                 opener=lambda *a, **kw: Response(data, {"Content-Range": f"bytes 0-{len(data)-1}/{len(data)}"}))
    assert partial.read_bytes() == data[:7]


def test_hash_mismatch_is_not_published_or_removed(tmp_path):
    candidate, data = fixture()
    corrupted = b"x" * len(data)
    with pytest.raises(RuntimeError, match="failed verification"):
        downloader.download_file(candidate, tmp_path, "encoder.onnx", opener=ranged_opener(corrupted, []))
    assert (tmp_path / "encoder.onnx.part").read_bytes() == corrupted
    assert not (tmp_path / "encoder.onnx").exists()


def test_connection_eof_retries_from_actual_written_offset(tmp_path, monkeypatch):
    candidate, data = fixture()
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    observed = []
    regular = ranged_opener(data, observed)

    def first_truncated(request, timeout):
        if not observed:
            observed.append((0, len(data) - 1))
            return Response(data[:3], {"Content-Range": f"bytes 0-{len(data)-1}/{len(data)}",
                                      "Content-Length": str(len(data))})
        return regular(request, timeout)

    downloader.download_file(candidate, tmp_path, "encoder.onnx", opener=first_truncated)
    assert observed[1][0] == 3
    assert (tmp_path / "encoder.onnx").read_bytes() == data


def test_existing_invalid_final_is_preserved(tmp_path):
    candidate, data = fixture()
    final = tmp_path / "encoder.onnx"
    final.write_bytes(b"existing-invalid")
    with pytest.raises(RuntimeError, match="Existing final file"):
        downloader.download_file(candidate, tmp_path, "encoder.onnx", opener=ranged_opener(data, []))
    assert final.read_bytes() == b"existing-invalid"
