"""Synthetic model bytes and HTTP stubs only; no network or real model loading."""
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import threading

import pytest

from fanyiji.models_download import (
    DownloadCancelled, DownloadError, ModelCatalog, ModelDownloader, ModelEntry,
    ModelFile, _OfficialRedirects, validate_download_url,
)


URL = "https://huggingface.co/test/fake/resolve/0123456789abcdef/fake.gguf"


class Response(io.BytesIO):
    def __init__(self, data, *, status=200, headers=None, url=URL):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}
        self.url = url

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        return self.responses.pop(0)


def fake_file(path="translation/test.gguf", data=b"fake weight"):
    return ModelFile(path, URL, len(data), hashlib.sha256(data).hexdigest())


def fake_catalog():
    base = ModelEntry("base", "假基础模型", (fake_file("asr/test.onnx", b"base"),), URL,
                      "只用于测试", required=True)
    mt = ModelEntry("mt", "假翻译模型", (fake_file(),), URL, "只用于测试", model_kind="hy")
    return ModelCatalog((base, mt), "mt")


def place_fake_catalog(root, catalog):
    for entry in catalog.entries:
        for file in entry.files:
            path = root / file.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"base" if entry.required else b"fake weight")


def test_catalog_pins_manifest_and_all_ggufs():
    catalog = ModelCatalog.default()
    manifest = Path(__file__).parents[1] / "models" / "manifest.json"
    # A fresh source checkout and the final app intentionally contain no weights.
    # Compare the preparation manifest when available without requiring it.
    if manifest.is_file():
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        for file in catalog.required_entries[0].files:
            assert {"size": file.size, "sha256": file.sha256} == raw["files"][file.relative_path]
    assert catalog.default_model_id == "hy-mt2-7b-q6_k"
    assert catalog.get("hy-mt2-7b-q6_k").recommendation == "★★★★★"
    assert catalog.get("hy-mt2-1.8b-q6_k").recommendation == "★★★☆☆"
    assert catalog.get("qwen3-1.7b-q8_0").recommendation == "★★☆☆☆"
    assert catalog.get("hy-mt2-30b-a3b-q4_k_m").recommendation == "★★★☆☆"
    assert all("不是准确率或通用排行榜" in entry.recommendation_basis
               for entry in catalog.entries if not entry.required)
    assert catalog.get("hy-mt2-30b-a3b-q4_k_m").files[0].size == 18236702880


def test_range_resume_writes_only_tail(tmp_path):
    file = fake_file(data=b"abcdef")
    partial = tmp_path / (file.relative_path + ".part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"abc")
    opener = Opener(Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6", "Content-Length": "3"}))
    target = ModelDownloader(tmp_path, opener=opener).download_file(file)
    assert target.read_bytes() == b"abcdef"
    assert opener.requests[0].get_header("Range") == "bytes=3-"
    assert not partial.exists()


def test_range_ignored_restarts_from_zero(tmp_path):
    file = fake_file(data=b"abcdef")
    partial = tmp_path / (file.relative_path + ".part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"abc")
    target = ModelDownloader(tmp_path, opener=Opener(Response(b"abcdef"))).download_file(file)
    assert target.read_bytes() == b"abcdef"


def test_bad_range_preserves_partial(tmp_path):
    file = fake_file(data=b"abcdef")
    part = tmp_path / (file.relative_path + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"abc")
    opener = Opener(Response(b"def", status=206, headers={"Content-Range": "bytes 2-4/6"}))
    with pytest.raises(DownloadError, match="续传范围"):
        ModelDownloader(tmp_path, opener=opener).download_file(file)
    assert part.read_bytes() == b"abc"


def test_cancel_keeps_partial_and_does_not_publish(tmp_path):
    file = fake_file(data=b"abcdef")
    cancel = threading.Event()
    def progress(phase, done, total):
        if phase == "downloading" and done >= 2:
            cancel.set()
    downloader = ModelDownloader(tmp_path, opener=Opener(Response(b"abcdef")), chunk_size=2)
    with pytest.raises(DownloadCancelled):
        downloader.download_file(file, progress, cancel)
    assert (tmp_path / (file.relative_path + ".part")).read_bytes() == b"ab"
    assert not (tmp_path / file.relative_path).exists()


def test_wrong_hash_kept_for_explicit_restart(tmp_path):
    file = fake_file(data=b"abcdef")
    downloader = ModelDownloader(tmp_path, opener=Opener(Response(b"xxxxxx"), Response(b"abcdef")))
    with pytest.raises(DownloadError, match="SHA-256"):
        downloader.download_file(file)
    part = tmp_path / (file.relative_path + ".part")
    assert part.read_bytes() == b"xxxxxx"
    downloader.download_file(file, restart=True)
    assert (tmp_path / file.relative_path).read_bytes() == b"abcdef"
    assert next(part.parent.glob("*.part.invalid-*")).read_bytes() == b"xxxxxx"


def test_incomplete_http_body_keeps_part(tmp_path):
    file = fake_file(data=b"abcdef")
    opener = Opener(Response(b"abc", headers={"Content-Length": "6"}))
    with pytest.raises(DownloadError, match="大小或 SHA"):
        ModelDownloader(tmp_path, opener=opener, max_attempts=1).download_file(file)
    assert (tmp_path / (file.relative_path + ".part")).read_bytes() == b"abc"


def test_existing_invalid_weight_is_not_overwritten(tmp_path):
    file = fake_file()
    path = tmp_path / file.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old user file")
    opener = Opener()
    with pytest.raises(DownloadError, match="已保留"):
        ModelDownloader(tmp_path, opener=opener).download_file(file, restart=True)
    assert path.read_bytes() == b"old user file"
    assert not opener.requests


def test_enable_verifies_required_and_selected_before_atomic_backend(tmp_path):
    catalog = fake_catalog()
    downloader = ModelDownloader(tmp_path, opener=Opener())
    place_fake_catalog(tmp_path, catalog)
    backend = tmp_path / "translation/backend.json"
    backend.write_text('{"backend":"old"}', encoding="utf-8")
    (tmp_path / "asr/test.onnx").write_bytes(b"bad!")
    with pytest.raises(DownloadError, match="基础模型"):
        downloader.activate_model(catalog, "mt")
    assert json.loads(backend.read_text()) == {"backend": "old"}
    (tmp_path / "asr/test.onnx").write_bytes(b"base")
    downloader.activate_model(catalog, "mt")
    assert json.loads(backend.read_text()) == {
        "backend": "llama_cpp", "model_file": "test.gguf", "model_kind": "hy", "runtime_preference": "auto"}


def test_download_never_enables_model(tmp_path):
    file = fake_file()
    ModelDownloader(tmp_path, opener=Opener(Response(b"fake weight"))).download_file(file)
    assert not (tmp_path / "translation/backend.json").exists()


@pytest.mark.parametrize("path", ["../bad.gguf", "asr/../bad", "C:/bad", "asr\\bad", "/abs", "CON.gguf", "asr//bad", "asr/./bad"])
def test_catalog_rejects_unsafe_paths(path):
    with pytest.raises(ValueError):
        fake_file(path)


@pytest.mark.parametrize("url", ["http://huggingface.co/x", "https://huggingface.co.evil.test/x", "https://evil.test/x", "https://user:secret@github.com/x"])
def test_official_https_only(url):
    with pytest.raises(DownloadError):
        validate_download_url(url)


def test_redirect_rejected_before_following_external_url():
    from urllib.request import Request
    with pytest.raises(DownloadError):
        _OfficialRedirects().redirect_request(Request(URL), None, 302, "Found", {}, "https://evil.test/file")


def test_final_response_url_is_checked_without_writing(tmp_path):
    file = fake_file()
    opener = Opener(Response(b"fake weight", url="https://evil.test/file"))
    with pytest.raises(DownloadError):
        ModelDownloader(tmp_path, opener=opener).download_file(file)
    assert not (tmp_path / (file.relative_path + ".part")).exists()


def test_download_uses_single_read_for_responsive_cancellation(tmp_path):
    class SingleReadResponse(Response):
        def read(self, size=-1):
            raise AssertionError("不得使用可能等待整块的 read")
        def read1(self, size=-1):
            return io.BytesIO.read(self, size)
    file = fake_file()
    downloader = ModelDownloader(tmp_path, opener=Opener(SingleReadResponse(b"fake weight")))
    assert downloader.download_file(file).read_bytes() == b"fake weight"


def test_cancel_active_socket_does_not_close_buffered_reader(tmp_path):
    class FakeSocket:
        called = False
        def shutdown(self, mode):
            self.called = True
    class Raw:
        _sock = FakeSocket()
    class Buffered:
        raw = Raw()
    class Active:
        fp = Buffered()
    downloader = ModelDownloader(tmp_path)
    downloader._active_response = Active()
    downloader.cancel_active_download()
    assert downloader._active_response.fp.raw._sock.called


def test_symlink_parent_cannot_escape_model_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "models"
    root.mkdir()
    try:
        (root / "translation").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows account has no symlink privilege")
    with pytest.raises(DownloadError):
        ModelDownloader(root).path_for(fake_file())


def test_recommendation_is_replaceable_without_changing_pinned_files():
    catalog = ModelCatalog.default()
    original = catalog.get(catalog.default_model_id)
    updated = replace(original, recommendation="实测后填写")
    assert updated.files == original.files


def place_file(root, file, data=b"fake weight"):
    path = root / file.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def count_weight_reads(monkeypatch, paths):
    original = Path.open
    reads = []
    def observed(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path in paths and mode == "rb":
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", observed)
    return reads


def test_persisted_cache_skips_unchanged_weight_reads_on_reopen(tmp_path, monkeypatch):
    file = fake_file()
    path = place_file(tmp_path, file)
    assert ModelDownloader(tmp_path).verify_file(file)
    cache = json.loads((tmp_path / ".model-verification.json").read_text())
    assert cache["schema_version"] == 1
    assert list(cache["verified_files"].values())[0]["ino"] == path.stat().st_ino
    reads = count_weight_reads(monkeypatch, {path})
    assert ModelDownloader(tmp_path).verify_file(file)
    assert not reads


def test_explicit_force_recomputes_sha_even_when_file_unchanged(tmp_path, monkeypatch):
    file = fake_file()
    path = place_file(tmp_path, file)
    downloader = ModelDownloader(tmp_path)
    assert downloader.verify_file(file)
    reads = count_weight_reads(monkeypatch, {path})
    assert downloader.verify_file(file, force=True)
    assert reads == [path]


@pytest.mark.parametrize("change", ["mtime", "inode", "size", "expected_hash"])
def test_identity_or_expected_hash_change_invalidates_cache(tmp_path, monkeypatch, change):
    file = fake_file()
    path = place_file(tmp_path, file)
    downloader = ModelDownloader(tmp_path)
    assert downloader.verify_file(file)
    before = path.stat()
    if change == "mtime":
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    elif change == "inode":
        alternate = path.with_suffix(".replacement")
        alternate.write_bytes(b"fake weight")
        os.utime(alternate, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(alternate, path)
        assert path.stat().st_ino != before.st_ino
    elif change == "size":
        path.write_bytes(b"different size weight")
    else:
        file = replace(file, sha256="0" * 64)
    reads = count_weight_reads(monkeypatch, {path})
    result = ModelDownloader(tmp_path).verify_file(file)
    if change == "size":
        assert not result and not reads  # Size mismatch rejects before hashing.
        updated = fake_file(data=b"different size weight")
        assert ModelDownloader(tmp_path).verify_file(updated)
        assert reads == [path]
    else:
        assert result is (change != "expected_hash")
        assert reads == [path]


@pytest.mark.parametrize("raw", ["bad json", "[]", '{"schema_version":2,"verified_files":{}}',
                                   '{"schema_version":true,"verified_files":{}}',
                                   '{"schema_version":1,"verified_files":[]}'])
def test_bad_persistent_cache_is_ignored_and_full_hash_runs(tmp_path, monkeypatch, raw):
    file = fake_file()
    path = place_file(tmp_path, file)
    (tmp_path / ".model-verification.json").write_text(raw, encoding="utf-8")
    reads = count_weight_reads(monkeypatch, {path})
    assert ModelDownloader(tmp_path).verify_file(file)
    assert reads == [path]


def test_downloaded_weight_is_cached_after_verified_atomic_publish(tmp_path, monkeypatch):
    file = fake_file()
    opener = Opener(Response(b"fake weight"))
    downloader = ModelDownloader(tmp_path, opener=opener)
    final = downloader.download_file(file)
    reads = count_weight_reads(monkeypatch, {final})
    assert ModelDownloader(tmp_path, opener=Opener()).download_file(file) == final
    assert not reads and len(opener.requests) == 1


def test_cache_write_failure_does_not_reject_a_verified_weight(tmp_path, monkeypatch):
    file = fake_file()
    path = place_file(tmp_path, file)
    def failed_replace(*args):
        raise OSError("test read-only cache location")
    monkeypatch.setattr(os, "replace", failed_replace)
    assert ModelDownloader(tmp_path).verify_file(file)
    assert path.read_bytes() == b"fake weight"


def test_switch_uses_cached_base_and_preserves_old_weights_and_partials(tmp_path, monkeypatch):
    initial = fake_catalog()
    second = ModelEntry("second", "第二假模型", (fake_file("translation/second.gguf", b"second"),),
                        URL, "测试", model_kind="qwen")
    catalog = ModelCatalog((*initial.entries, second), initial.default_model_id)
    place_fake_catalog(tmp_path, initial)
    place_file(tmp_path, second.files[0], b"second")
    partial = tmp_path / "translation/future.gguf.part"
    partial.write_bytes(b"unfinished")
    downloader = ModelDownloader(tmp_path, opener=Opener())
    downloader.activate_model(catalog, "mt")
    assert downloader.verify_entry(second).ready
    expected = {tmp_path / file.relative_path for entry in catalog.entries for file in entry.files}
    reads = count_weight_reads(monkeypatch, expected)
    ModelDownloader(tmp_path, opener=Opener()).activate_model(catalog, "second")
    assert not reads
    assert json.loads((tmp_path / "translation/backend.json").read_text())["model_file"] == "second.gguf"
    assert (tmp_path / "translation/test.gguf").read_bytes() == b"fake weight"
    assert (tmp_path / "translation/second.gguf").read_bytes() == b"second"
    assert partial.read_bytes() == b"unfinished"


def codex_catalogs():
    from fanyiji.asr_catalog import AsrCatalog
    def entry(mid, paths, *, required=False, kind=None):
        return ModelEntry(mid, mid, tuple(fake_file(path, b"stub") for path in paths),
                          URL, "测试", required=required, model_kind=kind)
    vad = entry("vad", ("vad/silero_vad.onnx",), required=True)
    speech_filter = entry("speech-filter", ("speech_filter/filter.onnx",), required=True)
    mt = entry("mt", ("translation/test.gguf",), kind="hy")
    asr = entry("sensevoice", ("asr/model.int8.onnx", "asr/tokens.txt"), kind="sense_voice")
    return ModelCatalog((vad, speech_filter, mt), "mt"), AsrCatalog((asr,), "sensevoice")


@pytest.mark.parametrize("saved_translation", [False, True])
def test_codex_asr_activation_does_not_require_or_write_local_translation(tmp_path, saved_translation):
    from fanyiji.asr_catalog import profile_for_entry, vad_dependencies
    mt, asr = codex_catalogs()
    selected = asr.get(asr.default_model_id)
    for entry in (*vad_dependencies(mt), selected):
        for file in entry.files:
            place_file(tmp_path, file, b"stub")
    translation = tmp_path / "translation/backend.json"
    original = b'{"backend":"llama_cpp","model_file":"user-saved.gguf"}\n\n'
    if saved_translation:
        translation.parent.mkdir(parents=True)
        translation.write_bytes(original)
        original_stat = translation.stat()
    targets = ModelDownloader(tmp_path, opener=Opener()).activate_models(
        mt, None, asr, selected.id, translation_required=False)
    assert targets == (tmp_path / "asr/backend.json",)
    assert json.loads(targets[0].read_text()) == profile_for_entry(selected)
    assert not (tmp_path / "translation/test.gguf").exists()
    if saved_translation:
        assert translation.read_bytes() == original
        assert translation.stat().st_mtime_ns == original_stat.st_mtime_ns
    else:
        assert not translation.parent.exists()


@pytest.mark.parametrize("missing", ["base-vad", "speech-filter", "sensevoice"])
def test_codex_activation_still_rejects_missing_audio_dependencies(tmp_path, missing):
    from fanyiji.asr_catalog import vad_dependencies
    mt, asr = codex_catalogs()
    selected = asr.get(asr.default_model_id)
    for entry in (*vad_dependencies(mt), selected):
        if entry.id != missing:
            for file in entry.files:
                place_file(tmp_path, file, b"stub")
    asr_profile = tmp_path / "asr/backend.json"
    asr_profile.parent.mkdir(parents=True, exist_ok=True)
    original = b'{"backend":"old-asr"}\n'
    asr_profile.write_bytes(original)
    with pytest.raises(DownloadError, match="尚未就绪"):
        ModelDownloader(tmp_path, opener=Opener()).activate_models(
            mt, None, asr, selected.id, translation_required=False)
    assert asr_profile.read_bytes() == original
    assert not (tmp_path / "translation").exists()


def test_codex_activation_does_not_require_disabled_speech_filter(tmp_path):
    from fanyiji.asr_catalog import vad_dependencies
    mt, asr = codex_catalogs()
    selected = asr.get(asr.default_model_id)
    for entry in (*vad_dependencies(mt, speech_only=False), selected):
        for file in entry.files:
            place_file(tmp_path, file, b"stub")
    ModelDownloader(tmp_path, opener=Opener()).activate_models(
        mt, None, asr, selected.id, translation_required=False, speech_only=False)
    assert (tmp_path / "asr/backend.json").is_file()
    assert not (tmp_path / "speech_filter").exists()
    assert not (tmp_path / "translation").exists()
