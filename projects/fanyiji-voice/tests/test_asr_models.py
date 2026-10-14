"""Pinned metadata and synthetic file activation; never download model weights."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import threading

import pytest

from fanyiji.asr_catalog import ASRCatalog, AsrCatalog, profile_for_entry, vad_dependencies
from fanyiji.models_download import DownloadCancelled, DownloadError, ModelCatalog, ModelDownloader, validate_download_url


def synthetic_catalogs():
    digest = hashlib.sha256(b"stub").hexdigest()
    def fake(entry):
        return replace(entry, files=tuple(replace(file, size=4, sha256=digest) for file in entry.files))
    catalog = ModelCatalog.default()
    mt = ModelCatalog(tuple(fake(catalog.get(mid)) for mid in
                            (catalog.required_entries[0].id, catalog.default_model_id, "hy-mt2-1.8b-q6_k")),
                      catalog.default_model_id)
    asr = AsrCatalog.default()
    return mt, AsrCatalog(tuple(fake(entry) for entry in asr.entries), asr.default_model_id)


def place(root, entries):
    for entry in entries:
        for file in entry.files:
            path = root / file.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stub")


def test_catalog_exposes_only_sensevoice_and_keeps_vad_and_filter():
    assert ASRCatalog is AsrCatalog
    asr = AsrCatalog.default()
    assert AsrCatalog.default(ModelCatalog.default()) == asr
    assert len(asr.entries) == 1 and asr.default_model_id == "sensevoice-int8"
    selected = asr.get(asr.default_model_id)
    assert selected.name == "SenseVoice INT8" and "备用" not in selected.recommendation
    assert profile_for_entry(selected) == {"backend": "sense_voice", "model_id": selected.id}
    for file in selected.files:
        validate_download_url(file.url)
    vad = vad_dependencies(ModelCatalog.default(), speech_only=False)
    assert len(vad) == 1 and all(file.relative_path.startswith("vad/") for file in vad[0].files)
    assert any(entry.id == "speech-filter-zipformer" for entry in vad_dependencies(ModelCatalog.default()))


@pytest.mark.parametrize("kind", ["whisper", "qwen3_asr"])
def test_retired_asr_cannot_be_activated_through_profile_generation(kind):
    retired = replace(AsrCatalog.default().entries[0], model_kind=kind)
    with pytest.raises(DownloadError, match="仅支持 SenseVoice"):
        profile_for_entry(retired)


@pytest.mark.parametrize("url", [
    "https://modelscope.cn/models/fixed/resolve/pinned/file",
    "https://cdn-lfs-cn-1.modelscope.cn/path/file",
])
def test_actual_modelscope_delivery_hosts_allowed(url):
    validate_download_url(url)


def test_arbitrary_modelscope_subdomain_or_aliyun_not_allowed():
    for url in ("https://unexpected.modelscope.cn/file", "https://storage.aliyuncs.com/file"):
        with pytest.raises(DownloadError):
            validate_download_url(url)


def test_sensevoice_activation_requires_selected_asr_vad_and_mt(tmp_path):
    mt, asr = synthetic_catalogs()
    selected = asr.get("sensevoice-int8")
    place(tmp_path, (*vad_dependencies(mt), selected, mt.get(mt.default_model_id)))
    downloader = ModelDownloader(tmp_path)
    targets = downloader.activate_models(mt, mt.default_model_id, asr, selected.id)
    assert json.loads(targets[0].read_text()) == profile_for_entry(selected)
    assert json.loads(targets[1].read_text())["model_kind"] == "hy"
    assert (tmp_path / "asr/model.int8.onnx").exists()


def test_switch_unchanged_asr_and_mt_reuses_cache_and_preserves_all_weights(tmp_path, monkeypatch):
    mt, asr = synthetic_catalogs()
    place(tmp_path, (*mt.entries, *asr.entries))
    downloader = ModelDownloader(tmp_path)
    for entry in (*mt.entries, *asr.entries):
        assert downloader.verify_entry(entry).ready
    weights = {tmp_path / file.relative_path for entry in (*mt.entries, *asr.entries) for file in entry.files}
    original_open = Path.open
    def no_weight_read(path, *args, **kwargs):
        if path in weights and (args[0] if args else kwargs.get("mode", "r")) == "rb":
            raise AssertionError("未变化的已校验模型不应重新读取")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_weight_read)
    reopened = ModelDownloader(tmp_path)
    for mid, aid in ((mt.default_model_id, "sensevoice-int8"),
                     ("hy-mt2-1.8b-q6_k", "sensevoice-int8"),
                     (mt.default_model_id, "sensevoice-int8")):
        reopened.activate_models(mt, mid, asr, aid)
        assert all(path.is_file() and path.stat().st_size == 4 for path in weights)
    assert json.loads((tmp_path / "asr/backend.json").read_text())["backend"] == "sense_voice"


@pytest.mark.parametrize("previous_exists", [True, False])
def test_second_configuration_failure_restores_first_without_deleting(tmp_path, monkeypatch, previous_exists):
    mt, asr = synthetic_catalogs()
    place(tmp_path, (*mt.entries, *asr.entries))
    asr_target, mt_target = tmp_path / "asr/backend.json", tmp_path / "translation/backend.json"
    old_asr = b'{"backend":"sense_voice"}\n\n'
    old_mt = b'{"backend":"ctranslate2"}\n'
    if previous_exists:
        asr_target.write_bytes(old_asr)
        mt_target.write_bytes(old_mt)
    original_replace = os.replace
    def fail_second(source, target):
        if Path(target) == mt_target:
            raise OSError("synthetic second write failure")
        original_replace(source, target)
    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(OSError, match="second write"):
        ModelDownloader(tmp_path).activate_models(mt, mt.default_model_id, asr, "sensevoice-int8")
    if previous_exists:
        assert asr_target.read_bytes() == old_asr and mt_target.read_bytes() == old_mt
    else:
        assert json.loads(asr_target.read_text()) == {"backend": "sense_voice"}
        assert not mt_target.exists()
    assert all((tmp_path / file.relative_path).is_file() for entry in (*mt.entries, *asr.entries) for file in entry.files)


def test_cancellation_between_writes_restores_implicit_default(tmp_path, monkeypatch):
    mt, asr = synthetic_catalogs()
    place(tmp_path, (*mt.entries, *asr.entries))
    event = threading.Event()
    original_replace = os.replace
    target = tmp_path / "asr/backend.json"
    def cancel_after_first(source, destination):
        original_replace(source, destination)
        if Path(destination) == target:
            event.set()
    monkeypatch.setattr(os, "replace", cancel_after_first)
    with pytest.raises(DownloadCancelled):
        ModelDownloader(tmp_path).activate_models(mt, mt.default_model_id, asr, "sensevoice-int8", cancel=event)
    assert json.loads(target.read_text()) == {"backend": "sense_voice"}
    assert not (tmp_path / "translation/backend.json").exists()


def test_rollback_failure_is_explicit_and_never_reported_as_success(tmp_path, monkeypatch):
    mt, asr = synthetic_catalogs()
    place(tmp_path, (*mt.entries, *asr.entries))
    original_replace = os.replace
    backend_calls = []
    def fail_second_and_restore(source, target):
        if Path(target).name == "backend.json":
            backend_calls.append(Path(target))
            if len(backend_calls) >= 2:
                raise OSError("synthetic write/rollback failure")
        original_replace(source, target)
    monkeypatch.setattr(os, "replace", fail_second_and_restore)
    with pytest.raises(DownloadError, match="恢复未完成，未启用"):
        ModelDownloader(tmp_path).activate_models(mt, mt.default_model_id, asr, "sensevoice-int8")
