"""Synthetic pinned assets and local archives only; no real model or network."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import zipfile

import pytest

from fanyiji.model_import import ModelImporter, ModelImportCancelled, ModelImportError
from fanyiji.models_download import ModelEntry, ModelFile


def entry(files=None, *, kind="hy"):
    assets = files or {"translation/test.gguf": b"synthetic pinned GGUF bytes"}
    model_files = tuple(ModelFile(name, "https://huggingface.co/test/model/resolve/pinned/file",
                                 len(data), hashlib.sha256(data).hexdigest()) for name, data in assets.items())
    return ModelEntry("test-model", "Technical Model Name", model_files,
                      "https://huggingface.co/test/model", "test entry", model_kind=kind), assets


def folder(tmp_path, assets, prefix=""):
    root = tmp_path / "user-package"
    root.mkdir()
    for name, data in assets.items():
        path = root / prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def package(tmp_path, assets, *, prefix="", compression=zipfile.ZIP_STORED):
    path = tmp_path / "user-package.zip"
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, data in assets.items():
            archive.writestr(prefix + name, data)
    return path


@pytest.mark.parametrize("source_kind", ["file", "folder", "zip"])
def test_exact_pinned_assets_import_offline_with_provenance_and_without_activation(tmp_path, source_kind):
    model, assets = entry()
    data = next(iter(assets.values()))
    if source_kind == "file":
        source = tmp_path / "arbitrary-user-filename.gguf"
        source.write_bytes(data)
    elif source_kind == "folder":
        source = folder(tmp_path, assets, "outer-package")
    else:
        source = package(tmp_path, assets, prefix="outer-package/")
    before = source.read_bytes() if source.is_file() else {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    storage = tmp_path / "separate-models"
    importer = ModelImporter(storage)
    updates = []
    result = importer.import_entry(model, source, lambda *args: updates.append(args))
    assert result.model_id == model.id and result.imported_paths == tuple(assets)
    assert result.reused_paths == () and result.source_kind == source_kind
    assert result.verification == "pinned_catalog_sha256"
    assert (storage / "translation/test.gguf").read_bytes() == data
    assert (storage / "translation/test.gguf").stat().st_nlink == 1
    assert not (storage / "translation/backend.json").exists() and not (storage / "asr/backend.json").exists()
    record = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert record["verification"] == "pinned_catalog_sha256" and record["self_supplied_checksum"] is False
    assert record["source_kind"] == source_kind
    assert importer.has_provenance(model)
    assert updates and any(phase == "importing" for phase, _, _ in updates)
    assert not list((storage / ".model-import-staging").iterdir())
    assert source.exists()
    assert (source.read_bytes() if source.is_file() else {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}) == before


@pytest.mark.parametrize("kind,assets", [
    ("sense_voice", {"asr/model.int8.onnx": b"onnx", "asr/tokens.txt": b"tokens"}),
    (None, {"vad/silero_vad.onnx": b"vad"}),
    (None, {"speech_filter/model.int8.onnx": b"filter", "speech_filter/class_labels_indices.csv": b"labels"}),
])
def test_supported_multifile_asr_vad_and_filter_import_complete_flat_official_packages(tmp_path, kind, assets):
    model, _ = entry(assets, kind=kind)
    flat = {name.rsplit("/", 1)[-1]: data for name, data in assets.items()}
    source = folder(tmp_path, flat)
    storage = tmp_path / "models"
    result = ModelImporter(storage).import_entry(model, source)
    assert result.imported_paths == tuple(assets)
    for name, data in assets.items():
        assert (storage / name).read_bytes() == data


def test_valid_existing_files_are_reused_without_overwrite_or_mtime_change(tmp_path):
    model, assets = entry()
    source = folder(tmp_path, assets)
    storage = tmp_path / "models"
    target = storage / "translation/test.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(assets["translation/test.gguf"])
    before = target.stat()
    result = ModelImporter(storage).import_entry(model, source)
    assert result.imported_paths == () and result.reused_paths == tuple(assets)
    assert target.stat().st_ino == before.st_ino and target.stat().st_mtime_ns == before.st_mtime_ns


def test_conflicting_existing_target_refuses_before_writes_and_retains_all_sources(tmp_path):
    model, assets = entry()
    source = folder(tmp_path, assets)
    storage = tmp_path / "models"
    target = storage / "translation/test.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing personal file")
    with pytest.raises(ModelImportError, match="已存在"):
        ModelImporter(storage).import_entry(model, source)
    assert target.read_bytes() == b"existing personal file"
    assert not (storage / ".model-import-staging").exists()
    assert (source / "translation/test.gguf").exists()


@pytest.mark.parametrize("bad_kind", ["digest", "size"])
def test_wrong_pinned_digest_or_size_never_publishes_weight(tmp_path, bad_kind):
    model, assets = entry()
    valid = assets["translation/test.gguf"]
    bad = b"X" * len(valid) if bad_kind == "digest" else b"small"
    source = folder(tmp_path, {"test.gguf": bad})
    storage = tmp_path / "models"
    with pytest.raises(ModelImportError):
        ModelImporter(storage).import_entry(model, source)
    assert not (storage / "translation/test.gguf").exists()
    assert (source / "test.gguf").read_bytes() == bad


def test_all_files_are_verified_before_any_model_is_published(tmp_path):
    assets = {"asr/model.int8.onnx": b"first", "asr/tokens.txt": b"second"}
    model, _ = entry(assets, kind="sense_voice")
    source = folder(tmp_path, {"model.int8.onnx": b"first", "tokens.txt": b"XXXXXX"})
    storage = tmp_path / "models"
    with pytest.raises(ModelImportError, match="SHA-256"):
        ModelImporter(storage).import_entry(model, source)
    assert not (storage / "asr").exists()


@pytest.mark.parametrize("bad_path", ["../escaped", "/absolute", "C:/drive", "folder\\backslash", "CON", "trailing. ", "a//b"])
def test_zip_slip_and_windows_unsafe_extra_entries_refuse_before_writes(tmp_path, bad_path):
    model, assets = entry()
    source = package(tmp_path, {**assets, bad_path: b"bad extra"})
    if "\\" in bad_path:
        # ZipInfo constructors normalize native separators on Windows. Preserve
        # an actual hostile central-directory name rather than testing that normalization.
        with zipfile.ZipFile(source, "a") as archive:
            member = zipfile.ZipInfo("placeholder")
            member.filename = member.orig_filename = bad_path
            archive.writestr(member, b"raw malformed name")
    storage = tmp_path / "models"
    with pytest.raises(ModelImportError, match="路径"):
        ModelImporter(storage).import_entry(model, source)
    assert not storage.exists() and not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_zip_unix_links_or_special_files_are_rejected_even_if_not_selected(tmp_path, kind):
    model, assets = entry()
    source = package(tmp_path, assets)
    with zipfile.ZipFile(source, "a") as archive:
        info = zipfile.ZipInfo("unselected-extra")
        info.create_system = 3
        info.external_attr = (kind | 0o777) << 16
        archive.writestr(info, "outside target")
    with pytest.raises(ModelImportError, match="链接"):
        ModelImporter(tmp_path / "models").import_entry(model, source)
    assert not (tmp_path / "models").exists()


def test_zip_hardlink_extra_metadata_is_rejected(tmp_path):
    model, assets = entry()
    source = package(tmp_path, assets)
    with zipfile.ZipFile(source, "a") as archive:
        info = zipfile.ZipInfo("linked-extra")
        info.extra = b"\x0d\x00\x04\x00link"
        archive.writestr(info, b"payload")
    with pytest.raises(ModelImportError, match="链接扩展"):
        ModelImporter(tmp_path / "models").import_entry(model, source)


def test_zip_encryption_flag_rejected_before_extracting_even_unselected_entry(tmp_path):
    model, assets = entry()
    source = package(tmp_path, assets)
    raw = bytearray(source.read_bytes())
    central = raw.index(b"PK\x01\x02")
    raw[central + 8] |= 1
    source.write_bytes(raw)
    with pytest.raises(ModelImportError, match="加密"):
        ModelImporter(tmp_path / "models").import_entry(model, source)
    assert not (tmp_path / "models").exists()


def test_zip_crc_corruption_cannot_publish_any_asset(tmp_path):
    model, assets = entry()
    source = package(tmp_path, assets)
    raw = bytearray(source.read_bytes())
    payload = raw.index(next(iter(assets.values())))
    raw[payload] ^= 1
    source.write_bytes(raw)
    with pytest.raises(ModelImportError):
        ModelImporter(tmp_path / "models").import_entry(model, source)
    assert not (tmp_path / "models/translation/test.gguf").exists()


@pytest.mark.parametrize("extra", [{"translation/TEST.GGUF": b"another"}, {"translation": b"file conflicts with directory"}])
def test_zip_case_duplicates_and_file_directory_conflicts_rejected(tmp_path, extra):
    model, assets = entry()
    source = package(tmp_path, {**assets, **extra})
    with pytest.raises(ModelImportError):
        ModelImporter(tmp_path / "models").import_entry(model, source)
    assert not (tmp_path / "models").exists()


def test_zipbomb_ratio_declared_size_and_member_budget_are_bounded(tmp_path):
    model, assets = entry({"translation/test.gguf": b"0" * 10000})
    source = package(tmp_path, assets, compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(ModelImportError, match="压缩比例"):
        ModelImporter(tmp_path / "models", max_compression_ratio=2).import_entry(model, source)
    with pytest.raises(ModelImportError, match="大小"):
        ModelImporter(tmp_path / "models", max_file_bytes=50).import_entry(model, source)
    source = package(tmp_path, {**assets, "extra": b"extra"})
    with pytest.raises(ModelImportError, match="数量"):
        ModelImporter(tmp_path / "models", max_files=1).import_entry(model, source)


@pytest.mark.parametrize("field,value", [(zipfile._ECD_ENTRIES_TOTAL, 5000), (zipfile._ECD_SIZE, 40 * 1024 ** 2)])
def test_zip_member_and_central_directory_bounds_checked_before_allocating_zipfile(tmp_path, monkeypatch, field, value):
    model, assets = entry()
    source = package(tmp_path, assets)
    original_end = zipfile._EndRecData
    def declared_end(stream):
        end = original_end(stream)
        end[field] = value
        return end
    monkeypatch.setattr(zipfile, "_EndRecData", declared_end)
    def must_not_parse(*args, **kwargs):
        raise AssertionError("No full central-directory allocation should occur")
    monkeypatch.setattr(zipfile, "ZipFile", must_not_parse)
    with pytest.raises(ModelImportError, match="安全限制"):
        ModelImporter(tmp_path / "models").import_entry(model, source)
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("relation", ["same", "source_child", "target_child"])
def test_source_storage_aliases_or_nested_paths_refuse_without_touching_source(tmp_path, relation):
    model, assets = entry()
    source = folder(tmp_path, assets)
    storage = source if relation == "same" else source / "storage" if relation == "target_child" else tmp_path
    with pytest.raises(ModelImportError, match="嵌套"):
        ModelImporter(storage).import_entry(model, source)
    assert (source / "translation/test.gguf").read_bytes() == assets["translation/test.gguf"]


def test_folder_hardlink_rejected_even_when_extra_file_would_not_be_copied(tmp_path):
    model, assets = entry()
    source = folder(tmp_path, assets)
    outside = tmp_path / "outside"
    outside.write_bytes(b"private material")
    os.link(outside, source / "hardlink-extra")
    with pytest.raises(ModelImportError, match="hardlink"):
        ModelImporter(tmp_path / "models").import_entry(model, source)
    assert outside.read_bytes() == b"private material" and not (tmp_path / "models").exists()


def test_hardlinked_destination_even_with_correct_hash_is_never_reused_or_overwritten(tmp_path):
    model, assets = entry()
    source = folder(tmp_path, assets)
    external = tmp_path / "external-model"
    external.write_bytes(next(iter(assets.values())))
    storage = tmp_path / "models"
    (storage / "translation").mkdir(parents=True)
    os.link(external, storage / "translation/test.gguf")
    with pytest.raises(ModelImportError, match="hardlink"):
        ModelImporter(storage).import_entry(model, source)
    assert external.read_bytes() == next(iter(assets.values()))
    assert not (storage / ".model-import-staging").exists()


def test_conflicting_provenance_refuses_before_copying_models(tmp_path):
    model, assets = entry()
    source = folder(tmp_path, assets)
    storage = tmp_path / "models"
    importer = ModelImporter(storage)
    record = importer.provenance_path(model)
    record.parent.mkdir(parents=True)
    record.write_text("{}", encoding="utf-8")
    with pytest.raises(ModelImportError, match="来源记录"):
        importer.import_entry(model, source)
    assert not (storage / "translation").exists()
    assert not (storage / ".model-import-staging").exists()
    assert record.read_text(encoding="utf-8") == "{}"


def test_symlink_source_or_target_ancestor_is_rejected(tmp_path):
    model, assets = entry()
    source = folder(tmp_path, assets)
    link = tmp_path / "linked-source"
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation requires a privilege on this Windows host")
    with pytest.raises(ModelImportError, match="symlink"):
        ModelImporter(tmp_path / "models").import_entry(model, link)
    storage = tmp_path / "models"
    storage.mkdir()
    (storage / "translation").symlink_to(source, target_is_directory=True)
    with pytest.raises(ModelImportError, match="symlink"):
        ModelImporter(storage).import_entry(model, source)


def test_cancel_before_or_during_copy_never_creates_final_or_changes_source(tmp_path):
    model, assets = entry({"translation/test.gguf": b"x" * 100})
    source = folder(tmp_path, assets)
    storage = tmp_path / "models"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ModelImportCancelled):
        ModelImporter(storage).import_entry(model, source, cancel=cancel)
    assert not storage.exists()
    cancel.clear()
    def progress(phase, done, total):
        if phase == "importing":
            cancel.set()
    with pytest.raises(ModelImportCancelled):
        ModelImporter(storage, chunk_size=10).import_entry(model, source, progress, cancel)
    assert not (storage / "translation/test.gguf").exists()
    assert list((storage / ".model-import-staging").iterdir())
    assert (source / "translation/test.gguf").read_bytes() == assets["translation/test.gguf"]


def test_target_created_after_preflight_is_not_overwritten(tmp_path):
    model, assets = entry()
    source = folder(tmp_path, assets)
    storage = tmp_path / "models"
    target = storage / "translation/test.gguf"
    def progress(phase, done, total):
        if phase == "importing" and done == total:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"racing existing material")
    with pytest.raises(ModelImportError):
        ModelImporter(storage).import_entry(model, source, progress)
    assert target.read_bytes() == b"racing existing material"


def test_unknown_architecture_and_missing_required_or_ambiguous_files_are_rejected(tmp_path):
    model, assets = entry()
    source = folder(tmp_path, {"one/test.gguf": next(iter(assets.values())), "two/test.gguf": next(iter(assets.values()))})
    with pytest.raises(ModelImportError, match="架构"):
        ModelImporter(tmp_path / "models").import_entry(replace(model, model_kind="unknown"), source)
    with pytest.raises(ModelImportError, match="歧义"):
        ModelImporter(tmp_path / "models").import_entry(model, source)
    with pytest.raises(ModelImportError, match="必要文件"):
        ModelImporter(tmp_path / "models").import_entry(replace(model, files=(replace(model.files[0], relative_path="translation/missing.gguf"),)), source)


def test_import_never_overwrites_active_configuration_even_if_bad_entry_attempts_to_target_it(tmp_path):
    model, assets = entry({"asr/backend.json": b"injected profile"})
    source = folder(tmp_path, assets)
    with pytest.raises(ModelImportError, match="支持"):
        ModelImporter(tmp_path / "models").import_entry(model, source)
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("kwargs", [{"max_files": True}, {"chunk_size": 0}, {"max_total_bytes": -1},
                                    {"max_file_bytes": 0}, {"max_compression_ratio": float("inf")},
                                    {"max_compression_ratio": float("nan")}])
def test_import_resource_bounds_reject_invalid_settings(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ModelImporter(tmp_path / "models", **kwargs)


@pytest.mark.parametrize("kind", ["whisper", "qwen3_asr"])
def test_retired_asr_import_is_rejected_before_copy_or_source_read(tmp_path, kind):
    model, _ = entry({"asr/retired/encoder.int8.onnx": b"encoder"}, kind=kind)
    storage = tmp_path / "models"
    with pytest.raises(ModelImportError, match="仅支持 SenseVoice"):
        ModelImporter(storage).import_entry(model, tmp_path / "missing-source")
    assert not storage.exists()
