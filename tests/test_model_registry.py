"""Offline synthetic containers and temporary registries; no native model loads."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import zipfile

import pytest

from fanyiji.asr_catalog import AsrCatalog, profile_for_entry
from fanyiji.gguf_inspection import GgufInspectionError, inspect_gguf
from fanyiji.model_import import ModelImporter, ModelImportCancelled, ImportOption
from fanyiji.model_registry import ModelRegistry, validate_display_name
from fanyiji.models_download import DownloadError, ModelCatalog, ModelDownloader, ModelEntry, ModelFile

URL = "https://huggingface.co/test/fixture/resolve/pinned/model"


def gguf_bytes(architecture="qwen3", *, offsets=None, dtype=1, split=1, tokens=True):
    def string(value):
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw
    def text(key, value):
        return string(key) + struct.pack("<I", 8) + string(value)
    fields = [text("general.architecture", architecture), text("general.name", "Test local GGUF"),
              text("tokenizer.ggml.model", "gpt2"),
              text("tokenizer.chat_template", "{{ messages }}"),
              string("split.count") + struct.pack("<II", 4, split)]
    for key, value in (("block_count", 1), ("context_length", 2048), ("embedding_length", 32),
                       ("feed_forward_length", 32), ("attention.head_count", 1)):
        fields.append(string(architecture + "." + key) + struct.pack("<II", 4, value))
    if tokens:
        fields.append(string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + string("hello") + string("world"))
    names = ["token_embd.weight", "output_norm.weight", *("blk.0." + suffix for suffix in
             ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_norm.weight",
              "ffn_norm.weight", "ffn_up.weight", "ffn_down.weight", "ffn_gate.weight"))]
    raw = b"GGUF" + struct.pack("<IQQ", 3, len(names), len(fields)) + b"".join(fields)
    for index, name in enumerate(names):
        offset = offsets[index] if offsets is not None and index < len(offsets) else index * 64
        raw += string(name) + struct.pack("<IQIQ", 1, 32, dtype, offset)
    raw += b"\0" * (-len(raw) % 32)
    return raw + b"x" * (64 * len(names))


def fake_catalogs():
    def entry(mid, kind, assets):
        files = tuple(ModelFile(path, URL, len(raw), hashlib.sha256(raw).hexdigest()) for path, raw in assets.items())
        return ModelEntry(mid, mid, files, URL, "synthetic", model_kind=kind)
    mt = entry("preset-mt", "hy", {"translation/preset.gguf": b"preset mt"})
    asr = entry("preset-asr", "sense_voice", {"asr/model.int8.onnx": b"preset onnx", "asr/tokens.txt": b"preset tokens"})
    return ModelCatalog((mt,), mt.id), AsrCatalog((asr,), asr.id)


@pytest.mark.parametrize("architecture,kind", [("qwen2", "qwen"), ("qwen3", "qwen"), ("hy_v3", "hy"), ("hunyuan-dense", "hy")])
def test_bounded_gguf_container_family_detection(architecture, kind):
    raw = gguf_bytes(architecture)
    metadata = inspect_gguf(io.BytesIO(raw), len(raw))
    assert metadata.architecture == architecture and metadata.model_kind == kind
    assert metadata.tensor_count == 11 and metadata.token_count == 2
    assert metadata.header_bytes < len(raw)


@pytest.mark.parametrize("raw", [
    b"fake.gguf", gguf_bytes("llama"), gguf_bytes(tokens=False), gguf_bytes(split=2),
    gguf_bytes(offsets=(0, 0)), gguf_bytes(offsets=(0, 65)), gguf_bytes(dtype=99),
    gguf_bytes()[:-1], b"GGUF" + struct.pack("<IQQ", 3, 2 ** 63, 1),
])
def test_unknown_truncated_or_overlapping_gguf_is_refused(raw):
    with pytest.raises(GgufInspectionError):
        inspect_gguf(io.BytesIO(raw), len(raw))


@pytest.mark.parametrize("name", ["", " " * 5, "a" * 81, "text\nnewline", "invisible\u200bname", 42])
def test_display_name_validation(name):
    with pytest.raises(DownloadError):
        validate_display_name(name)
    assert validate_display_name("  我的模型  ") == "我的模型"


@pytest.mark.parametrize("role,source_kind", [(role, kind) for role in ("asr", "translation") for kind in ("zip", "folder", "file") if role != "asr" or kind != "file"])
def test_catalog_auto_detection_independent_name_and_roundtrip(tmp_path, role, source_kind):
    mt, asr = fake_catalogs()
    template = (mt if role == "translation" else asr).entries[0]
    data = {"translation/preset.gguf": b"preset mt", "asr/model.int8.onnx": b"preset onnx", "asr/tokens.txt": b"preset tokens"}
    selected = {file.relative_path: data[file.relative_path] for file in template.files}
    source = tmp_path / "source"
    if source_kind == "zip":
        source = source.with_suffix(".zip")
        with zipfile.ZipFile(source, "w") as archive:
            for filename, raw in selected.items():
                archive.writestr("wrapper/" + filename, raw)
    elif source_kind == "file":
        source = source.with_suffix(".gguf")
        source.write_bytes(next(iter(selected.values())))
    else:
        for filename, raw in selected.items():
            path = source / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    models = tmp_path / "models"
    importer = ModelImporter(models)
    registry = ModelRegistry(models, catalog=mt, asr_catalog=asr)
    inspection = importer.inspect_source(source, (*mt.entries, *asr.entries))
    assert len(inspection.options) == 1 and inspection.options[0].template.id == template.id
    entry = importer.import_named(inspection.options[0], source, "我的独立模型", registry)
    assert entry.id.startswith("local-") and entry.id != template.id
    assert entry.name == "我的独立模型" and registry.display_name_for(entry.id) == entry.name
    assert entry == registry.entries()[0]
    assert not (models / "asr/backend.json").exists() and not (models / "translation/backend.json").exists()
    assert ModelDownloader(models).verify_entry(entry).ready
    for file in entry.files:
        assert file.relative_path.startswith(role + "/imported/" + entry.id + "/")
        assert (models / file.relative_path).read_bytes() in selected.values()
    catalogs = registry.catalogs()
    assert catalogs[role == "asr"].get(entry.id).name == entry.name
    if role == "asr":
        profile = profile_for_entry(entry)
        assert profile["display_name"] == entry.name and profile["model_id"] == entry.id
        assert profile["model"].startswith("imported/" + entry.id)
    else:
        ModelDownloader(models).activate_model(catalogs[0], entry.id)
        profile = json.loads((models / "translation/backend.json").read_text(encoding="utf-8"))
        assert profile["display_name"] == entry.name and profile["model_id"] == entry.id
        assert profile["model_file"].startswith("imported/" + entry.id)
    assert source.exists()


@pytest.mark.parametrize("source_kind", ["file", "folder", "zip"])
def test_custom_gguf_is_local_digest_only_and_can_be_explicitly_activated(tmp_path, source_kind):
    raw = gguf_bytes()
    source = tmp_path / "source.gguf"
    if source_kind == "file":
        source.write_bytes(raw)
    elif source_kind == "folder":
        source.mkdir()
        (source / "my-model.gguf").write_bytes(raw)
    else:
        source = source.with_suffix(".zip")
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("model/my-model.gguf", raw)
    models = tmp_path / "models"
    mt, asr = fake_catalogs()
    importer, registry = ModelImporter(models), ModelRegistry(models, catalog=mt, asr_catalog=asr)
    option = importer.inspect_source(source, (*mt.entries, *asr.entries)).options[0]
    assert option.template.verification == "local_sha256" and option.template.files[0].url is None
    entry = importer.import_named(option, source, "My Qwen", registry)
    assert registry.entries()[0] == entry and entry.source_model_id is None
    record = json.loads(importer.provenance_path(entry).read_text(encoding="utf-8"))
    assert record["verification"] == "local_sha256" and record["self_supplied_checksum"] is True
    state = ModelDownloader(models).verify_entry(entry)
    assert state.ready and "官方来源未经验证" in state.detail and "目录固定" not in state.detail
    catalogs = registry.catalogs()
    assert not (models / "translation/backend.json").exists()
    ModelDownloader(models).activate_model(catalogs[0], entry.id)
    profile = json.loads((models / "translation/backend.json").read_text(encoding="utf-8"))
    assert profile["model_kind"] == "qwen" and profile["display_name"] == "My Qwen"
    with pytest.raises(DownloadError, match="没有下载地址"):
        ModelDownloader(models).download_entry(entry)


def test_tampered_pinned_registry_cannot_supply_its_own_official_hash(tmp_path):
    mt, asr = fake_catalogs()
    registry = ModelRegistry(tmp_path, catalog=mt, asr_catalog=asr)
    entry = registry.named_entry(mt.entries[0], "test")
    registry.register(entry)
    raw = json.loads(registry.path.read_text(encoding="utf-8"))
    raw["entries"][0]["files"][0]["sha256"] = "0" * 64
    registry.path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DownloadError, match="固定 SHA-256"):
        registry.entries()
    with pytest.raises(DownloadError):
        registry.register(registry.named_entry(mt.entries[0], "second"))
    assert json.loads(registry.path.read_text(encoding="utf-8")) == raw


def test_unknown_onnx_and_ct2_and_wrong_preset_hash_are_refused(tmp_path):
    mt, asr = fake_catalogs()
    source = tmp_path / "unknown"
    source.mkdir()
    (source / "model.int8.onnx").write_bytes(b"X" * len(b"preset onnx"))
    (source / "tokens.txt").write_bytes(b"preset tokens")
    with pytest.raises(DownloadError, match="未知 ONNX / CT2"):
        ModelImporter(tmp_path / "models").inspect_source(source, (*mt.entries, *asr.entries))
    assert not (tmp_path / "models").exists()


def test_inspection_cancel_and_postinspection_source_change_do_not_register(tmp_path):
    source = tmp_path / "model.gguf"
    raw = gguf_bytes()
    source.write_bytes(raw)
    models = tmp_path / "models"
    mt, asr = fake_catalogs()
    registry = ModelRegistry(models, catalog=mt, asr_catalog=asr)
    importer = ModelImporter(models, chunk_size=64)
    cancel = threading.Event()
    def progress(*_):
        cancel.set()
    with pytest.raises(ModelImportCancelled):
        importer.inspect_source(source, mt.entries, progress, cancel)
    assert not models.exists()
    option = importer.inspect_source(source, mt.entries).options[0]
    source.write_bytes(raw[:-1] + b"Y")
    with pytest.raises(DownloadError, match="SHA-256"):
        importer.import_named(option, source, "modified", registry)
    assert not registry.path.exists()
    assert not (models / "translation/backend.json").exists()


def test_duplicate_name_rejected_before_copy_and_existing_weights_preserved(tmp_path):
    mt, asr = fake_catalogs()
    source = tmp_path / "preset.gguf"
    source.write_bytes(b"preset mt")
    registry = ModelRegistry(tmp_path / "models", catalog=mt, asr_catalog=asr)
    importer = ModelImporter(registry.models_root)
    option = importer.inspect_source(source, mt.entries).options[0]
    first = importer.import_named(option, source, "One Model", registry)
    before = registry.path.read_bytes()
    with pytest.raises(DownloadError, match="已存在"):
        importer.import_named(option, source, "one model", registry)
    assert registry.path.read_bytes() == before and registry.entries() == (first,)


def test_registry_paths_and_model_kind_cannot_escape_or_invent_adapters(tmp_path):
    mt, asr = fake_catalogs()
    registry = ModelRegistry(tmp_path, catalog=mt, asr_catalog=asr)
    entry = registry.named_entry(mt.entries[0], "test")
    raw = registry._record(entry)
    raw["files"][0]["path"] = "translation/elsewhere.gguf"
    with pytest.raises(DownloadError, match="超出"):
        registry._entry(raw)
    raw = registry._record(entry)
    raw.update(verification="local_sha256", source_model_id=None, architecture="llama")
    with pytest.raises(DownloadError, match="尚未适配"):
        registry._entry(raw)


@pytest.mark.parametrize("kind", ["whisper", "qwen3_asr"])
def test_retired_asr_is_not_available_for_independent_named_import(tmp_path, kind):
    retired = replace(AsrCatalog.default().entries[0], model_kind=kind)
    registry = ModelRegistry(tmp_path)
    with pytest.raises(DownloadError, match="尚未支持"):
        registry.named_entry(retired, "Retired recognizer")
    assert not registry.path.exists()


@pytest.mark.parametrize("kind", ["whisper", "qwen3_asr"])
def test_legacy_retired_asr_records_are_filtered_without_losing_translation_records(tmp_path, kind):
    mt, asr = fake_catalogs()
    registry = ModelRegistry(tmp_path, catalog=mt, asr_catalog=asr)
    first_translation = registry.named_entry(mt.entries[0], "Translation one")
    registry.register(first_translation)
    raw = json.loads(registry.path.read_text(encoding="utf-8"))
    retired_record = registry._record(registry.named_entry(asr.entries[0], "Retired recognizer"))
    retired_record.update(model_kind=kind, source_model_id="retired-original")
    raw["entries"].append(retired_record)
    registry.path.write_text(json.dumps(raw), encoding="utf-8")
    before = registry.path.read_bytes()
    assert registry.entries() == (first_translation,)
    assert registry.path.read_bytes() == before
    translation_catalog, asr_catalog = registry.catalogs()
    assert translation_catalog.get(first_translation.id) == first_translation
    assert all(entry.model_kind == "sense_voice" for entry in asr_catalog.entries)
    second_translation = registry.named_entry(mt.entries[0], "Translation two")
    registry.register(second_translation)
    preserved = json.loads(registry.path.read_text(encoding="utf-8"))["entries"]
    assert preserved[:2] == raw["entries"]
    assert registry.entries() == (first_translation, second_translation)


@pytest.mark.parametrize("kind", ["whisper", "qwen3_asr"])
def test_explicit_legacy_asr_catalog_cannot_restore_retired_selection(tmp_path, kind):
    mt, asr = fake_catalogs()
    retired = replace(asr.entries[0], id="retired-asr", model_kind=kind,
                      files=tuple(replace(file, relative_path="asr/retired/" + Path(file.relative_path).name)
                                  for file in asr.entries[0].files))
    legacy = AsrCatalog((*asr.entries, retired), retired.id)
    registry = ModelRegistry(tmp_path, catalog=mt, asr_catalog=legacy)
    translation, recognition = registry.catalogs()
    assert translation == mt
    assert recognition == asr


@pytest.mark.parametrize("kind", ["whisper", "qwen3_asr"])
def test_explicit_retired_import_templates_are_not_detected(tmp_path, kind):
    mt, asr = fake_catalogs()
    retired = replace(asr.entries[0], model_kind=kind)
    source = tmp_path / "retired-package"
    source.mkdir()
    (source / "model.int8.onnx").write_bytes(b"preset onnx")
    (source / "tokens.txt").write_bytes(b"preset tokens")
    with pytest.raises(DownloadError, match="未知 ONNX / CT2"):
        ModelImporter(tmp_path / "models").inspect_source(source, (*mt.entries, retired))
    assert not (tmp_path / "models").exists()


def test_names_change_only_display_metadata_not_id_paths_or_hashes(tmp_path):
    mt, asr = fake_catalogs()
    registry = ModelRegistry(tmp_path, catalog=mt, asr_catalog=asr)
    named = registry.named_entry(mt.entries[0], "My model")
    renamed = replace(named, name="別の表示名")
    assert named.id == renamed.id and named.files == renamed.files
    registry.register(renamed)
    assert registry.display_name_for(named.id) == "別の表示名"
    assert registry.catalogs()[0].get(named.id).files == named.files


def test_invalid_or_large_registry_is_not_silently_replaced(tmp_path):
    registry = ModelRegistry(tmp_path)
    for raw in (b'[]', b'{"schema_version":true,"entries":[]}', b'x' * (2 * 1024 * 1024 + 1)):
        registry.path.write_bytes(raw)
        with pytest.raises(DownloadError):
            registry.entries()
        assert registry.display_name_for("local-" + "1" * 32) is None
        assert registry.path.read_bytes() == raw


def test_custom_gguf_profile_factory_reports_alias_without_starting_helper(tmp_path):
    from fanyiji.engines import create_translation_engine
    mt, asr = fake_catalogs()
    registry = ModelRegistry(tmp_path / "models", catalog=mt, asr_catalog=asr)
    source = tmp_path / "custom.gguf"
    source.write_bytes(gguf_bytes())
    importer = ModelImporter(registry.models_root)
    option = importer.inspect_source(source, mt.entries).options[0]
    named = importer.import_named(option, source, "翻訳モデル", registry)
    ModelDownloader(registry.models_root).activate_model(registry.catalogs()[0], named.id)
    engine = create_translation_engine(registry.models_root / "translation")
    try:
        assert engine.backend_info["model_id"] == named.id
        assert engine.backend_info["display_name"] == "翻訳モデル"
        assert engine.backend_info["helper_pid"] is None and engine.backend_info["device"] == "not_loaded"
        assert engine.model_file == registry.models_root / named.files[0].relative_path
    finally:
        engine.close()


def test_windows_uppercase_gguf_extension_imports_without_renaming_source(tmp_path):
    source = tmp_path / "MODEL.GGUF"
    source.write_bytes(gguf_bytes())
    mt, asr = fake_catalogs()
    registry = ModelRegistry(tmp_path / "models", catalog=mt, asr_catalog=asr)
    importer = ModelImporter(registry.models_root)
    option = importer.inspect_source(source, mt.entries).options[0]
    named = importer.import_named(option, source, "Uppercase file", registry)
    assert source.exists() and named.files[0].relative_path.endswith("MODEL.gguf")
    ModelDownloader(registry.models_root).activate_model(registry.catalogs()[0], named.id)
    assert (registry.models_root / named.files[0].relative_path).exists()
