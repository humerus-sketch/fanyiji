"""Downloaded-file import and explicit activation, without network or inference."""
import hashlib
import json
from pathlib import PurePosixPath

import pytest

from fanyiji.asr_catalog import AsrCatalog
from fanyiji.model_import import ModelImporter
from fanyiji.model_registry import ModelRegistry
from fanyiji.models_download import (
    DownloadError, ModelCatalog, ModelDownloader, ModelEntry, ModelFile,
)
from test_model_registry import gguf_bytes


URL = "https://huggingface.co/test/manual-import/resolve/pinned/model"


class NoNetwork:
    def open(self, *_args, **_kwargs):
        raise AssertionError("Manual import and activation must stay offline")


def prepared_models(tmp_path):
    """Only GGUF structure is real; synthetic ONNX bytes are hash fixtures."""
    models = tmp_path / "model-storage"
    assets = {
        "translation/original.gguf": gguf_bytes("hy_v3"),
        "asr/model.int8.onnx": b"synthetic SenseVoice weights, not inference",
        "asr/tokens.txt": b"synthetic SenseVoice vocabulary",
        "vad/silero_vad.onnx": b"synthetic VAD weights, not inference",
    }

    def file(relative_path):
        raw = assets[relative_path]
        return ModelFile(relative_path, URL, len(raw), hashlib.sha256(raw).hexdigest())

    original = ModelEntry(
        "original-translation", "Original translation",
        (file("translation/original.gguf"),), URL, "Offline fixture", model_kind="hy")
    vad = ModelEntry(
        "required-vad", "Silero VAD", (file("vad/silero_vad.onnx"),),
        URL, "Offline fixture", required=True)
    asr = ModelEntry(
        "sensevoice-int8", "SenseVoice INT8",
        (file("asr/model.int8.onnx"), file("asr/tokens.txt")),
        URL, "Offline fixture", model_kind="sense_voice")
    catalog = ModelCatalog((vad, original), original.id)
    asr_catalog = AsrCatalog((asr,), asr.id)
    for relative_path, raw in assets.items():
        target = models / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    downloader = ModelDownloader(models, opener=NoNetwork())
    downloader.activate_models(catalog, original.id, asr_catalog, asr.id, speech_only=False)
    return models, catalog, asr_catalog, downloader, assets


def backend_bytes(models):
    return tuple((models / role / "backend.json").read_bytes() for role in ("asr", "translation"))


def stored_bytes(models):
    return {path.relative_to(models): path.read_bytes() for path in models.rglob("*") if path.is_file()}


@pytest.mark.parametrize("origin", ["pinned-direct-download", "custom-compatible-gguf"])
def test_downloaded_translation_file_import_verify_and_explicit_enable_preserves_sensevoice(tmp_path, origin):
    models, catalog, asr_catalog, downloader, assets = prepared_models(tmp_path)
    raw = assets["translation/original.gguf"] if origin == "pinned-direct-download" else gguf_bytes("qwen3")
    source = tmp_path / "Downloads" / "user-selected-file.gguf"
    source.parent.mkdir()
    source.write_bytes(raw)
    source_stat = source.stat()
    prior_backends = backend_bytes(models)
    importer = ModelImporter(models)
    registry = ModelRegistry(models, catalog=catalog, asr_catalog=asr_catalog)

    inspection = importer.inspect_source(source, (*catalog.entries, *asr_catalog.entries))
    assert inspection.source_kind == "file"
    assert len(inspection.options) == 1 and inspection.options[0].role == "translation"
    imported = importer.import_named(inspection.options[0], source, "我下载的翻译模型", registry)
    assert imported.verification == ("pinned_catalog_sha256" if origin == "pinned-direct-download" else "local_sha256")
    assert imported.source_model_id == (catalog.default_model_id if origin == "pinned-direct-download" else None)
    assert backend_bytes(models) == prior_backends, "Importing must never select a new active model"
    assert downloader.verify_entry(imported, force=True).ready
    assert backend_bytes(models) == prior_backends, "Verification must never activate the imported model"
    assert all((models / relative_path).read_bytes() == value for relative_path, value in assets.items())

    # Reload the registry as on the next app launch, then use the same activation
    # operation as the model manager rather than writing a backend by hand.
    reloaded = ModelRegistry(models, catalog=catalog, asr_catalog=asr_catalog)
    assert reloaded.entries() == (imported,)
    translation_choices, recognition_choices = reloaded.catalogs()
    targets = downloader.activate_models(
        translation_choices, imported.id, recognition_choices,
        asr_catalog.default_model_id, speech_only=False)
    assert targets == (models / "asr/backend.json", models / "translation/backend.json")
    assert backend_bytes(models)[0] == prior_backends[0]
    asr_profile = json.loads(targets[0].read_text(encoding="utf-8"))
    translation_profile = json.loads(targets[1].read_text(encoding="utf-8"))
    assert asr_profile == {"backend": "sense_voice", "model_id": "sensevoice-int8"}
    assert translation_profile["backend"] == "llama_cpp"
    assert translation_profile["model_id"] == imported.id
    assert translation_profile["display_name"] == "我下载的翻译模型"
    expected_file = str(PurePosixPath(imported.files[0].relative_path).relative_to("translation"))
    assert translation_profile["model_file"] == expected_file
    assert (models / "translation" / expected_file).read_bytes() == raw
    assert all((models / relative_path).read_bytes() == value for relative_path, value in assets.items())
    assert source.read_bytes() == raw
    assert source.stat().st_ino == source_stat.st_ino and source.stat().st_mtime_ns == source_stat.st_mtime_ns


@pytest.mark.parametrize("filename,raw,message", [
    ("download.html", b"<!doctype html><html>Login required</html>", "网页.*不是模型权重"),
    ("download.gguf", b"<!doctype html><html>Access denied</html>", "完整.*GGUF"),
    ("unfinished.gguf.part", gguf_bytes()[:-100], "下载临时文件.*下载完成"),
    ("truncated.gguf", gguf_bytes()[:-1], "文件不完整"),
    ("unsupported.gguf", gguf_bytes("llama"), "架构尚未适配"),
])
def test_invalid_download_is_rejected_before_import_without_changing_active_models(tmp_path, filename, raw, message):
    models, catalog, asr_catalog, _downloader, _assets = prepared_models(tmp_path)
    source = tmp_path / filename
    source.write_bytes(raw)
    before = stored_bytes(models)
    registry = ModelRegistry(models, catalog=catalog, asr_catalog=asr_catalog)
    with pytest.raises(DownloadError, match=message) as rejected:
        ModelImporter(models).inspect_source(source, (*catalog.entries, *asr_catalog.entries))
    if filename.endswith(".html"):
        assert str(rejected.value) == "选择的是网页或网页快捷方式，不是模型权重；请下载完整 .gguf 文件后再导入。"
    elif filename.endswith(".part"):
        assert str(rejected.value) == "文件仍是下载临时文件，请等浏览器下载完成后选择 .gguf 模型文件。"
    assert stored_bytes(models) == before
    assert registry.entries() == ()
    assert source.read_bytes() == raw


@pytest.mark.parametrize("temporary_suffix", [".part", ".crdownload", ".download", ".tmp"])
def test_browser_temporary_file_cannot_import_even_when_its_bytes_already_match_a_catalog_model(tmp_path, temporary_suffix):
    models, catalog, asr_catalog, _downloader, assets = prepared_models(tmp_path)
    source = tmp_path / ("original.gguf" + temporary_suffix)
    raw = assets["translation/original.gguf"]
    source.write_bytes(raw)
    before = stored_bytes(models)
    with pytest.raises(DownloadError) as rejected:
        ModelImporter(models).inspect_source(source, (*catalog.entries, *asr_catalog.entries))
    assert str(rejected.value) == "文件仍是下载临时文件，请等浏览器下载完成后选择 .gguf 模型文件。"
    assert stored_bytes(models) == before
    assert source.read_bytes() == raw


@pytest.mark.parametrize("missing_dependency", ["vad/silero_vad.onnx", "asr/model.int8.onnx"])
def test_imported_translation_cannot_replace_active_backend_when_dependencies_are_incomplete(tmp_path, missing_dependency):
    models, catalog, asr_catalog, downloader, _assets = prepared_models(tmp_path)
    source = tmp_path / "downloaded.gguf"
    source.write_bytes(gguf_bytes("qwen3"))
    importer = ModelImporter(models)
    registry = ModelRegistry(models, catalog=catalog, asr_catalog=asr_catalog)
    inspection = importer.inspect_source(source, (*catalog.entries, *asr_catalog.entries))
    imported = importer.import_named(inspection.options[0], source, "已导入但依赖未就绪", registry)
    assert downloader.verify_entry(imported).ready
    prior_backends = backend_bytes(models)
    # Keep the dependency's bytes in an unfinished file, as after an interrupted
    # download. A complete MT weight alone must not be enough to enable a mode.
    dependency = models / missing_dependency
    dependency.rename(dependency.with_name(dependency.name + ".part"))
    translation_choices, recognition_choices = registry.catalogs()
    with pytest.raises(DownloadError, match="尚未就绪.*下载未完成"):
        downloader.activate_models(
            translation_choices, imported.id, recognition_choices,
            asr_catalog.default_model_id, speech_only=False)
    assert backend_bytes(models) == prior_backends
    assert registry.entries() == (imported,)
    assert source.read_bytes() == gguf_bytes("qwen3")
    assert downloader.verify_entry(imported).ready
