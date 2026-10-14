"""Offscreen UI, synthetic local files and stub operations; never use model/audio/network."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from dataclasses import replace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog

from fanyiji.model_manager import ModelManagerDialog
from fanyiji.model_manager import _Worker
from unittest.mock import Mock
from fanyiji.models_download import (
    DownloadCancelled, ModelCatalog, ModelDownloader, ModelEntry, ModelFile,
)
from fanyiji.asr_catalog import AsrCatalog, profile_for_entry, vad_dependencies


APP = QApplication.instance() or QApplication([])


def test_disabled_speech_filter_excludes_assets_and_activation_forwards_choice(tmp_path):
    dialog = ModelManagerDialog(tmp_path, speech_only=False)
    try:
        assert dialog._required_entries
        assert all(not file.relative_path.startswith("speech_filter/")
                   for entry in dialog._required_entries for file in entry.files)
        downloader = Mock()
        worker = _Worker(downloader, dialog.catalog, "activate", dialog.catalog.default_model_id,
                         threading.Event(), asr_catalog=dialog.asr_catalog,
                         asr_model_id=dialog.asr_catalog.default_model_id, speech_only=False)
        worker.run()
        assert downloader.activate_models.call_args.kwargs["speech_only"] is False
    finally:
        dispose(dialog)


def catalog():
    entries = []
    for mid, relative, data, required in (
        ("base", "asr/fake.onnx", b"fake base", True),
        ("mt", "translation/fake.gguf", b"fake mt", False),
    ):
        file = ModelFile(relative, "https://huggingface.co/test/fake/resolve/pinned/file",
                         len(data), hashlib.sha256(data).hexdigest())
        entries.append(ModelEntry(mid, mid, (file,), file.url, "测试假数据",
                                  required=required, model_kind=None if required else "hy"))
    return ModelCatalog(tuple(entries), "mt")


def events_until(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        APP.processEvents()
        time.sleep(0.005)
    APP.processEvents()
    assert predicate()


def dispose(dialog):
    dialog.reject()
    events_until(lambda: dialog._thread is None)
    dialog.deleteLater()
    APP.processEvents()


def test_hybrid_model_manager_marks_actual_local_draft_model_loaded(tmp_path):
    model_catalog = catalog()
    dialog = ModelManagerDialog(tmp_path, catalog=model_catalog, active_models={"translation": {
        "mode": "hybrid", "backend": "codex_cli", "model_id": "cloud-test",
        "draft_model": {"model_id": "mt", "model_kind": "hy", "model_file": "fake.gguf", "device": "cpu"}}})
    try:
        assert "mt" in dialog.active_labels
        assert "CPU" in dialog.active_labels["mt"].text()
        assert "cloud-test" in dialog.current_models_label.text()
        assert not dialog.radios["mt"].isEnabled() or dialog.translation_required
    finally:
        dispose(dialog)


def test_empty_folder_verify_does_not_download_or_allow_activation(tmp_path):
    class NoNetwork:
        def open(self, *args, **kwargs):
            raise AssertionError("不应启动网络")
    model_catalog = catalog()
    downloader = ModelDownloader(tmp_path, opener=NoNetwork())
    dialog = ModelManagerDialog(tmp_path, catalog=model_catalog, downloader=downloader)
    try:
        dialog.show()
        events_until(lambda: dialog._thread is None)
        assert dialog.selected_model_id is None
        assert not dialog.use_button.isEnabled()
        assert not (tmp_path / "translation/backend.json").exists()
        assert all(label.text() == "未下载" for label in dialog.status_labels.values())
    finally:
        dispose(dialog)


def test_ready_files_explicit_activation_signal_and_backend(tmp_path):
    model_catalog = catalog()
    for relative, data in (("asr/fake.onnx", b"fake base"), ("translation/fake.gguf", b"fake mt")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    dialog = ModelManagerDialog(tmp_path, catalog=model_catalog)
    signals = []
    dialog.model_selected.connect(signals.append)
    try:
        dialog.show()
        events_until(lambda: dialog._thread is None)
        assert dialog.use_button.isEnabled()
        assert not (tmp_path / "translation/backend.json").exists()
        dialog.use_button.click()
        events_until(lambda: dialog._thread is None and dialog.selected_model_id == "mt")
        assert signals == ["mt"]
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert json.loads((tmp_path / "translation/backend.json").read_text())['model_file'] == "fake.gguf"
    finally:
        dispose(dialog)


def test_close_requests_cancel_and_keeps_worker_until_it_finishes(tmp_path):
    entered = threading.Event()
    released = threading.Event()
    class SlowDownloader(ModelDownloader):
        def verify_entry(self, entry, progress=None, cancel=None):
            entered.set()
            if not released.wait(3):
                raise AssertionError("测试未解除等待")
            if cancel.is_set():
                raise DownloadCancelled("测试取消")
            return super().verify_entry(entry, progress, cancel)
    dialog = ModelManagerDialog(tmp_path, catalog=catalog(), downloader=SlowDownloader(tmp_path))
    try:
        dialog.show()
        events_until(entered.is_set)
        dialog.close()
        assert dialog._thread is not None
        assert dialog._cancel.is_set()
        assert dialog.isVisible()
        released.set()
        events_until(lambda: dialog._thread is None)
        assert not dialog.isVisible()
        assert dialog.selected_model_id is None
        assert not (tmp_path / "translation/backend.json").exists()
    finally:
        released.set()
        dispose(dialog)


def test_shutdown_does_not_require_gui_events_to_quit_worker_thread(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class SlowDownloader(ModelDownloader):
        def verify_entry(self, entry, progress=None, cancel=None):
            entered.set()
            release.wait(2)
            if cancel.is_set():
                raise DownloadCancelled("测试取消")
            return super().verify_entry(entry, progress, cancel)
    dialog = ModelManagerDialog(tmp_path, catalog=catalog(), downloader=SlowDownloader(tmp_path))
    timer = None
    try:
        dialog.show()
        events_until(entered.is_set)
        thread = dialog._thread
        timer = threading.Timer(0.05, release.set)
        timer.start()
        dialog._shutdown()
        assert not thread.isRunning()
        events_until(lambda: dialog._thread is None)
    finally:
        release.set()
        if timer:
            timer.join()
        dispose(dialog)


def two_models():
    original = catalog()
    other = replace(original.get("mt"), id="other", name="other",
                    files=(replace(original.get("mt").files[0], relative_path="translation/other.gguf"),))
    return ModelCatalog((*original.entries, other), "mt")


def test_default_radio_reads_active_backend_and_explicit_id_still_wins(tmp_path):
    model_catalog = two_models()
    profile = tmp_path / "translation/backend.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({"backend": "llama_cpp", "model_file": "other.gguf"}), encoding="utf-8")
    dialog = ModelManagerDialog(tmp_path, catalog=model_catalog)
    explicit = ModelManagerDialog(tmp_path, catalog=model_catalog, initial_model_id="mt")
    try:
        assert dialog.radios["other"].isChecked()
        assert explicit.radios["mt"].isChecked()
    finally:
        dispose(dialog)
        dispose(explicit)


def test_unsafe_or_bad_backend_falls_back_without_reading_weight(tmp_path):
    model_catalog = two_models()
    profile = tmp_path / "translation/backend.json"
    profile.parent.mkdir(parents=True)
    for raw in ("broken", "[]", '{"backend":"llama_cpp","model_file":"../other.gguf"}'):
        profile.write_text(raw, encoding="utf-8")
        dialog = ModelManagerDialog(tmp_path, catalog=model_catalog)
        try:
            assert dialog.radios["mt"].isChecked()
        finally:
            dispose(dialog)


def test_reopen_uses_cache_but_recheck_button_forces_hash(tmp_path, monkeypatch):
    model_catalog = catalog()
    for relative, data in (("asr/fake.onnx", b"fake base"), ("translation/fake.gguf", b"fake mt")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    initial = ModelDownloader(tmp_path)
    for entry in model_catalog.entries:
        assert initial.verify_entry(entry).ready
    original_open = Path.open
    weights = {tmp_path / file.relative_path for entry in model_catalog.entries for file in entry.files}
    reads = []
    def observed(path, *args, **kwargs):
        if path in weights and (args[0] if args else kwargs.get("mode", "r")) == "rb":
            reads.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", observed)
    dialog = ModelManagerDialog(tmp_path, catalog=model_catalog)
    try:
        dialog.show()
        events_until(lambda: dialog._thread is None)
        assert dialog.use_button.isEnabled() and not reads
        dialog.verify_button.click()
        events_until(lambda: dialog._thread is None)
        assert dialog.use_button.isEnabled() and set(reads) == weights and len(reads) == 2
    finally:
        dispose(dialog)


def dual_catalogs():
    digest = hashlib.sha256(b"stub").hexdigest()
    def fake(entry):
        return replace(entry, files=tuple(replace(file, size=4, sha256=digest) for file in entry.files))
    mt = ModelCatalog.default()
    mt = ModelCatalog((fake(mt.required_entries[0]), fake(mt.get(mt.default_model_id))), mt.default_model_id)
    asr = AsrCatalog.default()
    sensevoice = fake(asr.entries[0])
    alias_id = "local-" + "1" * 32
    alias = replace(sensevoice, id=alias_id, name="Imported SenseVoice", files=tuple(
        replace(file, relative_path="asr/imported/" + alias_id + "/" + Path(file.relative_path).name)
        for file in sensevoice.files))
    asr = AsrCatalog((sensevoice, alias), sensevoice.id)
    return mt, asr


def test_dual_tabs_keep_independent_selection_and_read_current_asr(tmp_path):
    mt, asr = dual_catalogs()
    profile = tmp_path / "asr/backend.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps(profile_for_entry(asr.get("local-" + "1" * 32))), encoding="utf-8")
    dialog = ModelManagerDialog(tmp_path, catalog=mt, asr_catalog=asr)
    try:
        assert dialog.tabs.count() == 2
        assert dialog.radios[mt.default_model_id].isChecked()
        assert dialog.asr_radios["local-" + "1" * 32].isChecked()
        dialog.asr_radios[asr.default_model_id].setChecked(True)
        assert dialog.radios[mt.default_model_id].isChecked()
        assert dialog.selected_asr_model_id is None
        assert not dialog.use_button.isEnabled()
    finally:
        dispose(dialog)


def test_dual_selection_uses_imported_sensevoice_and_needs_explicit_click(tmp_path):
    mt, asr = dual_catalogs()
    selected = asr.get("local-" + "1" * 32)
    for entry in (*vad_dependencies(mt), mt.get(mt.default_model_id), selected):
        for file in entry.files:
            path = tmp_path / file.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stub")
    dialog = ModelManagerDialog(tmp_path, catalog=mt, asr_catalog=asr, initial_asr_model_id=selected.id)
    try:
        dialog.show()
        events_until(lambda: dialog._thread is None)
        assert dialog.use_button.isEnabled()
        assert dialog.selected_model_id is None and dialog.selected_asr_model_id is None
        assert not (tmp_path / "asr/backend.json").exists()
        dialog.use_button.click()
        events_until(lambda: dialog._thread is None and dialog.selected_asr_model_id == selected.id)
        assert dialog.selected_model_id == mt.default_model_id
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert json.loads((tmp_path / "asr/backend.json").read_text()) == profile_for_entry(selected)
        assert not (tmp_path / "asr/model.int8.onnx").exists()
    finally:
        dispose(dialog)


def test_missing_selected_asr_prevents_activation_even_when_mt_ready(tmp_path):
    mt, asr = dual_catalogs()
    for entry in (*vad_dependencies(mt), mt.get(mt.default_model_id)):
        for file in entry.files:
            path = tmp_path / file.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stub")
    dialog = ModelManagerDialog(tmp_path, catalog=mt, asr_catalog=asr)
    try:
        dialog.show()
        events_until(lambda: dialog._thread is None)
        assert not dialog.use_button.isEnabled()
        assert dialog._states[mt.default_model_id].ready
        assert not dialog._states[asr.default_model_id].ready
    finally:
        dispose(dialog)


def test_retired_saved_asr_falls_back_to_sensevoice_without_changing_translation(tmp_path):
    profile = tmp_path / "asr/backend.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({"backend": "qwen3_asr", "model_id": "qwen3-asr-1.7b-int8"}), encoding="utf-8")
    translation = tmp_path / "translation/backend.json"
    translation.parent.mkdir(parents=True)
    original = b'{"backend":"llama_cpp","model_file":"HY-MT2-1.8B-Q6_K.gguf"}\n'
    translation.write_bytes(original)
    dialog = ModelManagerDialog(tmp_path, initial_asr_model_id="whisper-turbo-int8", translation_required=False)
    try:
        assert set(dialog.asr_radios) == {"sensevoice-int8"}
        assert dialog.asr_radios["sensevoice-int8"].isChecked()
        assert dialog.radios["hy-mt2-1.8b-q6_k"].isChecked()
        assert translation.read_bytes() == original
        assert {entry.id for entry in dialog.catalog.entries if not entry.required} == {
            entry.id for entry in ModelCatalog.default().entries if not entry.required}
    finally:
        dispose(dialog)


def test_active_models_are_marked_independently_of_radio_selection(tmp_path):
    mt = ModelCatalog.default()
    asr = AsrCatalog.default()
    loaded = {"asr": {"model_id": "sensevoice-int8", "model_kind": "sense_voice", "device": "cpu"},
              "translation": {"model_file": "HY-MT2-7B-Q6_K.gguf", "model_kind": "hy", "device": "cuda"}}
    dialog = ModelManagerDialog(tmp_path, active_models=loaded)
    try:
        assert set(dialog.active_labels) == {"sensevoice-int8", "hy-mt2-7b-q6_k"}
        dialog.radios["hy-mt2-1.8b-q6_k"].setChecked(True)
        assert set(dialog.asr_radios) == {"sensevoice-int8"}
        assert set(dialog.active_labels) == {"sensevoice-int8", "hy-mt2-7b-q6_k"}
        assert "CPU" in dialog.current_models_label.text() and "CUDA" in dialog.current_models_label.text()
        assert "当前已加载" in dialog.active_labels["hy-mt2-7b-q6_k"].text()
        assert dialog.selected_model_id is None and dialog.selected_asr_model_id is None
    finally:
        dispose(dialog)


def test_saved_profile_does_not_claim_any_active_model_without_runtime_metadata(tmp_path):
    profile = tmp_path / "translation/backend.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({"backend": "llama_cpp", "model_file": "HY-MT2-7B-Q6_K.gguf"}), encoding="utf-8")
    dialog = ModelManagerDialog(tmp_path)
    try:
        assert dialog.radios["hy-mt2-7b-q6_k"].isChecked()
        assert not dialog.active_labels
        assert "识别：未加载" in dialog.current_models_label.text()
        assert "翻译：未加载" in dialog.current_models_label.text()
    finally:
        dispose(dialog)


def codex_dual_catalogs():
    mt, asr = dual_catalogs()
    speech_filter = ModelCatalog.default().get("speech-filter-zipformer")
    digest = hashlib.sha256(b"stub").hexdigest()
    speech_filter = replace(speech_filter, files=tuple(
        replace(file, size=4, sha256=digest) for file in speech_filter.files))
    return ModelCatalog((*mt.entries, speech_filter), mt.default_model_id), asr


def test_codex_model_manager_activates_only_asr_with_no_mt_weight(tmp_path):
    mt, asr = codex_dual_catalogs()
    selected = asr.get("local-" + "1" * 32)
    for entry in (*vad_dependencies(mt), selected):
        for file in entry.files:
            path = tmp_path / file.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stub")
    saved_translation = tmp_path / "translation/backend.json"
    saved_translation.parent.mkdir(parents=True)
    original = b'{"backend":"llama_cpp","model_file":"remembered.gguf"}\n'
    saved_translation.write_bytes(original)
    dialog = ModelManagerDialog(tmp_path, catalog=mt, asr_catalog=asr,
                                initial_asr_model_id=selected.id, translation_required=False)
    selected_signals = []
    dialog.model_selected.connect(selected_signals.append)
    try:
        assert "当前使用所选在线翻译源" in dialog.instructions_label.text()
        assert "Codex" not in dialog.instructions_label.text()
        assert dialog.tabs.currentIndex() == 1
        dialog.show()
        events_until(lambda: dialog._thread is None)
        assert dialog.use_button.isEnabled()
        assert not dialog._states[mt.default_model_id].ready
        assert all(not radio.isEnabled() for radio in dialog.radios.values())
        assert dialog.asr_radios[selected.id].isEnabled()
        assert dialog.download_buttons[mt.default_model_id].isEnabled()
        dialog.use_button.click()
        events_until(lambda: dialog._thread is None and dialog.selected_asr_model_id == selected.id)
        assert dialog.selected_model_id is None
        assert selected_signals == [selected.id]
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert json.loads((tmp_path / "asr/backend.json").read_text()) == profile_for_entry(selected)
        assert saved_translation.read_bytes() == original
    finally:
        dispose(dialog)


@pytest.mark.parametrize("missing", ["base-vad", "speech-filter-zipformer", "local-" + "1" * 32])
def test_codex_model_manager_requires_selected_asr_vad_and_speech_filter(tmp_path, missing):
    mt, asr = codex_dual_catalogs()
    selected = asr.get("local-" + "1" * 32)
    for entry in (*vad_dependencies(mt), selected):
        if entry.id != missing:
            for file in entry.files:
                path = tmp_path / file.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"stub")
    dialog = ModelManagerDialog(tmp_path, catalog=mt, asr_catalog=asr,
                                initial_asr_model_id=selected.id, translation_required=False)
    try:
        dialog.show()
        events_until(lambda: dialog._thread is None)
        assert not dialog.use_button.isEnabled()
        assert not dialog._states[missing].ready
        assert not (tmp_path / "asr/backend.json").exists()
        assert not (tmp_path / "translation/backend.json").exists()
    finally:
        dispose(dialog)


def test_codex_worker_forwards_translation_mode_and_reports_only_asr(tmp_path):
    mt, asr = dual_catalogs()
    downloader = Mock()
    worker = _Worker(downloader, mt, "activate", None, threading.Event(),
                     asr_catalog=asr, asr_model_id=asr.default_model_id,
                     translation_required=False)
    results = []
    worker.result.connect(results.append)
    worker.run()
    assert downloader.activate_models.call_args.kwargs["translation_required"] is False
    assert results == [{"states": {}, "activated": None, "activated_asr": asr.default_model_id}]


@pytest.mark.parametrize("source_kind", ["file", "folder", "zip"])
def test_import_menu_copies_known_local_package_async_without_selection_or_activation(
    tmp_path, monkeypatch, source_kind
):
    import zipfile
    from fanyiji.model_manager import QFileDialog, ModelImportDialog

    models = tmp_path / "models"
    (models / "asr").mkdir(parents=True)
    (models / "asr/fake.onnx").write_bytes(b"fake base")
    model_catalog = catalog()
    source = tmp_path / "source-package"
    source.mkdir()
    (source / "fake.gguf").write_bytes(b"fake mt")
    if source_kind == "folder":
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(source))
    else:
        source = source / "fake.gguf"
        if source_kind == "zip":
            source = tmp_path / "model.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("outer/fake.gguf", b"fake mt")
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    dialog = ModelManagerDialog(models, catalog=model_catalog)
    def confirm(wizard):
        assert len(wizard.inspection.options) == 1
        assert wizard.selected_option.template.id == "mt"
        wizard.name_edit.setText("专用翻译模型")
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(ModelImportDialog, "exec", confirm)
    signals = []
    dialog.model_selected.connect(signals.append)
    try:
        dialog.show()
        events_until(lambda: not dialog.is_busy)
        assert not dialog.import_buttons
        assert len(dialog.import_button.menu().actions()) == 2
        assert not dialog.use_button.isEnabled()
        dialog._choose_source(folder=source_kind == "folder")
        assert dialog.is_busy and not dialog.import_button.isEnabled()
        events_until(lambda: not dialog.is_busy)
        imported = dialog.registry.entries()[0]
        assert imported.name == "专用翻译模型" and imported.id.startswith("local-")
        assert (models / imported.files[0].relative_path).read_bytes() == b"fake mt"
        assert not (models / "translation/fake.gguf").exists()
        assert not dialog.use_button.isEnabled() and dialog.radios["mt"].isChecked()
        assert "本地导入" in dialog.status_labels[imported.id].text()
        assert dialog.selected_model_id is None and not signals
        assert source.exists() and not (models / "translation/backend.json").exists()
        assert dialog.import_button.isEnabled()
        dialog.verify_button.click()
        events_until(lambda: not dialog.is_busy)
        assert "本地导入" in dialog.status_labels[imported.id].text()
        dialog.radios[imported.id].setChecked(True)
        assert dialog.use_button.isEnabled()
        dialog.use_button.click()
        events_until(lambda: not dialog.is_busy)
        profile = json.loads((models / "translation/backend.json").read_text(encoding="utf-8"))
        assert signals == [imported.id]
        assert profile["display_name"] == imported.name and profile["model_id"] == imported.id
        assert profile["model_file"].startswith("imported/" + imported.id)
    finally:
        dispose(dialog)


def test_cancelled_file_picker_starts_no_task_and_keeps_radio_selection(tmp_path, monkeypatch):
    from fanyiji.model_manager import QFileDialog
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    model_catalog = two_models()
    dialog = ModelManagerDialog(tmp_path / "models", catalog=model_catalog, initial_model_id="other")
    try:
        dialog._choose_source(folder=False)
        assert not dialog.is_busy and dialog.radios["other"].isChecked()
        assert not (tmp_path / "models").exists()
    finally:
        dispose(dialog)


def test_naming_wizard_cancel_keeps_models_and_backend_untouched(tmp_path, monkeypatch):
    from fanyiji.model_import import ImportInspection, ImportOption
    from fanyiji.model_manager import ModelImportDialog
    dialog = ModelManagerDialog(tmp_path / "models", catalog=catalog())
    inspection = ImportInspection((ImportOption(catalog().get("mt"), "translation", "mt"),), "file")
    monkeypatch.setattr(ModelImportDialog, "exec", lambda _: QDialog.DialogCode.Rejected)
    try:
        dialog._confirm_import(inspection, tmp_path / "source.gguf")
        assert not dialog.is_busy and dialog.radios["mt"].isChecked()
        assert not (tmp_path / "models").exists()
    finally:
        dispose(dialog)


def test_direct_download_explains_import_without_switching_model(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QPushButton
    from fanyiji.model_manager import QDesktopServices

    dialog = ModelManagerDialog(tmp_path / "models", catalog=catalog())
    opened, help_calls = [], []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    monkeypatch.setattr(dialog, "_manual_download_help", lambda: help_calls.append(True))
    try:
        # The second catalog entry is the translation weight. Opening a direct
        # link must never download on the GUI thread, import, or activate it.
        buttons = [button for button in dialog.findChildren(QPushButton) if button.text() == "下载直链"]
        buttons[1].menu().actions()[0].trigger()
        assert opened == [catalog().get("mt").files[0].url]
        assert help_calls == [True]
        assert "导入本地模型" in dialog.message.text() and ".gguf" in dialog.message.text()
        assert not dialog.is_busy and dialog.selected_model_id is None
        assert dialog.radios["mt"].isChecked()
        assert not (tmp_path / "models").exists()
        assert dialog.findChildren(QPushButton, "unifiedModelImport") == [dialog.import_button]
        assert dialog.import_buttons == {}
    finally:
        dispose(dialog)


def test_download_help_and_model_folder_are_available_from_top_level(tmp_path, monkeypatch):
    from fanyiji.model_manager import QDesktopServices, QMessageBox
    dialog = ModelManagerDialog(tmp_path / "models", catalog=catalog())
    dialogs, opened = [], []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: dialogs.append((box.windowTitle(), box.text())))
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    try:
        dialog.download_help_button.click()
        assert len(dialogs) == 1
        assert "不需要解压" in dialogs[0][1] and ".crdownload" in dialogs[0][1]
        assert "Codex CLI、Google、有道" in dialogs[0][1]
        assert "不会自动切换" in dialogs[0][1]
        dialog.open_directory_button.click()
        assert (tmp_path / "models").is_dir()
        assert len(opened) == 1 and opened[0].isLocalFile()
        assert Path(opened[0].toLocalFile()) == (tmp_path / "models").resolve()
        assert not dialog.is_busy and dialog.selected_model_id is None
    finally:
        dispose(dialog)


def test_download_worker_uses_safe_diagnosis_in_signal_and_log(tmp_path, monkeypatch, caplog):
    from fanyiji.models_download import DownloadError
    import fanyiji.model_manager as module
    raw = DownloadError("https://host.invalid/model?token=PRIVATE_VALUE")
    diagnosed = []
    monkeypatch.setattr(module, "describe_download_error", lambda exc: diagnosed.append(exc) or "网络连接失败（host.invalid）；可续传")
    downloader = Mock()
    downloader.download_entry.side_effect = raw
    worker = _Worker(downloader, catalog(), "download", "mt", threading.Event())
    results = []
    worker.result.connect(results.append)
    worker.run()
    assert diagnosed == [raw]
    assert results[0]["error"] == "网络连接失败（host.invalid）；可续传"
    assert not results[0]["cancelled"]
    assert "网络连接失败" in caplog.text and "PRIVATE_VALUE" not in caplog.text


def test_download_error_details_are_preserved_on_model_row_and_copyable(tmp_path):
    from PySide6.QtCore import Qt
    dialog = ModelManagerDialog(tmp_path, catalog=catalog())
    try:
        dialog._active_action, dialog._active_id = "download", "mt"
        detail = "HTTP 429（huggingface.co）；下载源限制请求，请稍后续传"
        dialog._on_result({"states": {}, "error": detail, "cancelled": False})
        assert dialog.message.text() == detail
        assert dialog.status_labels["mt"].text() == detail
        assert dialog.message.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    finally:
        dispose(dialog)


@pytest.mark.parametrize("phase,expected", [("retrying", "自动续传"), ("source_switching", "官方下载源")])
def test_download_retry_phase_keeps_progress_and_reports_recovery(tmp_path, phase, expected):
    dialog = ModelManagerDialog(tmp_path, catalog=catalog())
    try:
        dialog._on_progress("mt", phase, 37)
        assert expected in dialog.status_labels["mt"].text()
        assert "37%" in dialog.status_labels["mt"].text()
        assert dialog.progress_bar.value() == 37
    finally:
        dispose(dialog)


def test_online_mode_import_success_explains_activation_requirement(tmp_path):
    dialog = ModelManagerDialog(tmp_path, catalog=catalog(), translation_required=False)
    try:
        dialog._on_result({"states": {}, "imported": "mt"})
        assert "更改设置" in dialog.message.text() and "本地暂译 + CLI 终稿" in dialog.message.text()
        assert "当前在线翻译源保持不变" in dialog.message.text()
        assert not dialog.radios["mt"].isEnabled()
        assert dialog.selected_model_id is None
    finally:
        dispose(dialog)


def test_naming_wizard_rejects_duplicate_and_invalid_name_before_copy(tmp_path):
    from fanyiji.model_import import ImportInspection, ImportOption
    from fanyiji.model_manager import ModelImportDialog
    inspection = ImportInspection((ImportOption(catalog().get("mt"), "translation", "mt"),), "file")
    wizard = ModelImportDialog(inspection, tmp_path / "source.gguf", existing_names=("mt",))
    try:
        assert wizard.name_edit.text() == "mt (2)" and wizard.confirm_button.isEnabled()
        wizard.name_edit.setText("MT")
        assert not wizard.confirm_button.isEnabled() and "已存在" in wizard.error_label.text()
        wizard.name_edit.setText(" ")
        assert not wizard.confirm_button.isEnabled()
        wizard.name_edit.setText("我的模型")
        assert wizard.display_name == "我的模型" and wizard.confirm_button.isEnabled()
    finally:
        wizard.deleteLater()
        APP.processEvents()


def test_cancel_after_inspection_result_does_not_open_naming_wizard(tmp_path, monkeypatch):
    from fanyiji.model_import import ImportInspection, ImportOption
    dialog = ModelManagerDialog(tmp_path, catalog=catalog())
    inspections = []
    monkeypatch.setattr(dialog, "_confirm_import", lambda *args: inspections.append(args))
    try:
        dialog._active_action = "inspect"
        inspection = ImportInspection((ImportOption(catalog().get("mt"), "translation", "mt"),), "file")
        dialog._on_result({"states": {}, "inspection": inspection, "source": tmp_path / "source.gguf"})
        dialog._cancel.set()
        dialog._task_finished()
        assert not inspections and dialog._inspection_pending is None
        assert not (tmp_path / ".model-registry.json").exists()
    finally:
        dispose(dialog)


def test_import_close_is_async_cancelled_without_destroying_thread_or_activating(tmp_path):
    from fanyiji.model_import import ModelImporter, ModelImportCancelled
    entered, release = threading.Event(), threading.Event()
    class SlowImporter(ModelImporter):
        def import_entry(self, entry, source, progress=None, cancel=None):
            entered.set()
            release.wait(3)
            if cancel.is_set():
                raise ModelImportCancelled("导入已取消，原模型包保留")
            raise AssertionError("expected cancellation")
    models = tmp_path / "models"
    dialog = ModelManagerDialog(models, catalog=catalog(), importer=SlowImporter(models))
    try:
        dialog._verified_once = True
        dialog.show()
        dialog._start_task("import", "mt", import_source=tmp_path / "source.gguf")
        events_until(entered.is_set)
        dialog.close()
        assert dialog.is_busy and dialog._cancel.is_set() and dialog.isVisible()
        release.set()
        events_until(lambda: not dialog.is_busy)
        assert not dialog.isVisible() and dialog.selected_model_id is None
        assert not models.exists()
    finally:
        release.set()
        dispose(dialog)


def test_failed_import_preserves_previously_ready_model_and_does_not_activate_or_modify_it(tmp_path):
    models = tmp_path / "models"
    for relative, data in (("asr/fake.onnx", b"fake base"), ("translation/fake.gguf", b"fake mt")):
        path = models / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    source = tmp_path / "wrong-model.gguf"
    source.write_bytes(b"WRONG!!")
    dialog = ModelManagerDialog(models, catalog=catalog())
    try:
        dialog.show()
        events_until(lambda: not dialog.is_busy)
        assert dialog.use_button.isEnabled()
        dialog._start_task("import", "mt", import_source=source)
        events_until(lambda: not dialog.is_busy)
        assert dialog._states["mt"].ready and dialog.use_button.isEnabled()
        assert (models / "translation/fake.gguf").read_bytes() == b"fake mt"
        assert not (models / "translation/backend.json").exists()
        assert dialog.selected_model_id is None
    finally:
        dispose(dialog)
