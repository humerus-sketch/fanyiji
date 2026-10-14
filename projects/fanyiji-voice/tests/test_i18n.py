"""UI language persistence and display tests; no audio, models or user files."""
import ast
from dataclasses import replace
import json
import os
from pathlib import Path
from string import Formatter
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from fanyiji.config import AppConfig, ConfigError, SourceConfig, load_config, save_config
from fanyiji.i18n import _MESSAGES, get_language, set_language, tr
from fanyiji.identity import APP_NAME
from fanyiji.model_manager import ModelManagerDialog, ModelImportDialog
from fanyiji.model_import import ImportInspection, ImportOption
from fanyiji.models_download import ModelCatalog
from fanyiji.specialized_dialog import SpecializedModelDialog
from fanyiji.ui import MainWindow, SettingsDialog, loaded_models_summary


@pytest.fixture(autouse=True)
def language_scope():
    previous = get_language()
    set_language("zh-CN")
    yield
    set_language(previous)


@pytest.fixture
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_interface_language_round_trip_preserves_audio_settings(tmp_path, language):
    path = tmp_path / "settings.json"
    config = AppConfig(ui_language=language, microphone=SourceConfig(False, "ja_to_zh"),
                       playback=SourceConfig(True, "zh_to_ja"), multilingual_asr=False)
    save_config(path, config)
    assert load_config(path) == config


def test_legacy_configuration_defaults_to_simplified_chinese(tmp_path):
    path = tmp_path / "settings.json"
    save_config(path, AppConfig())
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("ui_language")
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_config(path).ui_language == "zh-CN"


@pytest.mark.parametrize("language", ["zh", "JA", "english", "", None, True, [], {}])
def test_invalid_language_does_not_replace_known_good_settings(tmp_path, language):
    path = tmp_path / "settings.json"
    save_config(path, AppConfig())
    before = path.read_bytes()
    with pytest.raises(ConfigError):
        save_config(path, replace(AppConfig(), ui_language=language))
    assert path.read_bytes() == before
    raw = json.loads(before)
    raw["ui_language"] = language
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_catalog_preserves_every_template_field_and_format_specifier():
    def fields(text):
        return sorted((field, spec, conversion) for _, field, spec, conversion in Formatter().parse(text)
                      if field is not None)
    for source, translations in _MESSAGES.items():
        assert len(translations) == 2
        for translated in translations:
            assert translated.strip(), source
            assert fields(translated) == fields(source), source


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_recovery_actions_translate_only_registered_actions(language):
    from fanyiji.i18n import _RECOVERY_ACTION_LABELS, recovery_action_text
    set_language(language)
    for action, source in _RECOVERY_ACTION_LABELS.items():
        assert source in _MESSAGES
        assert recovery_action_text(action) == tr(source)
        assert recovery_action_text(source) == tr(source)
    assert recovery_action_text("PRIVATE_UNKNOWN") == tr("自动处理当前失败消息")
    assert recovery_action_text(["PRIVATE_UNKNOWN"]) == tr("自动处理当前失败消息")


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_recovery_categories_and_fixed_diagnostics_have_translations(language):
    from fanyiji.i18n import _RECOVERY_CATEGORY_LABELS, recovery_category_text
    from fanyiji.danmaku_recovery import failure_decision
    set_language(language)
    for category, source in _RECOVERY_CATEGORY_LABELS.items():
        assert source in _MESSAGES
        assert recovery_category_text(category) == tr(source)
    assert recovery_category_text(["PRIVATE_UNKNOWN"]) == tr("原因未确定")
    cases = [("platform_f", {}), ("result_unknown", {}), ("platform_rejected", {}),
             ("platform_rejected", {"code": 1003212}), ("not_submitted", {"error": "not_logged_in"}),
             ("not_submitted", {"error": "csrf_missing"}), ("not_submitted", {"error": "feature_unavailable"}),
             ("not_submitted", {})]
    for reason, payload in cases:
        decision = failure_decision(reason, payload)
        assert decision.message in _MESSAGES
        if language != "zh-CN":
            assert tr(decision.message) != decision.message


def test_registered_ui_messages_have_both_translations():
    root = Path(__file__).resolve().parents[1] / "src" / "fanyiji"
    for name in ("ui.py", "specialized_dialog.py", "model_manager.py"):
        tree = ast.parse((root / name).read_bytes())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tr"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                message = node.args[0].value
                if any("\u4e00" <= char <= "\u9fff" for char in message):
                    assert message in _MESSAGES, (name, node.lineno, message)


@pytest.mark.parametrize("language", ["ja", "en"])
def test_ui_language_choice_is_saved_without_immediate_partial_switch(qt_app, language):
    config = AppConfig(playback=SourceConfig(True, "ja_to_zh"), microphone=SourceConfig(False, "zh_to_ja"))
    dialog = SettingsDialog(config)
    assert dialog.ui_language.count() == 3
    dialog.ui_language.setCurrentIndex(dialog.ui_language.findData(language))
    dialog.accept()
    assert dialog.result_config.ui_language == language
    assert dialog.result_config.playback == config.playback
    assert dialog.result_config.microphone == config.microphone
    assert get_language() == "zh-CN"
    restored = SettingsDialog(dialog.result_config)
    assert restored.ui_language.currentData() == language
    assert "重新启动" in restored.ui_language.toolTip()


@pytest.mark.parametrize("language,pause,settings_title,source_title", [
    ("en", "Pause", "Subtitle settings", "Computer audio"),
    ("ja", "一時停止", "字幕設定", "PC の音声"),
])
def test_window_settings_and_dynamic_states_follow_selected_language(qt_app, language, pause, settings_title, source_title):
    set_language(language)
    window = MainWindow(AppConfig(ui_language=language))
    assert window.pause_button.text() == pause
    assert source_title in [label.text() for label in window.findChildren(QLabel)]
    settings = SettingsDialog(AppConfig(ui_language=language))
    assert window.windowTitle() == APP_NAME
    assert APP_NAME in [label.text() for label in window.findChildren(QLabel)]
    assert settings.windowTitle() == APP_NAME + " · " + settings_title
    settings.set_codex_models([{"model": "gpt-unit", "displayName": "模型管理"}])
    assert "1" in settings.codex_models_status.text()
    assert "已读取" not in settings.codex_models_status.text()
    assert settings.codex_model.itemText(1) == "模型管理"
    window.set_session_state("running")
    assert window.session_state.text() == tr("正在听")
    window.set_source_status("system", "已连接：设备名 {原文}")
    assert window._panel("system").status_label.text() == tr("已连接：{0}", "设备名 {原文}")
    window.append_notice("界面语言已保存，重新启动软件后生效。")
    assert window.notice_label.text() == tr("界面语言已保存，重新启动软件后生效。")
    assert tr("更改设置") in [button.text() for button in window.findChildren(QPushButton)]


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_product_name_is_preserved_in_every_interface_language(language):
    set_language(language)
    assert tr(APP_NAME) == APP_NAME


@pytest.mark.parametrize("language", ["ja", "en"])
def test_subtitle_text_and_model_ids_are_never_translated(qt_app, language):
    set_language(language)
    window = MainWindow(AppConfig())
    # These exact strings also occur as catalog keys: subtitle data must stay untouched.
    window.set_original("system", "更改设置", "segment")
    window.set_translation("system", "暂停", "segment")
    row = window._panel("system")._rows["segment"]
    assert row.original.text() == "更改设置"
    assert row.translation.text() == "暂停"
    info = {"translation": {"backend": "codex_cli", "model_kind": "codex", "device": "cloud",
                            "model_id": "gpt-{unchanged}", "effort": "low", "speed": "fast"}}
    summary = loaded_models_summary(info)
    assert "gpt-{unchanged}" in summary
    assert "推理：" not in summary


def test_translation_uses_whole_registered_templates_and_preserves_values():
    set_language("en")
    assert tr("数据目录：{0}", r"D:\模型管理\{name}") == r"Data directory: D:\模型管理\{name}"
    assert tr("已连接：麦克风 {name}") == "Connected: 麦克风 {name}"
    raw = "这是字幕原文，暂停是原话中的内容。"
    assert tr(raw) == raw
    assert tr("未知模板 {value}") == "未知模板 {value}"
    with pytest.raises(ValueError):
        set_language("fr")
    assert get_language() == "en"


class _SpecializedService:
    def __init__(self, root):
        self.root = root
    def summary(self):
        return {"enabled": False, "busy": False, "status": "正在对照 模型原名 · 已保存 2 条候选",
                "samples": 1, "audio_files": 1, "local_candidates": 1, "codex_candidates": 0}
    def list_samples(self, **kwargs):
        return [{"id": "s1", "source": "更改设置", "channel": "system", "direction": "ja_to_zh",
                 "candidates": [{"id": "c1", "origin": "local", "model": "模型管理", "text": "暂停"}]}]


@pytest.mark.parametrize("language", ["ja", "en"])
def test_specialized_ui_localizes_status_and_keeps_corpus_unchanged(qt_app, tmp_path, language):
    set_language(language)
    dialog = SpecializedModelDialog(_SpecializedService(tmp_path))
    assert dialog.windowTitle() == tr("专用本地模型 · 数据与训练准备")
    assert dialog.compare_button.text() == tr("对照本地模型（自动暂停字幕）")
    assert dialog.source_text.toPlainText() == "更改设置"
    assert dialog.target_text.toPlainText() == "暂停"
    assert "模型管理" in dialog.candidates_combo.currentText()
    assert dialog.phase_label.text() == tr("{0} · 暂未开放使用", tr("正在对照 {0} · 已保存 {1} 条候选", "模型原名", "2"))
    assert "1" in dialog.summary_label.text()
    assert "样本 " not in dialog.summary_label.text()
    assert not dialog.use_button.isEnabled()


@pytest.mark.parametrize("language", ["ja", "en"])
def test_model_manager_import_labels_and_descriptions_are_localized(qt_app, tmp_path, language):
    set_language(language)
    with patch.object(ModelManagerDialog, "_start_task"):
        dialog = ModelManagerDialog(tmp_path / "模型管理", translation_required=False)
        dialog.show()
        qt_app.processEvents()
        assert dialog.windowTitle() == tr("离线模型管理")
        assert dialog.path_label.text() == str((tmp_path / "模型管理").resolve())
        assert dialog.tabs.tabText(1) == tr("识别模型")
        assert dialog.use_button.text() == tr("使用选中识别模型")
        assert dialog.import_buttons == {}
        assert dialog.import_button.text() == tr("导入本地模型")
        assert dialog.open_directory_button.text() == tr("打开模型目录")
        assert dialog.download_help_button.text() == tr("手动下载使用说明")
        assert dialog.import_button.toolTip() == tr("导入浏览器直链下载的 .gguf、ZIP 或完整模型文件夹，并自定义模型名称。")
        assert ".gguf" in dialog.import_instructions_label.text()
        assert dialog.import_instructions_label.text() != next(source for source in _MESSAGES if source.startswith("浏览器直链下载的 .gguf"))
        for step in dialog._download_help_lines():
            assert step not in _MESSAGES  # All six steps are localized.
        assert dialog.findChildren(QPushButton, "unifiedModelImport") == [dialog.import_button]
        actions = dialog.import_button.menu().actions()
        assert [action.text() for action in actions] == [tr("选择 ZIP / 模型文件"), tr("选择模型文件夹")]
        assert "已有模型包" not in dialog.import_instructions_label.text()
        assert "当前翻译源" not in dialog.instructions_label.text()
        assert tr("CPU运行；识别结果交由所选翻译源处理") in "\n".join(label.text() for label in dialog.findChildren(QLabel))
        assert set(dialog.asr_radios) == {"sensevoice-int8"}
        dialog.close()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_download_error_composed_templates_translate_reason_and_preserve_host(language):
    from fanyiji.model_manager import _status_text
    set_language(language)
    reason = "下载源暂时限流，保留临时文件，请稍后续传。"
    raw = f"下载 HTTP 429；{reason} 已尝试 4 次；保留进度，可续传或导入已下载文件。（下载主机：modelscope.cn）"
    expected_http = tr("下载 HTTP {0}；{1}", "429", tr(reason))
    expected_attempts = tr("{0} 已尝试 {1} 次；保留进度，可续传或导入已下载文件。", expected_http, "4")
    assert _status_text(raw) == tr("{0}（下载主机：{1}）", expected_attempts, "modelscope.cn")
    assert "429" in _status_text(raw) and "modelscope.cn" in _status_text(raw)
    if language != "zh-CN":
        assert "下载源暂时限流" not in _status_text(raw)


@pytest.mark.parametrize("language", ["ja", "en"])
def test_import_browser_temporary_file_diagnostics_are_localized(language):
    set_language(language)
    for text in ("文件仍是下载临时文件，请等浏览器下载完成后选择 .gguf 模型文件。",
                 "选择的是网页或网页快捷方式，不是模型权重；请下载完整 .gguf 文件后再导入。"):
        assert text in _MESSAGES and tr(text) != text
        assert ".gguf" in tr(text)


@pytest.mark.parametrize("language", ["ja", "en"])
def test_import_wizard_translates_labels_and_preserves_user_name(qt_app, tmp_path, language):
    set_language(language)
    template = next(entry for entry in ModelCatalog.default().entries if entry.model_kind == "hy")
    option = ImportOption(template, "translation", "用户模型原名")
    inspection = ImportInspection((option,), "file")
    dialog = ModelImportDialog(inspection, tmp_path / "模型来源.gguf")
    assert dialog.name_edit.text() == "用户模型原名"
    assert dialog.confirm_button.text() == tr("校验并导入")
    assert dialog.selected_option.template.name == template.name
    labels = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert tr("模型显示名称（1–80 字符）") in labels
    assert str(tmp_path / "模型来源.gguf") in labels
    dialog.name_edit.setText("")
    assert not dialog.confirm_button.isEnabled()
    assert dialog.error_label.text() == tr("模型显示名称必须为 1 到 80 个字符，不能包含控制字符。")
    dialog.close()
