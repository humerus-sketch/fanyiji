"""Web-provider settings and rendering; no website, real models or user settings."""
from dataclasses import replace
import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fanyiji.app import AppController, configured_models_available
from fanyiji.config import AppConfig, SourceConfig, load_config, save_config
from fanyiji.i18n import get_language, set_language, tr
from fanyiji.ui import MainWindow, SettingsDialog, loaded_model_display, loaded_models_summary


@pytest.fixture
def qt_app():
    app = QApplication.instance() or QApplication([])
    language = get_language()
    set_language("zh-CN")
    yield app
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.processEvents()
    set_language(language)


@pytest.mark.parametrize("provider", ["google_web", "youdao_web"])
def test_web_provider_round_trip_keeps_old_provider_settings(tmp_path, provider):
    path = tmp_path / "settings.json"
    original = AppConfig(translation_provider="hybrid", codex_model="saved-cli-model",
                         codex_effort="low", codex_speed="fast",
                         microphone=SourceConfig(False, "ja_to_zh"), playback=SourceConfig(True, "zh_to_ja"))
    selected = replace(original, translation_provider=provider)
    save_config(path, selected)
    assert load_config(path) == selected
    for old_provider in ("local", "codex", "hybrid"):
        restored = replace(load_config(path), translation_provider=old_provider)
        save_config(path, restored)
        assert load_config(path) == replace(original, translation_provider=old_provider)
    assert AppConfig().translation_provider == "local"


@pytest.mark.parametrize("provider", ["google_web", "youdao_web"])
@pytest.mark.parametrize("direction", ["ja_to_zh", "zh_to_ja"])
def test_web_provider_settings_are_added_without_changing_audio_direction(qt_app, provider, direction):
    original = AppConfig(translation_provider="codex", codex_model="saved-cli-model",
                         microphone=SourceConfig(True, direction), playback=SourceConfig(False, direction))
    dialog = SettingsDialog(original)
    dialog.show()
    qt_app.processEvents()
    assert [dialog.translation_provider.itemData(index) for index in range(dialog.translation_provider.count())] == [
        "local", "codex", "hybrid", "google_web", "youdao_web", "domestic_cli"]
    dialog.translation_provider.setCurrentIndex(dialog.translation_provider.findData(provider))
    qt_app.processEvents()
    assert not dialog.codex_options.isVisible()
    assert not dialog.hybrid_notice.isVisible()
    assert dialog.web_translation_notice.isVisible()
    assert "无需 API Key" in dialog.web_translation_notice.text()
    dialog.accept()
    selected = dialog.result_config
    assert selected.translation_provider == provider
    assert selected.microphone == original.microphone and selected.playback == original.playback
    assert selected.codex_model == original.codex_model
    restored = SettingsDialog(selected)
    assert restored.translation_provider.currentData() == provider


@pytest.mark.parametrize("provider", ["google_web", "youdao_web"])
def test_web_asset_gate_requires_asr_vad_and_enabled_speech_filter_only(tmp_path, provider):
    for name in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    (tmp_path / "translation").mkdir()
    (tmp_path / "translation/backend.json").write_text("invalid old local profile", encoding="utf-8")
    assert configured_models_available(tmp_path, speech_only=False, translation_provider=provider)
    assert not configured_models_available(tmp_path, speech_only=True, translation_provider=provider)
    assert not configured_models_available(tmp_path, speech_only=False, translation_provider="local")
    assert not configured_models_available(tmp_path, speech_only=False, translation_provider="hybrid")
    (tmp_path / "vad/silero_vad.onnx").write_bytes(b"")
    assert not configured_models_available(tmp_path, speech_only=False, translation_provider=provider)


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("provider,name", [("google_web", "Google 翻译"), ("youdao_web", "有道翻译")])
def test_loaded_web_source_and_final_label_name_actual_provider(qt_app, language, provider, name):
    set_language(language)
    metadata = {"provider": provider, "backend": "web_translation", "device": "web",
                "model_name": name, "network_required": True}
    assert loaded_model_display(metadata, "translation") == (tr(name), tr("网页（联网）"), None)
    summary = loaded_models_summary({"translation": metadata})
    assert tr(name) in summary and "Codex" not in summary
    assert loaded_model_display({**metadata, "available": False}, "translation")[0] == tr("未加载")
    window = MainWindow(AppConfig(translation_provider=provider))
    window.set_original("system", "テストです。", "test", revision=1, provisional=True)
    window.set_translation("system", "这是测试", "test", revision=1, provisional=True,
                           translation_stage="final", translation_provider=provider, translation_model=name)
    row = window._source_panels["playback"]._rows["test"]
    expected_pending = "Google 译文 · 等待后半句确认" if provider == "google_web" else "有道译文 · 等待后半句确认"
    assert row.stage_label.text() == tr(expected_pending)
    window.finalize_subtitle("system", "test", 1)
    assert row.stage_label.text() == tr("Google 终稿" if provider == "google_web" else "有道终稿")


@pytest.mark.parametrize("provider,name", [("google_web", "Google"), ("youdao_web", "有道")])
def test_controller_preparing_and_idle_queue_state_name_web_source(provider, name):
    window = SimpleNamespace(set_session_state=Mock())
    controller = SimpleNamespace(closed=False, epoch=0, config=AppConfig(translation_provider=provider),
        _danmaku_startup_checked=True, danmaku=None, chrome_danmaku=None, latency=SimpleNamespace(begin_session=Mock()),
        window=window, paused=False, active_models={}, _set_model_state=Mock(),
        executor=SimpleNamespace(submit=Mock()), _transition=Mock(), _schedule_transition=Mock())
    AppController.start(controller)
    assert window.set_session_state.call_args.args == ({
        "google_web": "正在准备本地识别和 Google 网页翻译…",
        "youdao_web": "正在准备本地识别和有道网页翻译…"}[provider],)
    AppController._on_event(controller, {"session": controller.epoch, "kind": "translation_queue",
                                        "pending": 0, "oldest_age_seconds": 0})
    text = window.set_session_state.call_args.args[0]
    assert name in text and "网页翻译" in text and "Codex" not in text and "完全本地" not in text


@pytest.mark.parametrize("provider", ["google_web", "youdao_web"])
@pytest.mark.parametrize("status_event", [False, True])
def test_web_failure_never_claims_codex_or_local_draft(qt_app, provider, status_event):
    window = MainWindow(AppConfig(translation_provider=provider))
    window.set_original("system", "テストです。", "failed")
    if status_event:
        window.set_translation_status("system", "failed", translation_stage="final",
                                      translation_provider=provider, failed=True, overall_failed=True)
    else:
        window.set_translation("system", "（翻译未完成）", "failed", translation_stage="final",
                               translation_provider=provider, failed=True, overall_failed=True)
    row = window._source_panels["playback"]._rows["failed"]
    assert row.stage_label.text() == tr("Google 翻译未完成" if provider == "google_web" else "有道翻译未完成")


def test_translation_page_button_tracks_selected_provider_and_emits(qt_app):
    window = MainWindow(AppConfig())
    window.show()
    qt_app.processEvents()
    calls = []
    window.translation_page_requested.connect(lambda: calls.append(True))
    for provider in ("local", "codex", "hybrid", "google_web", "youdao_web", "local"):
        window.update_config(AppConfig(translation_provider=provider))
        qt_app.processEvents()
        assert window.translation_page_button.isVisible() is (provider in ("google_web", "youdao_web"))
        if window.translation_page_button.isVisible():
            window.translation_page_button.click()
    assert calls == [True, True]


@pytest.mark.parametrize("provider", ["google_web", "youdao_web"])
@pytest.mark.parametrize("opened", [False, True])
def test_controller_opens_web_translation_page_without_creating_live_browser(provider, opened):
    controller = SimpleNamespace(closed=False, config=AppConfig(translation_provider=provider),
        window=SimpleNamespace(append_notice=Mock()), pipeline=SimpleNamespace(show_translation_window=Mock(return_value=opened)),
        _ensure_browser_service=Mock())
    AppController.open_translation_page(controller)
    controller.pipeline.show_translation_window.assert_called_once_with()
    controller._ensure_browser_service.assert_not_called()
    if opened:
        controller.window.append_notice.assert_not_called()
    else:
        controller.window.append_notice.assert_called_once_with(tr("网页翻译尚未准备好，请等待识别和翻译启动。"))


@pytest.mark.parametrize("provider,closed", [("local", False), ("codex", False), ("hybrid", False), ("google_web", True)])
def test_controller_does_not_open_translation_page_for_non_web_or_closed(provider, closed):
    controller = SimpleNamespace(closed=closed, config=AppConfig(translation_provider=provider),
                                 pipeline=SimpleNamespace(show_translation_window=Mock()))
    AppController.open_translation_page(controller)
    controller.pipeline.show_translation_window.assert_not_called()


def test_controller_reports_only_safe_translation_page_error():
    controller = SimpleNamespace(closed=False, config=AppConfig(translation_provider="google_web"),
        pipeline=SimpleNamespace(show_translation_window=Mock(side_effect=RuntimeError("PRIVATE ERROR"))),
        window=SimpleNamespace(append_notice=Mock()))
    AppController.open_translation_page(controller)
    controller.window.append_notice.assert_called_once_with(tr("网页翻译页面无法打开，请检查 WebView2。"))


@pytest.mark.parametrize("provider", ["google_web", "youdao_web"])
@pytest.mark.parametrize("checked", [False, True])
def test_web_fixed_language_preserves_multilingual_choice_for_other_sources(qt_app, provider, checked):
    config = AppConfig(translation_provider=provider, multilingual_asr=checked)
    dialog = SettingsDialog(config)
    assert not dialog.multilingual_asr.isEnabled()
    assert dialog.multilingual_asr.isChecked() is checked
    assert tr("网页模式按通道方向使用固定识别语言，仅支持中日互译。") in dialog.web_translation_notice.text()
    assert tr("Google 翻译需保持翻译窗口可见；有道翻译可在后台运行。") in dialog.web_translation_notice.text()
    for source in ("local", "codex", "hybrid"):
        dialog.translation_provider.setCurrentIndex(dialog.translation_provider.findData(source))
        assert dialog.multilingual_asr.isEnabled()
        assert dialog.multilingual_asr.isChecked() is checked
        dialog.translation_provider.setCurrentIndex(dialog.translation_provider.findData(provider))
        assert not dialog.multilingual_asr.isEnabled()
        assert dialog.multilingual_asr.isChecked() is checked
    dialog.accept()
    assert dialog.result_config.multilingual_asr is checked
