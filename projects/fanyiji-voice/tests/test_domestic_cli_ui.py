"""Account-CLI configuration and Qt controls using only temporary data and fake services."""
from dataclasses import replace
import json
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fanyiji.app import AppController, configured_models_available
from fanyiji.config import AppConfig, ConfigError, SourceConfig, load_config, save_config
from fanyiji.domestic_cli_catalog import BLOCKED_PROVIDERS, CLI_NAMES, CLI_PROVIDERS
from fanyiji.domestic_cli_dialog import DomesticCliManagerDialog, _RUNNING_WORKERS
from fanyiji.i18n import get_language, set_language, tr
from fanyiji.ui import MainWindow, SettingsDialog, loaded_model_display, loaded_models_summary


def events_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(5)
    assert predicate()


@pytest.fixture
def qt_app():
    app = QApplication.instance() or QApplication([])
    language = get_language()
    set_language("zh-CN")
    yield app
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    events_until(lambda: not _RUNNING_WORKERS)
    app.processEvents()
    set_language(language)


def ready_probe(provider="minimax", **overrides):
    return {"provider": provider, "status": "ready", "message": "synthetic account ready", "installed": True,
            "version": "test-version", "auth_mode": "oauth", "models": [{"id": "actual-model", "name": "Actual model"}],
            "capabilities": {"account_auth": True, "tool_free": True, "protocol": "json"}, **overrides}


def test_unconfirmed_cli_model_is_explicit_and_not_saved_selection(qt_app):
    metadata = {"provider": "domestic_cli", "cli_provider": "kimi", "device": "cloud",
                "model_id": "", "requested_model": "selected-only", "model_confirmed": False}
    name, _, _ = loaded_model_display(metadata, "translation")
    assert "模型待确认" in name and "selected-only" not in name


def test_controller_refreshes_cli_model_and_rejects_another_provider():
    controller = SimpleNamespace(closed=False, epoch=1, danmaku=None,
        config=AppConfig(translation_provider="domestic_cli", domestic_cli_provider="minimax"),
        active_models={"translation": {"worker_count": 1}}, model_state="ready", _set_model_state=Mock())
    event = {"session": 1, "kind": "translation_backend", "information": {
        "provider": "domestic_cli", "cli_provider": "minimax", "model_id": "observed-model"}}
    AppController._on_event(controller, event)
    models = controller._set_model_state.call_args.args[1]
    assert models["translation"]["model_id"] == "observed-model" and models["translation"]["worker_count"] == 1
    controller._set_model_state.reset_mock()
    AppController._on_event(controller, {**event, "information": {**event["information"], "cli_provider": "kimi"}})
    controller._set_model_state.assert_not_called()


def fake_service(result=None):
    return SimpleNamespace(probe=Mock(return_value=result or ready_probe()), login=Mock(),
        official_url=Mock(return_value="https://example.com/official"),
        check_translation=Mock(return_value={"translation": "今天稍微调低游戏音量", "seconds": 0.25, "model": "actual-model"}))


@pytest.mark.parametrize("provider", CLI_PROVIDERS)
@pytest.mark.parametrize("mode", ["domestic_cli", "hybrid"])
def test_domestic_settings_round_trip_and_switches_preserve_existing_settings(tmp_path, provider, mode):
    config = AppConfig(translation_provider=mode, domestic_cli_provider=provider, domestic_cli_model="selected-model",
        hybrid_final_provider="domestic_cli", codex_model="saved-codex-model", codex_speed="fast", codex_effort="low",
        microphone=SourceConfig(False, "ja_to_zh"), playback=SourceConfig(True, "zh_to_ja"))
    path = tmp_path / "config.json"
    save_config(path, config)
    assert load_config(path) == config
    for old_mode in ("local", "codex", "google_web", "youdao_web"):
        other = replace(config, translation_provider=old_mode)
        save_config(path, other)
        assert load_config(path) == other


def test_legacy_hybrid_keeps_codex_default_and_local_default_is_unchanged(tmp_path):
    path = tmp_path / "config.json"
    save_config(path, AppConfig(translation_provider="hybrid"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    for name in ("domestic_cli_provider", "domestic_cli_model", "hybrid_final_provider"):
        del raw[name]
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.hybrid_final_provider == "codex"
    assert loaded.domestic_cli_provider == "minimax" and loaded.domestic_cli_model == ""
    assert AppConfig().translation_provider == "local"


@pytest.mark.parametrize("changes", [{"domestic_cli_provider": "unknown"}, {"domestic_cli_provider": []},
    {"domestic_cli_model": "x" * 129}, {"domestic_cli_model": "model\nother"}, {"domestic_cli_model": None},
    {"hybrid_final_provider": "google_web"}, {"hybrid_final_provider": []}])
def test_invalid_domestic_settings_do_not_replace_saved_config(tmp_path, changes):
    path = tmp_path / "config.json"
    save_config(path, AppConfig())
    before = path.read_bytes()
    with pytest.raises(ConfigError):
        save_config(path, replace(AppConfig(), **changes))
    assert path.read_bytes() == before


def test_settings_hybrid_source_switch_preserves_both_accounts_and_audio_options(qt_app):
    config = AppConfig(translation_provider="hybrid", codex_model="saved-codex", domestic_cli_provider="kimi",
                       domestic_cli_model="saved-kimi", microphone=SourceConfig(False, "ja_to_zh"))
    dialog = SettingsDialog(config)
    dialog.show()
    qt_app.processEvents()
    assert dialog.hybrid_final_provider.currentData() == "codex"
    assert dialog.codex_options.isVisible() and not dialog.domestic_cli_options.isVisible()
    dialog.hybrid_final_provider.setCurrentIndex(dialog.hybrid_final_provider.findData("domestic_cli"))
    assert dialog.domestic_cli_options.isVisible() and not dialog.codex_options.isVisible()
    assert "Codex" not in dialog.hybrid_notice.text()
    dialog.translation_provider.setCurrentIndex(dialog.translation_provider.findData("local"))
    assert not dialog.domestic_cli_options.isVisible() and not dialog.hybrid_final_options.isVisible()
    dialog.accept()
    saved = dialog.result_config
    assert saved.translation_provider == "local" and saved.hybrid_final_provider == "domestic_cli"
    assert saved.codex_model == "saved-codex" and saved.domestic_cli_model == "saved-kimi"
    assert saved.microphone == config.microphone


def test_domestic_setting_disables_qwen_and_management_signal_does_not_claim_ready(qt_app):
    dialog = SettingsDialog(AppConfig(translation_provider="domestic_cli"))
    index = dialog.domestic_cli_provider.findData("qwen")
    assert not dialog.domestic_cli_provider.model().item(index).isEnabled()
    assert dialog.domestic_cli_provider.model().item(index).toolTip() == tr(BLOCKED_PROVIDERS["qwen"])
    assert "尚未检查" in dialog.domestic_cli_status.text()
    requests = []
    dialog.domestic_cli_requested.connect(lambda *values: requests.append(values))
    dialog.domestic_cli_manage_button.click()
    assert requests == [("minimax", "")]


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_actual_domestic_provider_is_used_for_model_draft_final_and_error_labels(qt_app, language):
    set_language(language)
    metadata = {"provider": "domestic_cli", "cli_provider": "kimi", "model_id": "actual-model", "device": "cloud"}
    name, device, identifier = loaded_model_display(metadata, "translation")
    assert name == "Kimi Code CLI · actual-model" and device == tr("云端 · 账号登录") and identifier is None
    summary = loaded_models_summary({"translation": {**metadata, "mode": "hybrid",
        "draft_model": {"model_kind": "hy", "model_file": "draft.gguf", "device": "cpu"}}})
    assert "Kimi Code CLI" in summary and "Codex" not in summary
    config = AppConfig(translation_provider="hybrid", hybrid_final_provider="domestic_cli", domestic_cli_provider="kimi")
    window = MainWindow(config)
    window.set_original("system", "テストです。", "sample")
    window.set_translation("system", "本地暂译", "sample", translation_stage="draft", translation_provider="local")
    row = window._source_panels["playback"]._rows["sample"]
    assert row.stage_label.text() == tr("本地暂译 · 等待 {0} 终稿", "Kimi Code CLI")
    window.set_translation("system", "终稿", "sample", translation_stage="final", translation_provider="domestic_cli",
                           translation_cli_provider="kimi", translation_model="actual-model")
    assert row.stage_label.text() == tr("{0} 终稿", "Kimi Code CLI")
    window.set_original("system", "次のテスト。", "error")
    window.set_translation_status("system", "error", translation_stage="final", translation_provider="domestic_cli",
                                  translation_cli_provider="kimi", failed=True, overall_failed=True)
    assert "Codex" not in window._source_panels["playback"]._rows["error"].stage_label.text()


def test_domestic_model_asset_gate_still_requires_asr_and_vad_but_hybrid_requires_mt(tmp_path):
    for name in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    assert configured_models_available(tmp_path, speech_only=False, translation_provider="domestic_cli")
    assert not configured_models_available(tmp_path, speech_only=True, translation_provider="domestic_cli")
    assert not configured_models_available(tmp_path, speech_only=False, translation_provider="hybrid")


def test_controller_preserves_actual_cli_provider_metadata_for_final_and_failure():
    controller = SimpleNamespace(closed=False, epoch=1, danmaku=None, config=AppConfig(),
        window=SimpleNamespace(set_translation=Mock(), set_translation_status=Mock()))
    event = {"session": 1, "kind": "translation", "channel": "system", "text": "译文", "segment_id": "sample",
             "translation_provider": "domestic_cli", "translation_cli_provider": "kimi", "translation_stage": "final"}
    AppController._on_event(controller, event)
    assert controller.window.set_translation.call_args.kwargs["translation_cli_provider"] == "kimi"
    AppController._on_event(controller, {**event, "kind": "translation_status", "failed": True})
    assert controller.window.set_translation_status.call_args.kwargs["translation_cli_provider"] == "kimi"


def test_management_starts_without_network_or_model_calls_and_qwen_is_blocked(qt_app):
    service = fake_service()
    dialog = DomesticCliManagerDialog(service=service)
    service.probe.assert_not_called()
    service.login.assert_not_called()
    service.check_translation.assert_not_called()
    assert not dialog.use_button.isEnabled()
    dialog = DomesticCliManagerDialog(provider="qwen", service=service)
    assert not dialog.check_button.isEnabled() and not dialog.login_button.isEnabled() and not dialog.test_button.isEnabled()
    assert dialog.status.text() == tr(BLOCKED_PROVIDERS["qwen"])
    dialog._start("test")
    service.check_translation.assert_not_called()


@pytest.mark.parametrize("status", ["not_installed", "not_logged_in", "auth_unverified", "unsupported", "api_key_blocked"])
def test_management_never_enables_usage_for_unverified_or_key_based_provider(qt_app, status):
    service = fake_service(ready_probe(status=status))
    dialog = DomesticCliManagerDialog(provider="codebuddy", service=service)
    dialog.check_button.click()
    events_until(lambda: dialog._thread is None)
    assert not dialog.use_button.isEnabled()
    assert dialog.last_probe["status"] == status


def test_probe_and_login_are_worker_operations_and_only_real_capabilities_enable_choice(qt_app):
    observed = []
    service = fake_service()
    def probe(*args, **kwargs):
        observed.append(QThread.currentThread() != QApplication.instance().thread())
        return ready_probe()
    service.probe.side_effect = probe
    dialog = DomesticCliManagerDialog(service=service, model="saved-model")
    dialog.check_button.click()
    events_until(lambda: dialog._thread is None)
    assert observed == [True]
    assert dialog.use_button.isEnabled()
    assert dialog.model.findData("actual-model") > 0
    assert dialog._selected_model() == "saved-model"
    dialog.login_button.click()
    events_until(lambda: dialog._thread is None)
    service.login.assert_called_once()
    assert service.probe.call_count == 2
    dialog.accept()
    assert dialog.result_choice == ("minimax", "saved-model")


def test_trial_sends_only_after_user_click_and_uses_selected_model(qt_app):
    service = fake_service()
    dialog = DomesticCliManagerDialog(provider="kimi", model="chosen-model", service=service)
    service.check_translation.assert_not_called()
    dialog.test_button.click()
    events_until(lambda: dialog._thread is None)
    args = service.check_translation.call_args.args
    assert args == ("kimi", "chosen-model", "今日はゲームの音量を少し下げます。")
    assert "今天稍微调低游戏音量" in dialog.test_result.text()
    assert not dialog.use_button.isEnabled()  # A translation result is not an account-capability check.


def test_official_link_opens_only_when_clicked(qt_app, monkeypatch):
    service = fake_service()
    opened = []
    monkeypatch.setattr("fanyiji.domestic_cli_dialog.QDesktopServices.openUrl", lambda url: opened.append(url.toString()))
    dialog = DomesticCliManagerDialog(service=service)
    assert not opened
    dialog.download_button.click()
    assert opened == ["https://example.com/official"]


def test_close_cancels_worker_and_drops_late_account_result(qt_app):
    started = threading.Event()
    cancelled = threading.Event()
    service = fake_service()
    def delayed_probe(provider, *, cancellation_event):
        started.set()
        assert cancellation_event.wait(2)
        cancelled.set()
        return ready_probe()
    service.probe.side_effect = delayed_probe
    dialog = DomesticCliManagerDialog(service=service)
    dialog.check_button.click()
    events_until(started.is_set)
    dialog.reject()
    events_until(cancelled.is_set)
    events_until(lambda: not _RUNNING_WORKERS)
    assert dialog.last_probe is None and not dialog.use_button.isEnabled()


def test_application_shutdown_joins_cancelled_worker_without_gui_delivery(qt_app):
    from fanyiji.domestic_cli_dialog import _shutdown_workers
    started = threading.Event()
    service = fake_service()
    def waiting_probe(provider, *, cancellation_event):
        started.set()
        assert cancellation_event.wait(2)
        return ready_probe()
    service.probe.side_effect = waiting_probe
    dialog = DomesticCliManagerDialog(service=service)
    dialog.check_button.click()
    events_until(started.is_set)
    thread = dialog._thread
    _shutdown_workers()
    assert not thread.isRunning()
    events_until(lambda: not _RUNNING_WORKERS)
    assert dialog.last_probe is None


def test_repeated_worker_completion_keeps_native_thread_until_joined(qt_app):
    for _ in range(12):
        dialog = DomesticCliManagerDialog(service=fake_service(ready_probe(status="not_logged_in")))
        dialog.check_button.click()
        events_until(lambda: dialog._thread is None and not _RUNNING_WORKERS)
        dialog.reject()
        dialog.deleteLater()
        QApplication.processEvents()
