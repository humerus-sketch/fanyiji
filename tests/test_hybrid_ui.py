"""Hybrid rendering and settings contracts without real accounts or audio."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fanyiji.app import AppController, configured_models_available
from fanyiji.config import AppConfig, load_config, save_config
from fanyiji.i18n import get_language, set_language, tr
from fanyiji.ui import MainWindow, SettingsDialog, loaded_models_summary


@pytest.fixture
def qt_app():
    app = QApplication.instance() or QApplication([])
    previous = get_language()
    set_language("zh-CN")
    yield app
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.processEvents()
    set_language(previous)


def _window():
    window = MainWindow(AppConfig(translation_provider="hybrid"))
    window.set_original("system", "今は行かない。", "a", context_generation=3)
    return window, window._source_panels["playback"], window._source_panels["playback"]._rows["a"]


def _result(window, text, stage, **options):
    window.set_translation("system", text, "a", translation_stage=stage,
                           translation_provider="local" if stage == "draft" else "codex",
                           translation_model="hy-test" if stage == "draft" else "codex-test",
                           context_generation=3, **options)


def _fail(window, stage="final", **options):
    window.set_translation_status("system", "a", translation_stage=stage,
                                  translation_provider="codex" if stage == "final" else "local",
                                  failed=True, context_generation=3, error="fixture failure", **options)


@pytest.mark.parametrize("language,retry_text,counter_text", [
    ("zh-CN", "自动重试", "f 自动重试累计 3 次"),
    ("ja", "自動再送", "f 自動再送合計 3 回"),
    ("en", "retry", "Total f retries 3")])
@pytest.mark.parametrize("state", ["waiting_f_retry", "retrying_f"])
def test_main_window_reports_f_retry_progress_and_wait(qt_app, language, retry_text, counter_text, state):
    set_language(language)
    window = MainWindow(AppConfig(ui_language=language))
    window.set_danmaku_status({"enabled": True, "state": state, "queued": 5,
        "page_visible": 2, "f_retry_active": True, "f_retries": 3, "cooldown_remaining": 1.5})
    text = window._danmaku_status_label.text()
    assert retry_text in text
    assert counter_text in text
    assert "1.5" in text
    assert "暂停" not in text


def test_main_window_unknown_after_f_retry_is_not_described_as_automatic(qt_app):
    window, _, _ = _window()
    window.set_danmaku_status({"enabled": True, "state": "waiting_f_retry", "queued": 1,
        "f_retry_active": True, "f_retries": 1, "cooldown_remaining": 2.5})
    window.set_danmaku_status({"enabled": True, "state": "result_unknown", "queued": 1,
        "f_retry_active": False, "f_retries": 1, "cooldown_remaining": 1.0})
    text = window._danmaku_status_label.text()
    assert "结果未知" in text
    assert "等待核实" in text
    assert "还需等待" not in text


def test_main_window_filter_error_and_candidate_warning_are_distinct(qt_app):
    window, _, _ = _window()
    window.set_danmaku_status({"enabled": True, "state": "filter_error", "queued": 4,
        "filter_error": "PRIVATE_RULE_ERROR", "filter_rule_count": 2, "filter_candidate_count": 3})
    text = window._danmaku_status_label.text()
    assert "过滤名单异常，待发消息保留" in text
    assert "过滤规则 2 · 未处理失败 3" in text
    assert "PRIVATE_RULE_ERROR" not in text
    window.set_danmaku_status({"enabled": True, "state": "waiting_f_retry", "queued": 4,
        "f_retry_active": True, "filter_candidate_error": "PRIVATE_CANDIDATE_ERROR"})
    text = window._danmaku_status_label.text()
    assert "自动重试" in text
    assert "失败候选保存失败" in text
    assert "PRIVATE_CANDIDATE_ERROR" not in text


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("state,description", [
    ("auto_verifying", "发送结果未知，正在自动只读核验"),
    ("auto_rechecking", "连续发送失败，正在自动复检账号和页面"),
    ("auto_retry", "正在自动恢复当前条，保留队列顺序")])
def test_main_window_reports_recovery_progress_and_archived_record_counts(qt_app, language, state, description):
    set_language(language)
    window = MainWindow(AppConfig(ui_language=language))
    window.set_danmaku_status({"enabled": True, "state": state, "queued": 2,
        "filter_rule_count": 1, "filter_candidate_count": 1, "filter_resolved_count": 7})
    text = window._danmaku_status_label.text()
    assert tr(description) in text
    assert tr(" · 过滤规则 {0} · 未处理失败 {1}", 1, 1) in text
    assert tr(" · 已处理历史 {0}", 7) in text
    assert "auto_" not in text
    assert "自动送达" not in text


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("state,description", [
    ("ai_analyzing", "正在请求 Codex 分析失败正文"),
    ("ai_repair_ready", "修改后正文已通过本地检查，等待独立测试验证"),
    ("ai_retrying", "正在按发送间隔重试测试正文")])
def test_main_window_reports_ai_recovery_without_rendering_arbitrary_model_content(qt_app, language, state, description):
    set_language(language)
    window = MainWindow(AppConfig(ui_language=language))
    window.set_danmaku_status({"enabled": True, "state": state, "queued": 2,
        "recovery_category": "ai_review", "recovery_action": "retry_current"})
    text = window._danmaku_status_label.text()
    assert tr(description) in text
    assert tr(" · 处置 {0}", tr("按发送间隔自动重试当前条")) in text
    assert "retry_current" not in text
    window.set_danmaku_status({"enabled": True, "state": state, "recovery_action": "PRIVATE_TEXT"})
    assert "PRIVATE_TEXT" not in window._danmaku_status_label.text()


@pytest.mark.parametrize("state", ["ai_analyzing", "waiting_validation", "validating", "validation_failed",
    "waiting_validation_browser", "validation_storage_error", "validation_passed"])
def test_startup_analysis_and_validation_progress_is_visible_without_enabled_production_sender(qt_app, state):
    window = MainWindow(AppConfig())
    window.set_danmaku_status({"enabled": False, "state": state, "queued": 0})
    assert not window._danmaku_status_label.isHidden()
    assert state not in window._danmaku_status_label.text()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("state", ["failure_transferred", "validation_configuration_changed"])
def test_independent_failure_processing_is_visible_with_disabled_production_sender(qt_app, language, state):
    set_language(language)
    window = MainWindow(AppConfig(ui_language=language))
    window.set_danmaku_status({"enabled": False, "state": state, "queued": 0, "deferred_analysis": 2})
    assert not window._danmaku_status_label.isHidden()
    assert state not in window._danmaku_status_label.text()
    assert tr(" · 累计移入独立处理 {0} 段", 2) in window._danmaku_status_label.text()


def test_hybrid_settings_round_trip(tmp_path):
    config = AppConfig(translation_provider="hybrid", codex_model="codex-test", codex_effort="low", codex_speed="fast")
    path = tmp_path / "settings.json"
    save_config(path, config)
    assert load_config(path) == config


def test_settings_exposes_hybrid_and_preserves_cloud_selection(qt_app):
    config = AppConfig(translation_provider="hybrid", codex_model="codex-test", codex_effort="low", codex_speed="fast")
    dialog = SettingsDialog(config)
    assert dialog.translation_provider.currentData() == "hybrid"
    assert not dialog.codex_options.isHidden()
    assert not dialog.hybrid_notice.isHidden()
    dialog.accept()
    assert dialog.result_config == config
    dialog.translation_provider.setCurrentIndex(dialog.translation_provider.findData("local"))
    assert dialog.codex_options.isHidden()
    assert dialog.hybrid_notice.isHidden()


def test_draft_is_marked_pending_then_final_replaces_same_row(qt_app):
    window, panel, row = _window()
    _result(window, "现在不去", "draft")
    assert row.translation.text() == "现在不去"
    assert "暂译" in row.stage_label.text()
    assert not row.completed
    _result(window, "我现在不去。", "final")
    assert row.translation.text() == "我现在不去。"
    assert row.stage_label.text() == "Codex 终稿"
    assert row.completed
    assert list(panel._rows) == ["a"]


def test_final_first_rejects_late_draft(qt_app):
    window, _, row = _window()
    _result(window, "终稿", "final")
    _result(window, "迟到暂译", "draft")
    assert row.translation.text() == "终稿"
    assert row.translation_stage == "final"


@pytest.mark.parametrize("failure_first", [False, True])
def test_cloud_failure_retains_local_text_and_marks_unconfirmed(qt_app, failure_first):
    window, _, row = _window()
    if failure_first:
        _fail(window)
    _result(window, "本地可用译文", "draft")
    if not failure_first:
        _fail(window)
    assert row.translation.text() == "本地可用译文"
    assert row.stage_label.text() == "本地暂译 · Codex 未确认"
    assert row.translation_stage == "draft"
    assert row.completed


def test_failure_flag_on_draft_is_sufficient_without_prior_status(qt_app):
    window, _, row = _window()
    _result(window, "可用暂译", "draft", final_failed=True)
    assert row.translation.text() == "可用暂译"
    assert "未确认" in row.stage_label.text()


def test_local_failure_does_not_prevent_cloud_final(qt_app):
    window, _, row = _window()
    _fail(window, "draft")
    assert "等待 Codex" in row.stage_label.text()
    _result(window, "可靠终稿", "final")
    assert row.translation.text() == "可靠终稿"
    assert row.completed


def test_context_generation_and_revision_must_match_original(qt_app):
    window, _, row = _window()
    window.set_translation("system", "旧上下文", "a", translation_stage="final", context_generation=2)
    window.set_translation("system", "未知新版本", "a", translation_stage="final", revision=2, context_generation=3)
    assert row.translation.text() == "翻译中…"
    _result(window, "暂译", "draft")
    window.set_original("system", "後半句。", "a", revision=2, context_generation=3)
    _result(window, "旧终稿", "final")
    assert row.translation.text() == "翻译中…"
    assert row.stage_label.isHidden()
    _result(window, "新暂译", "draft", revision=2)
    assert row.translation.text() == "新暂译"


def test_old_status_cannot_mark_new_revision_failed(qt_app):
    window, _, row = _window()
    window.set_original("system", "改訂", "a", revision=2, context_generation=4)
    _fail(window)
    assert not row.final_failed
    assert row.stage_label.isHidden()


def test_source_confirmation_does_not_promote_draft(qt_app):
    window, _, row = _window()
    window.set_original("system", "言いかけ", "a", revision=2, provisional=True, context_generation=3)
    _result(window, "半句暂译", "draft", revision=2, provisional=True)
    window.finalize_subtitle("system", "a", 2)
    assert not row.completed
    assert row.translation_stage == "draft"


@pytest.mark.parametrize("failure_first", [False, True])
def test_failed_cloud_half_sentence_waits_for_source_confirmation(qt_app, failure_first):
    window, _, row = _window()
    window.set_original("system", "言いかけ", "a", revision=2, provisional=True, context_generation=3)
    if failure_first:
        _fail(window, revision=2)
    _result(window, "半句暂译", "draft", revision=2, provisional=True)
    if not failure_first:
        _fail(window, revision=2)
    assert not row.completed
    window.finalize_subtitle("system", "a", 2)
    assert row.completed
    assert row.translation_stage == "draft"


def test_pending_draft_survives_completed_history_pruning(qt_app):
    window, panel, row = _window()
    _result(window, "待终稿", "draft")
    for index in range(8):
        window.set_original("system", "source", index)
        window.set_translation("system", "final", index)
    assert panel._rows["a"] is row
    assert not row.completed
    _result(window, "完成", "final")
    assert panel._rows["a"] is row
    assert len(panel._rows) == 5


def test_cancelled_or_removed_session_row_rejects_late_results(qt_app):
    window, panel, row = _window()
    _result(window, "暂译", "draft")
    window.cancel_pending()
    _result(window, "旧终稿", "final")
    _fail(window)
    assert row.cancelled
    assert row.translation.text() == "已暂停，此片段已取消"
    window.update_config(AppConfig(translation_provider="hybrid"))
    _result(window, "旧会话", "final")
    assert not panel._rows


def test_both_failed_has_clear_failure_state(qt_app):
    window, _, row = _window()
    _fail(window, "draft")
    _fail(window, overall_failed=True)
    assert row.translation.text() == "（翻译未完成）"
    assert row.stage_label.text() == "本地与 Codex 翻译均未完成"
    assert row.completed


def test_failure_placeholder_is_marked_as_failure_instead_of_success_final(qt_app):
    window, _, row = _window()
    _result(window, "（翻译未完成）", "final", failed=True, overall_failed=True)
    assert row.stage_label.text() == "本地与 Codex 翻译均未完成"
    assert row.completed


@pytest.mark.parametrize("provider", ["local", "codex"])
@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("provisional", [False, True])
def test_standalone_failure_is_never_labelled_as_successful_final(qt_app, provider, language, provisional):
    set_language(language)
    window = MainWindow(AppConfig(translation_provider=provider))
    window.set_original("system", "source", "failed", provisional=provisional)
    window.set_translation("system", "fixture failure placeholder", "failed", provisional=provisional,
                           translation_stage="final", translation_provider=provider, failed=True)
    row = window._source_panels["playback"]._rows["failed"]
    assert row.translation.text() == "fixture failure placeholder"
    assert row.stage_label.text() == tr("翻译未完成")
    assert "#e4b96a" in row.stage_label.styleSheet()
    assert row.completed is not provisional


def test_future_source_confirmation_does_not_finalize_current_draft(qt_app):
    window, _, row = _window()
    _result(window, "暂译", "draft")
    window.finalize_subtitle("system", "a", 2)
    assert row.confirmed_revision == 1
    assert not row.completed


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_hybrid_model_summary_names_both_loaded_models(qt_app, language):
    set_language(language)
    summary = loaded_models_summary({"translation": {
        "mode": "hybrid", "backend": "codex_cli", "model_id": "codex-test", "auth_mode": "chatgpt",
        "draft_model": {"model_kind": "hy", "model_file": "local-test.gguf", "device": "cuda"}}})
    assert "codex-test" in summary
    assert "local-test.gguf" in summary
    assert "CUDA" in summary
    if language != "zh-CN":
        assert "暂译" not in summary and "终稿" not in summary


def test_unavailable_cloud_is_not_shown_as_loaded(qt_app):
    summary = loaded_models_summary({"translation": {
        "mode": "hybrid", "backend": "codex_cli", "model_id": "codex-test", "available": False,
        "draft_model": {"model_kind": "hy", "model_file": "local-test.gguf", "device": "cpu"}}})
    assert "终稿：未加载" in summary
    assert "local-test.gguf" in summary


def test_hybrid_asset_gate_requires_local_translation_weights(tmp_path):
    for name in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    assert configured_models_available(tmp_path, speech_only=False, translation_provider="codex")
    assert not configured_models_available(tmp_path, speech_only=False, translation_provider="hybrid")
    translation = tmp_path / "translation"
    translation.mkdir()
    (translation / "backend.json").write_text(json.dumps({"backend": "llama_cpp", "model_file": "fixture.gguf"}))
    (translation / "fixture.gguf").write_bytes(b"fixture")
    assert configured_models_available(tmp_path, speech_only=False, translation_provider="hybrid")


def test_controller_forwards_stages_status_and_rejects_old_session():
    window = SimpleNamespace(set_original=Mock(), set_translation=Mock(), set_translation_status=Mock())
    controller = SimpleNamespace(closed=False, epoch=8, window=window, danmaku=None)
    original = {"session": 8, "kind": "original", "channel": "system", "segment_id": "a", "text": "source",
                "revision": 2, "context_generation": 4}
    AppController._on_event(controller, original)
    assert window.set_original.call_args.kwargs["context_generation"] == 4
    translated = {**original, "kind": "translation", "text": "draft", "translation_stage": "draft",
                  "translation_provider": "local", "translation_model": "hy-test", "final_failed": True}
    AppController._on_event(controller, translated)
    assert window.set_translation.call_args.kwargs["translation_stage"] == "draft"
    assert window.set_translation.call_args.kwargs["final_failed"] is True
    status = {**translated, "kind": "translation_status", "failed": True, "error": "fixture"}
    AppController._on_event(controller, status)
    assert window.set_translation_status.call_args.kwargs["failed"] is True
    AppController._on_event(controller, {**translated, "session": 7})
    AppController._on_event(controller, {**translated, "kind": "translation_candidate"})
    assert window.set_translation.call_count == 1
