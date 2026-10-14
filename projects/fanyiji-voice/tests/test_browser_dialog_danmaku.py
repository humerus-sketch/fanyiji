"""Push settings must be explicitly saved; UI tests never open a live website."""
import ast
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from fanyiji.browser_dialog import BrowserDialog
from fanyiji.browser_service import room_url
from fanyiji.i18n import _MESSAGES, get_language, set_language


class BrowserStub(QObject):
    event = Signal(dict)
    snapshot = Signal(dict)
    prefs_changed = Signal(dict)
    danmaku_status = Signal(dict)

    def __init__(self):
        super().__init__()
        self.room = "1698225"
        self.auto_check = False
        self.active_version = None
        self.danmaku_available = True
        self.danmaku_enabled = False
        self.danmaku_repair_enabled = False
        self.danmaku_source = "system"
        self.danmaku_interval_seconds = 3.5
        self.danmaku_validation_room = ""
        self.danmaku_snapshot = {}
        self.saved = []
        self.opened = []
        self.resumed = 0
        self.clear_requested = 0
        self.filter_requested = 0

    def save_preferences(self, room, auto_check, **updates):
        if room.strip():
            room = room_url(room).rsplit("/", 1)[-1]
        validation = updates.get("danmaku_validation_room", self.danmaku_validation_room)
        validation = room_url(validation).rsplit("/", 1)[-1] if validation.strip() else ""
        if validation and validation == room:
            raise ValueError("same_validation_room")
        if "danmaku_validation_room" in updates:
            updates["danmaku_validation_room"] = validation
        self.saved.append((room, auto_check, updates))
        self.room = room
        self.auto_check = auto_check
        for name, value in updates.items():
            setattr(self, name, value)
        self.prefs_changed.emit({"room": self.room, "enabled": self.danmaku_enabled,
            "source": self.danmaku_source, "interval_seconds": self.danmaku_interval_seconds,
            "validation_room": self.danmaku_validation_room, "repair_enabled": self.danmaku_repair_enabled})

    def set_danmaku_repair_enabled(self, enabled):
        if enabled and not self.danmaku_validation_room:
            raise ValueError("validation_room_required")
        self.save_preferences(self.room, self.auto_check, danmaku_repair_enabled=enabled)
        if enabled:
            self.open(room_url(self.danmaku_validation_room), get_language())

    def resume_danmaku(self):
        self.resumed += 1

    def clear_danmaku_queue(self):
        self.clear_requested += 1

    def request_content_filter(self):
        self.filter_requested += 1

    def open(self, url, language):
        self.opened.append((url, language))

    def activate_checks(self):
        pass

    def check_updates(self):
        pass

    def refresh(self):
        pass

    def stop_browser(self):
        pass


@pytest.fixture
def qt_app():
    application = QApplication.instance() or QApplication([])
    previous = get_language()
    set_language("zh-CN")
    yield application
    set_language(previous)
    for widget in application.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    application.processEvents()


def make_dialog(service):
    with patch("fanyiji.browser_dialog.runtime_snapshot", return_value={"state": "unknown"}):
        return BrowserDialog(service)


def test_opening_and_editing_push_controls_never_starts_sending(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    assert not dialog.danmaku_enabled.isChecked()
    assert dialog.danmaku_source.currentData() == "system"
    assert dialog.danmaku_interval.minimum() == 3.5
    assert dialog.danmaku_interval.value() == 3.5
    dialog.danmaku_enabled.setChecked(True)
    dialog.danmaku_interval.setValue(1.0)
    assert dialog.danmaku_interval.value() == 3.5
    dialog.auto_check.setChecked(True)
    assert service.saved == []
    assert service.opened == []
    assert service.danmaku_enabled is False


def test_only_save_button_submits_all_push_preferences(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    dialog.danmaku_enabled.setChecked(True)
    dialog.danmaku_source.setCurrentIndex(dialog.danmaku_source.findData("mic"))
    dialog.danmaku_interval.setValue(6.0)
    dialog.save_button.click()
    assert service.saved == [("1698225", False, {"danmaku_enabled": True,
        "danmaku_source": "mic", "danmaku_interval_seconds": 6.0,
        "danmaku_validation_room": ""})]
    assert "已保存" in dialog.status.text()


@pytest.mark.parametrize("action,url", [("login", "https://passport.bilibili.com/login"),
    ("open_room", "https://live.bilibili.com/1698225")])
def test_sign_in_and_navigation_preserve_unsaved_push_controls(qt_app, action, url):
    service = BrowserStub()
    dialog = make_dialog(service)
    dialog.danmaku_enabled.setChecked(True)
    getattr(dialog, action)()
    assert service.saved == [("1698225", False, {})]
    assert service.danmaku_enabled is False
    assert dialog.danmaku_enabled.isChecked()
    assert service.opened == [(url, "zh-CN")]


def test_saved_push_settings_and_initial_queue_status_are_restored(qt_app):
    service = BrowserStub()
    service.danmaku_enabled = True
    service.danmaku_source = "all"
    service.danmaku_interval_seconds = 8.5
    service.danmaku_snapshot = {"enabled": True, "state": "cooldown", "queued": 7,
                                "accepted": 3, "page_visible": 2}
    dialog = make_dialog(service)
    assert dialog.danmaku_enabled.isChecked()
    assert dialog.danmaku_source.currentData() == "all"
    assert dialog.danmaku_interval.value() == 8.5
    assert "待发 7 段" in dialog.danmaku_status.text()
    assert "接口回执 3" in dialog.danmaku_status.text()
    assert "页面收到 2" in dialog.danmaku_status.text()
    assert service.saved == []


def test_validation_room_is_saved_independently_of_unsaved_formal_push_controls(qt_app):
    service = BrowserStub()
    service.room = "21399689"
    dialog = make_dialog(service)
    dialog.room.setText("999")
    dialog.danmaku_enabled.setChecked(True)
    dialog.danmaku_source.setCurrentIndex(dialog.danmaku_source.findData("mic"))
    dialog.validation_room.setText("https://live.bilibili.com/001698225/")
    dialog.validation_open_button.click()
    assert service.saved == [("21399689", False, {"danmaku_validation_room": "1698225"})]
    assert service.room == "21399689"
    assert service.danmaku_validation_room == "1698225"
    assert service.opened == [("https://live.bilibili.com/1698225", "zh-CN")]
    assert service.danmaku_enabled is False
    assert dialog.room.text() == "999"
    assert dialog.danmaku_enabled.isChecked()
    assert dialog.danmaku_source.currentData() == "mic"
    assert dialog.validation_room.text() == "1698225"


def test_same_formal_and_validation_rooms_are_rejected_without_opening_or_saving(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    dialog.validation_room.setText("001698225")
    assert dialog.save_danmaku_preferences() is False
    assert "不能与正式直播间相同" in dialog.status.text()
    assert dialog.open_validation_room() is False
    assert service.saved == []
    assert service.opened == []


def test_open_validation_room_does_not_open_when_room_is_missing_or_invalid(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    for invalid in ("", "https://other.example/123", "bad"):
        dialog.validation_room.setText(invalid)
        assert dialog.open_validation_room() is False
    assert service.saved == []
    assert service.opened == []


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("state,description", [
    ("waiting_validation", "修改后正文等待独立测试房间验证，尚未生成规则"),
    ("validating", "正在独立测试房间验证修改后正文"),
    ("validation_failed", "测试正文未通过，保留记录且不生成规则"),
    ("waiting_validation_browser", "等待手动打开已登录的验证直播间"),
    ("validation_storage_error", "验证记录暂存失败 自动重试保存"),
    ("validation_passed", "独立测试房间验证通过，正式房间效果未确认")])
def test_validation_states_preserve_no_false_delivery_claims(qt_app, language, state, description):
    from fanyiji.i18n import tr
    set_language(language)
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"enabled": True, "state": state, "queued": 2, "can_resume": True})
    assert tr(description) in dialog.danmaku_status.text()
    if state in {"validating", "validation_storage_error"}:
        assert not dialog.resume_button.isEnabled()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_validation_and_manual_browser_start_requirements_are_explained(qt_app, language):
    from fanyiji.i18n import tr
    set_language(language)
    dialog = make_dialog(BrowserStub())
    notes = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert tr("修改后正文须在独立测试房间验证通过，才比对原句生成局部规则并归档。测试通过只说明该测试房间可发，不代表正式房间禁词原因已确认。") in notes
    assert dialog.validation_open_button.text() == tr("打开验证直播间")
    expected = {"zh-CN": "不自动启动浏览器", "ja": "ブラウザーは自動起動しません", "en": "browser does not start automatically"}
    assert expected[language] in notes


@pytest.mark.parametrize("state", ["waiting_validation", "waiting_validation_browser", "validating",
    "validation_failed", "validation_storage_error", "validation_passed"])
def test_validation_progress_is_visible_when_production_sending_is_disabled(qt_app, state):
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"enabled": False, "state": state, "queued": 0})
    assert state not in dialog.danmaku_status.text()
    assert "弹幕推送未开启" not in dialog.danmaku_status.text()
    assert not dialog.resume_button.isEnabled()


def test_filter_manager_can_open_without_browser_audio_or_enabled_push(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    assert dialog.filter_button.isEnabled()
    dialog.filter_button.click()
    assert service.filter_requested == 1
    assert service.saved == []
    assert service.opened == []
    assert service.resumed == 0


@pytest.mark.parametrize("language,error_text,count_text", [
    ("zh-CN", "过滤名单异常，待发消息保留", "过滤规则 2 · 未处理失败 3"),
    ("ja", "ローカルルールに問題があります", "置換ルール 2 · 未処理の失敗 3"),
    ("en", "Filter rules have an error", "Filter rules 2 · Unhandled failures 3")])
def test_filter_error_shows_retained_queue_and_safe_counts(qt_app, language, error_text, count_text):
    set_language(language)
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"enabled": True, "state": "filter_error", "queued": 4,
        "filter_error": "PRIVATE_READ_ERROR", "filter_rule_count": 2, "filter_candidate_count": 3})
    assert error_text in dialog.danmaku_status.text()
    assert count_text in dialog.danmaku_status.text()
    assert "PRIVATE_READ_ERROR" not in dialog.danmaku_status.text()
    assert dialog.filter_button.isEnabled()


def test_candidate_save_failure_does_not_offer_manual_resume_for_automatic_f(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    dialog.on_danmaku_status({"enabled": True, "state": "waiting_f_retry", "f_retry_active": True,
        "filter_candidate_error": "PRIVATE_WRITE_ERROR", "filter_rule_count": True,
        "filter_candidate_count": "PRIVATE_COUNT"})
    text = dialog.danmaku_status.text()
    assert "自动重试" in text
    assert "失败候选保存失败" in text
    assert "PRIVATE" not in text
    assert not dialog.resume_button.isEnabled()


def test_api_acknowledgement_is_not_described_as_delivery(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    service.danmaku_status.emit({"enabled": True, "state": "accepted", "queued": 1,
                                "accepted": 1, "page_visible": 0})
    assert "尚未确认页面收到" in dialog.danmaku_status.text()
    assert "服务器广播确认" not in dialog.danmaku_status.text()
    assert "页面收到 0" in dialog.danmaku_status.text()


def test_unknown_send_is_retained_and_explicit_button_only_requests_verification(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    service.danmaku_status.emit({"enabled": True, "state": "unknown", "queued": 1,
                                "can_resume": True})
    assert "结果未知" in dialog.danmaku_status.text()
    assert "保留" in dialog.danmaku_status.text()
    assert dialog.resume_button.isEnabled()
    assert dialog.resume_button.text() == "核验待定消息"
    assert "不重新发送" in dialog.resume_button.toolTip()
    assert service.resumed == 0
    dialog.resume_button.click()
    assert service.resumed == 1
    service.danmaku_status.emit({"enabled": True, "state": "paused", "can_resume": True})
    dialog.resume_button.click()
    assert service.resumed == 2


def test_unavailable_distribution_build_keeps_browser_but_hides_push(qt_app):
    service = BrowserStub()
    service.danmaku_available = False
    dialog = make_dialog(service)
    assert dialog.push_controls.isHidden()
    assert not dialog.push_controls.isEnabled()
    assert not dialog.danmaku_access_note.isHidden()
    assert "分发版不开放" in dialog.danmaku_access_note.text()
    assert dialog.save_danmaku_preferences() is False
    service.danmaku_status.emit({"clearable": 3})
    assert not dialog.clear_button.isEnabled()
    assert dialog.clear_pending() is False
    assert service.clear_requested == 0
    dialog.open_room()
    assert service.opened == [("https://live.bilibili.com/1698225", "zh-CN")]
    assert service.saved == [("1698225", False, {})]


@pytest.mark.parametrize("language,label", [("ja", "確定訳"), ("en", "final translations")])
def test_push_controls_follow_interface_language(qt_app, language, label):
    set_language(language)
    dialog = make_dialog(BrowserStub())
    assert label in dialog.danmaku_enabled.text()
    assert "电脑声音" not in dialog.danmaku_source.itemText(0)
    assert "待发" not in dialog.danmaku_status.text()


@pytest.mark.parametrize("language,title", [("zh-CN", "清空待发队列"),
    ("ja", "未送信キューを消去"), ("en", "Clear unsent queue")])
@pytest.mark.parametrize("reply,requests", [(QMessageBox.StandardButton.Yes, 1),
    (QMessageBox.StandardButton.No, 0)])
def test_clear_queue_requires_explicit_confirmation_and_preserves_unknown_count(
        qt_app, language, title, reply, requests):
    set_language(language)
    service = BrowserStub()
    service.danmaku_snapshot = {"clearable": 7, "retained_uncertain": 2, "queued": 9}
    dialog = make_dialog(service)
    assert dialog.clear_button.isEnabled()
    assert service.clear_requested == 0
    with patch("fanyiji.browser_dialog.QMessageBox.question", return_value=reply) as question:
        dialog.clear_button.click()
    assert service.clear_requested == requests
    args = question.call_args.args
    assert args[1] == title
    assert "7" in args[2] and "2" in args[2]
    assert args[4] == QMessageBox.StandardButton.No
    assert service.saved == []
    assert service.opened == []
    assert service.resumed == 0


def test_empty_or_only_uncertain_queue_cannot_be_cleared(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    service.danmaku_status.emit({"clearable": 0, "retained_uncertain": 3, "queued": 3})
    assert not dialog.clear_button.isEnabled()
    with patch("fanyiji.browser_dialog.QMessageBox.question") as question:
        dialog.clear_button.click()
        assert dialog.clear_pending() is False
    question.assert_not_called()
    assert service.clear_requested == 0


@pytest.mark.parametrize("language,cleared_text,code_text", [
    ("zh-CN", "已清理 5 段未发送消息", "平台码 1003212"),
    ("ja", "未送信 5 件を消去済み", "プラットフォームコード 1003212"),
    ("en", "Cleared 5 unsent messages", "Platform code 1003212")])
def test_clear_results_and_safe_diagnostics_do_not_change_delivery_meaning(
        qt_app, language, cleared_text, code_text):
    set_language(language)
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"enabled": True, "state": "platform_rejected", "blocked": True,
        "queued": 1, "clearable": 0, "cleared": 5, "accepted": 8, "page_visible": 3,
        "last_platform_code": 1003212, "head_characters": 78,
        "platform_message": "PRIVATE_PLATFORM_MESSAGE"})
    text = dialog.danmaku_status.text()
    assert cleared_text in text
    assert code_text in text
    assert "78" in text and "UTF-16" in text
    assert "PRIVATE_PLATFORM_MESSAGE" not in text
    assert "服务器广播确认" not in text
    assert not dialog.clear_button.isEnabled()


def test_invalid_diagnostic_fields_are_not_rendered(qt_app):
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"state": "platform_rejected", "last_platform_code": "secret-code",
                             "head_characters": "secret-length"})
    assert "secret" not in dialog.danmaku_status.text()


@pytest.mark.parametrize("language,duplicate_text,limit_text,punctuation_text,short_text", [
    ("zh-CN", "10 秒内重复内容跳过", "账号上限更低时取较小值", "正文去标点，断句用空格", "不足 2 字保留原位"),
    ("ja", "10 秒以内の重複は省略", "アカウントの上限が低ければそちら", "句読点を空白に置き換え", "2 文字未満は元の位置で保持"),
    ("en", "Skip duplicates within 10 seconds", "account's lower limit", "Replace punctuation with spaces", "shorter fragments retain their position")])
def test_notes_explain_deduplication_window_and_account_length_limit(
        qt_app, language, duplicate_text, limit_text, punctuation_text, short_text):
    set_language(language)
    dialog = make_dialog(BrowserStub())
    notes = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert duplicate_text in notes
    assert limit_text in notes
    assert punctuation_text in notes
    assert short_text in notes
    assert "40" in notes
    assert "76" not in notes
    assert "【】" in notes


@pytest.mark.parametrize("language,retry_text,counter_text", [
    ("zh-CN", "自动重试", "f 自动重试累计 7 次"),
    ("ja", "自動再送", "f 自動再送合計 7 回"),
    ("en", "retry", "Total f retries 7")])
@pytest.mark.parametrize("state", ["platform_f", "waiting_f_retry", "retrying_f"])
def test_platform_f_reports_automatic_retry_without_manual_resume(
        qt_app, language, retry_text, counter_text, state):
    set_language(language)
    service = BrowserStub()
    dialog = make_dialog(service)
    dialog.on_danmaku_status({"enabled": True, "state": state, "blocked": False,
        "queued": 1, "clearable": 0, "retained_uncertain": 1,
        "can_resume": True, "f_retry_active": True, "f_retries": 7,
        "cooldown_remaining": 2.5,
        "last_platform_code": 0, "head_characters": 29, "max_weighted_length": 20})
    assert retry_text in dialog.danmaku_status.text()
    assert counter_text in dialog.danmaku_status.text()
    assert "2.5" in dialog.danmaku_status.text()
    assert "拒绝" not in dialog.danmaku_status.text()
    assert "20" in dialog.danmaku_status.text()
    assert not dialog.resume_button.isEnabled()
    assert not dialog.clear_button.isEnabled()
    assert service.resumed == 0
    dialog.resume_button.click()
    assert service.resumed == 0
    assert service.clear_requested == 0


@pytest.mark.parametrize("state", ["checking_login", "verifying", "sending"])
def test_f_retry_active_cannot_offer_manual_resume_during_other_operations(qt_app, state):
    service = BrowserStub()
    dialog = make_dialog(service)
    dialog.on_danmaku_status({"enabled": True, "state": state,
        "f_retry_active": True, "f_retries": 2, "can_resume": True})
    assert dialog.resume_button.text() == "自动重试中"
    assert "无需手动恢复" in dialog.resume_button.toolTip()
    assert not dialog.resume_button.isEnabled()
    dialog.resume_button.click()
    assert service.resumed == 0


def test_unknown_after_f_retry_restores_readonly_verification_button(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    dialog.on_danmaku_status({"enabled": True, "state": "waiting_f_retry",
        "f_retry_active": True, "f_retries": 4})
    dialog.on_danmaku_status({"enabled": True, "state": "result_unknown", "blocked": True,
        "f_retry_active": False, "f_retries": 4})
    assert "结果未知" in dialog.danmaku_status.text()
    assert dialog.resume_button.text() == "核验待定消息"
    assert dialog.resume_button.isEnabled()
    assert "不重新发送" in dialog.resume_button.toolTip()
    dialog.resume_button.click()
    assert service.resumed == 1


@pytest.mark.parametrize("invalid", [0, -1, True, 7.0, "private-counter", None])
def test_invalid_f_retry_counters_are_not_rendered(qt_app, invalid):
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"enabled": True, "state": "waiting_f_retry", "f_retries": invalid})
    assert "累计" not in dialog.danmaku_status.text()
    assert "private-counter" not in dialog.danmaku_status.text()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("state,description", [
    ("auto_verifying", "发送结果未知，正在自动只读核验"),
    ("auto_rechecking", "连续发送失败，正在自动复检账号和页面"),
    ("auto_retry", "正在自动恢复当前条，保留队列顺序")])
def test_automatic_recovery_reports_progress_without_enabling_manual_send(qt_app, language, state, description):
    from fanyiji.i18n import tr
    set_language(language)
    service = BrowserStub()
    dialog = make_dialog(service)
    dialog.on_danmaku_status({"enabled": True, "state": state, "queued": 2,
                             "can_resume": True, "filter_rule_count": 1,
                             "filter_candidate_count": 1, "filter_resolved_count": 5})
    assert tr(description) in dialog.danmaku_status.text()
    assert tr(" · 已处理历史 {0}", 5) in dialog.danmaku_status.text()
    assert tr(" · 过滤规则 {0} · 未处理失败 {1}", 1, 1) in dialog.danmaku_status.text()
    assert not dialog.resume_button.isEnabled()
    if state == "auto_verifying":
        assert dialog.resume_button.text() == tr("自动核验中")
        assert dialog.resume_button.toolTip() == tr("正在自动只读核验，不会重复提交结果未知的消息。")
    dialog.resume_button.click()
    assert service.resumed == 0


@pytest.mark.parametrize("invalid", [True, -1, "PRIVATE_COUNT", 1.5, None])
def test_invalid_resolved_counters_are_not_rendered(qt_app, invalid):
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"enabled": True, "state": "auto_verifying", "filter_resolved_count": invalid})
    assert "已处理历史" not in dialog.danmaku_status.text()
    assert "PRIVATE" not in dialog.danmaku_status.text()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("state,description", [
    ("ai_analyzing", "正在请求 Codex 分析失败正文"),
    ("ai_repair_ready", "修改后正文已通过本地检查，等待独立测试验证"),
    ("ai_retrying", "正在按发送间隔重试测试正文")])
def test_ai_recovery_states_explain_automatic_action_without_duplicate_manual_submission(qt_app, language, state, description):
    from fanyiji.i18n import tr
    set_language(language)
    service = BrowserStub()
    dialog = make_dialog(service)
    dialog.on_danmaku_status({"enabled": True, "state": state, "queued": 2, "can_resume": True,
                             "recovery_category": "ai_review", "recovery_action": "analyze_with_codex"})
    assert tr(description) in dialog.danmaku_status.text()
    assert dialog.resume_button.text() == tr("自动处理中")
    assert not dialog.resume_button.isEnabled()
    assert "analyze_with_codex" not in dialog.danmaku_status.text()
    assert "自动送达" not in dialog.danmaku_status.text()
    dialog.resume_button.click()
    assert service.resumed == 0


@pytest.mark.parametrize("invalid", ["PRIVATE_DIAGNOSTIC", ["private"], {"private": 1}, True])
def test_unknown_recovery_action_never_exposes_raw_backend_details(qt_app, invalid):
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"enabled": True, "state": "ai_analyzing", "recovery_action": invalid})
    assert "自动处理当前失败消息" in dialog.danmaku_status.text()
    assert "PRIVATE" not in dialog.danmaku_status.text()
    assert "private" not in dialog.danmaku_status.text()


@pytest.mark.parametrize("language,automatic_text", [
    ("zh-CN", "失败消息保存到独立处理记录，开启失败译文修正后交给 Codex 处理，后续队列继续发送"),
    ("ja", "後続の送信を続けます"),
    ("en", "while later messages continue")])
def test_notes_explain_independent_failure_processing_and_retained_content(qt_app, language, automatic_text):
    set_language(language)
    dialog = make_dialog(BrowserStub())
    notes = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert automatic_text in notes
    assert "3.5" in notes


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_failure_transfer_status_displays_independent_count_and_disables_manual_head_retry(qt_app, language):
    from fanyiji.i18n import tr
    set_language(language)
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"enabled": True, "state": "failure_transferred", "queued": 5,
        "deferred_analysis": 2, "can_resume": True, "recovery_action": "transfer_failure"})
    assert tr("失败已移入独立处理，后续队列继续发送") in dialog.danmaku_status.text()
    assert tr(" · 累计移入独立处理 {0} 段", 2) in dialog.danmaku_status.text()
    assert not dialog.resume_button.isEnabled()


@pytest.mark.parametrize("invalid", [True, -1, "PRIVATE_COUNTER", 2.0, None])
def test_invalid_independent_processing_count_is_not_displayed(qt_app, invalid):
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"enabled": True, "state": "failure_transferred", "deferred_analysis": invalid})
    assert "独立处理 " not in dialog.danmaku_status.text()
    assert "PRIVATE_COUNTER" not in dialog.danmaku_status.text()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_notes_explain_startup_queue_reset_while_failure_history_is_retained(qt_app, language):
    from fanyiji.i18n import tr
    set_language(language)
    dialog = make_dialog(BrowserStub())
    notes = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert tr("每次启动清空上一轮待发队列与运行统计，失败处理记录和过滤规则保留；仅开启失败译文修正时继续分析。") in notes


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_repair_button_saves_only_switch_and_opens_saved_test_room_without_submitting_push_edits(qt_app, language):
    from fanyiji.i18n import tr
    set_language(language)
    service = BrowserStub()
    service.danmaku_enabled = True
    service.danmaku_validation_room = "888"
    dialog = make_dialog(service)
    dialog.room.setText("999")
    dialog.danmaku_enabled.setChecked(False)
    dialog.validation_room.setText("777")
    assert dialog.repair_button.text() == tr("开启失败译文修正")
    assert dialog.repair_status.text() == tr("失败译文修正已关闭")
    dialog.repair_button.click()
    assert service.saved == [("1698225", False, {"danmaku_repair_enabled": True})]
    assert service.danmaku_repair_enabled is True
    assert service.danmaku_enabled is True
    assert dialog.room.text() == "999"
    assert dialog.validation_room.text() == "777"
    assert not dialog.danmaku_enabled.isChecked()
    assert dialog.repair_button.isChecked()
    assert dialog.repair_button.text() == tr("停止失败译文修正")
    dialog.repair_button.click()
    assert service.danmaku_repair_enabled is False
    assert dialog.repair_status.text() == tr("失败译文修正已关闭")
    assert service.opened == [("https://live.bilibili.com/888", language)]
    assert service.resumed == 0


def test_repair_switch_write_failure_restores_browser_button_and_does_not_expose_error(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    with patch.object(service, "set_danmaku_repair_enabled", side_effect=OSError("PRIVATE_ERROR")):
        dialog.repair_button.click()
    assert not dialog.repair_button.isChecked()
    assert dialog.repair_status.text() == "失败译文修正已关闭"
    assert dialog.status.text() == "失败译文修正开关保存失败，原设置保留"
    assert "PRIVATE_ERROR" not in dialog.status.text()
    assert service.saved == []


def test_public_browser_repair_button_cannot_change_switch(qt_app):
    service = BrowserStub()
    service.danmaku_available = False
    dialog = make_dialog(service)
    assert not dialog.repair_button.isEnabled()
    assert dialog.toggle_repair() is False
    assert service.saved == []


def test_browser_repair_controls_follow_external_saved_switch_updates(qt_app):
    service = BrowserStub()
    service.danmaku_validation_room = "888"
    dialog = make_dialog(service)
    service.set_danmaku_repair_enabled(True)
    assert dialog.repair_button.isChecked()
    assert dialog.repair_status.text() == "失败译文修正已开启"
    service.set_danmaku_repair_enabled(False)
    assert not dialog.repair_button.isChecked()


def test_browser_repair_start_without_saved_test_room_keeps_switch_off(qt_app):
    service = BrowserStub()
    dialog = make_dialog(service)
    dialog.validation_room.setText("888")
    dialog.repair_button.click()
    assert not dialog.repair_button.isChecked()
    assert "请先配置并保存" in dialog.status.text()
    assert service.saved == []
    assert service.opened == []


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_repair_progress_is_visible_and_stopping_hides_round_counter(qt_app, language):
    from fanyiji.i18n import tr
    set_language(language)
    service = BrowserStub()
    service.danmaku_repair_enabled = True
    dialog = make_dialog(service)
    service.danmaku_status.emit({"enabled": False, "state": "repair_retry_wait",
        "repair_strategy": "word_by_word", "repair_round": 7, "repair_wait_seconds": 5,
        "can_resume": True})
    assert tr(" · 逐词修正") in dialog.repair_status.text()
    assert tr(" · 第 {0} 轮", 7) in dialog.repair_status.text()
    assert tr(" · 等待 {0} 秒", 5) in dialog.repair_status.text()
    assert tr("等待下一轮局部修正") in dialog.danmaku_status.text()
    assert not dialog.resume_button.isEnabled()
    service.set_danmaku_repair_enabled(False)
    assert dialog.repair_status.text() == tr("失败译文修正已关闭")
    service.danmaku_status.emit({"enabled": True, "state": "validation_paused", "can_resume": True})
    assert tr("失败译文修正已停止，验证进度保留") in dialog.danmaku_status.text()
    assert not dialog.resume_button.isEnabled()


@pytest.mark.parametrize("invalid", [True, -1, "PRIVATE", 2.5, None])
def test_repair_progress_counters_reject_invalid_values(qt_app, invalid):
    service = BrowserStub()
    service.danmaku_repair_enabled = True
    dialog = make_dialog(service)
    service.danmaku_status.emit({"repair_strategy": "PRIVATE", "repair_round": invalid,
                                "repair_wait_seconds": invalid})
    assert dialog.repair_status.text() == "失败译文修正已开启"


@pytest.mark.parametrize("invalid", [0, -1, True, 20.0, "secret-limit", None])
def test_weighted_limit_is_rendered_only_as_a_positive_integer(qt_app, invalid):
    dialog = make_dialog(BrowserStub())
    dialog.on_danmaku_status({"max_weighted_length": invalid})
    assert "加权上限" not in dialog.danmaku_status.text()
    assert "secret-limit" not in dialog.danmaku_status.text()


def test_every_browser_dialog_message_has_japanese_and_english_translations():
    path = Path(__file__).resolve().parents[1] / "src/fanyiji/browser_dialog.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tr"
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            message = node.args[0].value
            if any("\u4e00" <= char <= "\u9fff" for char in message):
                assert message in _MESSAGES, (node.lineno, message)
