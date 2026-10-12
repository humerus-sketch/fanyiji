"""Local rule management never treats a failed sentence as an active rule."""
import ast
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from fanyiji.danmaku_filter import DanmakuContentFilter, DanmakuFilterError
from fanyiji.danmaku_filter_dialog import DanmakuFilterDialog
from fanyiji.i18n import _MESSAGES, get_language, set_language


@pytest.fixture
def qt_app():
    app = QApplication.instance() or QApplication([])
    previous = get_language()
    set_language("zh-CN")
    yield app
    set_language(previous)
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.processEvents()


@pytest.fixture
def content_filter(tmp_path):
    return DanmakuContentFilter(tmp_path / "local-filter.json")


def test_opening_manager_is_readonly_and_requires_no_browser_or_models(qt_app, content_filter):
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    assert dialog.rules_table.rowCount() == 0
    assert dialog.candidates_table.rowCount() == 0
    assert dialog.enabled.isChecked()
    assert dialog.controls.isEnabled()
    assert not content_filter.path.exists()


def test_add_local_rule_defaults_empty_replacement_and_persists(qt_app, content_filter):
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.term.setText("待改用词")
    assert dialog.save_rule()
    rule = DanmakuContentFilter(content_filter.path).rules()[0]
    assert rule["term"] == "待改用词"
    assert rule["replacement"] == "已过滤"
    assert rule["action"] == "replace"
    assert rule["room"] == ""
    assert rule["enabled"] is True
    assert content_filter.apply("原文待改用词保留", "1698225") == ("原文已过滤保留", 1)
    assert content_filter.apply("待改用词", "777") == ("已过滤", 1)
    assert dialog.rules_table.item(0, 3).text() == "全部直播间"


@pytest.mark.parametrize("language,replace_text,remove_text", [
    ("zh-CN", "中性替换", "从弹幕删除"),
    ("ja", "中立的な置換", "コメントから削除"),
    ("en", "Neutral replacement", "Remove from chat")])
def test_explicit_rule_actions_follow_interface_language(qt_app, content_filter, language, replace_text, remove_text):
    set_language(language)
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    assert dialog.rule_action.currentData() == "replace"
    assert dialog.rule_action.itemText(dialog.rule_action.findData("replace")) == replace_text
    assert dialog.rule_action.itemText(dialog.rule_action.findData("remove")) == remove_text
    assert dialog.replacement.isEnabled()
    dialog.rule_action.setCurrentIndex(dialog.rule_action.findData("remove"))
    assert not dialog.replacement.isEnabled()
    dialog.new_rule()
    assert dialog.rule_action.currentData() == "replace"
    assert dialog.replacement.isEnabled()


def test_explicit_remove_action_discards_stale_replacement_and_only_changes_chat(qt_app, content_filter):
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.term.setText("评价")
    dialog.replacement.setText("旧替换文字")
    dialog.rule_action.setCurrentIndex(dialog.rule_action.findData("remove"))
    assert not content_filter.path.exists()
    assert dialog.save_rule()
    rule = DanmakuContentFilter(content_filter.path).rules()[0]
    assert rule["action"] == "remove" and rule["replacement"] == ""
    assert rule["room"] == ""
    assert content_filter.apply("恭喜获得评价", "1698225") == ("恭喜获得", 1)
    assert content_filter.apply("恭喜获得评价", "777") == ("恭喜获得", 1)
    assert dialog.rules_table.item(0, 2).text() == "从弹幕删除"
    dialog.rules_table.selectRow(0)
    assert dialog.rule_action.currentData() == "remove"
    assert dialog.replacement.text() == "" and not dialog.replacement.isEnabled()


def test_editing_replace_into_remove_requires_save_and_preserves_identity(qt_app, content_filter):
    original = content_filter.add_rule("评价", "感想", room="")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.rules_table.selectRow(0)
    assert dialog.rule_action.currentData() == "replace"
    dialog.rule_action.setCurrentIndex(dialog.rule_action.findData("remove"))
    assert content_filter.rules()[0] == original
    assert dialog.save_rule()
    changed = DanmakuContentFilter(content_filter.path).rules()[0]
    assert changed["id"] == original["id"]
    assert changed["action"] == "remove" and changed["replacement"] == ""
    dialog.rules_table.selectRow(0)
    dialog.rule_action.setCurrentIndex(dialog.rule_action.findData("replace"))
    assert dialog.replacement.isEnabled()
    assert dialog.save_rule()
    changed = content_filter.rules()[0]
    assert changed["id"] == original["id"]
    assert changed["action"] == "replace" and changed["replacement"] == "已过滤"


def test_candidate_remove_is_explicit_and_original_failure_is_preserved(qt_app, content_filter):
    candidate = content_filter.record_failure("1698225", "【恭喜获得评价】", "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.candidates_table.selectRow(0)
    dialog.candidate_button.click()
    dialog.term.setText("评价")
    dialog.rule_action.setCurrentIndex(dialog.rule_action.findData("remove"))
    assert content_filter.rules() == []
    assert dialog.save_rule()
    rule = content_filter.rules()[0]
    assert rule["action"] == "remove" and rule["replacement"] == ""
    assert content_filter.candidates()[0]["text"] == candidate["text"]
    assert content_filter.candidates()[0]["count"] == candidate["count"]
    dialog.candidates_table.selectRow(0)
    analysis = dialog.candidate_analysis.toPlainText()
    assert "已确认删除规则：评价" in analysis
    assert "只从弹幕删除" in analysis
    assert "评价 →" not in analysis


def test_editor_changes_replacement_and_enabled_only_on_save_with_global_scope(qt_app, content_filter):
    original = content_filter.add_rule("激烈表达", "中性表达", room="")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.rules_table.selectRow(0)
    dialog.replacement.setText("温和表达")
    dialog.rule_enabled.setChecked(False)
    assert content_filter.rules()[0] == original
    assert dialog.save_rule()
    changed = DanmakuContentFilter(content_filter.path).rules()[0]
    assert changed["id"] == original["id"]
    assert changed["replacement"] == "温和表达"
    assert changed["room"] == ""
    assert changed["enabled"] is False


@pytest.mark.parametrize("room", ["", "1698225", "777"])
def test_rule_editor_has_no_room_selector_and_always_saves_global_rules(qt_app, content_filter, room):
    dialog = DanmakuFilterDialog(content_filter, room)
    dialog.term.setText("原词")
    assert not hasattr(dialog, "scope")
    assert not hasattr(dialog, "scope_room")
    assert dialog.save_rule()
    assert content_filter.rules()[0]["room"] == ""
    assert content_filter.apply("原词", "123") == ("已过滤", 1)


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_saved_rule_scope_is_global_in_every_interface_language(qt_app, content_filter, language):
    from fanyiji.i18n import tr
    rule = content_filter.add_rule("旧词", "替换词", room="1698225")
    legacy_rule = {**rule, "room": "1698225"}
    set_language(language)
    with patch.object(content_filter, "list_rules", return_value=[legacy_rule]):
        dialog = DanmakuFilterDialog(content_filter, "777")
    note = "所有已保存规则统一适用于全部直播间；失败记录中的房间号仅表示来源和验证记录。"
    assert dialog.rules_table.item(0, 3).text() == tr("全部直播间")
    assert not hasattr(dialog, "scope_room")
    assert tr(note) in [label.text() for label in dialog.findChildren(QLabel)]
    if language != "zh-CN":
        assert tr(note) != note


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("message", ["已存在相同过滤词，请编辑原规则",
    "该直播间范围已存在相同过滤词，请编辑原规则", "该直播间范围已存在相同过滤词"])
def test_duplicate_rule_errors_use_global_message_in_all_languages(qt_app, content_filter, language, message):
    from fanyiji.i18n import tr
    set_language(language)
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog._error(DanmakuFilterError(message))
    assert dialog.status.text() == tr("已存在相同过滤词，请编辑原规则")


def test_enabling_whole_filter_is_explicit_and_persisted(qt_app, content_filter):
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.enabled.setChecked(False)
    assert content_filter.enabled is True
    assert dialog.save_enabled()
    assert DanmakuContentFilter(content_filter.path).enabled is False


def test_failed_candidate_text_count_time_and_reason_are_readonly(qt_app, content_filter):
    text = "<b>保留完整失败句和标点！</b>"
    content_filter.record_failure("1698225", text, "platform_f")
    content_filter.record_failure("1698225", text, "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.candidates_table.selectRow(0)
    assert dialog.candidate_text.toPlainText() == text
    assert dialog.candidate_text.isReadOnly()
    assert dialog.candidates_table.item(0, 1).text() == "1698225"
    assert dialog.candidates_table.item(0, 2).text() == "2"
    assert dialog.candidates_table.item(0, 3).text().endswith("+08:00")
    assert "原因未确定" in dialog.candidates_table.item(0, 4).text()
    assert content_filter.rules() == []


def test_candidate_does_not_activate_until_user_confirms_term_and_saves(qt_app, content_filter):
    content_filter.record_failure("1698225", "这句中的激烈表达需要人工判断", "platform_f")
    before = content_filter.path.read_bytes()
    dialog = DanmakuFilterDialog(content_filter, "999")
    dialog.candidates_table.selectRow(0)
    dialog.candidate_button.click()
    assert dialog.term.text() == ""
    assert not hasattr(dialog, "scope_room")
    assert dialog.candidates_table.item(0, 1).text() == "1698225"
    assert content_filter.path.read_bytes() == before
    assert content_filter.rules() == []
    assert not dialog.save_rule()
    assert content_filter.path.read_bytes() == before
    dialog.term.setText("激烈表达")
    dialog.replacement.setText("中性表达")
    assert dialog.save_rule()
    assert content_filter.rules()[0]["term"] == "激烈表达"
    assert content_filter.rules()[0]["replacement"] == "中性表达"
    assert content_filter.rules()[0]["room"] == ""
    assert content_filter.apply("激烈表达", "777") == ("中性表达", 1)
    assert len(content_filter.candidates()) == 1


def test_promoting_disabled_candidate_preserves_disabled_setting_atomically(qt_app, content_filter):
    content_filter.record_failure("1698225", "需要人工确认的原词", "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.candidates_table.selectRow(0)
    dialog.candidate_button.click()
    dialog.term.setText("原词")
    dialog.rule_enabled.setChecked(False)
    assert dialog.save_rule()
    assert DanmakuContentFilter(content_filter.path).rules()[0]["enabled"] is False


def test_term_outside_selected_candidate_is_rejected_without_activating(qt_app, content_filter):
    content_filter.record_failure("1698225", "未经确认的失败句", "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.candidates_table.selectRow(0)
    dialog.candidate_button.click()
    dialog.term.setText("这词不在候选中")
    assert not dialog.save_rule()
    assert "来自所选失败内容" in dialog.status.text()
    assert content_filter.rules() == []


@pytest.mark.parametrize("answer,remaining", [(QMessageBox.StandardButton.Yes, 0),
    (QMessageBox.StandardButton.No, 1)])
def test_rule_removal_requires_confirmation_and_preserves_candidates(qt_app, content_filter, answer, remaining):
    content_filter.add_rule("原词", room="1698225")
    content_filter.record_failure("1698225", "原词未确认", "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.rules_table.selectRow(0)
    with patch("fanyiji.danmaku_filter_dialog.QMessageBox.question", return_value=answer) as question:
        dialog.remove_button.click()
    assert question.call_args.args[-1] == QMessageBox.StandardButton.No
    assert len(content_filter.rules()) == remaining
    assert len(content_filter.candidates()) == 1


@pytest.mark.parametrize("language,expected", [("zh-CN", "原规则未修改"),
    ("ja", "既存ルールは変更されていません"), ("en", "existing rules are unchanged")])
def test_save_error_retains_editor_and_old_rules_without_raw_details(qt_app, content_filter, language, expected):
    set_language(language)
    original = content_filter.add_rule("原词", "中性表达")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.rules_table.selectRow(0)
    dialog.replacement.setText("新的表达")
    with patch("fanyiji.danmaku_filter.os.replace", side_effect=OSError("PRIVATE_STORAGE_DETAIL")):
        assert dialog.save_rule() is False
    assert content_filter.rules()[0] == original
    assert dialog.replacement.text() == "新的表达"
    assert expected in dialog.status.text()
    assert "PRIVATE_STORAGE_DETAIL" not in dialog.status.text()


@pytest.mark.parametrize("language,title,validation_text", [
    ("zh-CN", "本地过滤词管理", "过滤词不能为空"),
    ("ja", "ローカル置換ルール管理", "原語を入力してください"),
    ("en", "Local text filter rules", "original term cannot be empty")])
def test_manager_and_validation_follow_interface_language(qt_app, content_filter, language, title, validation_text):
    set_language(language)
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    assert dialog.windowTitle() == title
    assert not dialog.save_rule()
    assert validation_text in dialog.status.text()
    if language != "zh-CN":
        assert "原词" not in dialog.rules_table.horizontalHeaderItem(1).text()


def test_unavailable_filter_never_creates_or_overwrites_files(qt_app, tmp_path):
    damaged = tmp_path / "local-filter.json"
    damaged.write_text("bad-content", encoding="utf-8")
    before = damaged.read_bytes()
    dialog = DanmakuFilterDialog(None, "1698225")
    assert not dialog.controls.isEnabled()
    assert "暂不可用" in dialog.status.text()
    assert dialog.save_rule() is False
    assert dialog.save_enabled() is False
    assert damaged.read_bytes() == before


def test_notes_distinguish_failed_candidates_from_confirmed_rules(qt_app, content_filter):
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    notes = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert "返回 f 不代表句中存在禁词" in notes
    assert "候选不会自动生效" in notes
    assert "不修改字幕" in notes
    assert "手工规则由你确认" in notes
    assert "自动恢复使用 Codex 生成修改后正文" in notes
    assert "发送中或结果未知的消息不会因编辑规则而改写" in notes
    assert "已确认失败的消息重试前应用最新规则" in notes
    assert "失败消息移入独立处理记录，后续队列继续发送" in notes
    assert "记录保留并自动更新处理状态" in notes


def test_filter_dialog_literals_have_japanese_and_english_translations():
    path = Path(__file__).resolve().parents[1] / "src/fanyiji/danmaku_filter_dialog.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"tr", "_table"}:
            for argument in node.args[:1]:
                values = argument.elts if isinstance(argument, (ast.Tuple, ast.List)) else [argument]
                for value in values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        if any("\u4e00" <= character <= "\u9fff" for character in value.value):
                            assert value.value in _MESSAGES, (node.lineno, value.value)


def _load_test_lexicon(content_filter):
    path = content_filter.path.parent / "danmaku-wordlist.json"
    path.write_text(json.dumps({"version": 1, "sources": [{
        "id": "test", "name": "Test lexicon", "url": "https://example.test/lexicon",
        "revision": "fixture", "license": "MIT", "license_text": "MIT fixture"}],
        "entries": [{"term": "待核对词", "category": "test", "source_id": "test"}],
        "allowlist": []}, ensure_ascii=False), encoding="utf-8")
    content_filter.reload_detector()
    return path


def test_candidate_is_screened_and_zero_matches_are_not_marked_as_banned(qt_app, content_filter):
    _load_test_lexicon(content_filter)
    content_filter.record_failure("1698225", "【不过 谢谢 恭喜获得评价 谢谢】", "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.candidates_table.selectRow(0)
    assert "未命中参考词" in dialog.candidate_analysis.toPlainText()
    assert "原因仍未确定" in dialog.candidate_analysis.toPlainText()
    assert "不能保证发送成功" in dialog.candidate_analysis.toPlainText()
    assert content_filter.rules() == []
    assert "1 个词" in dialog.dictionary_status.text()


def test_reference_hits_show_source_and_never_enable_a_replacement(qt_app, content_filter):
    _load_test_lexicon(content_filter)
    content_filter.record_failure("1698225", "【待 核 对 词】", "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.candidates_table.selectRow(0)
    assert "待核对词" in dialog.candidate_analysis.toPlainText()
    assert "来源：test" in dialog.candidate_analysis.toPlainText()
    assert content_filter.rules() == []
    dialog.dictionary_search.setText("not-present")
    assert dialog.dictionary_table.isRowHidden(0)
    dialog.dictionary_search.setText("待核对")
    assert not dialog.dictionary_table.isRowHidden(0)


def test_allowlist_rescans_failed_sentence_without_changing_original_or_rules(qt_app, content_filter):
    path = _load_test_lexicon(content_filter)
    content_filter.record_failure("1698225", "【待核对词】", "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.allowlist.setPlainText("待核对词\n")
    assert dialog.save_allowlist()
    assert json.loads(path.read_text(encoding="utf-8"))["allowlist"] == ["待核对词"]
    dialog.candidates_table.selectRow(0)
    assert "未命中参考词" in dialog.candidate_analysis.toPlainText()
    assert content_filter.candidates()[0]["text"] == "【待核对词】"
    assert content_filter.rules() == []


def test_dictionary_failure_does_not_disable_rule_management_or_expose_details(qt_app, content_filter):
    path = _load_test_lexicon(content_filter)
    path.write_text("PRIVATE_BROKEN_DICTIONARY", encoding="utf-8")
    content_filter.reload_detector()
    content_filter.record_failure("1698225", "正常句", "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    assert dialog.controls.isEnabled()
    assert "词库不可用" in dialog.dictionary_status.text()
    assert "PRIVATE_BROKEN" not in dialog.dictionary_status.text()
    dialog.term.setText("旧用词")
    assert dialog.save_rule()


def test_known_rule_match_is_not_reported_as_no_confirmed_rule(qt_app, content_filter):
    _load_test_lexicon(content_filter)
    content_filter.add_rule("确认用词", "中性词")
    content_filter.record_failure("1698225", "【确认用词】", "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.candidates_table.selectRow(0)
    result = dialog.candidate_analysis.toPlainText()
    assert "但命中已确认规则" in result
    assert "确认用词 → 中性词" in result
    assert "未命中参考词或已确认规则" not in result


def test_user_observed_term_is_labeled_with_its_room_scope(qt_app, content_filter):
    path = _load_test_lexicon(content_filter)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["entries"].append({"term": "评价", "category": "user-confirmed-trigger", "source_id": "test", "room": "1698225"})
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    content_filter.reload_detector()
    content_filter.record_failure("1698225", "【恭喜获得评价】", "platform_f")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.candidates_table.selectRow(0)
    assert "当前直播间的实测触发词" in dialog.candidate_analysis.toPlainText()
    assert dialog.dictionary_table.item(1, 3).text() == "1698225"
    assert content_filter.detect("评价", "123")["hits"] == []
    assert content_filter.rules() == []


@pytest.mark.parametrize("language,expected", [("zh-CN", "未命中"), ("ja", "一致はありません"), ("en", "no reference terms")])
def test_screening_result_is_translated_and_plain_text(qt_app, content_filter, language, expected):
    _load_test_lexicon(content_filter)
    content_filter.record_failure("1698225", "<b>正常文字</b>", "platform_f")
    set_language(language)
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.candidates_table.selectRow(0)
    assert dialog.candidate_text.toPlainText() == "<b>正常文字</b>"
    assert expected in dialog.candidate_analysis.toPlainText()


def test_resolved_history_is_hidden_by_default_but_never_removed(qt_app, content_filter):
    active = content_filter.record_failure("1698225", "【继续处理】", "platform_f")
    history = {**active, "id": "resolved-history", "text": "【已经处理】", "status": "resolved",
               "resolution": "page_observed", "resolved_at": "2026-10-11T12:00:00+08:00"}
    active = {**active, "status": "recovering", "queue_id": 8}
    before = content_filter.path.read_bytes()
    with patch.object(content_filter, "list_candidates", return_value=[active, history]):
        dialog = DanmakuFilterDialog(content_filter, "1698225")
        assert dialog.candidates_table.rowCount() == 1
        assert dialog.candidates_table.item(0, 6).text() == "自动处理中"
        assert "未处理 1 条" in dialog.candidate_status.text()
        assert "历史 1 条" in dialog.candidate_status.text()
        dialog.show_history.setChecked(True)
        assert dialog.candidates_table.rowCount() == 2
        dialog.candidates_table.selectRow(1)
        assert "页面出现同文，来源未确认" in dialog.candidates_table.item(1, 7).text()
        assert "人工发送也可能" in dialog.candidate_analysis.toPlainText()
        assert "不能证明由程序自动发送" in dialog.candidate_analysis.toPlainText()
        assert "处理时间（UTC+8）" in dialog.candidate_analysis.toPlainText()
    assert content_filter.path.read_bytes() == before
    assert len(content_filter.candidates()) == 1


def test_candidate_refresh_updates_progress_preserves_selection_and_unsaved_rule(qt_app, content_filter):
    active = content_filter.record_failure("1698225", "【完整失败原句】", "platform_f")
    active = {**active, "status": "recovering", "queue_id": 42}
    with patch.object(content_filter, "list_candidates", return_value=[active]) as candidates:
        dialog = DanmakuFilterDialog(content_filter, "1698225")
        dialog.candidates_table.selectRow(0)
        dialog.prepare_candidate()
        dialog.term.setText("失败")
        dialog.replacement.setText("待保存内容")
        candidate_id = dialog._candidate_id
        candidates.return_value = [{**active, "status": "waiting_confirmation",
                                    "diagnosis": {"category": "unknown", "action": "check",
                                                  "message": "保留消息，自动检查发送结果"}}]
        assert dialog.refresh_candidates()
        assert dialog.candidate_text.toPlainText() == "【完整失败原句】"
        assert "等待结果确认" in dialog.candidate_analysis.toPlainText()
        assert "关联队列编号：42" in dialog.candidate_analysis.toPlainText()
        assert dialog.term.text() == "失败"
        assert dialog.replacement.text() == "待保存内容"
        assert dialog._candidate_id == candidate_id
        candidates.return_value = [{**active, "status": "resolved", "resolution": "page_observed"}]
        assert dialog.refresh_candidates()
        assert dialog.candidates_table.rowCount() == 0
        assert not dialog.candidate_text.toPlainText()
        assert not dialog.candidate_button.isEnabled()
        assert dialog.term.text() == "失败"
        assert dialog.replacement.text() == "待保存内容"


@pytest.mark.parametrize("status,expected", [("active", "待处理"), ("recovering", "自动处理中"),
    ("waiting_confirmation", "等待结果确认"), ("resolved", "已处理（历史）"),
    ("action_required", "需要登录或人工处理")])
@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_recovery_statuses_and_history_controls_follow_interface_language(qt_app, content_filter, status, expected, language):
    from fanyiji.i18n import tr
    set_language(language)
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    assert DanmakuFilterDialog.recovery_status_text({"status": status}) == tr(expected)
    assert dialog.tabs.tabText(1) == tr("失败处理记录")
    assert dialog.show_history.text() == tr("显示已处理历史")
    assert dialog.candidates_table.horizontalHeaderItem(6).text() == tr("处理状态")
    assert expected in _MESSAGES
    if language != "zh-CN":
        assert tr(expected) != expected


def test_legacy_candidate_defaults_to_active_and_diagnostic_message_stays_plain_text(qt_app, content_filter):
    candidate = content_filter.record_failure("1698225", "【旧版记录】", "platform_f")
    candidate = {key: value for key, value in candidate.items() if key not in {"status", "diagnosis"}}
    candidate["diagnosis"] = {"category": "unknown", "action": "check", "message": "<b>诊断文字</b>"}
    with patch.object(content_filter, "list_candidates", return_value=[candidate]):
        dialog = DanmakuFilterDialog(content_filter, "1698225")
        dialog.candidates_table.selectRow(0)
    assert dialog.candidates_table.item(0, 6).text() == "待处理"
    assert "<b>诊断文字</b>" in dialog.candidate_analysis.toPlainText()
    assert "发送失败原因" not in dialog.candidates_table.item(0, 7).text()
    assert content_filter.rules() == []


def test_background_refresh_only_reads_visible_recovery_tab(qt_app, content_filter):
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    assert dialog.recovery_timer.interval() == 2000
    assert dialog.recovery_timer.isActive()
    with patch.object(dialog, "refresh_candidates") as refresh:
        dialog._refresh_visible_candidates()
        refresh.assert_not_called()
        dialog.show()
        dialog._refresh_visible_candidates()
        refresh.assert_not_called()
        dialog.tabs.setCurrentIndex(1)
        refresh.assert_called_once_with()
        refresh.reset_mock()
        dialog._refresh_visible_candidates()
        refresh.assert_called_once_with()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("origin,label", [
    ("ai_success_observed", "AI修正后可见（经验规则，原因未证实）"),
    ("ai_analysis", "AI建议（尚未实测，原因未证实）")])
def test_ai_rule_origins_are_distinguished_from_manual_rules_and_remain_editable(qt_app, content_filter, language, origin, label):
    from fanyiji.i18n import tr
    rule = content_filter.add_rule("评价", "感想", room="1698225")
    ai_rule = {**rule, "origin": origin, "evidence": {"confirmed_cause": False}}
    before = content_filter.path.read_bytes()
    set_language(language)
    with patch.object(content_filter, "list_rules", return_value=[ai_rule]):
        dialog = DanmakuFilterDialog(content_filter, "1698225")
        assert dialog.rules_table.item(0, 5).text() == tr(label)
        dialog.rules_table.selectRow(0)
        assert dialog.term.text() == "评价"
        assert dialog.replacement.text() == "感想"
        assert dialog.remove_button.isEnabled()
        assert dialog.rule_enabled.isEnabled() is (origin != "ai_analysis")
        assert dialog.save_button.isEnabled()
        if origin == "ai_analysis":
            assert dialog.editor_note.text() == tr("编辑AI建议规则：未验证时保持停用，可修改内容，验证通过后才可启用。")
    assert content_filter.path.read_bytes() == before


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_ai_diagnosis_details_are_translated_without_claiming_confirmed_cause(qt_app, content_filter, language):
    from fanyiji.i18n import tr
    candidate = content_filter.record_failure("1698225", "【完整原文】", "platform_f")
    candidate = {**candidate, "status": "recovering", "queue_id": 42,
                 "diagnosis": {"category": "ai_review", "action": "analyze_with_codex",
                               "message": "正在请求 Codex 分析失败正文"}}
    set_language(language)
    with patch.object(content_filter, "list_candidates", return_value=[candidate]):
        dialog = DanmakuFilterDialog(content_filter, "1698225")
        dialog.candidates_table.selectRow(0)
    details = dialog.candidate_analysis.toPlainText()
    assert tr("正在请求 Codex 分析失败正文") in details
    assert tr("AI分析（原因尚未证实）") in details
    assert tr("处理动作：{0}", tr("正在请求 Codex 分析失败正文")) in details
    assert "ai_review" not in details
    assert "analyze_with_codex" not in details


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_previous_failed_test_text_is_visible_without_learning_or_changing_current_plan(qt_app, content_filter, language):
    from fanyiji.i18n import tr
    candidate = content_filter.record_failure("21399689", "【感谢你的评价我会继续努力】", "platform_f")
    candidate = {**candidate, "status": "waiting_validation", "validation": {"state": "ready",
        "test_room": "1698225", "revised_text": "感谢你的评语我会继续努力"},
        "last_validation": {"state": "failed", "test_room": "1698225", "revised_text": "感谢你的感想我会继续努力"}}
    before = content_filter.path.read_bytes()
    set_language(language)
    details = DanmakuFilterDialog.candidate_details(candidate)
    assert tr("上一测试正文：{0}", "感谢你的感想我会继续努力") in details
    assert tr("上一测试直播间：{0}", "1698225") in details
    assert tr("待验证正文：{0}", "感谢你的评语我会继续努力") in details
    assert tr("上一测试未通过，记录保留；当前正文须重新独立验证。") in details
    assert content_filter.path.read_bytes() == before
    assert content_filter.rules() == []


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("resolution,message", [
    ("ai_no_local_edit", "分析已完成，未提出局部修改"),
    ("ai_rules_saved", "分析已完成，已保存局部建议规则（尚未实测）")])
def test_archived_ai_analysis_results_are_distinct_from_page_delivery(qt_app, content_filter, language, resolution, message):
    from fanyiji.i18n import tr
    candidate = content_filter.record_failure("1698225", "【失败原文】", "platform_f")
    candidate = {**candidate, "status": "resolved", "resolution": resolution,
                 "resolved_at": "2026-10-11T14:00:00+08:00"}
    set_language(language)
    with patch.object(content_filter, "list_candidates", return_value=[candidate]):
        dialog = DanmakuFilterDialog(content_filter, "1698225")
        assert dialog.candidates_table.rowCount() == 0
        dialog.show_history.setChecked(True)
        dialog.candidates_table.selectRow(0)
    assert dialog.candidates_table.item(0, 7).text() == tr(message)
    assert tr("处理记录归档不代表弹幕已经送达。") in dialog.candidate_analysis.toPlainText()
    assert "自动送达" not in dialog.candidate_analysis.toPlainText()


def test_automatic_rule_list_refresh_preserves_unsaved_user_edits(qt_app, content_filter):
    rule = content_filter.add_rule("旧词", "替换词", room="1698225")
    dialog = DanmakuFilterDialog(content_filter, "1698225")
    dialog.rules_table.selectRow(0)
    dialog.term.setText("未保存的原词")
    suggestion = {**rule, "id": "new-ai-suggestion", "term": "建议词", "origin": "ai_analysis"}
    before = content_filter.path.read_bytes()
    with patch.object(content_filter, "list_rules", return_value=[rule, suggestion]):
        assert dialog.refresh_rule_records()
    assert dialog.rules_table.rowCount() == 2
    assert dialog.rules_table.item(1, 5).text() == "AI建议（尚未实测，原因未证实）"
    assert dialog.rules_table.currentRow() == 0
    assert dialog.term.text() == "未保存的原词"
    assert dialog._rule_id == rule["id"]
    assert content_filter.path.read_bytes() == before


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_unvalidated_ai_rule_stays_disabled_and_is_never_labeled_as_confirmed(qt_app, content_filter, language):
    from fanyiji.i18n import tr
    rule = content_filter.add_rule("建议词", "替换", room="21399689", enabled=False)
    rule = {**rule, "origin": "ai_analysis"}
    set_language(language)
    with patch.object(content_filter, "list_rules", return_value=[rule]):
        dialog = DanmakuFilterDialog(content_filter, "21399689")
        dialog.rules_table.selectRow(0)
    assert dialog.rules_table.item(0, 5).text() == tr("AI建议（尚未验证，已停用，原因未证实）")
    assert not dialog.rule_enabled.isChecked()
    assert not dialog.rule_enabled.isEnabled()
    dialog.new_rule()
    assert dialog.rule_enabled.isEnabled()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_ai_unavailability_waits_for_next_startup_instead_of_requesting_manual_login(qt_app, content_filter, language):
    from fanyiji.i18n import tr
    candidate = content_filter.record_failure("21399689", "【失败原文】", "platform_f")
    candidate = {**candidate, "status": "action_required", "diagnosis": {
        "category": "ai_review", "action": "review_on_restart", "message": "AI分析暂不可用 下次启动自动重试"}}
    set_language(language)
    with patch.object(content_filter, "list_candidates", return_value=[candidate]):
        dialog = DanmakuFilterDialog(content_filter, "21399689")
        dialog.candidates_table.selectRow(0)
    assert dialog.candidates_table.item(0, 6).text() == tr("下次启动自动重试分析")
    assert tr("AI分析暂不可用 下次启动自动重试") in dialog.candidate_analysis.toPlainText()
    assert tr("需要登录或人工处理") not in dialog.candidate_analysis.toPlainText()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("state,label", [
    ("waiting_room", "等待配置独立测试房间"), ("ready", "等待打开验证直播间"),
    ("testing", "正在验证修改后正文"), ("failed", "验证未通过，等待继续测试")])
def test_pending_ai_validation_remains_unarchived_and_displays_its_exact_stage(qt_app, content_filter, language, state, label):
    from fanyiji.i18n import tr
    candidate = content_filter.record_failure("21399689", "【完整失败正文】", "platform_f")
    candidate = {**candidate, "status": "waiting_validation", "validation": {"state": state,
        "test_room": "1698225", "formal_room": "21399689", "original_text": "【完整失败正文】",
        "revised_text": "【完整修改正文】", "attempts": 2, "message": ""}}
    set_language(language)
    with patch.object(content_filter, "list_candidates", return_value=[candidate]):
        dialog = DanmakuFilterDialog(content_filter, "21399689")
        assert dialog.candidates_table.rowCount() == 1
        dialog.candidates_table.selectRow(0)
    assert dialog.candidates_table.item(0, 6).text() == tr(label)
    assert dialog.candidates_table.item(0, 7).text() == tr("测试正文等待验证，成功后才比对原句生成局部规则")
    details = dialog.candidate_analysis.toPlainText()
    assert tr("独立测试直播间：{0}", "1698225") in details
    assert tr("正式直播间：{0}", "21399689") in details
    assert tr("测试尝试次数：{0}", 2) in details
    assert "【完整修改正文】" in details
    assert tr("测试通过只说明测试房间可发，正式房间效果和禁词原因仍未确认。") in details
    assert tr("处理记录归档不代表弹幕已经送达。") not in details


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_test_room_validated_rules_can_be_disabled_without_claiming_production_proof(qt_app, content_filter, language):
    from fanyiji.i18n import tr
    rule = content_filter.add_rule("建议词", "替换", room="21399689")
    rule = {**rule, "origin": "ai_analysis", "validation": {"test_room": "1698225", "formal_room": "21399689"}}
    set_language(language)
    with patch.object(content_filter, "list_rules", return_value=[rule]):
        dialog = DanmakuFilterDialog(content_filter, "21399689")
        dialog.rules_table.selectRow(0)
    assert dialog.rules_table.item(0, 5).text() == tr("专用测试房间验证通过（正式房间原因未证实）")
    assert dialog.rule_enabled.isEnabled()
    dialog.rule_enabled.setChecked(False)
    assert not dialog.rule_enabled.isChecked()


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_test_room_validated_history_explains_production_delivery_is_unconfirmed(qt_app, content_filter, language):
    from fanyiji.i18n import tr
    candidate = content_filter.record_failure("21399689", "【失败正文】", "platform_f")
    candidate = {**candidate, "status": "resolved", "resolution": "test_room_validated",
                 "validation": {"state": "passed", "test_room": "1698225", "formal_room": "21399689"}}
    set_language(language)
    with patch.object(content_filter, "list_candidates", return_value=[candidate]):
        dialog = DanmakuFilterDialog(content_filter, "21399689")
        assert dialog.candidates_table.rowCount() == 0
        dialog.show_history.setChecked(True)
        dialog.candidates_table.selectRow(0)
    assert dialog.candidates_table.item(0, 7).text() == tr("独立测试房间验证通过，正式房间效果未确认")
    assert tr("处理记录归档不代表弹幕已经送达。") in dialog.candidate_analysis.toPlainText()
