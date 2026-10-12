"""Durable subtitle delivery tests; no websites, cookies or live POSTs."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from fanyiji.danmaku_service import DanmakuService, split_danmaku
from fanyiji.danmaku_store import DanmakuStore
from fanyiji.danmaku_protocol import danmaku_weighted_length, danmaku_utf16_length
from fanyiji.danmaku_filter import DanmakuContentFilter, DanmakuFilterError


class FakeBrowser(QObject):
    danmaku_result = Signal(dict)
    event = Signal(dict)

    def __init__(self):
        super().__init__()
        self.is_browser_ready = True
        self.requests = []
        self.accept_requests = True

    def request_danmaku(self, operation, request_id, room, text=""):
        self.requests.append({"operation": operation, "request_id": request_id, "room": room, "text": text})
        return self.accept_requests

    def reply(self, **result):
        request = self.requests[-1]
        if request["operation"] == "login":
            result.setdefault("max_weighted_length", 80)
        elif request["operation"] == "send" and result.get("outcome") != "not_sent":
            result.setdefault("before_count", 0)
            result.setdefault("request_started_epoch_ms", 1000000)
        elif request["operation"] == "check":
            result.setdefault("occurrence_count", 1)
        self.danmaku_result.emit({"request_id": request["request_id"], "operation": request["operation"], "result": result})


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def setup_service(app, tmp_path):
    browser = FakeBrowser()
    service = DanmakuService(browser, store_path=tmp_path / "queue.sqlite3")
    now = [100.0]
    service._clock = lambda: now[0]
    service.configure(True, "1698225")
    service._apply_limits({"max_weighted_length": 80})
    service.timer.stop()
    service.begin_session(1)
    yield service, browser, now
    service.close()


def event(service, kind, *, segment="1", revision=1, text="你好", provisional=False,
          channel="system", session=1, **extra):
    service.handle_event({"kind": kind, "session": session, "channel": channel,
                          "segment_id": segment, "revision": revision, "text": text,
                          "provisional": provisional, **extra})


def enqueue(service, text="用于验证队列的完整测试译文", segment="1"):
    event(service, "original", segment=segment, text="日语原文")
    event(service, "translation", segment=segment, text=text, translation_stage="final")


def login(service, browser, now):
    service.tick()
    assert browser.requests[-1]["operation"] == "login"
    browser.reply(is_login=True, code=0)
    now[0] += 3.5
    service.tick()
    assert browser.requests[-1]["operation"] == "send"


def finish(service, browser, now):
    browser.reply(outcome="receipt_accepted", code=0, http_status=200)
    now[0] += 3.5
    service.tick()
    assert browser.requests[-1]["operation"] == "check"
    browser.reply(present=True, room_matches=True, chat_ready=True)


@pytest.mark.parametrize("text", [None, "", "None", "null", "语言 None<asr_text>",
    "&lt;asr_text&gt;你好", "<|im_start|>你好", "<script>你好</script>",
    "[此片段翻译失败]", "[识别控制标记异常，已跳过此片段]", "翻译中…", "等待后半句…", "hello\x01"])
def test_protocol_and_placeholders_are_not_sendable(text):
    assert split_danmaku(text) == []


@pytest.mark.parametrize("text", [
    "首先 大家请冷静一点 レレン 谢谢你的留言 等一下 クちゃん也 谢谢你的留言",
    "谢谢レレン的留言", "游戏进入 DLC 关卡", "谢谢 ＡＢＣ 的留言",
    "谢谢ﾚﾚﾝ的留言", "主播说한국谢谢", "主播说Привет谢谢", "谢谢🙂的留言",
])
def test_foreign_final_is_recorded_outside_fifo_and_next_chinese_can_send(setup_service, text):
    service, browser, now = setup_service
    enqueue(service, text, "foreign")
    assert service.store.head() is None
    assert service.snapshot()["queued"] == 0
    candidate = service.content_filter.candidates()[0]
    assert candidate["text"] == text
    assert candidate["reason"] == "non_chinese_translation"
    assert candidate["diagnosis"]["category"] == "output_language"
    assert service.store.failed_handoffs() == []
    assert browser.requests == []
    # The same model event is not counted twice, and names were not stripped.
    event(service, "translation", segment="foreign", text=text, translation_stage="final")
    assert service.content_filter.candidates()[0]["count"] == 1
    enqueue(service, "第2局开始 大家谢谢雷伦的留言", "next")
    login(service, browser, now)
    assert browser.requests[-1]["text"] == "【第2局开始 大家谢谢雷伦的留言】"
    assert not any(text == row["text"] for row in browser.requests)


def test_same_segment_chinese_correction_after_foreign_rejection_can_enter_fifo(setup_service):
    service, _, _ = setup_service
    enqueue(service, "谢谢レレン的留言", "one")
    event(service, "translation", segment="one", text="谢谢雷伦的留言 大家请冷静", translation_stage="final")
    assert service.store.head()["text"] == "【谢谢雷伦的留言 大家请冷静】"


def test_foreign_final_is_submitted_to_independent_ai_review(setup_service):
    service, browser, _ = setup_service
    class Recorder:
        busy = False
        tasks = []
        def submit(self, token, data, settings):
            self.tasks.append(data)
            self.busy = True
            return True
        def close(self):
            pass
    analyzer = Recorder()
    service._ai_analyzer = analyzer
    text = "首先 大家请冷静一点 レレン 谢谢你的留言"
    enqueue(service, text)
    service._review_history_tick()
    assert analyzer.tasks[0]["text"] == text
    assert service.content_filter.candidates()[0]["status"] == "recovering"
    assert service.store.head() is None
    assert browser.requests == []


def test_japanese_asr_can_confirm_an_all_chinese_translation(setup_service):
    service, browser, now = setup_service
    event(service, "original", text="まずみんな落ち着いて レレン コメントありがとう")
    event(service, "translation", text="首先大家冷静 雷伦谢谢留言", translation_stage="final")
    login(service, browser, now)
    assert browser.requests[-1]["text"] == "【首先大家冷静 雷伦谢谢留言】"


def test_filter_injected_foreign_output_is_rejected_before_enqueue(setup_service, monkeypatch):
    service, browser, _ = setup_service
    monkeypatch.setattr(service, "_filter_translation", lambda text, room: ("主播进入 DLC 关卡", ""))
    enqueue(service, "主播进入下载内容关卡")
    assert service.store.head() is None
    assert service.content_filter.candidates()[0]["text"] == "主播进入 DLC 关卡"
    service.tick()
    assert browser.requests == []


@pytest.mark.parametrize("state", ["pending", "holding_short", "retry_pending"])
def test_legacy_or_manual_foreign_head_is_deferred_before_any_browser_request(setup_service, state):
    service, browser, now = setup_service
    service.store.enqueue("1698225", "system", "old", 1, "旧句含レレン名字", ["【旧句含レレン名字】"])
    identifier = service.store.head()["id"]
    service.store.update(identifier, state)
    enqueue(service, "后续纯中文字幕继续发送 不要卡住", "next")
    service.tick()
    assert browser.requests == []
    assert service.content_filter.candidates()[0]["text"] == "【旧句含レレン名字】"
    assert service.snapshot()["deferred_analysis"] == 1
    login(service, browser, now)
    assert browser.requests[-1]["text"] == "【后续纯中文字幕继续发送 不要卡住】"


def test_direct_dispatch_foreign_head_cannot_bypass_language_gate(setup_service):
    service, browser, _ = setup_service
    service.store.enqueue("1698225", "system", "manual", 1, "手工混入ABC名字", ["【手工混入ABC名字】"])
    service._dispatch("send", service.store.head())
    assert browser.requests == []
    assert service.store.head() is None
    assert service.content_filter.candidates()[0]["text"] == "【手工混入ABC名字】"


def test_historical_foreign_text_can_still_be_checked_without_post(setup_service):
    service, browser, _ = setup_service
    text = "【旧句含レレン名字】"
    service.store.enqueue("1698225", "system", "old", 1, text, [text])
    identifier = service.store.head()["id"]
    service.store.update(identifier, "verify", reason="restart_verify")
    service._login_room = service.room
    service.tick()
    assert [request["operation"] for request in browser.requests] == ["check"]
    assert browser.requests[0]["text"] == text


def test_foreign_candidate_json_write_failure_survives_restart_without_blocking_chinese(app, tmp_path, monkeypatch):
    path = tmp_path / "queue.sqlite3"
    browser = FakeBrowser()
    service = DanmakuService(browser, store_path=path)
    service.configure(True, "1698225")
    service.timer.stop()
    service.begin_session(1)
    def denied(*args, **kwargs):
        raise DanmakuFilterError("write denied")
    monkeypatch.setattr(service.content_filter, "record_failure", denied)
    text = "谢谢レレン的留言 大家请冷静"
    enqueue(service, text, "foreign")
    assert service.store.head() is None
    assert service.store.failed_handoffs()[0]["text"] == text
    assert service.snapshot()["filter_candidate_error"]
    enqueue(service, "后续纯中文字幕应当正常发送", "next")
    service.tick()
    browser.reply(is_login=True, code=0)
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert "后续" in browser.requests[-1]["text"]
    service.close()
    restored = DanmakuService(FakeBrowser(), store_path=path, reset_on_start=True)
    try:
        assert restored.store.head() is None
        assert restored.content_filter.candidates()[0]["text"] == text
        assert restored.store.failed_handoffs() == []
    finally:
        restored.close()


def test_foreign_candidate_sqlite_write_failure_is_reported_and_never_posts(setup_service, monkeypatch):
    service, browser, _ = setup_service
    def denied(*args, **kwargs):
        raise OSError("disk unavailable")
    monkeypatch.setattr(service.store, "stage_failed_handoff", denied)
    enqueue(service, "谢谢レレン的留言")
    assert service.store.head() is None
    assert service.snapshot()["state"] == "storage_error"
    service.tick()
    assert browser.requests == []


@pytest.mark.parametrize("raw,text", [(20, "字" * 36), (40, "字" * 36),
    (20, "abcdefghijklmnopqrstuvwxyz"), (20, "😀" * 20),
    (40, "hello 日本語 friend 😀" * 8), (80, "word " * 100)])
def test_weighted_splitter_obeys_both_limits_without_losing_text(raw, text):
    parts = split_danmaku(text, max_weighted_length=raw)
    assert parts
    assert all(danmaku_weighted_length(part) <= raw and danmaku_utf16_length(part) <= 40 for part in parts)
    assert "".join(part[1:-1] for part in parts).replace(" ", "") == text.replace(" ", "")


@pytest.mark.parametrize("raw,chars,text", [(6, 40, "😀"), (20, 3, "😀"),
    (True, 40, "hello"), (20, True, "hello"), (5, 40, "hello")])
def test_weighted_splitter_rejects_impossible_or_invalid_caps(raw, chars, text):
    assert split_danmaku(text, max_weighted_length=raw, max_characters=chars) == []


def test_initial_queue_uses_eighty_but_invalid_account_cap_falls_back_to_twenty(app, tmp_path):
    service = DanmakuService(FakeBrowser(), store_path=tmp_path / "queue.sqlite3")
    try:
        assert service.snapshot()["max_weighted_length"] == service.store.max_weighted_length == 80
        service._apply_limits({"max_weighted_length": True})
        assert service.max_weighted_length == service.store.max_weighted_length == 20
    finally:
        service.close()


def low_limit_login(service, browser, raw=24):
    service.tick()
    assert browser.requests[-1]["operation"] == "login"
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=raw)
    service.tick()
    assert browser.requests[-1]["operation"] == "send"


def test_account_packets_keep_fifo_and_short_tail_remains_sendable(setup_service):
    service, browser, now = setup_service
    original = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉"
    enqueue(service, original, "first")
    enqueue(service, "后来完整字幕不能越过原句", "later")
    first_id = service.store.head()["id"]
    low_limit_login(service, browser)
    expected = split_danmaku(original, max_weighted_length=24)
    assert service.snapshot()["queued"] == len(expected) + 1
    for index, packet in enumerate(expected):
        assert browser.requests[-1]["text"] == packet
        assert service.store.head()["id"] == first_id
        finish(service, browser, now)
        assert service.snapshot()["page_visible"] == index + 1
        if index != len(expected) - 1:
            assert service.store.head()["state"] == "pending"
            service.tick()
            assert browser.requests[-1]["operation"] == "send"
    assert service.store.head()["id"] != first_id
    service.tick()
    assert browser.requests[-1]["text"] == split_danmaku("后来完整字幕不能越过原句", max_weighted_length=24)[0]


def test_known_length_rejection_is_reauthenticated_and_repacketized(setup_service):
    service, browser, now = setup_service
    enqueue(service, "这是超过账号默认长度限制的完整字幕")
    head = service.store.head()
    service.store.update(head["id"], "blocked", request_id="old", reason="platform_rejected", platform_code=1003212)
    low_limit_login(service, browser, raw=24)
    assert service.store.head()["id"] == head["id"]
    assert danmaku_weighted_length(browser.requests[-1]["text"]) <= 24
    assert service.store.remaining_parts(head["id"])


def test_fitting_length_rejection_is_not_blindly_retried(setup_service):
    service, browser, now = setup_service
    enqueue(service, "这是正常长度的完整字幕")
    head = service.store.head()
    service.store.update(head["id"], "blocked", request_id="old", reason="platform_rejected", platform_code=1003212)
    service.tick()
    browser.reply(is_login=True, code=0, max_weighted_length=80)
    service.tick()
    assert len(browser.requests) == 1
    assert service.store.head()["state"] == "blocked"


@pytest.mark.parametrize("reason,code", [("platform_f", 0), ("result_unknown", None), ("platform_rejected", 1003)])
def test_lower_account_cap_never_rewrites_other_blocked_heads(setup_service, reason, code):
    service, browser, now = setup_service
    enqueue(service, "这是超过账号默认长度限制的完整字幕")
    head = service.store.head()
    service.store.update(head["id"], "blocked", reason=reason, platform_code=code)
    service._apply_limits({"max_weighted_length": 20})
    service.tick()
    assert [row["operation"] for row in browser.requests] == (["login"] if reason in {"platform_f", "result_unknown"} else [])
    assert service.store.head()["text"] == head["text"]
    assert service.store.remaining_parts(head["id"]) == []


def test_send_runtime_lowered_cap_not_sent_is_preserved_for_analysis(setup_service):
    service, browser, now = setup_service
    enqueue(service, "这是超过账号默认长度限制的完整字幕")
    login(service, browser, now)
    browser.reply(outcome="not_sent", error="text_too_long", max_weighted_length=24)
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.content_filter.candidates()[0]["text"] == "【这是超过账号默认长度限制的完整字幕】"
    now[0] += 3.5
    service.tick()
    assert sum(item["operation"] == "send" for item in browser.requests) == 1
    assert service.snapshot()["page_visible"] == 0


def test_account_limit_change_does_not_reset_content_window(setup_service):
    service, browser, now = setup_service
    service.store._clock = lambda: 1000.0
    text = "这是超过账号默认长度限制的完整字幕"
    enqueue(service, text, "first")
    low_limit_login(service, browser, raw=24)
    count = service.snapshot()["queued"]
    service._apply_limits({"max_weighted_length": 40})
    enqueue(service, text, "duplicate")
    assert service.snapshot()["queued"] == count


def test_low_account_limit_keeps_unmergeable_original_short(setup_service):
    service, browser, now = setup_service
    service._apply_limits({"max_weighted_length": 20})
    enqueue(service, "这是一个短句", "first")
    assert service.store.head()["state"] == "holding_short"
    enqueue(service, "下一条很短句", "next")
    assert service.snapshot()["queued"] == 2
    assert service.store.head()["text"] == "【这是一个短句】"
    assert service.store.head()["state"] == "holding_short"


def test_weighted_packets_restart_after_partial_confirmation(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    browser = FakeBrowser()
    first = DanmakuService(browser, store_path=path)
    first.configure(True, "1698225")
    first.timer.stop()
    first.begin_session(1)
    now = [100.0]
    first._clock = lambda: now[0]
    original = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉"
    enqueue(first, original, "first")
    low_limit_login(first, browser)
    packets = split_danmaku(original, max_weighted_length=24)
    identifier = first.store.head()["id"]
    finish(first, browser, now)
    assert first.store.head()["text"] == packets[1]
    first.close()
    second_browser = FakeBrowser()
    second = DanmakuService(second_browser, store_path=path)
    try:
        second.configure(True, "1698225")
        second.timer.stop()
        second._clock = lambda: now[0]
        low_limit_login(second, second_browser)
        assert second.store.head()["id"] == identifier
        assert second_browser.requests[-1]["text"] == packets[1]
        assert second.snapshot()["page_visible"] == 1
        assert second.snapshot()["queued"] == len(packets) - 1
    finally:
        second.close()


@pytest.mark.parametrize("operation", ["send", "check", "login"])
def test_clear_continuations_preserves_inflight_packet_and_never_revives_tail(setup_service, operation):
    service, browser, now = setup_service
    text = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉"
    enqueue(service, text)
    if operation == "login":
        identifier = service.store.head()["id"]
        assert service.store.prepare_parts(identifier, split_danmaku(text, max_weighted_length=24))
        service.tick()
    else:
        low_limit_login(service, browser)
    identifier = service.store.head()["id"]
    if operation == "check":
        browser.reply(outcome="receipt_accepted", code=0)
        service.tick()
    assert service._inflight["operation"] == operation
    assert service.snapshot()["clearable"] == 1
    assert service.clear_pending() == {"removed": 1, "retained": 1, "retained_uncertain": 1}
    assert service.store.remaining_parts(identifier) == []
    if operation == "login":
        browser.reply(is_login=True, code=0, max_weighted_length=24)
        service.tick()
    if operation != "check":
        browser.reply(outcome="receipt_accepted", code=0)
        service.tick()
    browser.reply(present=True, room_matches=True, chat_ready=True)
    assert service.snapshot()["queued"] == 0
    assert service.snapshot()["cleared"] == 1
    assert service.snapshot()["page_visible"] == 1


def test_clear_unknown_packet_keeps_analysis_and_cancels_only_unsubmitted_continuations(setup_service):
    service, browser, now = setup_service
    enqueue(service, "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉")
    low_limit_login(service, browser)
    head = service.store.head()
    browser.reply(outcome="result_unknown")
    assert service.content_filter.candidates()[0]["text"] == head["text"]
    assert service.store.head()["id"] == head["id"]
    assert service.store.head()["part"] == head["part"] + 1
    assert service.store.head()["state"] == "pending"
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.clear_pending()["removed"] == 1
    assert service.store.head() is None
    assert service.snapshot()["queued"] == 0
    assert service.content_filter.candidates()[0]["text"] == head["text"]
    assert service.snapshot()["page_visible"] == 0


def test_delivery_length_and_outcome_logs_are_redacted(setup_service):
    service, browser, now = setup_service
    enqueue(service, "只在队列存储而不进入日志的完整正文")
    low_limit_login(service, browser)
    browser.reply(outcome="receipt_accepted", code=0, http_status=200,
                  max_weighted_length=24, request_started_epoch_ms=1720000000000, request_elapsed_ms=100)
    lines = [json.loads(line) for line in service.log_path.read_text(encoding="utf-8").splitlines()]
    sent = next(row for row in lines if row["kind"] == "result" and row["operation"] == "send")
    assert sent["outcome"] == "receipt_accepted"
    assert sent["weighted_length"] <= sent["max_weighted_length"] == 24
    assert sent["characters"] == danmaku_utf16_length(browser.requests[-1]["text"])
    assert "正文" not in service.log_path.read_text(encoding="utf-8")


def test_old_identical_dom_row_is_not_confirmation_of_new_post(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, before_count=1)
    service.tick()
    browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=1)
    assert service.snapshot()["page_visible"] == 0
    assert service.store.head()["state"] == "verify"
    now[0] += 0.5
    service.tick()
    browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=2)
    assert service.snapshot()["page_visible"] == 1
    assert len([r for r in browser.requests if r["operation"] == "send"]) == 1


@pytest.mark.parametrize("before,start", [(None, 100000), (True, 100000), (-1, 100000),
    (10001, 100000), (0, None), (0, True), (0, float("nan"))])
def test_missing_or_invalid_submission_baseline_stays_uncertain(setup_service, before, start):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, before_count=before, request_started_epoch_ms=start)
    assert service.store.delivery_baseline(service.store.head()["id"]) is None
    service.tick()
    browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=3)
    assert service.snapshot()["page_visible"] == 0
    now[0] += 15
    service.tick()
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.content_filter.candidates()[0]["reason"] == "receipt_unconfirmed"


def test_submission_baseline_survives_restart_and_new_part_resets_it(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    browser = FakeBrowser()
    first = DanmakuService(browser, store_path=path)
    first.configure(True, "1698225")
    first.timer.stop()
    first.begin_session(1)
    now = [100.0]
    first._clock = lambda: now[0]
    enqueue(first, "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉")
    low_limit_login(first, browser)
    identifier = first.store.head()["id"]
    first_text = first.store.head()["text"]
    browser.reply(outcome="receipt_accepted", code=0, before_count=1)
    first.close()
    second_browser = FakeBrowser()
    second = DanmakuService(second_browser, store_path=path)
    try:
        second.configure(True, "1698225")
        second.timer.stop()
        second._clock = lambda: now[0]
        assert second.store.delivery_baseline(identifier) == 1
        second.tick()
        second_browser.reply(is_login=True, code=0, max_weighted_length=24)
        second.tick()
        assert second_browser.requests[-1]["operation"] == "check"
        second_browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=1)
        assert second.store.head()["text"] == first_text
        now[0] += 0.5
        second.tick()
        second_browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=2)
        assert second.store.head()["id"] == identifier
        assert second.store.head()["text"] != first_text
        assert second.store.delivery_baseline(identifier) is None
        second.tick()
        assert second_browser.requests[-1]["operation"] == "send"
    finally:
        second.close()


def test_clearing_unknown_head_tails_keeps_submission_baseline(setup_service):
    service, browser, now = setup_service
    enqueue(service, "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉")
    low_limit_login(service, browser)
    identifier = service.store.head()["id"]
    browser.reply(outcome="receipt_accepted", code=0, before_count=1)
    service.clear_pending()
    assert service.store.delivery_baseline(identifier) == 1
    service.verify_pending()
    service.tick()
    browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=1)
    assert service.snapshot()["queued"] == 1
    assert service.snapshot()["page_visible"] == 0


@pytest.mark.parametrize("text,expected", [
    ("第一句。第二句，第三句！", "【第一句 第二句 第三句】"),
    ("日语：「ありがとう」 中文【谢谢】", "【日语 ありがとう 中文 谢谢】"),
    ("a~b～c〜d`e｀f|g｜h", "【a b c d e f g h】"),
    ("开始!?；…结束", "【开始 结束】"),
    ("价格 $12 + ￥8 = ￥20 ×2 ÷4", "【价格 $12 + ￥8 = ￥20 ×2 ÷4】"),
    ("数值 3.5 -2 +4", "【数值 3 5 2 +4】"),
    ("【第一句， 第二句。】", "【第一句 第二句】"),
])
def test_body_punctuation_becomes_collapsed_sentence_spaces(text, expected):
    assert split_danmaku(text) == [expected]


@pytest.mark.parametrize("wrapped_length,parts", [(39, 1), (40, 1), (41, 2)])
def test_default_global_limit_includes_brackets_and_splits_over_forty(wrapped_length, parts):
    chunks = split_danmaku("字" * (wrapped_length - 2))
    assert len(chunks) == parts
    assert all(danmaku_utf16_length(chunk) <= 40 for chunk in chunks)
    assert "".join(chunk[1:-1] for chunk in chunks) == "字" * (wrapped_length - 2)


@pytest.mark.parametrize("following,queued", [(34, 1), (35, 1), (36, 2)])
def test_short_merge_counts_brackets_and_sentence_space_at_39_40_41(setup_service, following, queued):
    service, _, _ = setup_service
    enqueue(service, "短句", "first")
    enqueue(service, "字" * following, "next")
    assert service.snapshot()["queued"] == queued
    if queued == 1:
        assert danmaku_utf16_length(service.store.head()["text"]) == following + 5
        assert service.store.head()["text"].startswith("【短句 ")
    else:
        head = service.store.head()
        parts = [head["text"], *service.store.remaining_parts(head["id"])]
        assert all(11 <= danmaku_utf16_length(part) <= 40 for part in parts)
        assert "".join(part[1:-1].replace(" ", "") for part in parts) == "短句" + "字" * following
        assert service.store.head()["state"] == "pending"


@pytest.mark.parametrize("emoji_count", [3, 4, 5])
def test_emoji_final_is_rejected_regardless_of_its_utf16_threshold(setup_service, emoji_count):
    service, _, _ = setup_service
    enqueue(service, "😀" * emoji_count)
    assert service.snapshot()["queued"] == 0
    assert service.content_filter.candidates()[0]["text"] == "😀" * emoji_count


def test_send_cleanup_does_not_rewrite_original_model_translation(setup_service):
    service, _, _ = setup_service
    original_translation = "主播说：这是第一句话，随后说第二句话。"
    enqueue(service, original_translation)
    assert service._segments[("system", "1")]["translation"] == original_translation
    assert service.store.head()["text"] == "【主播说 这是第一句话 随后说第二句话】"


@pytest.mark.parametrize("reason", ["pending", "length_rejected"])
def test_old_76_packets_are_normalized_and_split_without_losing_continuations(setup_service, reason):
    service, browser, now = setup_service
    enqueue(service, "可替换的完整旧版本字幕")
    identifier = service.store.head()["id"]
    old_head = "【" + "甲" * 30 + "，" + "乙" * 30 + "。】"
    old_tail = "【" + "丙" * 50 + "！】"
    with service.store.connection:
        service.store.connection.execute("UPDATE messages SET text=?,state=? WHERE id=?",
            (old_head, "blocked" if reason == "length_rejected" else "pending", identifier))
        service.store._set_metadata(f"remaining_parts:{identifier}", json.dumps([old_tail], ensure_ascii=False))
    if reason == "length_rejected":
        service.store.update(identifier, "blocked", reason="platform_rejected", platform_code=1003212)
    assert service.store.remaining_parts(identifier) == [old_tail]
    low_limit_login(service, browser, raw=80)
    current = service.store.head()
    packets = [current["text"], *service.store.remaining_parts(identifier)]
    assert current["id"] == identifier
    assert len(packets) == 4
    assert all(danmaku_utf16_length(packet) <= 40 for packet in packets)
    assert "".join(packet[1:-1] for packet in packets).replace(" ", "") == "甲" * 30 + "乙" * 30 + "丙" * 50
    assert "，" not in "".join(packets) and "。" not in "".join(packets) and "！" not in "".join(packets)


@pytest.mark.parametrize("state,reason", [("verify", "receipt_unconfirmed"), ("blocked", "result_unknown"), ("blocked", "platform_f")])
def test_old_unknown_or_f_packet_keeps_exact_76_text_and_remaining_parts(setup_service, state, reason):
    service, browser, now = setup_service
    enqueue(service, "可替换的完整旧版本字幕")
    identifier = service.store.head()["id"]
    old_text = "【" + "旧" * 60 + "，？】"
    old_tail = "【" + "尾" * 60 + "！】"
    with service.store.connection:
        service.store.connection.execute("UPDATE messages SET text=? WHERE id=?", (old_text, identifier))
        service.store._set_metadata(f"remaining_parts:{identifier}", json.dumps([old_tail], ensure_ascii=False))
    service.store.update(identifier, state, reason=reason, platform_code=0)
    service.store.set_delivery_baseline(identifier, 0)
    service._apply_limits({"max_weighted_length": 24})
    if state == "blocked":
        service.verify_pending()
    service.tick()
    browser.reply(is_login=True, code=0, max_weighted_length=24)
    service.tick()
    assert browser.requests[-1]["operation"] == "check"
    assert browser.requests[-1]["text"] == old_text
    assert service.store.remaining_parts(identifier) == [old_tail]
    browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=1)
    assert service.store.head()["id"] == identifier
    assert service.store.head()["text"] == old_tail
    # Only after the old current packet is verified may the next packet change.
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert danmaku_utf16_length(browser.requests[-1]["text"]) <= 40
    assert "！" not in browser.requests[-1]["text"]


@pytest.mark.parametrize("old_text", ["【字字字字字字字字】", "【字字字字字字字字字】"])
def test_old_holdings_below_eleven_remain_held_without_new_audio(setup_service, old_text):
    service, browser, now = setup_service
    enqueue(service, "旧短句")
    identifier = service.store.head()["id"]
    with service.store.connection:
        service.store.connection.execute("UPDATE messages SET text=?,state='holding_short' WHERE id=?", (old_text, identifier))
    if danmaku_utf16_length(old_text) < 11:
        service.tick()
        assert [row["operation"] for row in browser.requests] == ["login"]
        browser.reply(is_login=True, code=0, error="none")
        service.tick()
        assert not any(row["operation"] == "send" for row in browser.requests)
        assert service.store.head()["state"] == "holding_short"
        return
    low_limit_login(service, browser, raw=24)
    assert service.store.head()["id"] == identifier
    assert service.store.head()["state"] == "sending"
    assert danmaku_weighted_length(browser.requests[-1]["text"]) <= 24


def test_old_holding_threshold_is_measured_after_punctuation_cleanup(setup_service):
    service, browser, now = setup_service
    enqueue(service, "旧短句")
    identifier = service.store.head()["id"]
    with service.store.connection:
        service.store.connection.execute("UPDATE messages SET text=?,state='holding_short' WHERE id=?", ("【字字字字字字字！！】", identifier))
    service.tick()
    assert [row["operation"] for row in browser.requests] == ["login"]
    browser.reply(is_login=True, code=0, error="none")
    service.tick()
    assert not any(row["operation"] == "send" for row in browser.requests)
    assert service.store.head()["state"] == "holding_short"


@pytest.mark.parametrize("text", ["这是正常字幕。", "正常 " * 100, "字" * 300,
    "First sentence. " * 30, "【正常译文】", "一句中文，另一句。" * 20])
def test_long_subtitles_are_wrapped_and_bounded(text):
    chunks = split_danmaku(text)
    assert chunks
    assert all(chunk.startswith("【") and chunk.endswith("】") and len(chunk) <= 40 for chunk in chunks)
    assert all("【【" not in chunk for chunk in chunks)


def test_final_requires_confirmed_safe_original(setup_service):
    service, browser, now = setup_service
    event(service, "translation", text="等待原文确认", translation_stage="final", provisional=True)
    assert service.snapshot()["queued"] == 0
    event(service, "original", text="日语半句", provisional=True)
    assert service.snapshot()["queued"] == 0
    event(service, "subtitle_final")
    assert service.snapshot()["queued"] == 1
    event(service, "subtitle_final")
    assert service.snapshot()["queued"] == 1


def test_confirmation_before_translation_is_preserved(setup_service):
    service, _, _ = setup_service
    event(service, "original", provisional=True)
    event(service, "subtitle_final")
    event(service, "translation", text="真正终稿", translation_stage="final", provisional=True)
    assert service.store.head()["text"] == "【真正终稿】"


def test_confirmation_before_original_is_safe(setup_service):
    service, _, _ = setup_service
    event(service, "subtitle_final")
    event(service, "translation", translation_stage="final", provisional=True)
    assert service.snapshot()["queued"] == 0
    event(service, "original", text="原文", provisional=True)
    assert service.snapshot()["queued"] == 1


@pytest.mark.parametrize("extra", [{"translation_stage": "draft"}, {"failed": True},
    {"overall_failed": True}, {"translation_stage": "final", "failed": True}])
def test_only_successful_final_translations_enter_queue(setup_service, extra):
    service, _, _ = setup_service
    event(service, "original")
    event(service, "translation", **extra)
    assert service.snapshot()["queued"] == 0


def test_protocol_in_original_blocks_otherwise_valid_translation(setup_service):
    service, _, _ = setup_service
    event(service, "original", text="None<asr_text>")
    event(service, "translation", text="不能发送", translation_stage="final")
    event(service, "subtitle_final")
    assert service.snapshot()["queued"] == 0


def test_new_revision_rejects_old_result(setup_service):
    service, _, _ = setup_service
    event(service, "original", revision=1, provisional=True)
    event(service, "translation", revision=1, text="旧稿", provisional=True)
    event(service, "original", revision=2, text="修订原文", provisional=True)
    event(service, "translation", revision=1, text="迟到旧稿")
    event(service, "subtitle_final", revision=1)
    assert service.snapshot()["queued"] == 0
    event(service, "translation", revision=2, text="新终稿", provisional=True)
    event(service, "subtitle_final", revision=2)
    assert service.store.head()["text"] == "【新终稿】"


def test_old_session_other_channel_and_disabled_events_are_ignored(setup_service):
    service, _, _ = setup_service
    event(service, "original", session=0)
    event(service, "translation", session=0)
    event(service, "original", channel="mic")
    event(service, "translation", channel="mic")
    assert service.snapshot()["queued"] == 0
    service.configure(False, "1698225")
    enqueue(service)
    assert service.snapshot()["queued"] == 0


def test_pause_session_reset_and_room_change_keep_existing_fifo(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    service.configure(False, "1698225")
    service.begin_session(2)
    now[0] += 10000
    service.tick()
    assert not browser.requests
    assert service.snapshot()["queued"] == 1
    service.configure(True, "21399689")
    service.tick()
    assert service.snapshot()["state"] == "queued_room_mismatch"
    assert not browser.requests


def test_fifo_and_every_post_respects_minimum_after_result(setup_service):
    service, browser, now = setup_service
    enqueue(service, "第一条完整终稿字幕用于测试", "1")
    enqueue(service, "第二条完整终稿字幕用于测试", "2")
    login(service, browser, now)
    assert browser.requests[-1]["text"] == "【第一条完整终稿字幕用于测试】"
    browser.reply(outcome="receipt_accepted", code=0, http_status=200)
    service.tick()
    assert browser.requests[-1]["operation"] == "check"
    browser.reply(present=True, room_matches=True, chat_ready=True)
    assert service.snapshot()["page_visible"] == 1
    now[0] += 3.49
    service.tick()
    assert len(browser.requests) == 3
    now[0] += 0.01
    service.tick()
    assert browser.requests[-1]["text"] == "【第二条完整终稿字幕用于测试】"


@pytest.mark.parametrize("interval", [-10, 0, 1, 2, 3.49, 3.5, 4, 60])
def test_interval_has_a_hard_floor(setup_service, interval):
    service, _, _ = setup_service
    service.configure(True, "1698225", interval_seconds=interval)
    assert service.snapshot()["interval_seconds"] == max(3.5, interval)


@pytest.mark.parametrize("interval", [float("nan"), float("inf"), -float("inf"), "invalid"])
def test_invalid_interval_is_rejected(setup_service, interval):
    service, _, _ = setup_service
    with pytest.raises(ValueError):
        service.configure(True, "1698225", interval_seconds=interval)


def test_no_parallel_request_before_ack(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    service.tick()
    now[0] += 40
    service.tick()
    assert len(browser.requests) == 1


@pytest.mark.parametrize("outcome,code", [("platform_rejected", 10030)])
def test_platform_refusal_keeps_analysis_and_never_auto_retries_original(setup_service, outcome, code):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome=outcome, code=code, http_status=200)
    assert not service.snapshot()["blocked"]
    assert service.store.head() is None
    assert service.content_filter.candidates()[0]["reason"] == outcome
    assert service.snapshot()["deferred_analysis"] == 1
    now[0] += 3000
    service.tick()
    assert len(browser.requests) == 2
    assert service.snapshot()["queued"] == 0
    service.configure(True, "1698225")
    service.tick()
    assert len(browser.requests) == 2


def test_receipt_without_visibility_is_not_delivery_or_auto_retry(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, http_status=200)
    assert service.snapshot()["queued"] == 1
    assert service.snapshot()["page_visible"] == 0
    now[0] += 3.5
    service.tick()
    browser.reply(present=False, room_matches=True, chat_ready=True)
    assert not service.snapshot()["blocked"]
    now[0] += 1000
    service.tick()
    assert not service.snapshot()["blocked"]
    assert service.snapshot()["queued"] == 0
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.content_filter.candidates()[0]["reason"] == "receipt_unconfirmed"
    assert [row["operation"] for row in browser.requests] == ["login", "send", "check"]


@pytest.mark.parametrize("result", [{"present": True, "room_matches": False, "chat_ready": True},
    {"present": True, "room_matches": True, "chat_ready": False},
    {"present": True, "room_matches": True, "chat_ready": True, "error": "unknown"},
    {"broadcast_confirmed": True}])
def test_visibility_requires_correct_loaded_room(setup_service, result):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0)
    now[0] += 3.5
    service.tick()
    browser.reply(**result)
    assert service.snapshot()["page_visible"] == 0
    now[0] += 15
    service.tick()
    assert not service.snapshot()["blocked"]
    assert service.snapshot()["queued"] == 0
    assert service.snapshot()["deferred_analysis"] == 1


def test_not_sent_is_preserved_in_analysis_without_reposting_original(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="not_sent", error="wrong_room")
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.content_filter.candidates()[0]["reason"] == "not_sent"
    now[0] += 3.5
    service.tick()
    assert [item["operation"] for item in browser.requests] == ["login", "send"]


def test_timeout_preserves_uncertain_item_in_analysis_and_does_not_resend(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    now[0] += 45
    service.tick()
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.content_filter.candidates()[0]["reason"] == "result_unknown"
    service.resume()
    now[0] += 3.5
    service.tick()
    assert [item["operation"] for item in browser.requests] == ["login", "send"]


def test_manual_resume_after_transferred_refusal_does_not_repost_failed_original(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="platform_rejected", code=10030)
    failed = service.content_filter.candidates()[0]
    service.resume()
    now[0] += 3.5
    service.tick()
    now[0] += 100
    service.tick()
    assert len(browser.requests) == 2
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.content_filter.candidates()[0] == failed


def test_pending_final_and_duplicate_detection_survive_restart(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    first = DanmakuService(FakeBrowser(), store_path=path)
    first.configure(True, "1698225")
    first.begin_session(1)
    enqueue(first, "重复译文")
    first.close()
    second = DanmakuService(FakeBrowser(), store_path=path)
    try:
        second.configure(True, "1698225")
        second.begin_session(2)
        event(second, "original", session=2)
        event(second, "translation", session=2, text="重复译文")
        assert second.snapshot()["queued"] == 1
        assert second.store.head()["text"] == "【重复译文】"
    finally:
        second.close()


def test_inflight_after_restart_requires_check_not_another_post(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    store = DanmakuStore(path)
    store.enqueue("1698225", "system", "segment", 1, "已投递", ["【已投递】"])
    head = store.head()
    store.update(head["id"], "sending", request_id="previous-request")
    store.close()
    browser = FakeBrowser()
    service = DanmakuService(browser, store_path=path)
    try:
        service._clock = lambda: 100
        service.configure(True, "1698225")
        assert service.store.head()["state"] == "verify"
        service.tick()
        browser.reply(is_login=True, code=0)
        service._clock = lambda: 104
        service.tick()
        assert browser.requests[-1]["operation"] == "check"
        assert all(row["operation"] != "send" for row in browser.requests)
    finally:
        service.close()


def test_wrong_request_id_cannot_ack_another_message(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.danmaku_result.emit({"request_id": "stale", "operation": "send",
                                "result": {"outcome": "receipt_accepted", "code": 0}})
    assert service.snapshot()["inflight"] == "send"


def test_logs_do_not_contain_text_or_private_result_fields(setup_service):
    service, browser, now = setup_service
    enqueue(service, "隐私字幕不会进入投递日志")
    login(service, browser, now)
    browser.reply(outcome="platform_f", code=0, http_status=200,
                  msg="secret cookie example", cookie="sensitive", csrf="private")
    data = service.log_path.read_text(encoding="utf-8")
    assert "隐私字幕" not in data and "secret" not in data and "sensitive" not in data and "private" not in data
    assert all(json.loads(line)["time"].endswith("+08:00") for line in data.splitlines())


def test_browser_not_ready_makes_no_attempt(setup_service):
    service, browser, _ = setup_service
    enqueue(service)
    browser.is_browser_ready = False
    service.tick()
    assert not browser.requests
    assert service.snapshot()["state"] == "waiting_browser"


def test_explicit_not_dispatched_is_safe_to_retry(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    service._login_room = service.room
    browser.accept_requests = False
    service.tick()
    assert service.store.head()["state"] == "pending"
    assert not service.snapshot()["blocked"]
    now[0] += 3.5
    browser.accept_requests = True
    service.tick()
    assert browser.requests[-1]["operation"] == "login"


def test_dispatch_exception_is_uncertain_and_cannot_repeat_post(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    service._login_room = service.room
    def fails(*args):
        raise RuntimeError("transport failed after dispatch")
    browser.request_danmaku = fails
    service.tick()
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.content_filter.candidates()[0]["reason"] == "result_unknown"


def test_long_fifo_chunks_are_persisted_without_expiry(setup_service):
    service, _, now = setup_service
    enqueue(service, "字" * 200)
    assert service.snapshot()["queued"] == 6
    now[0] += 86400 * 60
    assert service.snapshot()["queued"] == 6


def test_emoji_respects_browser_utf16_length():
    chunks = split_danmaku("😀" * 100)
    assert len(chunks) == 6
    assert all(len(chunk.encode("utf-16-le")) // 2 <= 40 for chunk in chunks)
    assert "".join(chunk[1:-1] for chunk in chunks) == "😀" * 100


def test_unpaired_surrogate_cannot_break_storage_or_json():
    assert split_danmaku("坏字符\ud800") == []


def test_all_source_is_two_channel_fifo(setup_service):
    service, _, _ = setup_service
    service.configure(True, "1698225", source="all")
    event(service, "original", channel="mic")
    event(service, "translation", channel="mic", text="麦克风终稿")
    enqueue(service, "电脑声音终稿", "2")
    assert service.snapshot()["queued"] == 2
    assert service.store.head()["text"] == "【麦克风终稿】"


def test_metrics_are_receipts_and_page_visibility_separately(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, http_status=200)
    assert service.snapshot()["accepted"] == 1
    assert service.snapshot()["page_visible"] == 0
    now[0] += 3.5
    service.tick()
    browser.reply(present=True, room_matches=True, chat_ready=True)
    assert service.snapshot()["accepted"] == 1
    assert service.snapshot()["page_visible"] == 1


def test_real_sanitized_protocol_success_is_accepted(setup_service):
    from fanyiji.danmaku_protocol import sanitize_login, sanitize_receipt, sanitize_check
    service, browser, now = setup_service
    enqueue(service)
    service.tick()
    browser.reply(**sanitize_login({"is_login": True, "code": 0, "error": "none", "max_weighted_length": 80}))
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    browser.reply(**sanitize_receipt({"code": 0, "http_status": 200, "msg_class": "empty",
        "message_class": "empty", "error": "none", "request_elapsed_ms": 250,
        "request_started_epoch_ms": 100000, "before_count": 0}))
    service.tick()
    assert browser.requests[-1]["operation"] == "check"
    browser.reply(**sanitize_check({"room_matches": True, "chat_ready": True, "present": True, "occurrence_count": 1, "error": "none"}))
    assert service.snapshot()["page_visible"] == 1
    assert service.snapshot()["accepted"] == 1
    assert service.snapshot()["queued"] == 0


def test_readonly_polling_gives_dom_fifteen_seconds_without_extra_post(setup_service):
    from fanyiji.danmaku_protocol import sanitize_check
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, error="none")
    service.tick()
    browser.reply(**sanitize_check({"room_matches": True, "chat_ready": True, "present": False, "error": "none"}))
    count = len(browser.requests)
    now[0] += 0.49
    service.tick()
    assert len(browser.requests) == count
    now[0] += 0.01
    service.tick()
    assert browser.requests[-1]["operation"] == "check"
    browser.reply(**sanitize_check({"room_matches": True, "chat_ready": True, "present": True, "occurrence_count": 1, "error": "none"}))
    assert service.snapshot()["page_visible"] == 1
    assert sum(item["operation"] == "send" for item in browser.requests) == 1


def test_false_checks_transfer_to_analysis_after_fifteen_seconds(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, error="none")
    for _ in range(30):
        service.tick()
        browser.reply(present=False, room_matches=True, chat_ready=True, error="none")
        now[0] += 0.5
    service.tick()
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.content_filter.candidates()[0]["reason"] == "receipt_unconfirmed"
    assert sum(item["operation"] == "send" for item in browser.requests) == 1
    assert sum(item["operation"] == "check" for item in browser.requests) == 30


@pytest.mark.parametrize("text", ["包含\u200b控制符", "包含\u202e控制符", "<普通括号>"])
def test_protocol_validation_blocks_invalid_public_text(text):
    assert split_danmaku(text) == []


def test_internal_brackets_are_normalized_before_final_protocol_validation():
    from fanyiji.danmaku_protocol import valid_danmaku_text
    chunks = split_danmaku("他说【这是引用】的内容。")
    assert chunks == ["【他说 这是引用 的内容】"]
    assert valid_danmaku_text(chunks[0])


def test_login_failure_backoff_does_not_busy_loop(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    service.tick()
    browser.reply(is_login=False, code=0, error="none")
    service.tick()
    now[0] += 3.49
    service.tick()
    assert len(browser.requests) == 1
    now[0] += 0.01
    service.tick()
    assert len(browser.requests) == 2


def test_room_change_discards_only_unmatched_source_not_durable_fifo(setup_service):
    service, _, _ = setup_service
    enqueue(service, "已入队", "1")
    event(service, "original", segment="2", text="旧直播原文")
    service.configure(True, "21399689")
    event(service, "translation", segment="2", text="旧直播迟到终稿")
    assert service.snapshot()["queued"] == 1
    assert service.store.head()["room"] == "1698225"


def test_same_public_translation_format_variants_are_deduplicated(setup_service):
    service, _, _ = setup_service
    enqueue(service, "你好", "1")
    enqueue(service, "【你好】", "2")
    enqueue(service, " 你好 ", "3")
    assert service.snapshot()["queued"] == 1


def test_short_final_persistently_waits_for_next_utterance(setup_service):
    service, browser, now = setup_service
    enqueue(service, "没错", "1")
    service.tick()
    assert [row["operation"] for row in browser.requests] == ["login"]
    browser.reply(is_login=True, code=0, error="none")
    service.tick()
    assert service.snapshot()["queued"] == 1
    assert service.snapshot()["holding"] == 1
    assert service.snapshot()["state"] == "waiting_merge"
    assert not any(row["operation"] == "send" for row in browser.requests)
    now[0] += 86400 * 10
    service.tick()
    assert service.store.head()["text"] == "【没错】"
    assert service.store.head()["state"] == "holding_short"


def test_short_joins_following_complete_final_in_its_fifo_slot(setup_service):
    service, browser, now = setup_service
    enqueue(service, "是啊", "1")
    identifier = service.store.head()["id"]
    enqueue(service, "这就是主播正在解释的完整内容。", "2")
    assert service.snapshot()["queued"] == 1
    assert service.snapshot()["holding"] == 0
    assert service.store.head()["id"] == identifier
    assert service.store.head()["text"] == "【是啊 这就是主播正在解释的完整内容】"
    login(service, browser, now)
    assert browser.requests[-1]["text"] == "【是啊 这就是主播正在解释的完整内容】"


def test_multiple_short_finals_keep_waiting_until_eleven_with_brackets(setup_service):
    service, _, _ = setup_service
    enqueue(service, "你好", "1")
    enqueue(service, "好的", "2")
    assert service.store.head()["text"] == "【你好 好的】"
    assert service.snapshot()["holding"] == 1
    enqueue(service, "我也同意", "3")
    assert service.store.head()["text"] == "【你好 好的 我也同意】"
    assert service.snapshot()["holding"] == 0
    assert service.snapshot()["queued"] == 1


@pytest.mark.parametrize("length", [7, 8, 9])
def test_short_threshold_uses_body_characters(setup_service, length):
    service, _, _ = setup_service
    enqueue(service, "字" * length)
    assert service.snapshot()["holding"] == int(length + 2 < 11)


def test_emoji_translation_never_enters_the_send_queue(setup_service):
    service, _, _ = setup_service
    enqueue(service, "😀" * 5)
    assert service.store.head() is None
    assert service.content_filter.candidates()[0]["reason"] == "non_chinese_translation"


def test_merge_including_brackets_and_separator_accepts_exact_40(setup_service):
    service, _, _ = setup_service
    enqueue(service, "短句", "1")
    enqueue(service, "字" * 35, "2")
    head = service.store.head()
    assert len(head["text"]) == 40
    assert head["text"] == "【短句 " + "字" * 35 + "】"
    assert service.snapshot()["queued"] == 1


def test_over_40_merge_rebalances_and_keeps_all_parts_at_least_eleven(setup_service):
    service, _, _ = setup_service
    enqueue(service, "短句", "1")
    enqueue(service, "字" * 36, "2")
    head = service.store.head()
    parts = [head["text"], *service.store.remaining_parts(head["id"])]
    assert head["state"] == "pending"
    assert all(11 <= danmaku_utf16_length(part) <= 40 for part in parts)
    assert "".join(part[1:-1].replace(" ", "") for part in parts) == "短句" + "字" * 36
    assert service.snapshot()["queued"] == 2


def test_long_next_sentence_merges_short_head_without_losing_any_chunk(setup_service):
    service, _, _ = setup_service
    enqueue(service, "短句", "1")
    enqueue(service, "字" * 44, "2")
    head = service.store.head()
    parts = [head["text"], *service.store.remaining_parts(head["id"])]
    assert head["state"] == "pending"
    assert all(11 <= danmaku_utf16_length(part) <= 40 for part in parts)
    assert "".join(part[1:-1].replace(" ", "") for part in parts) == "短句" + "字" * 44
    assert service.snapshot()["queued"] == len(parts)
    assert service.snapshot()["holding"] == 0


def test_short_never_merges_with_other_channel(setup_service):
    service, _, _ = setup_service
    service.configure(True, "1698225", source="all")
    enqueue(service, "电脑声音", "1")
    event(service, "original", channel="mic", segment="2")
    event(service, "translation", channel="mic", segment="2", text="这是麦克风通道的完整终稿")
    rows = service.store.connection.execute("SELECT channel,text,state FROM messages ORDER BY id").fetchall()
    assert [(row["channel"], row["text"], row["state"]) for row in rows] == [
        ("system", "【电脑声音】", "holding_short"), ("mic", "【这是麦克风通道的完整终稿】", "pending")]


def test_short_never_merges_with_other_room(setup_service):
    service, _, _ = setup_service
    enqueue(service, "旧房短句", "1")
    service.configure(True, "21399689")
    enqueue(service, "新房间里的完整终稿内容", "2")
    assert service.snapshot()["queued"] == 2
    assert service.store.head()["text"] == "【旧房短句】"
    assert service.store.head()["state"] == "holding_short"


def test_short_hold_survives_restart_and_later_merges(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    first = DanmakuService(FakeBrowser(), store_path=path)
    first._apply_limits({"max_weighted_length": 80})
    first.configure(True, "1698225")
    first.begin_session(1)
    enqueue(first, "短句", "1")
    first.close()
    browser = FakeBrowser()
    second = DanmakuService(browser, store_path=path)
    second._apply_limits({"max_weighted_length": 80})
    try:
        second.configure(True, "1698225")
        second.begin_session(2)
        second.tick()
        assert second.snapshot()["holding"] == 1
        assert [row["operation"] for row in browser.requests] == ["login"]
        browser.reply(is_login=True, code=0, error="none")
        second.tick()
        assert not any(row["operation"] == "send" for row in browser.requests)
        event(second, "original", session=2, segment="2")
        event(second, "translation", session=2, segment="2", text="后续的新终稿让它完成合并")
        assert second.store.head()["text"] == "【短句 后续的新终稿让它完成合并】"
        assert second.snapshot()["holding"] == 0
    finally:
        second.close()


def test_duplicate_next_final_does_not_release_or_duplicate_short(setup_service):
    service, _, _ = setup_service
    enqueue(service, "相同短句", "1")
    enqueue(service, "相同短句", "2")
    assert service.snapshot()["queued"] == 1
    assert service.store.head()["text"] == "【相同短句】"
    assert service.snapshot()["holding"] == 1


@pytest.mark.parametrize("state", ["pending", "sending", "verify", "blocked"])
def test_submitted_or_ready_short_is_never_mutated_by_merge(setup_service, state):
    service, _, _ = setup_service
    enqueue(service, "旧短句", "1")
    head = service.store.head()
    service.store.update(head["id"], state, request_id="request", reason="test")
    enqueue(service, "这是随后才来到的新完整终稿", "2")
    assert service.store.head()["text"] == "【旧短句】"
    assert service.store.head()["state"] == state
    assert service.snapshot()["queued"] == 2


def test_pending_queue_ahead_of_short_holding_still_sends_first(setup_service):
    service, browser, now = setup_service
    enqueue(service, "队列第一条足够长的完整字幕", "1")
    enqueue(service, "短句", "2")
    assert service.snapshot()["queued"] == 2
    assert service.snapshot()["holding"] == 1
    login(service, browser, now)
    assert browser.requests[-1]["text"] == "【队列第一条足够长的完整字幕】"


def test_send_logs_browser_request_start_and_elapsed_without_raw_data(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, http_status=200,
        request_started_epoch_ms=1791676800123, request_elapsed_ms=250.12349,
        csrf="sensitive", raw_response="do not log", text="do not log")
    record = json.loads(service.log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["request_started_epoch_ms"] == 1791676800123
    assert record["request_elapsed_ms"] == 250.123
    assert not {"csrf", "raw_response", "text"} & record.keys()


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -float("inf"), -1, "100", None, {}, []])
def test_send_time_metrics_reject_non_finite_or_non_numeric_values(setup_service, value):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, request_started_epoch_ms=value, request_elapsed_ms=value)
    record = json.loads(service.log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["request_started_epoch_ms"] is None
    assert record["request_elapsed_ms"] is None


def test_send_time_metrics_reject_outside_protocol_range(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, request_started_epoch_ms=10**15+1, request_elapsed_ms=120001)
    record = json.loads(service.log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["request_started_epoch_ms"] is None
    assert record["request_elapsed_ms"] is None


def test_readonly_check_timeout_does_not_extend_send_cooldown(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, http_status=200)
    deadline = service._next_at
    service.tick()
    assert browser.requests[-1]["operation"] == "check"
    now[0] += 45
    service.tick()
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.snapshot()["page_visible"] == 0
    assert service._next_at == deadline


def test_readonly_login_timeout_only_backs_off_login(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    service.tick()
    assert browser.requests[-1]["operation"] == "login"
    now[0] += 45
    service.tick()
    assert service._next_at == 0
    assert service._read_not_before == now[0] + 3.5


def test_huge_python_integer_is_rejected_before_float_conversion():
    from fanyiji.danmaku_service import _finite_protocol_number
    assert _finite_protocol_number(10**400, 10**15) is None


def test_merged_public_payload_blocks_later_identical_complete_final(setup_service):
    service, _, _ = setup_service
    enqueue(service, "好的", "1")
    enqueue(service, "这是一句用于验证合并的完整译文", "2")
    assert service.store.head()["text"] == "【好的 这是一句用于验证合并的完整译文】"
    enqueue(service, "好的 这是一句用于验证合并的完整译文", "3")
    assert service.snapshot()["queued"] == 1
    assert service.snapshot()["holding"] == 0


def test_existing_combined_payload_preserves_short_components_without_force_release(setup_service):
    service, _, _ = setup_service
    enqueue(service, "好的 这是一句用于验证合并的完整译文", "1")
    enqueue(service, "好的", "2")
    enqueue(service, "这是一句用于验证合并的完整译文", "3")
    rows = service.store.connection.execute("SELECT text,state FROM messages ORDER BY id").fetchall()
    assert [(row["text"], row["state"]) for row in rows] == [
        ("【好的 这是一句用于验证合并的完整译文】", "pending"),
        ("【好的】", "holding_short"), ("【这是一句用于验证合并的完整译文】", "pending")]
    assert service.snapshot()["queued"] == 3
    assert service.snapshot()["holding"] == 1


def test_stepwise_short_merges_register_final_combined_digest(setup_service):
    service, _, _ = setup_service
    enqueue(service, "甲甲", "1")
    enqueue(service, "乙乙", "2")
    enqueue(service, "丙丙丙丙丙丙", "3")
    assert service.store.head()["text"] == "【甲甲 乙乙 丙丙丙丙丙丙】"
    enqueue(service, "甲甲 乙乙 丙丙丙丙丙丙", "4")
    assert service.snapshot()["queued"] == 1
    assert service.snapshot()["holding"] == 0


def test_merged_payload_duplicate_detection_survives_restart(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    first = DanmakuService(FakeBrowser(), store_path=path)
    first._apply_limits({"max_weighted_length": 80})
    first.configure(True, "1698225")
    first.begin_session(1)
    enqueue(first, "好的", "1")
    enqueue(first, "这是一句用于验证合并的完整译文", "2")
    first.close()
    second = DanmakuService(FakeBrowser(), store_path=path)
    try:
        second.configure(True, "1698225")
        second.begin_session(2)
        event(second, "original", session=2, segment="3")
        event(second, "translation", session=2, segment="3", text="好的 这是一句用于验证合并的完整译文")
        assert second.snapshot()["queued"] == 1
        assert second.store.head()["text"] == "【好的 这是一句用于验证合并的完整译文】"
    finally:
        second.close()


def test_cjk_merged_public_payload_blocks_later_normalized_whole_final(setup_service):
    service, _, _ = setup_service
    enqueue(service, "验证", "1")
    enqueue(service, "推送082200首条确认", "2")
    assert service.store.head()["text"] == "【验证 推送082200首条确认】"
    enqueue(service, "验证 推送082200首条确认", "3")
    assert service.snapshot()["queued"] == 1
    assert service.store.head()["text"] == "【验证 推送082200首条确认】"


def test_existing_cjk_whole_preserves_short_components_instead_of_duplicate_merge(setup_service):
    service, _, _ = setup_service
    enqueue(service, "验证 推送082200首条确认", "1")
    enqueue(service, "验证", "2")
    enqueue(service, "推送082200首条确认", "3")
    rows = service.store.connection.execute("SELECT text,state FROM messages ORDER BY id").fetchall()
    assert [(row["text"], row["state"]) for row in rows] == [
        ("【验证 推送082200首条确认】", "pending"), ("【验证】", "holding_short"), ("【推送082200首条确认】", "pending")]


def test_stepwise_cjk_merge_duplicates_survive_restart(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    first = DanmakuService(FakeBrowser(), store_path=path)
    first._apply_limits({"max_weighted_length": 80})
    first.configure(True, "1698225")
    first.begin_session(1)
    enqueue(first, "验证", "1")
    enqueue(first, "当前", "2")
    enqueue(first, "合并后的完整测试译文", "3")
    assert first.store.head()["text"] == "【验证 当前 合并后的完整测试译文】"
    first.close()
    second = DanmakuService(FakeBrowser(), store_path=path)
    try:
        second.configure(True, "1698225")
        second.begin_session(2)
        event(second, "original", session=2, segment="4")
        event(second, "translation", session=2, segment="4", text="验证 当前 合并后的完整测试译文")
        assert second.snapshot()["queued"] == 1
        assert second.store.head()["text"] == "【验证 当前 合并后的完整测试译文】"
    finally:
        second.close()


def test_digest_normalizes_only_non_ascii_adjacent_spaces_preserving_part_boundaries():
    from fanyiji.danmaku_store import content_digest
    assert content_digest("【验证 推送】") == content_digest("【验证推送】")
    assert content_digest("【OK 验证 word】") == content_digest("【OK验证word】")
    assert content_digest("【this is a sentence】") != content_digest("【thisisa sentence】")
    assert content_digest("【验证】\n【推送】") != content_digest("【验证】【推送】")


def test_prior_digest_version_migrates_cjk_merged_text_without_rewriting(tmp_path):
    import hashlib
    from datetime import datetime, timezone
    path = tmp_path / "queue.sqlite3"
    store = DanmakuStore(path, wall_clock=lambda: 1000)
    merged = "【验证 推送082200首条确认】"
    with store.connection:
        store.connection.execute("DELETE FROM metadata WHERE name='public_digest_normalization'")
        store.connection.execute("DELETE FROM metadata WHERE name='dedup_window_v1'")
        store.connection.execute("INSERT INTO contents VALUES (?,?,?)",
            ("1698225", hashlib.sha256(merged.encode("utf-8")).hexdigest(), "legacy"))
        store.connection.execute(
            "INSERT INTO messages(room,channel,segment_key,revision,part,text,created_at,state) VALUES(?,?,?,?,?,?,?,?)",
            ("1698225", "system", "legacy", 1, 0, merged, datetime.fromtimestamp(1000, timezone.utc).isoformat(), "pending"))
    store.close()
    upgraded = DanmakuStore(path, wall_clock=lambda: 1005)
    try:
        assert upgraded.enqueue("1698225", "system", "later", 1, "验证推送082200首条确认",
                                split_danmaku("验证推送082200首条确认")) is False
        assert upgraded.counts()["queued"] == 1
        assert upgraded.head()["text"] == merged
        assert upgraded.head()["state"] == "pending"
    finally:
        upgraded.close()


def test_clear_pending_cancels_unsent_final_and_short_but_keeps_history(setup_service):
    service, _, _ = setup_service
    enqueue(service, "这是尚未发送的第一条完整译文", "1")
    enqueue(service, "短句", "2")
    assert service.snapshot()["clearable"] == 2
    result = service.clear_pending()
    assert result == {"removed": 2, "retained": 0, "retained_uncertain": 0}
    assert service.snapshot()["queued"] == service.snapshot()["holding"] == 0
    assert service.snapshot()["clearable"] == 0
    assert service.store.head() is None
    assert service.store.connection.execute("SELECT COUNT(*) FROM messages WHERE state='cancelled'").fetchone()[0] == 2


def test_clear_pending_preserves_enabled_source_and_send_deadline(setup_service):
    service, _, _ = setup_service
    enqueue(service)
    service._next_at = 200
    before = (service.enabled, service.room, service.source, service.interval_seconds, service._next_at)
    service.clear_pending()
    after = (service.enabled, service.room, service.source, service.interval_seconds, service._next_at)
    assert after == before
    assert service.snapshot()["state"] == "empty"


def test_clear_pending_protects_login_head_from_late_callback(setup_service):
    service, browser, now = setup_service
    enqueue(service, "当前正在登录准备发送的完整译文", "1")
    enqueue(service, "后续尚未提交的另一条完整译文", "2")
    service.tick()
    request = dict(browser.requests[-1])
    assert request["operation"] == "login"
    assert service.snapshot()["clearable"] == 1
    result = service.clear_pending()
    assert result == {"removed": 1, "retained": 1, "retained_uncertain": 1}
    assert service.snapshot()["inflight"] == "login"
    browser.reply(is_login=True, code=0, error="none")
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == "【当前正在登录准备发送的完整译文】"
    assert service.snapshot()["queued"] == 1


def test_clear_pending_during_post_keeps_head_and_late_receipt(setup_service):
    service, browser, now = setup_service
    enqueue(service, "正在请求中的这条完整翻译内容", "1")
    enqueue(service, "后面的这条完整翻译可以取消", "2")
    login(service, browser, now)
    flight = dict(service._inflight)
    deadline = service._next_at
    result = service.clear_pending()
    assert result["removed"] == 1 and result["retained"] == 1
    assert service._inflight == flight and service._next_at == deadline
    browser.reply(outcome="receipt_accepted", code=0, http_status=200, error="none")
    service.tick()
    assert browser.requests[-1]["operation"] == "check"
    browser.reply(present=True, room_matches=True, chat_ready=True, error="none")
    assert service.snapshot()["queued"] == 0
    assert service.snapshot()["page_visible"] == 1


def test_clear_pending_during_presence_check_keeps_callback_identity(setup_service):
    service, browser, now = setup_service
    enqueue(service, "已经投递而仍等待页面核验的译文", "1")
    enqueue(service, "尚未投递所以可以取消的完整译文", "2")
    login(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, http_status=200, error="none")
    service.tick()
    assert browser.requests[-1]["operation"] == "check"
    flight = dict(service._inflight)
    assert service.clear_pending()["removed"] == 1
    assert service._inflight == flight
    browser.reply(present=True, room_matches=True, chat_ready=True, error="none")
    assert service.snapshot()["page_visible"] == 1
    assert service.snapshot()["queued"] == 0


@pytest.mark.parametrize("outcome,code,reason,clearable", [
    ("platform_rejected", 1003212, "platform_rejected", 1),
    ("platform_rejected", 0, "platform_rejected", 0),
    ("platform_rejected", None, "platform_rejected", 0),
    ("platform_f", 0, "platform_f", 0)])
def test_clear_pending_leaves_transferred_rejection_analysis_intact(setup_service, outcome, code, reason, clearable):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome=outcome, code=code, http_status=200, error="none")
    assert service.store.head() is None
    candidate = service.content_filter.candidates()[0]
    assert candidate["reason"] == reason
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.snapshot()["clearable"] == 0
    result = service.clear_pending()
    assert result["removed"] == result["retained"] == 0
    assert service.content_filter.candidates()[0] == candidate
    assert service.snapshot()["page_visible"] == 0


def test_clear_keeps_transferred_unknown_analysis_and_cancels_unsubmitted_next(setup_service):
    service, browser, now = setup_service
    enqueue(service, "未确认的这条完整字幕必须保留", "1")
    enqueue(service, "尚未提交的这条完整字幕可以取消", "2")
    login(service, browser, now)
    browser.reply(outcome="result_unknown", code=None, error="network_unknown")
    candidate = service.content_filter.candidates()[0]
    assert candidate["text"] == "【未确认的这条完整字幕必须保留】"
    assert service.store.head()["text"] == "【尚未提交的这条完整字幕可以取消】"
    assert not service.snapshot()["blocked"]
    result = service.clear_pending()
    assert result == {"removed": 1, "retained": 0, "retained_uncertain": 0}
    service.resume()
    service.tick()
    assert [row["operation"] for row in browser.requests] == ["login", "send"]
    assert service.content_filter.candidates()[0] == candidate
    assert sum(row["operation"] == "send" for row in browser.requests) == 1


def test_f_failure_analysis_survives_restart_without_reposting_original(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    browser = FakeBrowser()
    first = DanmakuService(browser, store_path=path)
    first._clock = lambda: 100
    first.configure(True, "1698225")
    first.begin_session(1)
    enqueue(first)
    first.tick()
    browser.reply(is_login=True, code=0, error="none")
    first.tick()
    browser.reply(outcome="platform_f", code=0, http_status=200, error="none")
    candidate = first.content_filter.candidates()[0]
    assert first.store.head() is None
    assert first.snapshot()["deferred_analysis"] == 1
    first.close()
    second_browser = FakeBrowser()
    second = DanmakuService(second_browser, store_path=path)
    try:
        second._clock = lambda: 200
        second.configure(True, "1698225")
        assert second.clear_pending()["removed"] == 0
        assert second.store.head() is None
        assert not second.snapshot()["f_retry_active"]
        assert second.content_filter.candidates()[0] == candidate
        second._clock = lambda: 203.5
        second.tick()
        assert second_browser.requests == []
        assert second.snapshot()["f_retries"] == 0
        assert second.snapshot()["deferred_analysis"] == 1
    finally:
        second.close()


def test_clear_pending_all_sources_and_rooms_without_changing_current_preferences(setup_service):
    service, _, _ = setup_service
    enqueue(service, "第一个房间电脑来源的待发完整字幕", "1")
    service.configure(True, "1698225", source="mic")
    event(service, "original", channel="mic", segment="2")
    event(service, "translation", channel="mic", segment="2", text="第一个房间麦克风来源的待发完整字幕")
    service.configure(True, "21399689", source="system")
    enqueue(service, "第二个房间电脑来源的待发完整字幕", "3")
    result = service.clear_pending()
    assert result["removed"] == 3 and result["retained"] == 0
    assert service.room == "21399689" and service.source == "system" and service.enabled


def test_cancellation_survives_restart_without_reviving_dedup_or_short_merge(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    first = DanmakuService(FakeBrowser(), store_path=path)
    first.configure(True, "1698225")
    first.begin_session(1)
    enqueue(first, "短句", "1")
    assert first.clear_pending()["removed"] == 1
    first.close()
    second = DanmakuService(FakeBrowser(), store_path=path)
    try:
        second.configure(True, "1698225")
        second.begin_session(2)
        event(second, "original", session=2, segment="2")
        event(second, "translation", session=2, segment="2", text="短句")
        assert second.snapshot()["queued"] == 0
        event(second, "original", session=2, segment="3")
        event(second, "translation", session=2, segment="3", text="取消之后新的独立完整字幕正常入队")
        assert second.snapshot()["queued"] == 1
        assert second.store.head()["text"] == "【取消之后新的独立完整字幕正常入队】"
    finally:
        second.close()


def test_cancelled_merged_text_is_not_revived_by_components_or_whole(setup_service):
    service, _, _ = setup_service
    enqueue(service, "验证", "1")
    enqueue(service, "推送082200首条确认", "2")
    assert service.clear_pending()["removed"] == 1
    enqueue(service, "验证", "3")
    enqueue(service, "推送082200首条确认", "4")
    enqueue(service, "验证 推送082200首条确认", "5")
    assert service.snapshot()["queued"] == 0
    assert service.snapshot()["holding"] == 0


def test_completed_page_visible_counts_are_unchanged_by_clear(setup_service):
    service, browser, now = setup_service
    enqueue(service, "已经核验成功的这条完整字幕", "1")
    login(service, browser, now)
    finish(service, browser, now)
    enqueue(service, "尚未发送的新完整字幕可以清空", "2")
    before = (service.snapshot()["page_visible"], service.snapshot()["accepted"])
    service.clear_pending()
    assert (service.snapshot()["page_visible"], service.snapshot()["accepted"]) == before
    assert service.snapshot()["queued"] == 0


def test_clear_pending_logs_counts_without_subtitle_content(setup_service):
    service, _, _ = setup_service
    enqueue(service, "不会泄漏到清空日志的私人字幕内容", "1")
    service.clear_pending()
    records = [json.loads(line) for line in service.log_path.read_text(encoding="utf-8").splitlines()]
    clear = records[-1]
    assert clear["kind"] == "clear_pending" and clear["removed"] == 1 and clear["retained"] == 0
    assert "不会泄漏" not in service.log_path.read_text(encoding="utf-8")


def test_old_message_schema_is_unchanged_and_unknown_rejection_is_not_clearable(tmp_path):
    import sqlite3
    path = tmp_path / "queue.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT,room TEXT NOT NULL,channel TEXT NOT NULL,segment_key TEXT NOT NULL,revision INTEGER NOT NULL,part INTEGER NOT NULL,text TEXT NOT NULL,created_at TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',request_id TEXT NOT NULL DEFAULT '',reason TEXT NOT NULL DEFAULT '')")
    connection.execute("INSERT INTO messages(room,channel,segment_key,revision,part,text,created_at,state,reason) VALUES(?,?,?,?,?,?,?,?,?)",
                       ("1698225", "system", "old", 1, 0, "【旧版本未知拒绝的完整字幕】", "time", "blocked", "platform_rejected"))
    connection.commit()
    connection.close()
    store = DanmakuStore(path)
    try:
        assert "platform_code" not in {row[1] for row in store.connection.execute("PRAGMA table_info(messages)")}
        assert store.head()["platform_code"] is None
        assert store.clear_pending() == {"removed": 0, "retained": 1, "retained_uncertain": 1}
    finally:
        store.close()


def test_clear_after_close_is_noop_and_does_not_access_closed_database(setup_service):
    service, _, _ = setup_service
    enqueue(service)
    service.close()
    assert service.clear_pending() == {"removed": 0, "retained": 0, "retained_uncertain": 0}


def test_content_duplicate_window_expires_without_refresh_on_skips(tmp_path):
    now = [1000.0]
    store = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: now[0])
    try:
        text = "这是用来验证十秒去重窗口的完整字幕"
        assert store.enqueue("1698225", "system", "1", 1, text, split_danmaku(text))
        now[0] = 1009
        assert not store.enqueue("1698225", "system", "2", 1, text, split_danmaku(text))
        now[0] = 1010
        assert not store.enqueue("1698225", "system", "3", 1, text, split_danmaku(text))
        now[0] = 1010.001
        assert store.enqueue("1698225", "system", "4", 1, text, split_danmaku(text))
        assert store.counts()["queued"] == 2
    finally:
        store.close()


def test_segment_callback_remains_permanently_deduplicated_after_window(tmp_path):
    now = [1000.0]
    store = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: now[0])
    try:
        text = "同一个字幕标识重复回调不能制造重复"
        assert store.enqueue("1698225", "system", "same-segment", 1, text, split_danmaku(text))
        now[0] += 100000
        assert not store.enqueue("1698225", "system", "same-segment", 2, text, split_danmaku(text))
        assert store.enqueue("1698225", "system", "different-segment", 1, text, split_danmaku(text))
        assert store.counts()["queued"] == 2
    finally:
        store.close()


def test_window_is_independent_for_each_room(tmp_path):
    store = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: 1000)
    try:
        text = "同一句字幕可以在另一个直播间排队"
        assert store.enqueue("1698225", "system", "1", 1, text, split_danmaku(text))
        assert store.enqueue("21399689", "system", "2", 1, text, split_danmaku(text))
        assert store.counts()["queued"] == 2
    finally:
        store.close()


def test_duplicate_window_survives_restart_with_original_epoch(tmp_path):
    path = tmp_path / "queue.sqlite3"
    text = "重启程序不会重置这条字幕的十秒窗口"
    first = DanmakuStore(path, wall_clock=lambda: 1000)
    assert first.enqueue("1698225", "system", "1", 1, text, split_danmaku(text))
    first.close()
    now = [1009.0]
    second = DanmakuStore(path, wall_clock=lambda: now[0])
    try:
        assert not second.enqueue("1698225", "system", "2", 1, text, split_danmaku(text))
        now[0] = 1011
        assert second.enqueue("1698225", "system", "3", 1, text, split_danmaku(text))
        assert second.counts()["queued"] == 2
    finally:
        second.close()


def test_wall_clock_rollback_conservatively_preserves_duplicate_window(tmp_path):
    now = [1000.0]
    store = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: now[0])
    try:
        text = "系统时间回拨时不能提前解除重复过滤"
        assert store.enqueue("1698225", "system", "1", 1, text, split_danmaku(text))
        now[0] = 990
        assert not store.enqueue("1698225", "system", "2", 1, text, split_danmaku(text))
        now[0] = 1011
        assert store.enqueue("1698225", "system", "3", 1, text, split_danmaku(text))
    finally:
        store.close()


def test_merged_actual_public_payload_uses_time_of_last_merge(tmp_path):
    now = [1000.0]
    store = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: now[0])
    try:
        assert store.enqueue("1698225", "system", "1", 1, "验证", split_danmaku("验证"))
        now[0] = 1009
        assert store.enqueue("1698225", "system", "2", 1, "推送082200首条确认", split_danmaku("推送082200首条确认"))
        now[0] = 1018
        whole = "验证 推送082200首条确认"
        assert not store.enqueue("1698225", "system", "3", 1, whole, split_danmaku(whole))
        now[0] = 1020
        assert store.enqueue("1698225", "system", "4", 1, whole, split_danmaku(whole))
        assert store.counts()["queued"] == 2
    finally:
        store.close()


def test_expired_existing_combined_payload_can_be_merged_again(tmp_path):
    now = [1000.0]
    store = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: now[0])
    try:
        text = "验证 推送082200首条确认"
        assert store.enqueue("1698225", "system", "1", 1, text, split_danmaku(text))
        now[0] = 1011
        assert store.enqueue("1698225", "system", "2", 1, "验证", split_danmaku("验证"))
        assert store.enqueue("1698225", "system", "3", 1, "推送082200首条确认", split_danmaku("推送082200首条确认"))
        assert store.counts()["queued"] == 2
        rows = store.connection.execute("SELECT text FROM messages ORDER BY id").fetchall()
        assert [row["text"] for row in rows] == ["【验证 推送082200首条确认】", "【验证 推送082200首条确认】"]
    finally:
        store.close()


@pytest.mark.parametrize("bad_value", ["none", "old-format", "NaN", "Infinity", "-1", "1e999"])
def test_invalid_old_window_metadata_is_ignored_safely(tmp_path, bad_value):
    from fanyiji.danmaku_store import content_digest
    store = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: 1000)
    try:
        text = "旧版无效时间格式不会导致永久不能排队"
        with store.connection:
            store._set_metadata("dedup_seen_at:1698225:" + content_digest("\n".join(split_danmaku(text))), bad_value)
        assert store.enqueue("1698225", "system", "1", 1, text, split_danmaku(text))
    finally:
        store.close()


def test_cleared_payload_can_reenter_after_ten_seconds_with_new_segment(tmp_path):
    now = [1000.0]
    store = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: now[0])
    try:
        text = "用户清空后新的字幕仍按十秒规则排队"
        assert store.enqueue("1698225", "system", "1", 1, text, split_danmaku(text))
        assert store.clear_pending()["removed"] == 1
        now[0] = 1009
        assert not store.enqueue("1698225", "system", "2", 1, text, split_danmaku(text))
        now[0] = 1011
        assert store.enqueue("1698225", "system", "3", 1, text, split_danmaku(text))
        assert store.counts()["queued"] == 1 and store.counts()["cleared"] == 1
    finally:
        store.close()


@pytest.mark.parametrize("match_field,value,restored", [(None, None, True), ("request_id", "another-request", False),
    ("room", "21399689", False), ("queue_id", 2, False), ("has_f", True, False),
    ("queue_id", True, False), ("has_f", None, False), ("platform_code", True, False), ("platform_code", "1003212", False)])
def test_legacy_head_code_restored_only_from_same_request_log(app, tmp_path, match_field, value, restored):
    path = tmp_path / "queue.sqlite3"
    store = DanmakuStore(path)
    text = "旧版等待队列中明确被拒绝的完整字幕"
    store.enqueue("1698225", "system", "old", 1, text, split_danmaku(text))
    head = store.head()
    store.update(head["id"], "blocked", request_id="exact-persisted-request", reason="platform_rejected")
    store.close()
    record = {"kind": "result", "operation": "send", "queue_id": head["id"], "room": "1698225",
              "request_id": "exact-persisted-request", "platform_code": 1003212, "has_f": False}
    if match_field:
        record[match_field] = value
    path.with_name("delivery.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    service = DanmakuService(FakeBrowser(), store_path=path)
    try:
        assert service.snapshot()["last_platform_code"] == (1003212 if restored else None)
        assert service.snapshot()["head_characters"] == len("【" + text + "】")
        assert service.snapshot()["clearable"] == int(restored)
    finally:
        service.close()


def import_legacy_f(service, browser, now, **fields):
    """Import an old persisted f receipt to exercise backward-compatible guards.

    New browser failures take the independent analysis path; this fixture never
    invokes that path and is used only by the legacy database compatibility cases.
    """
    head = service.store.head()
    service._inflight = None
    service.store.update(head["id"], "verify", request_id=browser.requests[-1]["request_id"], reason="platform_f", platform_code=0)
    service.store.set_delivery_baseline(head["id"], fields.get("before_count", 0))
    service.store.activate_f_recovery(head["id"], new_f=True)
    service.content_filter.record_failure(head["room"], head["text"], "platform_f", queue_id=head["id"])
    service._f_retry_queue_id = head["id"]
    service._verify_queue_id = service._verify_deadline = None
    service._next_at = now[0] + service.interval_seconds
    service._latest_reason = "waiting_f_retry"
    assert service.store.head()["state"] == "verify"
    assert service.snapshot()["f_retry_active"]
    return head


def start_f_recovery(service, browser, now, **fields):
    enqueue(service, "第一条完整字幕持续保留顺序", "f-head")
    login(service, browser, now)
    return import_legacy_f(service, browser, now, **fields)


def reach_f_retry(service, browser, now):
    now[0] = service._next_at
    service.tick()
    assert browser.requests[-1]["operation"] == "check"
    browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0, error="none")
    assert service.store.head()["state"] == "retry_pending"
    service.tick()
    assert browser.requests[-1]["operation"] == "send"


def test_legacy_f_waits_from_post_response_and_no_parallel_retry(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    first_started = now[0]
    now[0] += 2.0
    import_legacy_f(service, browser, now)
    due = now[0] + 3.5
    service.tick()
    browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
    now[0] = due - 0.001
    service.tick()
    assert sum(row["operation"] == "send" for row in browser.requests) == 1
    # A check still running at the deadline prevents any new request.
    now[0] = due + 2
    service.tick()
    assert service._inflight["operation"] == "check"
    browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
    service.tick()
    assert now[0] - first_started >= 5.5
    assert service.snapshot()["f_retries"] == 1
    assert service.snapshot()["state"] == "retrying_f"


def test_continuous_f_moves_each_failed_packet_to_analysis_and_advances_fifo(setup_service):
    service, browser, now = setup_service
    bodies = ["第一条完整字幕持续保留记录", "第二条完整字幕按顺序推进", "第三条完整字幕不会被前条卡住"]
    for index, body in enumerate(bodies):
        enqueue(service, body, str(index))
    login(service, browser, now)
    for index, body in enumerate(bodies):
        assert browser.requests[-1]["operation"] == "send"
        assert browser.requests[-1]["text"] == "【" + body + "】"
        browser.reply(outcome="platform_f", code=0, http_status=200, error="none")
        assert not service.snapshot()["f_retry_active"]
        assert service.snapshot()["deferred_analysis"] == index + 1
        assert len(service.content_filter.candidates()) == index + 1
        posts = sum(row["operation"] == "send" for row in browser.requests)
        now[0] = service._next_at - 0.001
        service.tick()
        assert sum(row["operation"] == "send" for row in browser.requests) == posts
        now[0] = service._next_at
        service.tick()
    assert service.store.head() is None
    assert service.snapshot()["queued"] == 0
    assert service.snapshot()["page_visible"] == 0
    assert [row["text"] for row in browser.requests if row["operation"] == "send"] == ["【" + body + "】" for body in bodies]


def test_f_newly_visible_advances_without_another_post(setup_service):
    service, browser, now = setup_service
    head = start_f_recovery(service, browser, now, before_count=2)
    enqueue(service, "排在后面的完整字幕等待前条投递", "later")
    service.tick()
    browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=2)
    assert service.store.head()["id"] == head["id"]
    now[0] += 0.5
    service.tick()
    browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=3)
    assert service.snapshot()["page_visible"] == 1
    assert service.snapshot()["f_retries"] == 0
    assert not service.snapshot()["f_retry_active"]
    service.tick()
    assert sum(row["operation"] == "send" for row in browser.requests) == 1
    now[0] = service._next_at
    service.tick()
    assert browser.requests[-1]["text"] != head["text"]


def test_f_retry_accepted_reverts_to_normal_fifteen_second_confirmation(setup_service):
    service, browser, now = setup_service
    head = start_f_recovery(service, browser, now)
    reach_f_retry(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, http_status=200, error="none")
    assert not service.snapshot()["f_retry_active"]
    assert service._verify_deadline == now[0] + 15
    service.tick()
    browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
    now[0] += 15
    service.tick()
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.content_filter.candidates()[0]["text"] == head["text"]
    assert sum(row["operation"] == "send" for row in browser.requests) == 2


@pytest.mark.parametrize("failure", ["result_unknown", "timeout", "exception"])
def test_f_retry_new_post_uncertainty_cannot_inherit_old_f_retry_permission(setup_service, failure):
    service, browser, now = setup_service
    start_f_recovery(service, browser, now)
    if failure == "exception":
        now[0] = service._next_at
        service.tick()
        browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
        def fails(*args):
            raise RuntimeError("request may have reached browser")
        browser.request_danmaku = fails
        service.tick()
    else:
        reach_f_retry(service, browser, now)
        if failure == "timeout":
            now[0] += 45
            service.tick()
        else:
            browser.reply(outcome="result_unknown", code=None, error="network_unknown")
            service.tick()
            browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
            now[0] += 15
            service.tick()
    assert not service.snapshot()["f_retry_active"]
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    posts = sum(request["operation"] == "send" for request in browser.requests)
    now[0] += 1000
    service.tick()
    assert sum(request["operation"] == "send" for request in browser.requests) == posts
    assert service.content_filter.candidates()


@pytest.mark.parametrize("failure", ["not_sent", "not_dispatched"])
def test_legacy_retry_dispatch_refusal_retains_payload_but_failed_receipt_defers(setup_service, failure):
    service, browser, now = setup_service
    head = start_f_recovery(service, browser, now, before_count=2)
    now[0] = service._next_at
    service.tick()
    browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=2)
    if failure == "not_dispatched":
        browser.accept_requests = False
    service.tick()
    if failure == "not_sent":
        browser.reply(outcome="not_sent", error="wrong_room")
        assert service.store.head() is None
        assert not service.snapshot()["f_retry_active"]
        assert service.snapshot()["deferred_analysis"] == 1
        assert service.content_filter.candidates()[-1]["text"] == head["text"]
        return
    assert service.store.head()["state"] == "retry_pending"
    assert service.snapshot()["f_retry_active"]
    assert service.store.delivery_baseline(head["id"]) == 2
    assert service.clear_pending()["removed"] == 0
    assert service.store.head()["id"] == head["id"]
    assert service.store.head()["text"] == head["text"]


def test_f_retry_pause_preserves_head_and_reenable_without_manual_resume(setup_service):
    service, browser, now = setup_service
    head = start_f_recovery(service, browser, now)
    service.configure(False, "1698225")
    now[0] += 30
    service.tick()
    assert len(browser.requests) == 2
    assert service.store.head()["id"] == head["id"]
    service.configure(True, "1698225")
    service.timer.stop()
    service.tick()
    browser.reply(is_login=True, code=0)
    service.tick()
    browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == head["text"]


def test_f_retry_room_switch_and_queue_clear_preserve_current_head(setup_service):
    service, browser, now = setup_service
    head = start_f_recovery(service, browser, now)
    enqueue(service, "后续这一条可以由用户明确清空", "later")
    now[0] = service._next_at
    service.tick()
    browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
    assert service.store.head()["state"] == "retry_pending"
    service.configure(True, "21399689")
    service.timer.stop()
    service.tick()
    assert service.snapshot()["state"] == "queued_room_mismatch"
    assert service.clear_pending() == {"removed": 1, "retained": 1, "retained_uncertain": 1}
    assert service.store.head()["id"] == head["id"]
    service.configure(True, "1698225")
    service.timer.stop()
    service.tick()
    browser.reply(is_login=True, code=0)
    service.tick()
    assert browser.requests[-1]["text"] == head["text"]


@pytest.mark.parametrize("failure", ["not_ready", "exception", "timeout"])
def test_f_readonly_check_failure_retains_automatic_recovery(setup_service, failure):
    service, browser, now = setup_service
    head = start_f_recovery(service, browser, now)
    if failure == "exception":
        original = browser.request_danmaku
        def fails(*args):
            raise RuntimeError("check connection lost")
        browser.request_danmaku = fails
        service.tick()
        browser.request_danmaku = original
    else:
        service.tick()
        if failure == "timeout":
            now[0] += 45
            service.tick()
        else:
            browser.reply(present=False, room_matches=True, chat_ready=False, error="network_unknown")
    assert service.snapshot()["f_retry_active"]
    assert service.store.head()["id"] == head["id"]
    assert not service.snapshot()["blocked"]
    now[0] += 4
    service.tick()
    if browser.requests[-1]["operation"] == "login":
        browser.reply(is_login=True, code=0)
        service.tick()
    browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == head["text"]


def test_f_without_prepost_baseline_still_retries_but_does_not_claim_delivery(setup_service):
    service, browser, now = setup_service
    start_f_recovery(service, browser, now, before_count=None)
    now[0] = service._next_at
    service.tick()
    browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=1)
    assert service.snapshot()["page_visible"] == 0
    service.tick()
    assert service.snapshot()["f_retries"] == 1


@pytest.mark.parametrize("count", [None, True, -1, 10001, "0"])
def test_f_check_without_valid_page_count_waits_for_connection_not_retry(setup_service, count):
    service, browser, now = setup_service
    start_f_recovery(service, browser, now)
    now[0] = service._next_at
    service.tick()
    browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=count)
    assert service.store.head()["state"] == "verify"
    assert service.snapshot()["f_retry_active"]
    assert service.snapshot()["f_retries"] == 0
    now[0] += 0.5
    service.tick()
    assert browser.requests[-1]["operation"] == "login"
    assert sum(row["operation"] == "send" for row in browser.requests) == 1


def test_f_retry_logs_are_safe_and_expose_counts_and_timing(setup_service):
    service, browser, now = setup_service
    start_f_recovery(service, browser, now, cookie="private_cookie", csrf="secret_csrf", msg="private server message")
    reach_f_retry(service, browser, now)
    body = service.log_path.read_text(encoding="utf-8")
    assert "第一条完整字幕" not in body and "private" not in body and "secret" not in body
    sends = [json.loads(line) for line in body.splitlines() if json.loads(line).get("kind") == "dispatch" and json.loads(line).get("operation") == "send"]
    assert [row["head_f_retries"] for row in sends] == [0, 1]
    assert sends[-1]["f_retries"] == 1 and sends[-1]["f_retry_active"]
    assert sends[-1]["last_f_at"].endswith("+08:00")


@pytest.mark.parametrize("later_event,restored", [(None, True), ("dispatch", False), ("result_unknown", False)])
def test_legacy_manual_check_can_restore_only_last_completed_f_post(app, tmp_path, later_event, restored):
    path = tmp_path / "queue.sqlite3"
    store = DanmakuStore(path)
    text = "旧版人工核验覆盖来源的完整字幕"
    store.enqueue("1698225", "system", "old", 1, text, split_danmaku(text))
    head = store.head()
    store.update(head["id"], "blocked", request_id="manual-readonly-request", reason="result_unknown")
    store.close()
    records = [{"kind": "result", "operation": "send", "room": "1698225", "queue_id": head["id"],
                "request_id": "last-post", "outcome": "platform_f", "has_f": True, "platform_code": 0,
                "http_status": 200, "characters": danmaku_utf16_length(head["text"]), "weighted_length": danmaku_weighted_length(head["text"])}]
    records.append({"kind": "result", "operation": "check", "room": "1698225", "queue_id": head["id"], "reason": "result_unknown"})
    if later_event:
        records.append({"kind": "dispatch" if later_event == "dispatch" else "result", "operation": "send", "room": "1698225", "queue_id": head["id"], "outcome": "result_unknown"})
    path.with_name("delivery.jsonl").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    service = DanmakuService(FakeBrowser(), store_path=path)
    try:
        service._clock = lambda: 100
        service.configure(True, "1698225")
        service.timer.stop()
        assert service.snapshot()["f_retry_active"] is restored
        assert service.store.head()["state"] == ("verify" if restored else "blocked")
        assert service.store.head()["text"] == head["text"]
    finally:
        service.close()


def test_unknown_without_f_provenance_does_not_automatically_retry(setup_service):
    service, browser, now = setup_service
    enqueue(service)
    head = service.store.head()
    service.store.update(head["id"], "blocked", reason="result_unknown")
    now[0] += 1000
    service.tick()
    assert not service.snapshot()["f_retry_active"]
    assert all(request["operation"] in {"login", "check"} for request in browser.requests)
    assert not any(request["operation"] == "send" for request in browser.requests)


@pytest.mark.parametrize("payload", [{"code": 1651, "http_status": 200}, {"code": 0, "http_status": 403},
                                     {"code": 0, "http_status": 200, "error": "network_unknown"}])
def test_invalid_or_explicitly_rejected_f_response_does_not_auto_retry(setup_service, payload):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="platform_f", **payload)
    now[0] += 100
    service.tick()
    assert not service.snapshot()["f_retry_active"]
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert sum(request["operation"] == "send" for request in browser.requests) == 1
    assert all(request["operation"] in {"login", "check"} for request in browser.requests[2:])


@pytest.mark.parametrize("state,reason", [("blocked", "result_unknown"), ("blocked", "platform_f"), ("verify", "restart_verify")])
def test_inactive_f_metadata_cannot_be_resurrected_from_old_logs(app, tmp_path, state, reason):
    path = tmp_path / "queue.sqlite3"
    store = DanmakuStore(path)
    text = "新请求结果未知不能继承旧的重试资格"
    store.enqueue("1698225", "system", "old", 1, text, split_danmaku(text))
    head = store.head()
    store.activate_f_recovery(head["id"], new_f=True)
    store.stop_f_recovery(head["id"])
    store.update(head["id"], state, request_id="new-request", reason=reason, platform_code=0)
    store.close()
    record = {"kind": "result", "operation": "send", "room": "1698225", "queue_id": head["id"],
              "request_id": "old-request", "outcome": "platform_f", "has_f": True, "platform_code": 0,
              "http_status": 200, "characters": danmaku_utf16_length(head["text"]), "weighted_length": danmaku_weighted_length(head["text"])}
    path.with_name("delivery.jsonl").write_text(json.dumps(record), encoding="utf-8")
    service = DanmakuService(FakeBrowser(), store_path=path)
    try:
        service.configure(True, "1698225")
        service.timer.stop()
        assert not service.snapshot()["f_retry_active"]
        assert service.store.head()["state"] == state
    finally:
        service.close()


def test_restart_during_f_retry_post_requires_normal_verification(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    browser = FakeBrowser()
    first = DanmakuService(browser, store_path=path)
    now = [100.0]
    first._clock = lambda: now[0]
    first.configure(True, "1698225")
    first.timer.stop()
    first.begin_session(1)
    head = start_f_recovery(first, browser, now)
    reach_f_retry(first, browser, now)
    assert first.store.head()["state"] == "sending"
    first.close()
    second_browser = FakeBrowser()
    second = DanmakuService(second_browser, store_path=path)
    try:
        second._clock = lambda: 200
        second.configure(True, "1698225")
        second.timer.stop()
        assert second.store.head()["state"] == "verify"
        assert not second.snapshot()["f_retry_active"]
        assert second.snapshot()["head_f_retries"] == 1
        second.tick()
        second_browser.reply(is_login=True, code=0)
        second.tick()
        second_browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
        second._clock = lambda: 216
        second.tick()
        assert second.store.head() is None
        assert second.snapshot()["deferred_analysis"] == 1
        assert all(row["operation"] != "send" for row in second_browser.requests)
        assert second.content_filter.candidates()[0]["text"] == head["text"]
    finally:
        second.close()


def test_legacy_confirmed_f_packet_clears_retry_identity_before_next_split_packet(setup_service):
    service, browser, now = setup_service
    enqueue(service, "甲" * 24)
    low_limit_login(service, browser)
    head = service.store.head()
    assert service.store.remaining_parts(head["id"])
    import_legacy_f(service, browser, now)
    reach_f_retry(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, http_status=200)
    service.tick()
    browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=1)
    assert service.store.head()["id"] == head["id"]
    assert service.store.head()["text"] != head["text"] or service.store.head()["part"] == head["part"] + 1
    assert service.snapshot()["f_retries"] == 1
    assert service.snapshot()["head_f_retries"] == 0
    assert not service.snapshot()["f_retry_active"]


def test_content_filter_changes_only_public_payload_not_original_subtitle_event(setup_service):
    service, browser, now = setup_service
    service.content_filter.add_rule("旧词", "中性表达", room="1698225")
    original = "主播刚才说旧词但是字幕原文仍然保留"
    enqueue(service, original)
    assert service._segments[("system", "1")]["translation"] == original
    assert "旧词" not in service.store.head()["text"]
    assert "中性表达" in service.store.head()["text"]
    assert service.snapshot()["filter_hits"] == 1
    login(service, browser, now)
    assert "旧词" not in browser.requests[-1]["text"]
    assert "旧词" not in service.log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("room", ["1698225", "21399689"])
def test_rule_created_with_legacy_room_applies_to_public_payload_in_every_room(setup_service, room):
    service, browser, now = setup_service
    rule = service.content_filter.add_rule("旧词", "中性表达", room="1698225")
    assert rule["room"] == ""
    service.configure(True, room)
    original = "主播刚才说旧词然后继续讲完整的一句话"
    enqueue(service, original)
    head = service.store.head()
    assert head["room"] == room
    assert "中性表达" in head["text"] and "旧词" not in head["text"]
    assert service._segments[("system", "1")]["translation"] == original
    login(service, browser, now)
    assert browser.requests[-1]["room"] == room
    assert "中性表达" in browser.requests[-1]["text"]


def test_filter_replacement_obeys_wrapped_forty_and_account_weighted_limit(setup_service):
    service, browser, now = setup_service
    service.content_filter.add_rule("旧词", "长" * 70)
    enqueue(service, "主播使用了旧词随后继续说话")
    first_id = service.store.head()["id"]
    low_limit_login(service, browser, raw=24)
    packets = [service.store.head()["text"], *service.store.remaining_parts(first_id)]
    assert packets
    assert all(danmaku_utf16_length(packet) <= 40 and danmaku_weighted_length(packet) <= 24 for packet in packets)
    assert "旧词" not in "".join(packets)
    assert service.store.filter_context(first_id)["source"] == "主播使用了旧词随后继续说话"


def test_filter_short_default_waits_and_merges_following_original_source(setup_service):
    service, browser, now = setup_service
    service.content_filter.add_rule("被替换的完整原句", "")
    enqueue(service, "被替换的完整原句", "first")
    assert service.store.head()["text"] == "【已过滤】"
    assert service.store.head()["state"] == "holding_short"
    enqueue(service, "下一句话超过八个字符可以发送", "next")
    assert service.store.head()["text"] == "【已过滤 下一句话超过八个字符可以发送】"
    assert service.store.head()["state"] == "pending"
    assert service.store.filter_context(service.store.head()["id"])["source"] == "被替换的完整原句 下一句话超过八个字符可以发送"


def test_same_filter_version_never_cascades_and_two_updates_use_original_source(setup_service):
    service, browser, now = setup_service
    rule = service.content_filter.add_rule("旧词", "旧词加长")
    enqueue(service, "原句包含旧词并且比较长可以发送")
    initial = service.store.head()["text"]
    for _ in range(3):
        service.refresh_content_filter()
        assert service.store.head()["text"] == initial
    service.content_filter.update_rule(rule["id"], replacement="第二种说法")
    assert service.refresh_content_filter()
    assert "第二种说法" in service.store.head()["text"] and "加长" not in service.store.head()["text"]
    service.content_filter.update_rule(rule["id"], replacement="第三种说法")
    assert service.refresh_content_filter()
    assert "第三种说法" in service.store.head()["text"] and "第二种" not in service.store.head()["text"]


def test_two_filter_updates_repacketize_unsubmitted_multiple_rows_without_loss(setup_service):
    service, browser, now = setup_service
    original = "前" * 38 + "旧词" + "后" * 40
    enqueue(service, original)
    head_id = service.store.head()["id"]
    assert service.store.counts()["queued"] > 1
    rule = service.content_filter.add_rule("旧词", "中性替换")
    assert service.refresh_content_filter()
    first_context = service.store.filter_context(head_id)
    assert not first_context["frozen"]
    assert first_context["source"] == original
    service.content_filter.update_rule(rule["id"], replacement="另一个表达")
    assert service.refresh_content_filter()
    packets = [service.store.head()["text"], *service.store.remaining_parts(head_id)]
    assert "".join(packet[1:-1] for packet in packets) == original.replace("旧词", "另一个表达")
    assert service.snapshot()["cleared"] == 0
    assert service.store.head()["id"] == head_id


def test_f_candidate_never_activates_rules_and_confirmed_rule_repairs_after_check(setup_service):
    service, browser, now = setup_service
    head = start_f_recovery(service, browser, now)
    assert service.content_filter.rules() == []
    candidates = service.content_filter.candidates("1698225")
    assert len(candidates) == 1 and candidates[0]["reason"] == "platform_f"
    service.content_filter.add_rule("第一条", "新词")
    assert service.refresh_content_filter()
    assert service.store.head()["text"] == head["text"]
    now[0] = service._next_at
    service.tick()
    browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
    assert service.store.head()["state"] == "pending"
    service.tick()
    assert browser.requests[-1]["text"] == head["text"].replace("第一条", "新词")
    browser.reply(outcome="platform_f", code=0, http_status=200, error="none")
    assert len(service.content_filter.candidates("1698225")) == 2
    assert not service.snapshot()["f_retry_active"]
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1


def test_confirmed_removal_repairs_legacy_failed_head_preserving_fifo_and_interval(setup_service):
    service, browser, now = setup_service
    service.configure(True, "1698225", interval_seconds=5.0)
    service.timer.stop()
    enqueue(service, "不过 谢谢 恭喜获得评价 谢谢", "observed")
    enqueue(service, "后面的消息需要按顺序继续发送", "later")
    login(service, browser, now)
    old_head = service.store.head()
    import_legacy_f(service, browser, now)
    deadline = service._next_at
    original_context = service.store.filter_context(old_head["id"])
    service.content_filter.add_rule("评价", "", "1698225", action="remove")
    service.refresh_content_filter()
    assert service.store.head()["text"] == "【不过 谢谢 恭喜获得评价 谢谢】"
    service.tick()
    browser.reply(present=False, occurrence_count=0, room_matches=True, chat_ready=True)
    repaired = service.store.head()
    assert repaired["text"] == "【不过 谢谢 恭喜获得 谢谢】"
    assert repaired["id"] == old_head["id"] and repaired["part"] == old_head["part"]
    assert service.store.filter_context(old_head["id"])["source"] == original_context["source"]
    assert service.store.filter_context(old_head["id"])["frozen"]
    assert service.store.delivery_baseline(old_head["id"]) is None
    assert service.snapshot()["queued"] == 2 and service.snapshot()["cleared"] == 0
    assert service._next_at == deadline
    service.tick()
    assert sum(row["operation"] == "send" for row in browser.requests) == 1
    now[0] = deadline
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == repaired["text"]
    browser.reply(outcome="receipt_accepted", code=0, http_status=200)
    service.tick()
    browser.reply(present=True, occurrence_count=1, room_matches=True, chat_ready=True)
    assert service.store.head()["id"] != old_head["id"]
    assert service.snapshot()["page_visible"] == 1
    assert "评价" not in service.log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("before_count,fields", [
    (None, {"present": False, "occurrence_count": 0}),
    (True, {"present": False, "occurrence_count": 0}),
    (-1, {"present": False, "occurrence_count": 0}),
    (10001, {"present": False, "occurrence_count": 0}),
    (0, {"present": True, "occurrence_count": 0}),
    (2, {"present": True, "occurrence_count": 1}),
    (2, {"present": False, "occurrence_count": 2}),
    (0, {"present": False, "occurrence_count": 0, "room_matches": False}),
    (0, {"present": False, "occurrence_count": 0, "chat_ready": False}),
    (0, {"present": False, "occurrence_count": 0, "error": "connection_error"}),
])
def test_f_rule_repair_requires_consistent_fresh_page_observation(setup_service, before_count, fields):
    service, browser, now = setup_service
    old_head = start_f_recovery(service, browser, now, before_count=before_count)
    service.content_filter.add_rule("第一条", "", action="remove")
    service.refresh_content_filter()
    service.tick()
    browser.reply(**{"room_matches": True, "chat_ready": True, **fields})
    assert service.store.head()["text"] == old_head["text"]
    assert not any(json.loads(line)["kind"] == "f_head_rule_repaired"
                   for line in service.log_path.read_text(encoding="utf-8").splitlines())


def test_new_occurrence_wins_over_new_filter_rule_without_rewrite(setup_service):
    service, browser, now = setup_service
    old_head = start_f_recovery(service, browser, now)
    service.content_filter.add_rule("第一条", "", action="remove")
    service.refresh_content_filter()
    service.tick()
    browser.reply(present=True, occurrence_count=1, room_matches=True, chat_ready=True)
    assert service.store.head() is None
    saved = service.store.connection.execute("SELECT text,state FROM messages WHERE id=?", (old_head["id"],)).fetchone()
    assert saved["text"] == old_head["text"] and saved["state"] == "page_visible"


def test_legacy_f_repair_excludes_already_applied_rules_across_rule_versions(setup_service):
    service, browser, now = setup_service
    service.content_filter.add_rule("原词", "新词")
    service.content_filter.add_rule("新词", "不能级联")
    enqueue(service, "原词以及评价都在同一句完整字幕里")
    login(service, browser, now)
    assert "新词" in browser.requests[-1]["text"] and "不能级联" not in browser.requests[-1]["text"]
    import_legacy_f(service, browser, now)
    service.content_filter.add_rule("评价", "", action="remove")
    service.refresh_content_filter()
    service.tick()
    browser.reply(present=False, occurrence_count=0, room_matches=True, chat_ready=True)
    repaired = service.store.head()["text"]
    assert "评价" not in repaired and "新词" in repaired and "不能级联" not in repaired
    now[0] = service._next_at
    service.tick()
    import_legacy_f(service, browser, now)
    service.content_filter.add_rule("不在句子中的词", "", action="remove")
    service.refresh_content_filter()
    reach_f_retry(service, browser, now)
    assert browser.requests[-1]["text"] == repaired
    assert service.store.head()["text"] == repaired


def test_empty_removal_preserves_legacy_failed_message_and_blocks_unchanged_retry(setup_service):
    service, browser, now = setup_service
    original = "这一整句话都需要删除"
    enqueue(service, original)
    login(service, browser, now)
    import_legacy_f(service, browser, now)
    service.content_filter.add_rule(original, "", action="remove")
    service.refresh_content_filter()
    service.tick()
    browser.reply(present=False, occurrence_count=0, room_matches=True, chat_ready=True)
    assert service.snapshot()["filter_error"]
    assert service.store.head()["text"] == "【" + original + "】"
    assert service.snapshot()["queued"] == 1
    now[0] += 10
    service.tick()
    assert sum(row["operation"] == "send" for row in browser.requests) == 1


@pytest.mark.parametrize("outcome", ["receipt_accepted", "result_unknown"])
def test_rule_edits_never_repair_accepted_or_unknown_submission(setup_service, outcome):
    service, browser, now = setup_service
    enqueue(service, "原句含有评价但投递结果尚未明确")
    login(service, browser, now)
    old_head = service.store.head()
    browser.reply(outcome=outcome, code=0, http_status=200)
    service.content_filter.add_rule("评价", "", action="remove")
    service.refresh_content_filter()
    if outcome == "result_unknown":
        assert service.store.head() is None
        saved = service.store.connection.execute("SELECT * FROM messages WHERE id=?", (old_head["id"],)).fetchone()
        assert saved["text"] == old_head["text"] and saved["state"] == "cancelled"
        assert service.content_filter.candidates()[0]["text"] == old_head["text"]
        assert not service.snapshot()["f_retry_active"]
        return
    service.tick()
    browser.reply(present=False, occurrence_count=0, room_matches=True, chat_ready=True)
    assert service.store.head()["text"] == old_head["text"]
    assert not service.snapshot()["f_retry_active"]


def test_submitted_long_utterance_freezes_remaining_packets_across_new_rules(setup_service):
    service, browser, now = setup_service
    original = "前" * 35 + "旧词" + "后" * 35
    enqueue(service, original)
    low_limit_login(service, browser, raw=24)
    head_id = service.store.head()["id"]
    remaining_before = service.store.remaining_parts(head_id)
    service.content_filter.add_rule("旧词", "全新表达")
    service.refresh_content_filter()
    assert service.store.remaining_parts(head_id) == remaining_before
    finish(service, browser, now)
    service.tick()
    packets = [service.store.head()["text"], *service.store.remaining_parts(head_id)]
    assert "全新表达" not in "".join(packets)
    assert service.store.filter_context(head_id)["frozen"]


def test_candidate_write_failure_retains_exact_packet_until_it_can_be_safely_deferred(setup_service, monkeypatch):
    service, browser, now = setup_service
    original_record = service.content_filter.record_failure
    def cannot_save(*args, **kwargs):
        raise DanmakuFilterError("candidate write denied containing private words")
    monkeypatch.setattr(service.content_filter, "record_failure", cannot_save)
    enqueue(service)
    login(service, browser, now)
    head = service.store.head()
    browser.reply(outcome="platform_f", code=0, http_status=200, error="none")
    assert service.snapshot()["filter_candidate_error"]
    assert not service.snapshot()["filter_error"]
    assert service.store.head()["id"] == head["id"]
    assert service.store.head()["text"] == head["text"]
    assert service.snapshot()["deferred_analysis"] == 0
    now[0] += 100
    service.tick()
    assert sum(row["operation"] == "send" for row in browser.requests) == 1
    assert not service.content_filter.candidates()
    monkeypatch.setattr(service.content_filter, "record_failure", original_record)
    now[0] += 3.5
    service.tick()
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.content_filter.candidates()[0]["text"] == head["text"]
    logs = service.log_path.read_text(encoding="utf-8")
    assert "private words" not in logs
    assert "filter_candidate_error" in logs


def test_corrupt_filter_preserves_new_finals_and_repair_refresh_recovers(app, tmp_path):
    path, filter_path = tmp_path / "queue.sqlite3", tmp_path / "filters.json"
    filter_path.write_text("{invalid", encoding="utf-8")
    browser = FakeBrowser()
    service = DanmakuService(browser, store_path=path, filter_path=filter_path)
    now = [100.0]
    service._clock = lambda: now[0]
    try:
        service.configure(True, "1698225")
        service.timer.stop()
        service.begin_session(1)
        enqueue(service, "词库损坏时完整字幕仍需要保留", "first")
        assert service.snapshot()["queued"] == 1
        assert service.snapshot()["filter_error"]
        service.tick()
        assert browser.requests == [] and service.snapshot()["state"] == "filter_error"
        service.store.connection.commit()
        other = DanmakuStore(path)
        assert other.counts()["queued"] == 1
        other.close()
        filter_path.write_text(json.dumps({"version": 1, "enabled": True, "rules": [], "candidates": []}), encoding="utf-8")
        fixed = DanmakuContentFilter(filter_path)
        fixed.add_rule("损坏", "修复")
        assert service.refresh_content_filter()
        assert not service.snapshot()["filter_error"]
        assert "修复" in service.store.head()["text"]
        login(service, browser, now)
        assert "损坏" not in browser.requests[-1]["text"]
    finally:
        service.close()


def test_apply_failure_keeps_original_final_and_resume_after_repair(setup_service, monkeypatch):
    service, browser, now = setup_service
    original_apply = service.content_filter.apply
    def failure(*args):
        raise DanmakuFilterError("apply private phrase failed")
    monkeypatch.setattr(service.content_filter, "apply", failure)
    enqueue(service, "规则执行失败的完整字幕应当保留")
    assert service.snapshot()["queued"] == 1 and service.snapshot()["filter_error"]
    service.tick()
    assert browser.requests == []
    monkeypatch.setattr(service.content_filter, "apply", original_apply)
    assert service.refresh_content_filter()
    login(service, browser, now)
    assert "规则执行失败" in browser.requests[-1]["text"]
    assert "private phrase" not in service.log_path.read_text(encoding="utf-8")


def test_invalid_filter_output_is_not_dropped_or_sent(setup_service):
    service, browser, now = setup_service
    service.content_filter.add_rule("整句需要保留", "<asr_text>")
    enqueue(service, "整句需要保留")
    assert service.snapshot()["queued"] == 1
    assert service.snapshot()["filter_error"]
    service.tick()
    assert browser.requests == []
    assert service.store.head()["text"] == "【整句需要保留】"


@pytest.mark.parametrize("outcome,code,reason", [("platform_f", 0, "platform_f"), ("platform_rejected", 1651, "platform_rejected"), ("result_unknown", None, "result_unknown")])
def test_failure_candidates_are_reason_specific_and_never_automatically_enabled(setup_service, outcome, code, reason):
    service, browser, now = setup_service
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome=outcome, code=code, http_status=200, error="none")
    assert service.content_filter.rules() == []
    assert service.content_filter.candidates("1698225")[0]["reason"] == reason
    assert service.snapshot()["filter_candidate_count"] == 1


def test_clear_virtual_filter_tail_is_not_regenerated_by_rule_refresh(setup_service):
    service, browser, now = setup_service
    enqueue(service, "前" * 39 + "旧词" + "后" * 40)
    service.content_filter.add_rule("旧词", "新说法")
    service.refresh_content_filter()
    head_id = service.store.head()["id"]
    assert service.store.remaining_parts(head_id)
    service.tick()
    assert browser.requests[-1]["operation"] == "login"
    service.clear_pending()
    assert not service.store.remaining_parts(head_id)
    service.content_filter.add_rule("前", "新前")
    service.refresh_content_filter()
    assert not service.store.remaining_parts(head_id)
    assert service.store.filter_context(head_id)["frozen"]


def test_legacy_unsubmitted_filter_source_keeps_virtual_tail_packets(setup_service):
    service, browser, now = setup_service
    enqueue(service, "原始完整字幕旧词后续文字很多")
    head = service.store.head()
    assert service.store.prepare_parts(head["id"], ["【原始完整】", "【字幕旧词】", "【后续文字很多】"])
    with service.store.connection:
        service.store.connection.execute("DELETE FROM metadata WHERE name LIKE 'content_filter:%'")
    service.content_filter.add_rule("旧词", "新词")
    assert service.refresh_content_filter()
    packets = [service.store.head()["text"], *service.store.remaining_parts(head["id"])]
    body = "".join(packet[1:-1] for packet in packets).replace(" ", "")
    assert body == "原始完整字幕新词后续文字很多"


@pytest.mark.parametrize("failure", ["not_sent", "not_dispatched", "exception"])
def test_filter_version_can_change_only_if_post_was_proven_unsubmitted(setup_service, failure):
    service, browser, now = setup_service
    original = "旧词出现在可以正常发送的完整字幕中"
    enqueue(service, original)
    head = service.store.head()
    if failure == "not_dispatched":
        service._login_room = service.room
        browser.accept_requests = False
        service.tick()
    elif failure == "exception":
        service._login_room = service.room
        def exception(*args):
            raise RuntimeError("uncertain after dispatch")
        browser.request_danmaku = exception
        service.tick()
    else:
        login(service, browser, now)
        browser.reply(outcome="not_sent", error="wrong_room")
    service.content_filter.add_rule("旧词", "新词")
    service.refresh_content_filter()
    if failure == "not_dispatched":
        assert not service.store.filter_context(service.store.head()["id"])["frozen"]
        assert "新词" in service.store.head()["text"]
    else:
        assert service.store.head() is None
        archived = service.store.connection.execute("SELECT * FROM messages WHERE id=?", (head["id"],)).fetchone()
        assert archived["state"] == "cancelled" and "旧词" in archived["text"]
        assert service.content_filter.candidates()[0]["text"] == archived["text"]
        assert service.snapshot()["deferred_analysis"] == 1


def test_submitted_balanced_tail_does_not_freeze_following_new_final(setup_service):
    service, browser, now = setup_service
    enqueue(service, "甲" * 38 + "乙" * 3, "long")
    login(service, browser, now)
    finish(service, browser, now)
    old_tail = service.store.head()
    assert old_tail["state"] == "pending"
    assert 11 <= danmaku_utf16_length(old_tail["text"]) <= 40
    assert service.store.filter_context(old_tail["id"])["frozen"]
    enqueue(service, "新的完整字幕应当独立管理并且替换", "new")
    assert service.store.head()["id"] == old_tail["id"]
    assert service.store.head()["text"] == old_tail["text"]
    assert service.store.head()["state"] == "pending"
    next_row = service.store.connection.execute("SELECT * FROM messages WHERE id>? AND state='pending' ORDER BY id LIMIT 1", (old_tail["id"],)).fetchone()
    assert next_row is not None
    assert not service.store.filter_context(next_row["id"])["frozen"]
    service.content_filter.add_rule("新的", "最新")
    service.refresh_content_filter()
    new_row = service.store.connection.execute("SELECT * FROM messages WHERE id=?", (next_row["id"],)).fetchone()
    assert "最新" in new_row["text"]


def test_filter_repacketization_remembers_public_payload_for_ten_second_dedup(setup_service):
    service, browser, now = setup_service
    service.store._clock = lambda: 1000
    enqueue(service, "旧词重复出现但十秒以内只接受一次完整字幕", "first")
    service.content_filter.add_rule("旧词", "新词")
    service.refresh_content_filter()
    enqueue(service, "旧词重复出现但十秒以内只接受一次完整字幕", "second")
    assert service.store.counts()["queued"] == 1


def test_unchanged_filter_revision_does_not_rescan_queue_on_each_timer_tick(setup_service, monkeypatch):
    service, browser, now = setup_service
    enqueue(service)
    original = service.store.unsubmitted_filter_heads
    calls = []
    def tracked():
        calls.append(1)
        return original()
    monkeypatch.setattr(service.store, "unsubmitted_filter_heads", tracked)
    browser.is_browser_ready = False
    for _ in range(5):
        service.tick()
    assert not calls
    service.content_filter.add_rule("完整", "全句")
    service.tick()
    assert len(calls) == 1


@pytest.mark.parametrize("unsubmitted", ["not_sent", "not_dispatched"])
def test_rule_change_during_send_applies_after_proven_unsubmitted_unfreezes(setup_service, unsubmitted):
    service, browser, now = setup_service
    rule = service.content_filter.add_rule("旧词", "首次说法")
    enqueue(service, "完整原句包含旧词并且可以正常发送")
    if unsubmitted == "not_sent":
        login(service, browser, now)
        assert "首次说法" in browser.requests[-1]["text"]
        service.content_filter.update_rule(rule["id"], replacement="新表达")
        assert service.refresh_content_filter()
        assert service._last_filter_revision == service.content_filter.revision
        browser.reply(outcome="not_sent", error="wrong_room")
        assert service.store.head() is None
        assert service.snapshot()["deferred_analysis"] == 1
        assert "首次说法" in service.content_filter.candidates()[0]["text"]
        now[0] += 3.5
        service.tick()
        assert sum(row["operation"] == "send" for row in browser.requests) == 1
        return
    else:
        service._login_room = service.room
        normal_request = browser.request_danmaku
        def not_dispatched(operation, request_id, room, text=""):
            service.content_filter.update_rule(rule["id"], replacement="新表达")
            service.refresh_content_filter()
            normal_request(operation, request_id, room, text)
            return False
        browser.request_danmaku = not_dispatched
        service.tick()
        browser.request_danmaku = normal_request
    assert service._last_filter_revision is None
    assert not service.store.filter_context(service.store.head()["id"])["frozen"]
    now[0] += 3.5
    service.tick()
    assert browser.requests[-1]["operation"] == "login"
    browser.reply(is_login=True, code=0)
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert "新表达" in browser.requests[-1]["text"]
    assert "首次说法" not in browser.requests[-1]["text"]


def write_advisory_wordlist(path):
    path.write_text(json.dumps({"version": 1, "sources": [{"id": "test", "name": "Test",
        "url": "https://example.invalid", "revision": "fixture", "license": "MIT", "license_text": "test"}],
        "entries": [{"term": "测试词", "category": "test", "source_id": "test"}], "allowlist": []},
        ensure_ascii=False), encoding="utf-8")


def test_advisory_detection_records_actual_failed_packet_and_preserves_next_deadline(setup_service, monkeypatch):
    service, browser, now = setup_service
    write_advisory_wordlist(service.content_filter.detector_path)
    service.content_filter.reload_detector()
    calls = []
    real_detect = service.content_filter._detector.detect
    def tracked(text, room=""):
        calls.append(text)
        return real_detect(text, room)
    monkeypatch.setattr(service.content_filter._detector, "detect", tracked)
    enqueue(service, "含有测试词但不自动改写的完整句子", "first")
    enqueue(service, "随后完整句子仍必须在后面等待", "second")
    login(service, browser, now)
    original = browser.requests[-1]["text"]
    original_id = service.store.head()["id"]
    browser.reply(outcome="platform_f", code=0, http_status=200, error="none")
    deadline = service._next_at
    assert calls == [original]
    assert service._next_at == deadline
    assert service.store.head()["id"] != original_id
    next_text = service.store.head()["text"]
    assert next_text == "【随后完整句子仍必须在后面等待】"
    assert service.content_filter.rules() == []
    assert service.content_filter.candidates()[0]["analysis"]["summary_code"] == "advisory_hits"
    now[0] += 3.5
    service.tick()
    if browser.requests[-1]["operation"] == "check":
        browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
        service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == next_text
    assert calls == [original]
    assert service.content_filter.candidates()[0]["text"] == original
    assert service.snapshot()["filter_detection"]["status"] == "ready"


def test_bad_advisory_wordlist_never_stops_send_or_failure_recording(setup_service):
    service, browser, now = setup_service
    service.content_filter.detector_path.write_text("{invalid", encoding="utf-8")
    service.content_filter.reload_detector()
    enqueue(service)
    login(service, browser, now)
    browser.reply(outcome="platform_f", code=0, http_status=200, error="none")
    candidate = service.content_filter.candidates()[0]
    assert candidate["analysis"]["status"] == "error"
    assert not service._filter_blocked and not service.snapshot()["filter_error"]
    assert not service.snapshot()["f_retry_active"]
    assert service.store.head() is None
    assert service.snapshot()["deferred_analysis"] == 1
    assert service.snapshot()["filter_detection"]["status"] == "error"


def test_service_startup_backfills_legacy_candidates_without_network(app, tmp_path):
    path = tmp_path / "danmaku-content-filter.json"
    content_filter = DanmakuContentFilter(path)
    candidate = content_filter.record_failure("1698225", "【测试词】", "platform_f")
    state = json.loads(path.read_text(encoding="utf-8"))
    state["candidates"][0].pop("analysis")
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    write_advisory_wordlist(content_filter.detector_path)
    browser = FakeBrowser()
    service = DanmakuService(browser, store_path=tmp_path / "queue.sqlite3")
    try:
        backfilled = service.content_filter.candidates()[0]
        assert backfilled["id"] == candidate["id"] and backfilled["count"] == 1
        assert backfilled["analysis"]["summary_code"] == "advisory_hits"
        assert browser.requests == []
    finally:
        service.close()
