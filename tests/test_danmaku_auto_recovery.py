"""End-to-end recovery state tests using a fake browser and fake AI only."""
import copy
import sqlite3

import pytest
from PySide6.QtCore import QObject, Signal

from test_danmaku_service import app, FakeBrowser, enqueue
from fanyiji.danmaku_service import DanmakuService


SETTINGS = {"codex_model": "fake-model", "codex_effort": "low", "codex_speed": "fast"}
TEXT = "感谢你的评价我会继续努力"


class FakeAI(QObject):
    finished = Signal(str, object, str)

    def __init__(self):
        super().__init__()
        self.busy = False
        self.calls = []
        self.cancelled = []
        self.closed = False

    def submit(self, token, data, settings):
        if self.busy or self.closed:
            return False
        self.busy = True
        self.calls.append({"token": token, "data": copy.deepcopy(data), "settings": dict(settings)})
        return True

    def reply(self, result=None, error="", *, token=None):
        self.busy = False
        self.finished.emit(token or self.calls[-1]["token"], result, error)

    def cancel(self, token=None):
        self.cancelled.append(token)
        self.busy = False

    def close(self, timeout=2.0):
        self.closed = True
        self.busy = False
        return True


@pytest.fixture
def recovery(app, tmp_path):
    browser, ai = FakeBrowser(), FakeAI()
    service = DanmakuService(browser, store_path=tmp_path / "queue.sqlite3", ai_analyzer=ai)
    now = [100.0]
    service._clock = lambda: now[0]
    service.configure(True, "1698225", interval_seconds=5.0)
    service.configure_ai(SETTINGS)
    service._apply_limits({"max_weighted_length": 80})
    service.timer.stop()
    service.review_timer.stop()
    service.begin_session(1)
    yield service, browser, ai, now
    service.close()


def start(service, browser, now, text=TEXT, *, maximum=80):
    enqueue(service, text)
    service.tick()
    assert browser.requests[-1]["operation"] == "login"
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=maximum)
    now[0] += 5.0
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    return service.store.head()


def f_reply(browser, *, before_count=0):
    browser.reply(outcome="platform_f", code=0, http_status=200, error="none", before_count=before_count)


def patch(text):
    body = text[1:-1]
    return {"action": "rewrite", "revised_text": body.replace("评价", "感想"),
            "reason": "普通同义表述调整 实际拒绝原因未确定", "suspected_cause": "unknown",
            "edits": [{"term": "评价", "replacement": "感想"}]}


def check(browser, count=0):
    browser.reply(present=count > 0, room_matches=True, chat_ready=True, error="none", occurrence_count=count)


def apply_ai(service, browser, ai, head):
    service._review_history_tick()
    ai.reply({"revised_text": patch(head["text"])["revised_text"]})
    assert service._ai_proposal is None
    candidate = next(item for item in service.content_filter.candidates() if item["queue_id"] == head["id"])
    plan = service.content_filter.validation_plan(candidate["id"])
    assert plan is not None and plan["state"] == "waiting_room"
    assert plan["edits"] == [{"term": "评价", "replacement": "感想"}]
    assert not service.content_filter.rules()
    return candidate, plan


def send_count(browser):
    return sum(request["operation"] == "send" for request in browser.requests)


def next_send(service, browser, now):
    initial = send_count(browser)
    for _ in range(12):
        service.tick()
        if service._inflight:
            operation = service._inflight["operation"]
            if operation == "send":
                assert send_count(browser) == initial + 1
                return
            if operation == "login":
                browser.reply(is_login=True, code=0, error="none")
            else:
                check(browser, 0)
        else:
            now[0] += 5.0
    pytest.fail("retry did not reach the send state")


def test_failed_part_moves_to_analysis_while_unsent_parts_keep_fifo_and_interval(recovery):
    service, browser, ai, now = recovery
    head = start(service, browser, now, TEXT + "已经收到你的留言谢谢大家" * 5, maximum=40)
    enqueue(service, "后面还有第二条完整译文", segment="second")
    remaining = service.store.strict_remaining_parts(head["id"])
    assert remaining
    others = [dict(row) for row in service.store.connection.execute("SELECT * FROM messages WHERE id!=?", (head["id"],))]
    f_reply(browser)
    service._review_history_tick()
    assert len(ai.calls) == 1
    assert ai.calls[0]["settings"] == SETTINGS
    assert ai.calls[0]["data"]["text"] == head["text"]
    apply_ai(service, browser, ai, head)
    assert service.store.strict_remaining_parts(head["id"]) == remaining[1:]
    assert service.store.head()["text"] == remaining[0]
    assert service.store.head()["part"] == head["part"] + 1
    for original in others:
        current = service.store.connection.execute("SELECT * FROM messages WHERE id=?", (original["id"],)).fetchone()
        assert dict(current) == original
    service.tick()
    assert send_count(browser) == 1
    now[0] += 4.999
    service.tick()
    assert send_count(browser) == 1
    if service._inflight:
        check(browser, 0)
    now[0] += 0.0011
    service.tick()
    if service._inflight and service._inflight["operation"] == "check":
        check(browser, 0)
        service.tick()
    assert send_count(browser) == 2
    assert browser.requests[-1]["text"] == remaining[0]
    assert not service.content_filter.rules()


@pytest.mark.parametrize("new_baseline", [0, 1])
def test_original_page_observation_does_not_validate_the_ai_correction(recovery, new_baseline):
    service, browser, ai, now = recovery
    head = start(service, browser, now)
    enqueue(service, "后面还有第二条完整译文", segment="second")
    f_reply(browser)
    candidate, plan = apply_ai(service, browser, ai, head)
    assert not service.content_filter.rules()
    now[0] += 5.0
    next_send(service, browser, now)
    assert send_count(browser) == 2
    browser.reply(outcome="receipt_accepted", code=0, http_status=200, error="none", before_count=new_baseline)
    assert not service.content_filter.rules()
    service.tick()
    check(browser, new_baseline)
    assert not service.content_filter.rules()
    now[0] += 0.5
    service.tick()
    check(browser, new_baseline + 1)
    assert not service.content_filter.rules()
    assert service.store.head() is None
    candidates = service.content_filter.candidates("1698225")
    assert candidates and candidates[0]["status"] == "waiting_validation"
    assert service.content_filter.validation_plan(candidate["id"]) == plan


def test_late_manual_same_text_observation_cannot_validate_an_archived_failure(recovery):
    service, browser, ai, now = recovery
    head = start(service, browser, now)
    f_reply(browser)
    service._review_history_tick()
    assert ai.calls
    failed_request = browser.requests[-1]
    browser.danmaku_result.emit({"operation": "check", "request_id": failed_request["request_id"], "result": {
        "present": True, "room_matches": True, "chat_ready": True, "occurrence_count": 1, "error": "none"}})
    assert send_count(browser) == 1
    assert service.store.head() is None
    assert not service.content_filter.rules()
    assert all(candidate["status"] != "resolved" for candidate in service.content_filter.candidates("1698225"))
    assert not ai.cancelled
    ai.reply(patch(head["text"]))
    assert not service.content_filter.rules()
    assert service.content_filter.candidates()[0]["status"] == "waiting_validation"


@pytest.mark.parametrize("stale", ["token", "candidate_room", "candidate_text", "candidate_deleted", "model_generation"])
def test_stale_ai_result_cannot_change_analysis_candidate(recovery, stale):
    service, browser, ai, now = recovery
    head = start(service, browser, now)
    f_reply(browser)
    service._review_history_tick()
    result = patch(head["text"])
    if stale in {"candidate_room", "candidate_text", "candidate_deleted"}:
        state = copy.deepcopy(service.content_filter._state)
        if stale == "candidate_deleted":
            state["candidates"] = []
        elif stale == "candidate_room":
            state["candidates"][0]["room"] = "123456"
        else:
            state["candidates"][0]["text"] = "【这是一条不同的完整译文】"
        service.content_filter._save(state)
    elif stale == "model_generation":
        service.configure_ai({**SETTINGS, "codex_model": "another-model"})
    before = service.store.head()
    ai.reply(result, token="wrong-token" if stale == "token" else None)
    assert service.store.head() == before
    assert service._ai_proposal is None
    assert not service.content_filter.rules()
    assert send_count(browser) == 1


def test_unknown_after_f_moves_each_failure_to_analysis_without_duplicate_post(recovery):
    service, browser, ai, now = recovery
    head = start(service, browser, now)
    enqueue(service, "第二句完整内容作为后续队列", segment="second")
    f_reply(browser)
    service._review_history_tick()
    next_send(service, browser, now)
    assert send_count(browser) == 2
    assert browser.requests[-1]["text"] != head["text"]
    browser.reply(outcome="result_unknown", code=None, error="network_unknown", before_count=0)
    ai.reply(patch(head["text"]))
    for _ in range(8):
        now[0] += 5.0
        service.tick()
        if service._inflight:
            assert service._inflight["operation"] != "send"
            if service._inflight["operation"] == "login":
                browser.reply(is_login=True, code=0, error="none")
            else:
                check(browser, 0)
    assert service.store.head() is None
    assert send_count(browser) == 2
    assert not service.content_filter.rules()
    assert len(service.content_filter.candidates()) == 2
    assert service.content_filter.validation_plan(service.content_filter.candidates()[0]["id"]) is not None


def test_unknown_without_f_enters_analysis_without_blocking_or_reposting(recovery):
    service, browser, ai, now = recovery
    head = start(service, browser, now)
    browser.reply(outcome="result_unknown", code=None, error="request_timeout", before_count=0)
    service._review_history_tick()
    for _ in range(12):
        now[0] += 5.0
        service.tick()
        if service._inflight:
            assert service._inflight["operation"] != "send"
    assert send_count(browser) == 1
    assert not [request for request in browser.requests if request["operation"] == "check"]
    assert len(ai.calls) == 1 and ai.calls[0]["data"]["text"] == head["text"]
    assert service.store.head() is None


def test_not_sent_moves_to_analysis_and_reauthenticates_for_the_next_head(recovery):
    service, browser, ai, now = recovery
    head = start(service, browser, now)
    enqueue(service, "下一句完整内容仍然继续发送", segment="second")
    browser.reply(outcome="not_sent", error="not_logged_in")
    assert service.store.head()["state"] == "pending"
    assert service.store.head()["text"] != head["text"]
    service.tick()
    assert send_count(browser) == 1
    now[0] += 5.0
    service.tick()
    assert browser.requests[-1]["operation"] == "login"
    browser.reply(is_login=True, code=0, error="none")
    service.tick()
    assert send_count(browser) == 2
    assert browser.requests[-1]["text"] != head["text"]
    assert service.content_filter.candidates()[0]["text"] == head["text"]


def test_login_transport_failure_is_not_misreported_as_explicit_logout(recovery):
    service, browser, ai, now = recovery
    enqueue(service, TEXT)
    service.tick()
    browser.reply(is_login=False, code=None, error="network_unknown")
    assert service.snapshot()["state"] == "waiting_connection"
    assert not ai.calls
    now[0] += 5.0
    service.tick()
    browser.reply(is_login=False, code=0, error="none")
    assert service.snapshot()["state"] == "login_required"


@pytest.mark.parametrize("bad", ["missing", "extra", "nonlocal", "same", "negation", "error"])
def test_bad_ai_result_preserves_queue(recovery, bad):
    service, browser, ai, now = recovery
    head = start(service, browser, now)
    enqueue(service, "未被失败分析修改的后续完整译文", segment="second")
    f_reply(browser)
    service._review_history_tick()
    pending = service.store.head()
    result = patch(head["text"])
    if bad == "missing":
        del result["edits"]
    elif bad == "extra":
        result["unexpected"] = "extra"
    elif bad == "nonlocal":
        result["revised_text"] = "谢谢你的感想我很开心"
    elif bad == "same":
        result["revised_text"] = head["text"][1:-1]
    elif bad == "negation":
        result["revised_text"] = head["text"][1:-1].replace("继续", "不再")
        result["edits"] = [{"term": "继续", "replacement": "不再"}]
    ai.reply(result, error="ai_timeout" if bad == "error" else "")
    assert service.store.head() == pending
    assert not service.content_filter.rules()
    assert send_count(browser) == 1
    assert service.content_filter.validation_plan(service.content_filter.candidates()[0]["id"]) is None


def test_failed_background_ai_review_is_bounded_and_never_reposts_the_formal_sentence(recovery):
    service, browser, ai, now = recovery
    head = start(service, browser, now)
    f_reply(browser)
    service._review_history_tick()
    assert len(ai.calls) == 1
    ai.reply(error="ai_timeout")
    for _ in range(5):
        now[0] += 5.0
        service._review_history_tick()
        service.tick()
    assert len(ai.calls) == 1
    assert service.store.head() is None
    assert send_count(browser) == 1
    assert not service.content_filter.rules()


def test_constructor_default_preserves_existing_queue(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    first = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI())
    first.configure(True, "1698225")
    first.begin_session(1)
    enqueue(first, TEXT)
    first.timer.stop()
    expected = first.store.head()["text"]
    first.close()
    second = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI())
    try:
        assert second.store.head()["text"] == expected
    finally:
        second.close()


def test_restart_reset_clears_only_old_queue_and_metrics_preserves_candidates_and_rules(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    first = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI())
    first.configure(True, "1698225")
    first.begin_session(1)
    enqueue(first, TEXT)
    first.timer.stop()
    head = first.store.head()
    first.store.update(head["id"], "blocked", reason="result_unknown")
    first.store.increment("failures")
    first.content_filter.record_failure("1698225", head["text"], "result_unknown", queue_id=head["id"])
    first.content_filter.add_rule("评价", "感想", "1698225")
    first.close()
    second = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI(), reset_on_start=True)
    try:
        assert second.store.head() is None
        assert second.store.counts()["queued"] == 0
        assert second.store.counts()["failures"] == 0
        assert second.content_filter.candidates("1698225")
        assert second.content_filter.rules()[0]["term"] == "评价"
    finally:
        second.close()


def seed_history(path):
    first = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI())
    first.configure(True, "1698225")
    first.begin_session(1)
    enqueue(first, TEXT)
    first.timer.stop()
    first.review_timer.stop()
    head = first.store.head()
    first.store.update(head["id"], "blocked", reason="result_unknown")
    first.store.increment("failures")
    first.store.increment("accepted")
    candidate = first.content_filter.record_failure("1698225", head["text"], "result_unknown", queue_id=head["id"])
    first.content_filter.add_rule("旧规则", "保留规则", "1698225")
    first.close()
    return head, candidate


def test_restart_keeps_a_complete_sqlite_backup_before_reset(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    head, candidate = seed_history(path)
    second = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI(), reset_on_start=True)
    try:
        backup = path.with_name("queue-before-last-restart.sqlite3")
        assert backup.is_file()
        with sqlite3.connect(backup) as previous:
            row = previous.execute("SELECT id,text,state,reason FROM messages").fetchone()
            assert row == (head["id"], head["text"], "blocked", "result_unknown")
            assert dict(previous.execute("SELECT name,value FROM metrics")) == {"failures": 1, "accepted": 1}
        assert second.store.head() is None
        assert second.store.counts()["accepted"] == 0
        assert second.store.counts()["failures"] == 0
        assert second.content_filter.candidates()[0]["id"] == candidate["id"]
    finally:
        second.close()


def test_restart_does_not_reuse_old_message_ids_for_historical_candidates(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    head, candidate = seed_history(path)
    second = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI(), reset_on_start=True)
    try:
        second.configure(True, "1698225")
        second.begin_session(2)
        second._session = 1
        enqueue(second, "新的直播字幕与旧失败句独立", segment="new")
        second.timer.stop()
        assert second.store.head()["id"] > head["id"]
        assert second.content_filter.candidates()[0]["queue_id"] == head["id"]
        assert second.content_filter.candidates()[0]["id"] == candidate["id"]
    finally:
        second.close()


def test_historical_ai_review_saves_an_isolated_plan_without_reviving_queue_or_post(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    head, candidate = seed_history(path)
    browser, ai = FakeBrowser(), FakeAI()
    browser.is_browser_ready = False
    service = DanmakuService(browser, store_path=path, ai_analyzer=ai, reset_on_start=True)
    try:
        service.configure_ai(SETTINGS)
        service.review_timer.stop()
        service._review_history_tick()
        assert len(ai.calls) == 1
        assert ai.calls[0]["data"]["text"] == head["text"]
        assert service.store.head() is None
        ai.reply({"revised_text": TEXT.replace("评价", "感想")})
        plan = service.content_filter.validation_plan(candidate["id"])
        assert plan is not None and plan["state"] == "waiting_room"
        assert plan["edits"] == [{"term": "评价", "replacement": "感想"}]
        status = service.content_filter.candidates()[0]
        assert status["analysis_resolution"] == "ai_plan_saved"
        assert status["analysis_completed_at"]
        assert status["status"] == "waiting_validation"
        assert service.store.head() is None and service.store.counts()["queued"] == 0
        assert [rule["term"] for rule in service.content_filter.rules()] == ["旧规则"]
        for _ in range(5):
            service._review_history_tick()
            service.tick()
        assert len(ai.calls) == 1
        assert not browser.requests
    finally:
        service.close()


def test_completed_historical_analysis_and_plan_survive_next_restart_without_new_ai(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    head, candidate = seed_history(path)
    first_ai = FakeAI()
    first = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=first_ai, reset_on_start=True)
    first.configure_ai(SETTINGS)
    first.review_timer.stop()
    first._review_history_tick()
    first_ai.reply({"revised_text": TEXT.replace("评价", "感想")})
    expected = first.content_filter.validation_plan(candidate["id"])
    assert expected is not None
    first.close()
    browser, ai = FakeBrowser(), FakeAI()
    second = DanmakuService(browser, store_path=path, ai_analyzer=ai, reset_on_start=True)
    try:
        second.configure_ai(SETTINGS)
        second.review_timer.stop()
        second._review_history_tick()
        assert second.content_filter.validation_plan(candidate["id"]) == expected
        assert second.store.head() is None
        assert not ai.calls and not browser.requests
    finally:
        second.close()
