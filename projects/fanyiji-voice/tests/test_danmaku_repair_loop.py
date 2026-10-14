"""Opt-in repair loops use isolated fake AI/browser and temporary stores only."""
import pytest

from fanyiji import danmaku_filter
from fanyiji.danmaku_filter import DanmakuContentFilter
from fanyiji.danmaku_service import DanmakuService
from test_danmaku_service import app, FakeBrowser, enqueue
from test_danmaku_auto_recovery import FakeAI, SETTINGS, start, f_reply


FORMAL_ROOM = "21399689"
TEST_ROOM = "1698225"
BODY = "感谢你的评价我会继续努力"
ORIGINAL = "【" + BODY + "】"


@pytest.fixture
def repair_loop(app, tmp_path):
    browser, ai, now = FakeBrowser(), FakeAI(), [100.0]
    service = DanmakuService(browser, store_path=tmp_path / "queue.sqlite3", ai_analyzer=ai)
    service._clock = lambda: now[0]
    service.configure(True, FORMAL_ROOM, interval_seconds=3.5,
                      validation_room=TEST_ROOM, repair_enabled=True)
    service.configure_ai(SETTINGS)
    service.timer.stop()
    service.review_timer.stop()
    service.begin_session(1)
    candidate = service.content_filter.record_failure(FORMAL_ROOM, ORIGINAL, "platform_f", queue_id=44)
    yield service, browser, ai, now, candidate["id"]
    service.close()


def posts(browser, room=None):
    return [item for item in browser.requests if item["operation"] == "send"
            and (room is None or item["room"] == room)]


def next_ai(service, ai, now, expected):
    for _ in range(50):
        if len(ai.calls) >= expected:
            assert len(ai.calls) == expected and ai.busy
            return ai.calls[-1]
        service._review_history_tick()
        now[0] += 5.0
    pytest.fail("repair did not request the next AI round")


def correction(ai, replacement):
    ai.reply({"revised_text": BODY.replace("评价", replacement)})


def choose_focus(monkeypatch, term):
    from fanyiji import danmaku_repair
    original_context = danmaku_repair.repair_context
    selected = [term]

    def context(text, round_number, failed_plans, dictionary_hits=()):
        result = original_context(text, round_number, failed_plans, dictionary_hits)
        result["focus_term"] = selected[0]
        return result

    monkeypatch.setattr(danmaku_repair, "repair_context", context)
    return selected


def send_correction(service, browser, now):
    before = len(posts(browser, TEST_ROOM))
    for _ in range(20):
        service._review_history_tick()
        flight = service._validator._flight if service._validator else None
        if flight is not None:
            if flight["operation"] == "login":
                browser.reply(is_login=True, code=0, error="none", max_weighted_length=80)
            elif flight["operation"] == "send":
                assert len(posts(browser, TEST_ROOM)) == before + 1
                assert browser.requests[-1]["room"] == TEST_ROOM
                return browser.requests[-1]
            else:
                pytest.fail("a new correction unexpectedly reached a prior read-only check")
        now[0] += 3.5
    pytest.fail("correction never reached an isolated send")


def reject_correction(service, browser, candidate_id):
    browser.reply(outcome="platform_f", code=0, http_status=200, error="none", before_count=0)
    assert service.content_filter.validation_plan(candidate_id)["state"] == "failed"
    assert not service.content_filter.rules()


def page_check(service, browser, now, count):
    for _ in range(12):
        now[0] += 3.5
        service._review_history_tick()
        flight = service._validator._flight
        if flight is not None:
            assert flight["operation"] == "check"
            browser.reply(present=count > 0, room_matches=True, chat_ready=True,
                          error="none", occurrence_count=count)
            return
    pytest.fail("accepted test did not reach page verification")


def test_four_rejections_continue_fifth_round_and_learn_only_global_word_rule(repair_loop, monkeypatch):
    choose_focus(monkeypatch, "评价")
    service, browser, ai, now, candidate_id = repair_loop
    replacements = ("感想", "看法", "意见", "观点", "反馈")
    rejected = []
    for round_number, replacement in enumerate(replacements, 1):
        call = next_ai(service, ai, now, round_number)
        context = call["data"]["repair_context"]
        assert context["round"] >= round_number
        assert context["strategy"] == "word_by_word"
        assert [item["revised_text"] for item in context["failed_revisions"]] == rejected
        correction(ai, replacement)
        request = send_correction(service, browser, now)
        assert request["text"] == "【" + BODY.replace("评价", replacement) + "】"
        if round_number < 5:
            reject_correction(service, browser, candidate_id)
            rejected.append(BODY.replace("评价", replacement))
    browser.reply(outcome="receipt_accepted", code=0, http_status=200, error="none", before_count=2)
    assert not service.content_filter.rules()
    page_check(service, browser, now, 2)
    assert not service.content_filter.rules()
    page_check(service, browser, now, 3)
    rules = service.content_filter.rules()
    assert [(rule["term"], rule["replacement"], rule["room"]) for rule in rules] == [("评价", "反馈", "")]
    assert rules[0]["enabled"] and rules[0]["validation"]["receipt_accepted"]
    assert rules[0]["evidence"]["confirmed_cause"] is False
    assert service.content_filter.candidates()[0]["status"] == "resolved"
    assert len(posts(browser, TEST_ROOM)) == 5 and not posts(browser, FORMAL_ROOM)
    assert len(service.content_filter.candidates()[0]["repair_history"]) == 4
    assert DanmakuContentFilter(service.filter_path).rules() == rules


def test_repeated_current_and_older_failed_text_never_create_new_post(repair_loop, monkeypatch):
    choose_focus(monkeypatch, "评价")
    service, browser, ai, now, candidate_id = repair_loop
    next_ai(service, ai, now, 1)
    correction(ai, "感想")
    send_correction(service, browser, now)
    reject_correction(service, browser, candidate_id)
    failed = service.content_filter.validation_plan(candidate_id)
    next_ai(service, ai, now, 2)
    correction(ai, "感想")
    assert service.content_filter.validation_plan(candidate_id) == failed
    assert len(posts(browser)) == 1
    next_ai(service, ai, now, 3)
    correction(ai, "看法")
    send_correction(service, browser, now)
    reject_correction(service, browser, candidate_id)
    current = service.content_filter.validation_plan(candidate_id)
    next_ai(service, ai, now, 4)
    correction(ai, "感想")
    assert service.content_filter.validation_plan(candidate_id) == current
    assert len(posts(browser)) == 2
    next_ai(service, ai, now, 5)
    correction(ai, "意见")
    send_correction(service, browser, now)
    assert len(posts(browser, TEST_ROOM)) == 3
    assert not service.content_filter.rules()


def test_unchanged_ai_reply_waits_and_rotates_focus_before_next_round(repair_loop):
    service, browser, ai, now, candidate_id = repair_loop
    first = next_ai(service, ai, now, 1)
    ai.reply({"revised_text": BODY})
    for _ in range(5):
        service._review_history_tick()
    assert len(ai.calls) == 1 and not browser.requests
    second = next_ai(service, ai, now, 2)
    assert first["data"]["repair_context"]["focus_term"] != second["data"]["repair_context"]["focus_term"]
    assert service.content_filter.validation_plan(candidate_id) is None
    assert not service.content_filter.rules()


def test_ai_edit_outside_focus_is_rejected_then_correct_focus_can_continue(repair_loop, monkeypatch):
    selected = choose_focus(monkeypatch, "继续")
    service, browser, ai, now, candidate_id = repair_loop
    first = next_ai(service, ai, now, 1)
    assert first["data"]["repair_context"]["focus_term"] == "继续"
    correction(ai, "感想")
    assert service.content_filter.validation_plan(candidate_id) is None
    assert not browser.requests and not service.content_filter.rules()
    for _ in range(5):
        service._review_history_tick()
    assert len(ai.calls) == 1
    selected[0] = "评价"
    second = next_ai(service, ai, now, 2)
    assert second["data"]["repair_context"]["focus_term"] == "评价"
    assert second["data"]["repair_context"]["round"] > first["data"]["repair_context"]["round"]
    correction(ai, "感想")
    plan = service.content_filter.validation_plan(candidate_id)
    assert plan is not None and plan["edits"] == [{"term": "评价", "replacement": "感想"}]
    send_correction(service, browser, now)
    assert len(posts(browser, TEST_ROOM)) == 1
    assert not service.content_filter.rules()


def test_disable_cancels_ai_and_ignores_late_answer_without_losing_records(repair_loop):
    service, browser, ai, now, candidate_id = repair_loop
    rule = service.content_filter.add_rule("旧词", "旧表达")
    call = next_ai(service, ai, now, 1)
    original = service.content_filter.candidates()[0]["text"]
    service.set_repair_enabled(False)
    assert call["token"] in ai.cancelled
    ai.reply({"revised_text": BODY.replace("评价", "感想")}, token=call["token"])
    for _ in range(5):
        now[0] += 30.0
        service._review_history_tick()
    assert len(ai.calls) == 1 and not browser.requests
    assert service.content_filter.validation_plan(candidate_id) is None
    assert service.content_filter.candidates()[0]["text"] == original
    assert service.content_filter.rules() == [rule]
    assert not service.snapshot()["repair_enabled"]


def test_disable_stops_new_validation_and_ignores_late_page_success(repair_loop, monkeypatch):
    choose_focus(monkeypatch, "评价")
    service, browser, ai, now, candidate_id = repair_loop
    next_ai(service, ai, now, 1)
    correction(ai, "感想")
    send_correction(service, browser, now)
    browser.reply(outcome="receipt_accepted", code=0, http_status=200, error="none", before_count=0)
    now[0] += 3.5
    service._review_history_tick()
    request = browser.requests[-1]
    assert request["operation"] == "check"
    service.set_repair_enabled(False)
    count = len(browser.requests)
    browser.reply(present=True, room_matches=True, chat_ready=True, error="none", occurrence_count=1)
    for _ in range(5):
        now[0] += 30.0
        service._review_history_tick()
    assert len(browser.requests) == count and len(ai.calls) == 1
    assert service.content_filter.validation_plan(candidate_id)["state"] == "testing"
    assert not service.content_filter.rules()


def test_default_disabled_still_transfers_failure_and_sends_next_formal_subtitle(app, tmp_path):
    browser, ai, now = FakeBrowser(), FakeAI(), [100.0]
    service = DanmakuService(browser, store_path=tmp_path / "queue.sqlite3", ai_analyzer=ai)
    service._clock = lambda: now[0]
    service.configure(True, FORMAL_ROOM, interval_seconds=3.5, validation_room=TEST_ROOM)
    service.configure_ai(SETTINGS)
    service.timer.stop()
    service.review_timer.stop()
    service.begin_session(1)
    try:
        assert service.repair_enabled is False
        first = start(service, browser, now, BODY)
        enqueue(service, "下一句完整中文字幕继续发送", segment="second")
        f_reply(browser)
        assert service.store.head()["text"] != first["text"]
        for _ in range(3):
            service._review_history_tick()
        assert not ai.calls and service.content_filter.candidates()[0]["text"] == first["text"]
        now[0] += 3.5
        service.tick()
        assert browser.requests[-1]["operation"] == "send"
        assert browser.requests[-1]["text"] == "【下一句完整中文字幕继续发送】"
        assert len(posts(browser, FORMAL_ROOM)) == 2 and not posts(browser, TEST_ROOM)
        assert not service.content_filter.rules()
    finally:
        service.close()


def test_disk_recovery_reuses_validated_ai_answer_without_another_request(repair_loop, monkeypatch):
    choose_focus(monkeypatch, "评价")
    service, browser, ai, now, candidate_id = repair_loop
    next_ai(service, ai, now, 1)
    before = service.content_filter.candidates()[0]
    with monkeypatch.context() as patcher:
        def cannot_replace(*args):
            raise OSError("simulated disk failure")
        patcher.setattr(danmaku_filter.os, "replace", cannot_replace)
        correction(ai, "感想")
        assert service.content_filter.validation_plan(candidate_id) is None
        assert service._pending_ai_result is not None
        assert not browser.requests and len(ai.calls) == 1
    for _ in range(20):
        now[0] += 5.0
        service._review_history_tick()
        if service.content_filter.validation_plan(candidate_id) is not None:
            break
    plan = service.content_filter.validation_plan(candidate_id)
    assert plan is not None and plan["revised_text"] == BODY.replace("评价", "感想")
    assert service._pending_ai_result is None
    assert len(ai.calls) == 1
    assert service.content_filter.candidates()[0]["count"] == before["count"]
    assert not service.content_filter.rules()
    assert DanmakuContentFilter(service.filter_path).validation_plan(candidate_id) == plan
