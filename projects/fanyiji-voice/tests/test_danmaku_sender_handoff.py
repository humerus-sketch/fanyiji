"""Two live sender instances share a real temporary DB, never a real browser."""
from types import SimpleNamespace

import pytest

from fanyiji.danmaku_filter import DanmakuContentFilter
from fanyiji.danmaku_service import DanmakuService
from test_danmaku_service import app, FakeBrowser


ROOM = "1698225"
TEXT = "这一句正在等待浏览器实际投递结果"
PACKET = "【" + TEXT + "】"
FOLLOWING = "【这是后面等待发送的另一句完整译文】"


@pytest.fixture
def senders(app, tmp_path):
    values = []
    now = [100.0]
    def create(*, shared=True):
        browser = FakeBrowser()
        path = tmp_path / ("shared.sqlite3" if shared else "separate.sqlite3")
        service = DanmakuService(browser, store_path=path, reset_on_start=False)
        service._clock = lambda: now[0]
        service.configure(True, ROOM)
        service.begin_session(len(values) + 1)
        service.timer.stop()
        service.review_timer.stop()
        service._login_room = ROOM
        value = SimpleNamespace(service=service, browser=browser)
        values.append(value)
        return value
    yield SimpleNamespace(create=create, values=values, now=now)
    for value in values:
        value.service.close()


def dispatch(value, *, following=False):
    assert value.service.store.enqueue(ROOM, "system", "first", 1, TEXT, [PACKET])
    if following:
        assert value.service.store.enqueue(ROOM, "system", "following", 1, FOLLOWING, [FOLLOWING])
    value.service.tick()
    assert value.browser.requests[-1]["operation"] == "send"
    assert value.service.store.head()["state"] == "sending"
    return value.service.store.head()


def disable(value):
    value.service.configure(False, ROOM)
    value.service.timer.stop()
    assert value.service._inflight is not None


def receipt(value):
    value.browser.reply(outcome="receipt_accepted", code=0, http_status=200)
    assert value.service.store.head()["state"] == "verify"


def confirmed(value):
    value.browser.reply(present=True, room_matches=True, chat_ready=True, occurrence_count=1)


def test_preexisting_second_sender_waits_for_live_sending_row_and_only_checks_after_receipt(senders):
    original, next_sender = senders.create(), senders.create()
    row = dispatch(original)
    disable(original)
    next_sender.service.tick()
    assert next_sender.browser.requests == []
    assert next_sender.service.store.head()["state"] == "sending"
    assert next_sender.service.store.head()["request_id"] == row["request_id"]
    assert next_sender.service.snapshot()["page_visible"] == 0
    receipt(original)
    next_sender.service.tick()
    assert next_sender.browser.requests[-1]["operation"] == "check"
    assert next_sender.service.snapshot()["page_visible"] == 0
    confirmed(next_sender)
    assert next_sender.service.store.head() is None
    assert next_sender.service.snapshot()["page_visible"] == 1
    assert sum(request["operation"] == "send" for value in senders.values for request in value.browser.requests) == 1


def test_late_created_sender_does_not_treat_live_constructor_restart_verify_as_an_orphan(senders):
    original = senders.create()
    row = dispatch(original)
    disable(original)
    next_sender = senders.create()
    assert next_sender.service.store.head()["state"] == "verify"
    assert next_sender.service.store.head()["reason"] == "restart_verify"
    next_sender.service.tick()
    assert next_sender.browser.requests == []
    assert next_sender.service.store.head()["request_id"] == row["request_id"]
    receipt(original)
    next_sender.service.tick()
    assert next_sender.browser.requests[-1]["operation"] == "check"
    confirmed(next_sender)
    assert next_sender.service.snapshot()["accepted"] == 1
    assert next_sender.service.snapshot()["page_visible"] == 1
    assert len(original.browser.requests) == 1


def test_foreign_pending_presence_check_is_not_advanced_by_another_sender(senders):
    original, next_sender = senders.create(), senders.create()
    dispatch(original)
    receipt(original)
    original.service.tick()
    assert original.browser.requests[-1]["operation"] == "check"
    disable(original)
    next_sender.service.tick()
    assert next_sender.browser.requests == []
    assert next_sender.service.snapshot()["page_visible"] == 0
    confirmed(original)
    next_sender.service.tick()
    assert next_sender.browser.requests == []
    assert next_sender.service.snapshot()["page_visible"] == 1


def test_unconfirmed_presence_never_becomes_success_or_a_duplicate_send(senders):
    original, next_sender = senders.create(), senders.create()
    dispatch(original)
    disable(original)
    receipt(original)
    next_sender.service.tick()
    assert next_sender.browser.requests[-1]["operation"] == "check"
    next_sender.browser.reply(present=False, room_matches=True, chat_ready=True, occurrence_count=0)
    assert next_sender.service.snapshot()["page_visible"] == 0
    assert next_sender.service.store.head()["state"] == "verify"
    senders.now[0] += 1.0
    next_sender.service.tick()
    assert all(request["operation"] == "check" for request in next_sender.browser.requests)


def test_handoff_keeps_minimum_send_interval_for_following_message(senders):
    original, next_sender = senders.create(), senders.create()
    dispatch(original, following=True)
    disable(original)
    receipt(original)
    next_sender.service.tick()
    confirmed(next_sender)
    next_sender.service.tick()
    assert next_sender.service.snapshot()["state"] == "cooldown"
    assert sum(request["operation"] == "send" for request in next_sender.browser.requests) == 0
    senders.now[0] += 3.49
    next_sender.service.tick()
    assert next_sender.browser.requests[-1]["operation"] == "check"
    senders.now[0] += 0.01
    next_sender.service.tick()
    assert next_sender.browser.requests[-1]["operation"] == "send"
    assert next_sender.browser.requests[-1]["text"] == FOLLOWING


def test_old_timeout_records_failure_before_new_sender_moves_to_next_fifo_item(senders):
    original, next_sender = senders.create(), senders.create()
    dispatch(original, following=True)
    disable(original)
    next_sender.service.tick()
    assert next_sender.browser.requests == []
    senders.now[0] += original.service._request_timeout + 0.1
    original.service.tick()
    restored_filter = DanmakuContentFilter(original.service.filter_path)
    assert restored_filter.candidates()[0]["text"] == PACKET
    assert next_sender.service.store.head()["text"] == FOLLOWING
    assert next_sender.service.snapshot()["page_visible"] == 0
    next_sender.service.tick()
    assert next_sender.browser.requests == []
    senders.now[0] += 3.5
    next_sender.service.tick()
    assert next_sender.browser.requests[-1]["operation"] == "send"
    assert next_sender.browser.requests[-1]["text"] == FOLLOWING


def test_unknown_old_callback_cannot_be_treated_as_delivery_success(senders):
    original, next_sender = senders.create(), senders.create()
    dispatch(original, following=True)
    disable(original)
    original.browser.reply(outcome="result_unknown", error="request_timeout", code=None)
    assert next_sender.service.snapshot()["page_visible"] == 0
    assert next_sender.service.store.head()["text"] == FOLLOWING
    persisted = DanmakuContentFilter(original.service.filter_path).candidates()
    assert persisted[0]["text"] == PACKET
    next_sender.service.tick()
    assert next_sender.browser.requests == []


def test_real_restart_without_live_owner_only_verifies_and_never_resends(senders):
    original = senders.create()
    dispatch(original)
    original.service.close()
    restarted = senders.create()
    assert restarted.service.store.head()["state"] == "verify"
    restarted.service.tick()
    assert restarted.browser.requests[-1]["operation"] == "check"
    assert restarted.service.snapshot()["page_visible"] == 0


def test_unowned_sending_row_is_waited_without_claiming_it_was_delivered(senders):
    value = senders.create()
    assert value.service.store.enqueue(ROOM, "system", "orphan", 1, TEXT, [PACKET])
    row = value.service.store.head()
    value.service.store.update(row["id"], "sending", request_id="unowned-request")
    value.service.tick()
    assert value.browser.requests == []
    assert value.service.store.head()["state"] == "sending"
    assert value.service.snapshot()["page_visible"] == 0


def test_distinct_databases_do_not_block_each_other(senders):
    original, independent = senders.create(), senders.create(shared=False)
    dispatch(original)
    dispatch(independent)
    assert len(original.browser.requests) == len(independent.browser.requests) == 1
