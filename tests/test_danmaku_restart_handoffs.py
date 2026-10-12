"""Restart recovery uses temporary stores and fake browser/AI only."""
import sqlite3

import pytest

from fanyiji.danmaku_filter import DanmakuContentFilter
from fanyiji.danmaku_service import DanmakuService
from fanyiji.danmaku_store import DanmakuStore
from test_danmaku_auto_recovery import FakeAI, SETTINGS
from test_danmaku_service import app, FakeBrowser, enqueue


ROOM = "21399689"
TEXT = "感谢你的评价我会继续努力"
PACKET = "【" + TEXT + "】"


def cannot_save(*args, **kwargs):
    raise OSError("temporary failure containing private fixture text")


def marked_failure(store, *, text=PACKET, key="failed", marker=True, state="verify"):
    assert store.enqueue(ROOM, "system", key, 1, TEXT, [text])
    row = store.connection.execute("SELECT * FROM messages WHERE segment_key=?", (key,)).fetchone()
    store.update(row["id"], state, request_id="old-request", reason="platform_f", platform_code=0)
    if marker is not None:
        store.save_recovery_context(row["id"], row["part"], {"handoff_pending": marker})
    return dict(store.connection.execute("SELECT * FROM messages WHERE id=?", (row["id"],)).fetchone())


def unwritten_live_failure(path, monkeypatch):
    browser = FakeBrowser()
    service = DanmakuService(browser, store_path=path)
    now = [100.0]
    service._clock = lambda: now[0]
    service.configure(True, ROOM, interval_seconds=5.0)
    service.timer.stop()
    service.begin_session(1)
    rule = service.content_filter.add_rule("旧规则", "保留规则", ROOM)
    try:
        enqueue(service, TEXT)
        service.tick()
        browser.reply(is_login=True, code=0, error="none", max_weighted_length=80)
        now[0] += 5.0
        service.tick()
        assert browser.requests[-1]["operation"] == "send"
        head = service.store.head()
        with monkeypatch.context() as patcher:
            patcher.setattr(service.content_filter, "record_failure", cannot_save)
            browser.reply(outcome="platform_f", code=0, http_status=200, error="none")
        assert service.store.head()["text"] == PACKET
        assert service.store.recovery_context(head["id"], head["part"])["handoff_pending"] is True
        assert service.content_filter.candidates() == []
        return head, rule
    finally:
        service.close()


def assert_runtime_empty(store):
    assert store.head() is None
    assert all(value == 0 for value in store.counts().values())
    for table in ("messages", "contents", "segments", "metrics"):
        assert store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_unwritten_failure_survives_restart_and_enters_ai_review(app, tmp_path, monkeypatch):
    path = tmp_path / "queue.sqlite3"
    head, rule = unwritten_live_failure(path, monkeypatch)
    browser, ai = FakeBrowser(), FakeAI()
    service = DanmakuService(browser, store_path=path, ai_analyzer=ai, reset_on_start=True)
    try:
        assert_runtime_empty(service.store)
        assert service.store.failed_handoffs() == []
        candidate = service.content_filter.candidates()[0]
        assert (candidate["room"], candidate["text"], candidate["queue_id"], candidate["count"]) == (
            ROOM, PACKET, head["id"], 1)
        assert candidate["status"] == "active"
        assert candidate["diagnosis"]["action"] == "transfer_failure"
        assert service.content_filter.rules() == [rule]
        service.configure_ai(SETTINGS)
        service.review_timer.stop()
        service._review_history_tick()
        assert len(ai.calls) == 1 and ai.calls[0]["data"]["text"] == PACKET
        ai.reply({"revised_text": TEXT.replace("评价", "感想")})
        plan = service.content_filter.validation_plan(candidate["id"])
        assert plan["revised_text"] == TEXT.replace("评价", "感想")
        assert plan["state"] == "waiting_room"
        assert service.content_filter.rules() == [rule]
        assert_runtime_empty(service.store)
        assert browser.requests == []
    finally:
        service.close()


def test_continuous_restart_keeps_staging_until_filter_can_save(app, tmp_path, monkeypatch):
    path = tmp_path / "queue.sqlite3"
    head, rule = unwritten_live_failure(path, monkeypatch)
    profile = tmp_path / "browser-user-data" / "Default" / "login-state.fixture"
    profile.parent.mkdir(parents=True)
    profile.write_bytes(b"preserved-profile-fixture")
    with monkeypatch.context() as patcher:
        patcher.setattr(DanmakuContentFilter, "record_failure", cannot_save)
        for _ in range(3):
            browser, ai = FakeBrowser(), FakeAI()
            service = DanmakuService(browser, store_path=path, ai_analyzer=ai, reset_on_start=True)
            try:
                assert_runtime_empty(service.store)
                pending = service.store.failed_handoffs()
                assert len(pending) == 1
                assert (pending[0]["room"], pending[0]["text"], pending[0]["queue_id"]) == (ROOM, PACKET, head["id"])
                assert service.content_filter.candidates() == []
                assert service.content_filter.rules() == [rule]
                service.configure_ai(SETTINGS)
                service.review_timer.stop()
                service._review_history_tick()
                assert service.snapshot()["filter_candidate_error"]
                assert not browser.requests and not ai.calls
                service.store.clear_pending()
                assert service.store.failed_handoffs() == pending
            finally:
                service.close()
    browser, ai = FakeBrowser(), FakeAI()
    service = DanmakuService(browser, store_path=path, ai_analyzer=ai, reset_on_start=True)
    try:
        assert_runtime_empty(service.store)
        assert not service.store.failed_handoffs()
        assert service.content_filter.candidates()[0]["count"] == 1
        assert service.content_filter.rules() == [rule]
        assert profile.read_bytes() == b"preserved-profile-fixture"
        service.configure_ai(SETTINGS)
        service.review_timer.stop()
        service._review_history_tick()
        assert len(ai.calls) == 1 and not browser.requests
        assert "private fixture text" not in service.log_path.read_text(encoding="utf-8")
    finally:
        service.close()


def test_history_tick_restores_after_temporary_write_failure_without_restart(app, tmp_path, monkeypatch):
    path = tmp_path / "queue.sqlite3"
    unwritten_live_failure(path, monkeypatch)
    with monkeypatch.context() as patcher:
        patcher.setattr(DanmakuContentFilter, "record_failure", cannot_save)
        service = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI(), reset_on_start=True)
        service.configure_ai(SETTINGS)
        service.review_timer.stop()
        service._review_history_tick()
        assert len(service.store.failed_handoffs()) == 1
    try:
        service._review_history_tick()
        assert service.store.failed_handoffs() == []
        assert service.content_filter.candidates()[0]["count"] == 1
        assert not service.snapshot()["filter_candidate_error"]
        assert len(service._ai_analyzer.calls) == 1
        assert service.browser.requests == []
        assert_runtime_empty(service.store)
    finally:
        service.close()


def test_atomic_filter_save_failure_keeps_staging_and_defers_ai(app, tmp_path, monkeypatch):
    path = tmp_path / "queue.sqlite3"
    unwritten_live_failure(path, monkeypatch)
    with monkeypatch.context() as patcher:
        patcher.setattr(DanmakuContentFilter, "_save", cannot_save)
        service = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI(), reset_on_start=True)
        service.configure_ai(SETTINGS)
        service.review_timer.stop()
        for _ in range(2):
            service._review_history_tick()
            assert len(service.store.failed_handoffs()) == 1
            assert service.content_filter.candidates() == []
            assert not service._ai_analyzer.calls and not service.browser.requests
            assert_runtime_empty(service.store)
    try:
        service._review_history_tick()
        assert not service.store.failed_handoffs()
        assert service.content_filter.candidates()[0]["count"] == 1
        assert len(service._ai_analyzer.calls) == 1
        assert DanmakuContentFilter(service.filter_path).candidates()[0]["count"] == 1
    finally:
        service.close()


def test_in_memory_match_is_explicitly_saved_before_staging_ack(app, tmp_path, monkeypatch):
    path = tmp_path / "queue.sqlite3"
    head, _ = unwritten_live_failure(path, monkeypatch)
    with monkeypatch.context() as patcher:
        patcher.setattr(DanmakuContentFilter, "record_failure", cannot_save)
        service = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI(), reset_on_start=True)
    # Simulate a future writer exposing a candidate before persistence. The
    # handoff must not rely on that in-memory identity to delete its last copy.
    isolated = DanmakuContentFilter(tmp_path / "isolated-candidate.json")
    candidate = isolated.record_failure(ROOM, PACKET, "platform_f", queue_id=head["id"])
    service.content_filter._state["candidates"] = [candidate]
    service.configure_ai(SETTINGS)
    service.review_timer.stop()
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(service.content_filter, "_save", cannot_save)
            for _ in range(2):
                service._review_history_tick()
                assert len(service.store.failed_handoffs()) == 1
                assert service.content_filter.candidates()[0]["count"] == 1
                assert DanmakuContentFilter(service.filter_path).candidates() == []
                assert not service._ai_analyzer.calls and not service.browser.requests
        service._review_history_tick()
        assert not service.store.failed_handoffs()
        persisted = DanmakuContentFilter(service.filter_path).candidates()[0]
        assert persisted["id"] == candidate["id"] and persisted["count"] == 1
        assert len(service._ai_analyzer.calls) == 1
        assert_runtime_empty(service.store)
    finally:
        service.close()


def test_candidate_saved_before_ack_failure_is_not_counted_again(app, tmp_path, monkeypatch):
    path = tmp_path / "queue.sqlite3"
    head, _ = unwritten_live_failure(path, monkeypatch)
    with monkeypatch.context() as patcher:
        patcher.setattr(DanmakuStore, "acknowledge_failed_handoff", cannot_save)
        service = DanmakuService(FakeBrowser(), store_path=path, reset_on_start=True)
        try:
            assert len(service.store.failed_handoffs()) == 1
            candidate = service.content_filter.candidates()[0]
            assert candidate["queue_id"] == head["id"] and candidate["count"] == 1
            service._review_history_tick()
            assert service.content_filter.candidates()[0]["count"] == 1
        finally:
            service.close()
    service = DanmakuService(FakeBrowser(), store_path=path, reset_on_start=True)
    try:
        assert service.content_filter.candidates()[0]["id"] == candidate["id"]
        assert service.content_filter.candidates()[0]["count"] == 1
        assert service.store.failed_handoffs() == []
        assert_runtime_empty(service.store)
        assert service.browser.requests == []
    finally:
        service.close()


def test_exact_existing_resolved_candidate_is_not_reopened(app, tmp_path):
    path = tmp_path / "queue.sqlite3"
    store = DanmakuStore(path)
    head = marked_failure(store)
    store.close()
    content_filter = DanmakuContentFilter(tmp_path / "danmaku-content-filter.json")
    candidate = content_filter.record_failure(ROOM, PACKET, "platform_f", queue_id=head["id"])
    resolved = content_filter.set_candidate_status(candidate["id"], "resolved", resolution="page_observed")
    service = DanmakuService(FakeBrowser(), store_path=path, ai_analyzer=FakeAI(), reset_on_start=True)
    try:
        assert service.content_filter.candidates() == [resolved]
        service.configure_ai(SETTINGS)
        service.review_timer.stop()
        service._review_history_tick()
        assert not service._ai_analyzer.calls and not service.browser.requests
        assert not service.store.failed_handoffs()
    finally:
        service.close()


@pytest.mark.parametrize("marker", [None, False, 1, "true", {}, []])
@pytest.mark.parametrize("state", ["pending", "sending", "verify", "blocked"])
def test_normal_or_unproven_rows_are_not_staged(tmp_path, marker, state):
    store = DanmakuStore(tmp_path / "queue.sqlite3")
    try:
        marked_failure(store, marker=marker, state=state)
        store.reset_for_restart()
        assert_runtime_empty(store)
        assert not store.failed_handoffs()
    finally:
        store.close()


@pytest.mark.parametrize("target,operation", [("messages", "DELETE"), ("failed_handoffs", "INSERT")])
def test_reset_transaction_failure_preserves_runtime_and_does_not_half_stage(tmp_path, target, operation):
    store = DanmakuStore(tmp_path / "queue.sqlite3")
    try:
        head = marked_failure(store)
        store.connection.execute(f"CREATE TEMP TRIGGER deny_runtime_clear BEFORE {operation} ON {target} "
                                 "BEGIN SELECT RAISE(ABORT,'temporary reset failure'); END")
        with pytest.raises(sqlite3.IntegrityError, match="temporary reset failure"):
            store.reset_for_restart()
        assert store.head()["id"] == head["id"]
        assert store.recovery_context(head["id"], head["part"])["handoff_pending"] is True
        assert not store.failed_handoffs()
        assert not store.connection.in_transaction
        store.connection.execute("DROP TRIGGER deny_runtime_clear")
        store.reset_for_restart()
        assert len(store.failed_handoffs()) == 1
        assert_runtime_empty(store)
    finally:
        store.close()


def test_reset_rejects_outer_transaction_without_committing_it(tmp_path):
    store = DanmakuStore(tmp_path / "queue.sqlite3")
    try:
        head = marked_failure(store)
        store.connection.execute("BEGIN")
        store._set_metadata("caller_owned", "uncommitted")
        with pytest.raises(ValueError, match="restart_reset_requires_independent_transaction"):
            store.reset_for_restart()
        assert store.connection.in_transaction
        store.connection.rollback()
        assert store._metadata("caller_owned") is None
        assert store.head()["id"] == head["id"]
        assert not store.failed_handoffs()
    finally:
        store.close()


def test_staging_identity_and_ids_survive_reset_without_replaying_parts(tmp_path):
    path = tmp_path / "queue.sqlite3"
    store = DanmakuStore(path)
    try:
        head = marked_failure(store)
        with store.connection:
            store._set_metadata(f"remaining_parts:{head['id']}", '["【尚未发送的后续原文包】"]')
            store.connection.execute("INSERT INTO metrics VALUES('failures',5)")
        previous = store.reset_for_restart()
        assert previous["queued"] == 2 and previous["failures"] == 5
        assert len(store.failed_handoffs()) == 1
        assert store.failed_handoffs()[0]["text"] == PACKET
        assert_runtime_empty(store)
        pending = store.failed_handoffs()
        store.reset_for_restart()
        assert store.failed_handoffs() == pending
        assert not store.acknowledge_failed_handoff("1698225", PACKET, head["id"])
        assert not store.acknowledge_failed_handoff(ROOM, "其他正文", head["id"])
        assert store.failed_handoffs() == pending
        assert store.enqueue(ROOM, "system", "new-run", 1, "下一轮正常内容", ["【下一轮正常内容依次等待发送】"])
        assert store.head()["id"] > head["id"]
        assert store.acknowledge_failed_handoff(ROOM, PACKET, head["id"])
        assert not store.failed_handoffs()
    finally:
        store.close()
