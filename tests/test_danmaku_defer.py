"""Failed packets are handed to analysis without live browser requests."""
import json

import pytest

from fanyiji.danmaku_filter import DanmakuContentFilter
from fanyiji.danmaku_store import DanmakuStore


ROOM = "1698225"
FAILED = "【当前失败正文必须完整保留】"
TAIL = "【同一句后续内容仍然完整保留】"
FOLLOWING = "【下一句正常内容继续按顺序发送】"
SIGNATURE = "a" * 64


@pytest.fixture
def store(tmp_path):
    result = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: 1000.0)
    yield result
    result.close()


def enqueue(store, key, parts):
    assert store.enqueue(ROOM, "system", key, 1, "原始字幕保留", parts,
                         filter_revision="original", filter_rule_signatures=[SIGNATURE])
    return store.head()


def failed_head(store, *, remaining=(), state="verify", reason="platform_f", code=0):
    head = enqueue(store, "failed", [FAILED])
    if remaining:
        assert store.prepare_parts(head["id"], [FAILED, *remaining])
    store.set_filter_frozen(head["id"], True)
    store.begin_f_retry(head["id"], "failed-request")
    store.update(head["id"], state, request_id="failed-request", reason=reason, platform_code=code)
    store.set_delivery_baseline(head["id"], 4)
    store.activate_f_recovery(head["id"], new_f=True)
    return store.head()


def snapshot(store):
    return {table: [tuple(row) for row in store.connection.execute(f"SELECT * FROM {table} ORDER BY 1,2")]
            for table in ("messages", "metadata", "metrics", "contents", "segments", "sqlite_sequence")}


def test_failed_packet_moves_to_candidate_before_fifo_is_advanced(store, tmp_path):
    expected = failed_head(store)
    enqueue(store, "following", [FOLLOWING])
    content_filter = DanmakuContentFilter(tmp_path / "filter.json")
    candidate = content_filter.record_failure(expected["room"], expected["text"], expected["reason"],
                                             queue_id=expected["id"])
    assert store.defer_failed_head(expected)
    assert store.head()["text"] == FOLLOWING
    reloaded = DanmakuContentFilter(content_filter.path)
    saved = next(item for item in reloaded.candidates() if item["id"] == candidate["id"])
    assert saved["text"] == FAILED
    assert saved["queue_id"] == expected["id"]
    archived = dict(store.connection.execute("SELECT * FROM messages WHERE id=?", (expected["id"],)).fetchone())
    assert archived["state"] == "cancelled"
    assert archived["reason"] == "deferred_analysis"
    assert archived["request_id"] == ""
    assert store.counts()["deferred_analysis"] == 1
    assert store.counts()["page_visible"] == 0
    assert store.counts()["accepted"] == 0
    assert store.counts()["failures"] == 0
    assert store.counts()["queued"] == 1
    assert store.delivery_baseline(expected["id"]) is None
    assert store.delivery_code(expected["id"]) is None
    assert store.f_recovery(expected["id"]) == {"active": False, "retries": 0, "last_f_at": None}
    assert reloaded.rules() == []


def test_continuations_keep_original_fifo_id_and_full_body(store):
    expected = failed_head(store, remaining=[TAIL, FOLLOWING])
    enqueue(store, "later", ["【更加后面的完整内容不会插队】"])
    previous = store.counts()
    context = store.filter_context(expected["id"])
    assert store.defer_failed_head(expected)
    current = store.head()
    assert current["id"] == expected["id"]
    assert current["part"] == expected["part"] + 1
    assert current["text"] == TAIL
    assert current["state"] == "pending"
    assert current["request_id"] == current["reason"] == ""
    assert store.strict_remaining_parts(current["id"]) == [FOLLOWING]
    assert store.filter_context(current["id"]) == {**context, "frozen": True}
    assert store.applied_rule_signatures(current["id"]) == [SIGNATURE]
    assert store.delivery_code(current["id"]) is None
    assert store.delivery_baseline(current["id"]) is None
    assert store.f_recovery(current["id"])["active"] is False
    assert store.counts()["queued"] == previous["queued"] - 1
    assert store.counts()["page_visible"] == 0
    archived = store.connection.execute("SELECT * FROM messages WHERE reason='deferred_analysis'").fetchall()
    assert len(archived) == 1
    assert archived[0]["text"] == FAILED
    assert archived[0]["part"] == expected["part"]
    assert archived[0]["state"] == "cancelled"
    assert not store.defer_failed_head(expected)
    assert store.counts()["deferred_analysis"] == 1


def test_short_continuation_keeps_holding_without_releasing_it(store):
    expected = failed_head(store)
    # A historical queue can already have a short continuation after the
    # current packet was submitted. Deferring must preserve it literally.
    with store.connection:
        store._set_metadata(f"remaining_parts:{expected['id']}", json.dumps(["【短句】", TAIL], ensure_ascii=False))
    assert store.defer_failed_head(expected)
    assert store.head()["text"] == "【短句】"
    assert store.head()["state"] == "holding_short"
    assert store.strict_remaining_parts(expected["id"]) == [TAIL]


@pytest.mark.parametrize("field,value", [
    ("id", -1), ("id", True), ("room", "21399689"), ("text", FOLLOWING),
    ("part", 1), ("part", False), ("request_id", "stale-request"),
    ("state", "blocked"), ("reason", "result_unknown"),
    ("platform_code", 1), ("platform_code", False), ("platform_code", "0"),
])
def test_stale_or_invalid_snapshot_never_moves_queue(store, field, value):
    expected = failed_head(store, remaining=[TAIL])
    expected[field] = value
    before = snapshot(store)
    assert not store.defer_failed_head(expected)
    assert snapshot(store) == before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("field", ["id", "room", "text", "part", "request_id", "state", "reason", "platform_code"])
def test_missing_identity_is_rejected_without_mutation(store, field):
    expected = failed_head(store)
    expected.pop(field)
    before = snapshot(store)
    assert not store.defer_failed_head(expected)
    assert snapshot(store) == before


@pytest.mark.parametrize("state", ["pending", "holding_short", "sending", "retry_pending", "page_visible", "cancelled"])
def test_not_a_finished_failure_cannot_be_deferred(store, state):
    expected = failed_head(store)
    store.update(expected["id"], state, request_id=expected["request_id"], reason="platform_f")
    expected["state"] = state
    before = snapshot(store)
    assert not store.defer_failed_head(expected)
    assert snapshot(store) == before


@pytest.mark.parametrize("state,reason,code", [
    ("blocked", "platform_rejected", -403),
    ("verify", "result_unknown", None),
    ("verify", "receipt_unconfirmed", 0),
])
def test_known_saved_candidate_can_be_moved_without_assuming_cause(store, state, reason, code):
    expected = failed_head(store, state=state, reason=reason, code=code)
    assert store.defer_failed_head(expected)
    assert store.head() is None
    assert store.counts()["deferred_analysis"] == 1
    assert store.counts()["page_visible"] == 0


@pytest.mark.parametrize("value", ["{", "{}", '[null]', '["unwrapped"]'])
def test_corrupt_continuations_roll_back_without_losing_head(store, value):
    expected = failed_head(store)
    with store.connection:
        store._set_metadata(f"remaining_parts:{expected['id']}", value)
    before = snapshot(store)
    with pytest.raises(ValueError, match="invalid_remaining_parts"):
        store.defer_failed_head(expected)
    assert snapshot(store) == before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("prefix", ["remaining_parts", "delivery_code", "delivery_baseline", "f_recovery", "content_filter"])
def test_write_failure_rolls_back_archive_and_all_prior_changes(store, monkeypatch, prefix):
    expected = failed_head(store, remaining=[TAIL])
    before = snapshot(store)
    original = store._set_metadata

    def fail_after_write(name, value):
        original(name, value)
        if name.startswith(prefix + ":"):
            raise OSError("temporary write failure")

    monkeypatch.setattr(store, "_set_metadata", fail_after_write)
    with pytest.raises(OSError, match="temporary write failure"):
        store.defer_failed_head(expected)
    assert snapshot(store) == before
    assert not store.connection.in_transaction


def test_restart_preserves_deferred_metric_and_unsubmitted_continuation(store):
    expected = failed_head(store, remaining=[TAIL, FOLLOWING])
    assert store.defer_failed_head(expected)
    other = DanmakuStore(store.path)
    try:
        assert other.head()["text"] == TAIL
        assert other.head()["id"] == expected["id"]
        assert other.strict_remaining_parts(expected["id"]) == [FOLLOWING]
        assert other.counts()["deferred_analysis"] == 1
        assert other.counts()["page_visible"] == 0
    finally:
        other.close()


def test_already_archived_head_is_idempotent(store):
    expected = failed_head(store)
    assert store.defer_failed_head(expected)
    before = snapshot(store)
    assert not store.defer_failed_head(expected)
    assert snapshot(store) == before


def test_unrelated_fifo_head_is_not_removed_by_other_failure_snapshot(store):
    expected = failed_head(store)
    enqueue(store, "following", [FOLLOWING])
    other = dict(store.connection.execute("SELECT * FROM messages WHERE segment_key='following'").fetchone())
    store.update(other["id"], "verify", request_id="another-request", reason="platform_f", platform_code=0)
    other.update(state="verify", request_id="another-request", reason="platform_f", platform_code=0)
    before = snapshot(store)
    assert not store.defer_failed_head(other)
    assert snapshot(store) == before
    assert store.head()["id"] == expected["id"]


def test_outer_transaction_is_rejected_without_committing_callers_work(store):
    expected = failed_head(store)
    store.connection.execute("BEGIN")
    store._set_metadata("caller_owned", "value")
    with pytest.raises(ValueError, match="defer_requires_independent_transaction"):
        store.defer_failed_head(expected)
    assert store.connection.in_transaction
    store.connection.rollback()
    assert store._metadata("caller_owned") is None
