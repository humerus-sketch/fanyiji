"""Atomic rule repairs against temporary queues; no live accounts or POSTs."""
import json

import pytest

from fanyiji.danmaku_store import DanmakuStore, content_digest


ROOM = "1698225"
ORIGINAL = "【当前原始失败内容完整保留】"
TAIL = "【当前原始续段完整保留】"
REPAIRED = ["【当前中性修正内容完整保留】", "【当前中性修正续段完整保留】"]
SIGNATURE_A = "a" * 64
SIGNATURE_B = "b" * 64


@pytest.fixture
def store(tmp_path):
    result = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: 1000.0)
    yield result
    result.close()


def enqueue(store, key="head", parts=None, *, room=ROOM):
    parts = parts or [ORIGINAL]
    assert store.enqueue(room, "system", key, 1, "原始完整字幕不受弹幕修改影响", parts,
                         filter_revision="initial-revision", filter_rule_signatures=[SIGNATURE_A])
    rows = store.connection.execute(
        "SELECT * FROM messages WHERE room=? AND segment_key=? ORDER BY id", (room, key)
    ).fetchall()
    return [dict(row) for row in rows]


def reject_f(store, identifier):
    store.set_filter_frozen(identifier, True)
    for attempt in range(2):
        store.begin_f_retry(identifier, f"send-{attempt}")
        store.update(identifier, "verify", request_id=f"send-{attempt}",
                     reason="platform_f", platform_code=0)
        store.set_delivery_baseline(identifier, 2)
        store.activate_f_recovery(identifier, new_f=True)
    return store.head()


def make_head(store):
    identifier = enqueue(store)[0]["id"]
    assert store.prepare_parts(identifier, [ORIGINAL, TAIL])
    return reject_f(store, identifier)


def snapshot(store):
    return {
        name: [tuple(row) for row in store.connection.execute(f"SELECT * FROM {name} ORDER BY 1,2")]
        for name in ("messages", "metadata", "contents", "segments", "metrics", "sqlite_sequence")
    }


def repair(store, expected, *, remaining=None, baseline=2, parts=None, signatures=None):
    return store.repair_f_rejected_head(
        expected, [TAIL] if remaining is None else remaining, baseline,
        REPAIRED if parts is None else parts, "new-revision",
        [SIGNATURE_A, SIGNATURE_B] if signatures is None else signatures)


def test_repair_is_same_fifo_packet_with_atomic_delivery_reset(store):
    expected = make_head(store)
    recovery = store.f_recovery(expected["id"])
    original_context = store.filter_context(expected["id"])
    original_metrics = snapshot(store)["metrics"]
    assert repair(store, expected)
    actual = store.head()
    for field in ("id", "room", "channel", "segment_key", "revision", "part", "created_at"):
        assert actual[field] == expected[field]
    assert actual["text"] == REPAIRED[0]
    assert actual["state"] == "pending"
    assert actual["reason"] == "f_head_rule_repaired"
    assert actual["request_id"] == ""
    assert actual["platform_code"] is None
    assert store.delivery_baseline(actual["id"]) is None
    assert store.f_recovery(actual["id"]) == {**recovery, "active": False}
    assert store.strict_remaining_parts(actual["id"]) == REPAIRED[1:]
    assert store.filter_context(actual["id"]) == original_context
    assert snapshot(store)["metrics"] == original_metrics
    assert store.applied_rule_signatures(actual["id"]) == [SIGNATURE_A, SIGNATURE_B]
    assert json.loads(store._metadata(f"f_filter_repair:{actual['id']}")) == {
        "revision": "new-revision", "part": expected["part"],
        "source_digest": content_digest(expected["text"])}


@pytest.mark.parametrize("field,value", [
    ("id", -1), ("room", "21399689"), ("text", "【不同失败正文】"),
    ("part", 7), ("request_id", "stale-request")])
def test_expected_head_guard_cannot_mutate_any_table(store, field, value):
    expected = make_head(store)
    expected[field] = value
    before = snapshot(store)
    assert not repair(store, expected)
    assert snapshot(store) == before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("state", [
    "pending", "holding_short", "sending", "retry_pending", "blocked", "page_visible", "cancelled"])
def test_current_state_must_be_explicit_f_verification(store, state):
    expected = make_head(store)
    store.update(expected["id"], state, request_id=expected["request_id"], reason="platform_f", platform_code=0)
    before = snapshot(store)
    assert not repair(store, expected)
    assert snapshot(store) == before


@pytest.mark.parametrize("reason", ["receipt_unconfirmed", "result_unknown", "restart_verify", "platform_rejected", ""])
def test_another_verification_reason_cannot_inherit_f_permission(store, reason):
    expected = make_head(store)
    store.update(expected["id"], "verify", request_id=expected["request_id"], reason=reason, platform_code=0)
    before = snapshot(store)
    assert not repair(store, expected)
    assert snapshot(store) == before


@pytest.mark.parametrize("code", [None, 1, -403, True, "0"])
def test_platform_code_must_be_integer_zero(store, code):
    expected = make_head(store)
    store.update(expected["id"], "verify", request_id=expected["request_id"], reason="platform_f", platform_code=code)
    before = snapshot(store)
    assert not repair(store, expected)
    assert snapshot(store) == before


def test_inactive_f_recovery_cannot_modify_a_verify_packet(store):
    expected = make_head(store)
    store.stop_f_recovery(expected["id"])
    before = snapshot(store)
    assert not repair(store, expected)
    assert snapshot(store) == before


@pytest.mark.parametrize("baseline", [None, True, -1, 10001, "2", 1, 3])
def test_expected_baseline_must_be_valid_and_current(store, baseline):
    expected = make_head(store)
    before = snapshot(store)
    assert not repair(store, expected, baseline=baseline)
    assert snapshot(store) == before


@pytest.mark.parametrize("baseline", [None, 0, 3])
def test_changed_stored_baseline_cannot_be_repaired_with_stale_evidence(store, baseline):
    expected = make_head(store)
    store.set_delivery_baseline(expected["id"], baseline)
    before = snapshot(store)
    assert not repair(store, expected)
    assert snapshot(store) == before


@pytest.mark.parametrize("parts", [[], ["【】"], ["未包装正文"], ["【不允许。标点】"], ["【" + "字" * 39 + "】"], [None]])
def test_invalid_replacement_packets_are_never_persisted(store, parts):
    expected = make_head(store)
    before = snapshot(store)
    assert not repair(store, expected, parts=parts)
    assert snapshot(store) == before


@pytest.mark.parametrize("signatures", [["not-a-signature"], [True], "a" * 64])
def test_invalid_rule_signatures_leave_original_packets_intact(store, signatures):
    expected = make_head(store)
    before = snapshot(store)
    assert not repair(store, expected, signatures=signatures)
    assert snapshot(store) == before


@pytest.mark.parametrize("value", ["{", "{}", '[null]', '["unwrapped"]', json.dumps(["【" + "字" * 75 + "】"])])
def test_corrupt_original_continuations_fail_closed_and_rollback(store, value):
    expected = make_head(store)
    with store.connection:
        store._set_metadata(f"remaining_parts:{expected['id']}", value)
    before = snapshot(store)
    with pytest.raises(ValueError, match="invalid_remaining_parts"):
        repair(store, expected)
    assert snapshot(store) == before
    assert not store.connection.in_transaction


def test_cleared_continuations_are_not_restored_from_stale_check_snapshot(store):
    expected = make_head(store)
    removed = store.clear_pending(protected_id=expected["id"])
    assert removed["removed"] == 1
    assert store.head()["text"] == ORIGINAL
    assert store.strict_remaining_parts(expected["id"]) == []
    before = snapshot(store)
    assert not repair(store, expected, remaining=[TAIL])
    assert snapshot(store) == before


@pytest.mark.parametrize("failure_key", ["remaining_parts", "delivery_baseline", "f_filter_rules", "f_filter_repair", "content_filter"])
def test_metadata_failure_rolls_back_message_and_all_previous_metadata(store, monkeypatch, failure_key):
    expected = make_head(store)
    before = snapshot(store)
    original = store._set_metadata

    def fail_after_write(key, value):
        original(key, value)
        if key.startswith(failure_key + ":"):
            raise OSError("temporary database write failure")

    monkeypatch.setattr(store, "_set_metadata", fail_after_write)
    with pytest.raises(OSError, match="temporary database write failure"):
        repair(store, expected)
    assert snapshot(store) == before
    assert not store.connection.in_transaction


def test_dedup_failure_rolls_back_already_repaired_body_and_new_digest(store, monkeypatch):
    expected = make_head(store)
    before = snapshot(store)
    original = store._remember

    def fail_after_remember(*args):
        original(*args)
        raise OSError("temporary digest write failure")

    monkeypatch.setattr(store, "_remember", fail_after_remember)
    with pytest.raises(OSError, match="temporary digest write failure"):
        repair(store, expected)
    assert snapshot(store) == before
    assert not store.connection.in_transaction


def test_independent_transaction_requirement_does_not_rollback_callers_work(store):
    expected = make_head(store)
    before = snapshot(store)
    store.connection.execute("BEGIN IMMEDIATE")
    store._set_metadata("unrelated-uncommitted-setting", "keep")
    try:
        with pytest.raises(ValueError, match="repair_requires_independent_transaction"):
            repair(store, expected)
        assert store.connection.in_transaction
        assert store._metadata("unrelated-uncommitted-setting") == "keep"
    finally:
        store.connection.rollback()
    assert snapshot(store) == before


def make_partially_delivered_group(store):
    grouped = enqueue(store, "partial", [
        "【原先已确认广播的前缀内容】", ORIGINAL, "【同组另一原始待发正文】"])
    store.mark_page_visible(grouped[0]["id"], "prefix-confirmation")
    identifier = grouped[1]["id"]
    assert store.prepare_parts(identifier, [ORIGINAL, TAIL])
    expected = reject_f(store, identifier)
    enqueue(store, "later", ["【排在后面的独立完整字幕】"])
    enqueue(store, "other-room", ["【其他直播间完整待发字幕】"], room="21399689")
    return grouped, expected


def test_delivered_prefix_other_rows_and_original_group_source_never_rebuild(store):
    grouped, expected = make_partially_delivered_group(store)
    before = snapshot(store)
    context = store.filter_context(expected["id"])
    other_rows = [row for row in before["messages"] if row[0] != expected["id"]]
    assert expected["part"] == 1
    assert repair(store, expected)
    after = snapshot(store)
    assert [row for row in after["messages"] if row[0] != expected["id"]] == other_rows
    assert after["segments"] == before["segments"]
    assert after["sqlite_sequence"] == before["sqlite_sequence"]
    assert store.filter_context(expected["id"]) == context
    assert context["frozen"] is True
    assert context["source"] == "原始完整字幕不受弹幕修改影响"
    assert context["revision"] == "initial-revision"
    assert store.applied_rule_signatures(grouped[0]["id"]) == [SIGNATURE_A]
    assert store.applied_rule_signatures(grouped[2]["id"]) == [SIGNATURE_A]
    assert store.applied_rule_signatures(expected["id"]) == [SIGNATURE_A, SIGNATURE_B]
    assert store.head()["id"] == expected["id"]
    assert store.head()["part"] == 1
    assert not store.prepare_filtered_parts(expected["id"], ["【不得重建整句前缀】"], "another-revision")


def test_stale_head_evidence_cannot_repair_a_non_fifo_packet(store):
    grouped, expected = make_partially_delivered_group(store)
    store.update(grouped[0]["id"], "pending")
    before = snapshot(store)
    assert store.head()["id"] == grouped[0]["id"]
    assert not repair(store, expected)
    assert snapshot(store) == before


def test_continuation_advance_keeps_its_rule_signatures_and_fifo_id(store):
    grouped, expected = make_partially_delivered_group(store)
    assert repair(store, expected)
    store.mark_page_visible(expected["id"], "repaired-confirmation")
    actual = store.head()
    assert actual["id"] == expected["id"]
    assert actual["part"] == expected["part"] + 1
    assert actual["text"] == REPAIRED[1]
    assert store.strict_remaining_parts(actual["id"]) == []
    assert store.applied_rule_signatures(actual["id"]) == [SIGNATURE_A, SIGNATURE_B]
    assert store.applied_rule_signatures(grouped[2]["id"]) == [SIGNATURE_A]
    assert store.filter_context(actual["id"])["frozen"] is True
    archived = store.connection.execute(
        "SELECT * FROM messages WHERE state='page_visible' ORDER BY id").fetchall()
    assert [row["text"] for row in archived] == [grouped[0]["text"], REPAIRED[0]]


def test_restart_preserves_repair_and_does_not_resurrect_old_f_permission(store):
    expected = make_head(store)
    assert repair(store, expected)
    before = snapshot(store)
    store.close()
    reopened = DanmakuStore(store.path, wall_clock=lambda: 1001.0)
    try:
        assert snapshot(reopened) == before
        actual = reopened.head()
        assert actual["id"] == expected["id"]
        assert actual["state"] == "pending"
        assert actual["text"] == REPAIRED[0]
        assert reopened.strict_remaining_parts(actual["id"]) == REPAIRED[1:]
        assert reopened.applied_rule_signatures(actual["id"]) == [SIGNATURE_A, SIGNATURE_B]
        assert not reopened.f_recovery(actual["id"])["active"]
        assert reopened.f_recovery(actual["id"])["retries"] == 2
        assert reopened.delivery_baseline(actual["id"]) is None
        assert reopened.filter_context(actual["id"])["frozen"] is True
        assert not repair(reopened, expected)
        assert snapshot(reopened) == before
    finally:
        reopened.close()


def test_restart_during_new_post_cannot_restore_repair_from_prior_f(store):
    expected = make_head(store)
    assert repair(store, expected)
    store.update(expected["id"], "sending", request_id="new-payload-send", platform_code=None)
    store.close()
    reopened = DanmakuStore(store.path, wall_clock=lambda: 1001.0)
    try:
        head = reopened.head()
        assert head["state"] == "verify"
        assert head["reason"] == "restart_verify"
        before = snapshot(reopened)
        assert not repair(reopened, head)
        assert snapshot(reopened) == before
    finally:
        reopened.close()
