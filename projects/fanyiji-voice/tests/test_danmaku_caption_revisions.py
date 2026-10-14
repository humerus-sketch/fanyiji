"""Chrome-only revision transactions use temporary SQLite, never live accounts."""
import json

import pytest

from fanyiji.danmaku_service import split_danmaku
from fanyiji.danmaku_store import DanmakuStore, rebalance_short_parts


ROOM = "1698225"


@pytest.fixture
def store(tmp_path):
    value = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: 100.0)
    yield value
    value.close()


def enqueue(store, key, text, revision=1):
    return store.enqueue(ROOM, "chrome_caption", key, revision, text, split_danmaku(text),
                         revise_unsubmitted=True)


def payload(store):
    result = []
    for row in store.connection.execute("SELECT * FROM messages WHERE state NOT IN ('page_visible','cancelled') ORDER BY id"):
        result.extend([row["text"], *store.strict_remaining_parts(row["id"])])
    return result


def legacy_merged_group(store, first, second):
    """Synthesize a pre-policy merged group without changing real user storage."""
    assert enqueue(store, "one", first)
    assert enqueue(store, "two", second)
    head = store.head()
    members = [{"key": key, "revision": 1, "source": text, "parts": split_danmaku(text)}
               for key, text in (("one", first), ("two", second))]
    combined = rebalance_short_parts([part for member in members for part in member["parts"]])
    with store.connection:
        store.connection.execute("UPDATE messages SET text=? WHERE id=?", (combined[0], head["id"]))
        store.connection.execute("UPDATE messages SET state='cancelled',reason='caption_merged' WHERE segment_key='two'")
        store._set_metadata(f"remaining_parts:{head['id']}", json.dumps(combined[1:], ensure_ascii=False))
        store._save_filter_context(ROOM, "one", first + " " + second, "", False)
        store._save_caption_members(ROOM, "one", members)
    return head["id"]


def test_caption_revision_keeps_fifo_head_and_growing_continuations(store):
    assert enqueue(store, "one", "最初的字幕正在等待发送。")
    first = store.head()["id"]
    assert enqueue(store, "two", "后来的字幕仍然排在后面。")
    assert enqueue(store, "one", "改" * 83 + "。", 2)
    assert store.head()["id"] == first
    assert store.head()["revision"] == 2
    parts = payload(store)
    assert "".join(packet[1:-1] for packet in parts[:-1]) == "改" * 83
    assert parts[-1] == "【后来的字幕仍然排在后面】"
    assert store.caption_enqueue_state(ROOM, "one")["mutable"]


def test_merged_short_members_can_both_be_revised_without_losing_either(store):
    first = legacy_merged_group(store, "谢谢大家。", "现在可以继续游戏了。")
    assert store.head()["id"] == first
    assert enqueue(store, "one", "请等一下。", 2)
    assert payload(store) == ["【请等一下 现在可以继续游戏了】"]
    assert enqueue(store, "two", "现在请进入下一场比赛。", 2)
    assert payload(store) == ["【请等一下 现在请进入下一场比赛】"]
    assert store.filter_context(first)["source"] == "请等一下。 现在请进入下一场比赛。"
    assert store.caption_enqueue_state(ROOM, "two")["head_id"] == first


@pytest.mark.parametrize("state,reason", [
    ("sending", ""), ("verify", "result_unknown"), ("blocked", "result_unknown"),
    ("retry_pending", "f_retry_pending"), ("page_visible", ""), ("cancelled", ""),
])
def test_post_unknown_and_terminal_states_never_revise(store, state, reason):
    assert enqueue(store, "one", "这一句已经开始投递不能重复。")
    row = store.head()
    store.update(row["id"], state, request_id="already-started", reason=reason)
    assert not enqueue(store, "one", "不得重新生成并投递这段内容。", 2)
    after = store.connection.execute("SELECT * FROM messages WHERE id=?", (row["id"],)).fetchone()
    assert after["text"] == row["text"]
    assert after["state"] == state
    assert after["request_id"] == "already-started"


def test_post_frozen_merged_member_never_regenerates_delivered_text(store):
    first = legacy_merged_group(store, "谢谢。", "甲" * 50)
    store.set_filter_frozen(first, True)
    store.mark_page_visible(first, "visible-first")
    before = payload(store)
    assert not enqueue(store, "two", "乙" * 50, 2)
    assert payload(store) == before


def test_default_enqueue_does_not_opt_in_to_caption_revisions(store):
    assert store.enqueue(ROOM, "system", "one", 1, "这一句来自原有翻译模式。", ["【这一句来自原有翻译模式】"])
    assert not store.enqueue(ROOM, "system", "one", 2, "旧行为不能自动覆盖。", ["【旧行为不能自动覆盖】"])
    with pytest.raises(ValueError, match="caption_revision_requires_chrome_channel"):
        store.enqueue(ROOM, "system", "one", 2, "不得绕过源限制。", ["【不得绕过源限制】"], revise_unsubmitted=True)


def test_same_or_older_revision_never_overwrites_saved_caption(store):
    assert enqueue(store, "one", "原始完整字幕保持不变。", 2)
    assert not enqueue(store, "one", "同一修订号偷换内容。", 2)
    assert not enqueue(store, "one", "过期修订号不能回滚。", 1)
    assert payload(store) == ["【原始完整字幕保持不变】"]


def test_explicit_clear_fences_caption_revisions(store):
    assert enqueue(store, "one", "已经被用户明确清理的字幕。")
    assert store.clear_pending()["removed"] == 1
    assert not enqueue(store, "one", "不能因为后续修订恢复已清理内容。", 2)
    assert store.head() is None


def test_revision_write_failure_rolls_back_text_members_and_revision(store, monkeypatch):
    assert enqueue(store, "one", "原始内容保存完整。")
    before = list(map(tuple, store.connection.execute("SELECT * FROM messages")))
    save = store._save_caption_members
    def fail(*args, **kwargs):
        raise OSError("synthetic write failure")
    monkeypatch.setattr(store, "_save_caption_members", fail)
    with pytest.raises(OSError):
        enqueue(store, "one", "这段新修订写盘失败必须回滚。", 2)
    assert list(map(tuple, store.connection.execute("SELECT * FROM messages"))) == before
    assert store.caption_enqueue_state(ROOM, "one")["revision"] == 1
    monkeypatch.setattr(store, "_save_caption_members", save)
    assert enqueue(store, "one", "这段新修订现在能够完整保存。", 2)


def test_saving_new_short_member_is_atomic_without_merging_the_fifo(store, monkeypatch):
    assert enqueue(store, "one", "谢谢。")
    before = payload(store)
    def fail(*args, **kwargs):
        raise OSError("synthetic membership failure")
    monkeypatch.setattr(store, "_save_caption_members", fail)
    with pytest.raises(OSError):
        enqueue(store, "two", "现在请大家继续观看下一场。")
    assert payload(store) == before
    assert store.caption_enqueue_state(ROOM, "two") is None


def test_existing_other_store_post_state_is_rechecked_under_transaction(store):
    assert enqueue(store, "one", "另一个发送器可能已经开始请求。")
    other = DanmakuStore(store.path)
    try:
        other.set_filter_frozen(store.head()["id"], True)
        other.update(store.head()["id"], "sending", request_id="foreign-live-request")
        assert not enqueue(store, "one", "已经开始请求不可以被更改。", 2)
        assert store.head()["request_id"] == "foreign-live-request"
    finally:
        other.close()


def test_corrupt_members_fail_closed_and_preserve_pending_row(store):
    assert enqueue(store, "one", "一段有效的中文完整字幕。")
    before = payload(store)
    with store.connection:
        store._set_metadata(f"caption_members:{ROOM}:one", json.dumps([{"key": "one"}]))
    with pytest.raises(ValueError, match="invalid_caption_members"):
        enqueue(store, "one", "另一段修订文本不能覆盖损坏元数据。", 2)
    assert payload(store) == before


def test_shrinking_caption_does_not_merge_already_queued_adjacent_utterance(store):
    assert enqueue(store, "one", "主播现在介绍下一场比赛。")
    first = store.head()["id"]
    assert enqueue(store, "two", "请大家等一下。")
    assert enqueue(store, "one", "好吧。", 2)
    assert store.head()["id"] == first
    assert store.head()["state"] == "pending"
    assert payload(store) == ["【好吧】", "【请大家等一下】"]
    assert store.caption_enqueue_state(ROOM, "two")["head_id"] != first
    assert enqueue(store, "two", "下一场比赛继续。", 2)
    assert payload(store) == ["【好吧】", "【下一场比赛继续】"]


def test_shrinking_caption_keeps_all_adjacent_utterances_separate_in_fifo(store):
    assert enqueue(store, "one", "主播现在介绍下一场比赛。")
    assert enqueue(store, "two", "谢谢。")
    assert enqueue(store, "three", "大家。")
    assert enqueue(store, "four", "继续。")
    assert enqueue(store, "one", "好吧。", 2)
    assert payload(store) == ["【好吧】", "【谢谢】", "【大家】", "【继续】"]
    assert store.head()["state"] == "pending"
    assert len({store.caption_enqueue_state(ROOM, key)["head_id"] for key in ("one", "two", "three", "four")}) == 4


def test_shrinking_revision_does_not_merge_across_other_channel(store):
    assert enqueue(store, "one", "主播现在介绍下一场比赛。")
    assert store.enqueue(ROOM, "mic", "other", 1, "麦克风中另一条完整字幕", ["【麦克风中另一条完整字幕】"])
    assert enqueue(store, "two", "请大家等一下。")
    assert enqueue(store, "one", "好吧。", 2)
    assert payload(store) == ["【好吧】", "【麦克风中另一条完整字幕】", "【请大家等一下】"]


def test_revision_duplicate_of_another_recent_segment_is_not_sent_twice(store):
    assert enqueue(store, "one", "第一句尚未开始发送。")
    assert enqueue(store, "two", "这句话已经在队列中了。")
    assert not enqueue(store, "one", "这句话已经在队列中了。", 2)
    assert payload(store) == ["【这句话已经在队列中了】"]
    assert not store.caption_enqueue_state(ROOM, "one")["mutable"]


def test_revision_duplicate_outside_ten_second_window_is_allowed(store):
    assert enqueue(store, "one", "第一句尚未开始发送。")
    assert enqueue(store, "two", "这句话可以在稍后重新出现。")
    store._clock = lambda: 111.0
    assert enqueue(store, "one", "这句话可以在稍后重新出现。", 2)
    assert len(payload(store)) == 2


def test_revision_to_one_character_merges_only_the_immediate_queued_successor(store):
    assert enqueue(store, "one", "主播介绍比赛。")
    first = store.head()["id"]
    assert enqueue(store, "two", "母亲。")
    assert enqueue(store, "three", "逃跑。")
    assert enqueue(store, "one", "嗯。", 2)
    assert store.head()["id"] == first
    assert payload(store) == ["【嗯 母亲】", "【逃跑】"]
    assert store.caption_enqueue_state(ROOM, "two")["head_id"] == first
