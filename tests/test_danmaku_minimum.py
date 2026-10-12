"""Strict public-message minimum; SQLite only, no live accounts or POSTs."""
import json

import pytest

from fanyiji.danmaku_protocol import danmaku_utf16_length, danmaku_weighted_length
from fanyiji.danmaku_store import DanmakuStore, MIN_SEND_CHARACTERS, rebalance_short_parts


@pytest.fixture
def store(tmp_path):
    instance = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: 100.0)
    yield instance
    instance.close()


def enqueue(store, body, key="one", channel="system", room="1698225"):
    parts = ["【" + body[index:index + 38] + "】" for index in range(0, len(body), 38)]
    assert store.enqueue(room, channel, key, 1, body, parts)


def packets(store):
    rows = store.connection.execute("SELECT * FROM messages WHERE state NOT IN ('page_visible','cancelled') ORDER BY id").fetchall()
    return [packet for row in rows for packet in [row["text"], *store.remaining_parts(row["id"])]]


@pytest.mark.parametrize("length", [3, 9, 10, 11, 12, 40])
def test_wrapped_length_is_the_minimum_and_never_times_out(store, length):
    enqueue(store, "字" * (length - 2))
    head = store.head()
    assert head["state"] == ("pending" if length >= 11 else "holding_short")
    store._clock = lambda: 10000000.0
    store.release_holding(head["id"])
    assert store.head()["state"] == head["state"]


def test_short_fragments_merge_until_the_total_reaches_eleven(store):
    enqueue(store, "谢谢", "one")
    identifier = store.head()["id"]
    enqueue(store, "大家", "two")
    assert store.head()["state"] == "holding_short"
    enqueue(store, "支持", "three")
    assert store.head()["state"] == "holding_short"
    enqueue(store, "继续", "four")
    assert store.head()["id"] == identifier
    assert store.head()["state"] == "pending"
    assert store.head()["text"] == "【谢谢 大家 支持 继续】"


def test_oversized_next_final_rebalances_without_releasing_short_or_losing_fifo(store):
    enqueue(store, "谢谢", "one")
    identifier = store.head()["id"]
    enqueue(store, "长" * 38, "two")
    enqueue(store, "下一段完整字幕保持顺序", "three")
    payload = packets(store)
    assert len(payload) == 3
    assert store.head()["id"] == identifier
    assert all(MIN_SEND_CHARACTERS <= danmaku_utf16_length(packet) <= 40 for packet in payload)
    assert "".join(item[1:-1].replace(" ", "") for item in payload) == "谢谢" + "长" * 38 + "下一段完整字幕保持顺序"
    first, second, last = payload
    store.mark_page_visible(identifier, "first")
    assert store.head()["id"] == identifier
    assert store.head()["text"] == second
    store.mark_page_visible(identifier, "second")
    assert store.head()["text"] == last


@pytest.mark.parametrize("body", ["字" * 39, "字" * 46, "字" * 77, "😀" * 21, "A" * 39])
def test_long_tail_is_balanced_before_any_post(body):
    # The service may supply a tiny first or final chunk; all text must survive.
    initial = ["【" + body[index:index + 18] + "】" for index in range(0, len(body), 18)]
    parts = rebalance_short_parts(initial)
    assert "".join(packet[1:-1].replace(" ", "") for packet in parts) == body
    assert all(11 <= danmaku_utf16_length(packet) <= 40 for packet in parts)
    assert all(danmaku_weighted_length(packet) <= 80 for packet in parts)


def test_account_limit_incompatible_with_minimum_preserves_the_queue(store):
    store.max_weighted_length = 20
    assert store.enqueue("1698225", "system", "one", 1, "汉字" * 7,
                         ["【汉字汉字汉字汉字】", "【汉字汉字汉字】"])
    assert store.head()["state"] == "holding_short"
    assert packets(store) == ["【汉字汉字汉字汉字】", "【汉字汉字汉字】"]
    store.release_holding(store.head()["id"])
    assert store.head()["state"] == "holding_short"


def test_ascii_can_reach_the_minimum_with_the_same_weighted_limit():
    parts = rebalance_short_parts(["【ABC】", "【DEFGHIJKLM】"], max_weighted_length=20)
    assert all(11 <= danmaku_utf16_length(part) <= 40 and danmaku_weighted_length(part) <= 20 for part in parts)


def test_cross_channel_or_room_does_not_overtake_a_short_head(store):
    enqueue(store, "谢谢", "one")
    enqueue(store, "麦克风另一条完整字幕", "two", channel="mic")
    enqueue(store, "电脑下一条完整字幕", "three")
    assert packets(store) == ["【谢谢】", "【麦克风另一条完整字幕】", "【电脑下一条完整字幕】"]
    assert store.head()["state"] == "holding_short"


def test_frozen_short_tail_joins_only_its_unsent_text(store):
    enqueue(store, "这部分已经发送并确认", "one")
    identifier = store.head()["id"]
    with store.connection:
        store._set_metadata(f"remaining_parts:{identifier}", json.dumps(["【谢谢】"], ensure_ascii=False))
    store.set_filter_frozen(identifier, True)
    store.mark_page_visible(identifier, "visible")
    assert store.head()["state"] == "holding_short"
    enqueue(store, "大家继续一起观看直播", "two")
    assert store.head()["id"] == identifier
    assert store.head()["text"] == "【谢谢 大家继续一起观看直播】"
    assert store.filter_context(identifier)["frozen"] is True
    assert "已经发送" not in "".join(packets(store))


def test_rule_repacking_cannot_turn_a_short_head_pending(store):
    enqueue(store, "这句原始字幕可以发送", "one")
    identifier = store.head()["id"]
    assert store.prepare_filtered_parts(identifier, ["【谢谢】"], "new-rule")
    assert store.head()["state"] == "holding_short"
    store.release_holding(identifier)
    assert store.head()["state"] == "holding_short"
