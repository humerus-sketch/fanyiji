"""Final two-body-character/weighted-forty policy, SQLite and pure fixtures only."""
import json

import pytest

from fanyiji.danmaku_protocol import (danmaku_body_length, danmaku_utf16_length,
                                     danmaku_weighted_length, sanitize_limits, validate_danmaku_command)
from fanyiji.danmaku_service import split_danmaku
from fanyiji.danmaku_store import DanmakuStore, rebalance_short_parts


@pytest.fixture
def store(tmp_path):
    instance = DanmakuStore(tmp_path / "queue.sqlite3", wall_clock=lambda: 100.0)
    yield instance
    instance.close()


def enqueue(store, body, key="one", channel="system", room="1698225"):
    return store.enqueue(room, channel, key, 1, body, split_danmaku(body))


def packets(store):
    rows = store.connection.execute("SELECT * FROM messages WHERE state NOT IN ('page_visible','cancelled') ORDER BY id").fetchall()
    return [packet for row in rows for packet in [row["text"], *store.remaining_parts(row["id"])]]


@pytest.mark.parametrize("body", ["谢谢", "母亲", "逃跑", "第33页", "12", "〇一", "字" * 18])
def test_two_body_characters_are_pending_and_not_joined_to_later_sentences(store, body):
    assert enqueue(store, body)
    assert store.head()["state"] == "pending"
    assert store.head()["text"] == "【" + body + "】"
    assert enqueue(store, "继续", "next")
    assert packets(store) == ["【" + body + "】", "【继续】"]


@pytest.mark.parametrize("body", ["嗯", "1"])
def test_below_two_characters_are_preserved_until_only_the_immediate_next_sentence(store, body):
    assert enqueue(store, body)
    first = store.head()["id"]
    assert store.head()["state"] == "holding_short"
    assert enqueue(store, "继续", "next")
    assert store.head()["id"] == first
    assert packets(store) == ["【" + body + " 继续】"]
    assert store.head()["state"] == "pending"


def test_two_one_character_utterances_merge_in_original_order(store):
    assert enqueue(store, "嗯")
    assert enqueue(store, "啊", "next")
    assert packets(store) == ["【嗯 啊】"]


def test_two_character_utterances_never_merge_even_when_total_weight_is_small(store):
    for index, text in enumerate(("你还好吗", "逃跑", "母亲", "第33页")):
        assert enqueue(store, text, str(index))
    assert packets(store) == ["【你还好吗】", "【逃跑】", "【母亲】", "【第33页】"]


@pytest.mark.parametrize("body", ["字" * 19, "字" * 37, "字" * 18 + "1234567890", "12" + "字" * 40])
def test_long_sentence_is_cut_in_order_without_rebalancing_or_borrowing(body):
    parts = split_danmaku(body)
    assert "".join(packet[1:-1] for packet in parts) == body
    assert all(danmaku_weighted_length(packet) <= 40 for packet in parts)
    assert all(danmaku_utf16_length(packet) <= 40 for packet in parts)
    if body.startswith("字" * 18):
        assert parts[0] == "【" + "字" * 18 + "】"


def test_single_character_long_tail_remains_at_fifo_tail_and_only_merges_following_sentence(store):
    assert enqueue(store, "字" * 19)
    assert packets(store) == ["【" + "字" * 18 + "】", "【字】"]
    assert enqueue(store, "继续", "next")
    assert packets(store) == ["【" + "字" * 18 + "】", "【字 继续】"]


def test_direct_cut_keeps_internal_spaces_and_counts_them_in_the_weight():
    text = "甲  乙" + "字" * 16 + " 12"
    parts = split_danmaku(text)
    assert "".join(packet[1:-1] for packet in parts) == text
    assert all(danmaku_weighted_length(packet) <= 40 for packet in parts)
    assert split_danmaku("甲  乙") == ["【甲  乙】"]
    assert [part for packet in parts for part in split_danmaku(packet)] == parts


def test_ascii_digits_use_the_weighted_cap_without_premature_raw_divided_by_two_cut():
    assert sanitize_limits({"max_weighted_length": 80}) == {"max_weighted_length": 40, "max_characters": 40}
    parts = split_danmaku("1" * 37)
    assert parts == ["【" + "1" * 36 + "】", "【1】"]
    assert danmaku_weighted_length(parts[0]) == 40
    assert danmaku_body_length(parts[0]) == 36


def test_existing_holding_two_character_packet_can_release_without_new_audio(store):
    assert enqueue(store, "第33页")
    identifier = store.head()["id"]
    store.update(identifier, "holding_short")
    store.release_holding(identifier)
    assert store.head()["state"] == "pending"
    assert store.head()["text"] == "【第33页】"


def test_one_character_holding_survives_restart_then_merges_next_sentence(store):
    assert enqueue(store, "嗯")
    identifier = store.head()["id"]
    other = DanmakuStore(store.path)
    try:
        assert other.head()["id"] == identifier and other.head()["state"] == "holding_short"
        assert enqueue(other, "继续", "next")
        assert other.head()["text"] == "【嗯 继续】"
    finally:
        other.close()


def test_undersized_tail_does_not_merge_across_an_intervening_channel(store):
    assert enqueue(store, "嗯")
    assert enqueue(store, "麦克风", "mic", channel="mic")
    assert enqueue(store, "继续", "next")
    assert packets(store) == ["【嗯】", "【麦克风】", "【继续】"]


def test_only_body_characters_count_for_minimum_not_spaces_or_brackets():
    assert danmaku_body_length("【第33页】") == 4
    assert danmaku_body_length("【嗯    】") == 1
    assert danmaku_body_length("【12】") == 2
    assert danmaku_body_length("【𠀀𠀁】") == 2
    assert danmaku_utf16_length("【𠀀𠀁】") == 6


def test_protocol_accepts_two_digits_but_refuses_one_digit_even_with_many_spaces():
    base = {"op": "danmaku_send", "request_id": "fixture", "room": "1698225"}
    assert validate_danmaku_command({**base, "text": "【12】"})["text"] == "【12】"
    with pytest.raises(ValueError):
        validate_danmaku_command({**base, "text": "【1    】"})


def test_compatibility_splitter_name_does_not_rebalance_single_character_tail():
    parts = rebalance_short_parts(["【" + "字" * 18 + "】", "【字】"])
    assert parts == ["【" + "字" * 18 + "】", "【字】"]


def test_merging_single_character_with_long_successor_never_invents_spaces_between_its_cuts(store):
    assert enqueue(store, "嗯")
    assert enqueue(store, "字" * 37, "next")
    assert "".join(part[1:-1] for part in packets(store)) == "嗯 " + "字" * 37
    assert all(danmaku_weighted_length(part) <= 40 for part in packets(store))


def test_combined_single_character_duplicate_obeys_ten_second_window_and_keeps_source_record(store):
    assert enqueue(store, "嗯啊", "earlier")
    store.mark_page_visible(store.head()["id"], "fake-observed")
    assert enqueue(store, "嗯", "one")
    identifier = store.head()["id"]
    assert not enqueue(store, "啊", "two")
    assert store.head() is None
    assert store.filter_context(identifier)["source"] == "嗯 啊"
    assert store.connection.execute("SELECT reason FROM messages WHERE id=?", (identifier,)).fetchone()[0] == "duplicate_skipped"
