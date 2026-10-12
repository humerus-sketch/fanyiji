"""Lossless scheduling, revision replacement and temporary spill ownership."""
from collections import Counter
from dataclasses import dataclass, replace
import json
import sqlite3
import threading
from unittest.mock import Mock

import pytest

from fanyiji.translation_queue import TranslationQueue


@dataclass
class Job:
    channel: str
    segment_id: str
    text: str = "こんにちは。"
    ended_at: float = 100.0
    direction: tuple[str, str] = ("ja", "zh")
    seconds: float = 1.0
    context: str = "前の文。"
    revision: int = 1
    provisional: bool = False
    source_segment_ids: tuple[str, ...] = ("source-a", "source-b")
    queued_at: float = 100.0
    context_generation: int = 0


def job(identifier, *, channel="system", **fields):
    return Job(channel, str(identifier), **fields)


def drain(queue):
    queue.close()
    results = []
    while (item := queue.get()) is not None:
        results.append(item)
    return results


def test_audio_duration_does_not_limit_text_admission():
    queue = TranslationQueue()
    for number in range(40):
        assert queue.put("system", job(number), 3600)
    assert queue.snapshot()["pending"] == 40
    assert len(queue._queues["system"]) == 40
    assert not queue._queues["mic"]
    assert [item.segment_id for item in drain(queue)] == list(map(str, range(40)))


def test_sources_alternate_without_reordering_independent_sentences():
    queue = TranslationQueue(memory_limit=2)
    for channel in ("system", "mic"):
        for number in range(4):
            queue.put(channel, job(number, channel=channel))
    results = drain(queue)
    assert [item.channel for item in results] == ["mic", "system"] * 4
    for channel in ("mic", "system"):
        assert [item.segment_id for item in results if item.channel == channel] == ["0", "1", "2", "3"]


@pytest.mark.parametrize("batch", [False, True])
def test_channel_filtered_consumer_does_not_take_other_source_and_closed_channel_drains(batch):
    queue = TranslationQueue(memory_limit=1)
    queue.put("mic", job("m", channel="mic"))
    queue.put("system", job("s"))
    queue.close()
    take = queue.get_batch if batch else queue.get
    result = take(channel="system")
    assert ([entry.segment_id for entry in result] if batch else result.segment_id) == (["s"] if batch else "s")
    assert take(channel="system") == ([] if batch else None)
    assert queue.snapshot()["pending_by_channel"] == {"mic": 1, "system": 0}
    result = take(channel="mic")
    assert ([entry.segment_id for entry in result] if batch else result.segment_id) == (["m"] if batch else "m")


@pytest.mark.parametrize("batch", [False, True])
def test_put_wakes_the_owner_among_two_channel_bound_waiters(batch):
    queue = TranslationQueue()
    done = {channel: threading.Event() for channel in ("mic", "system")}
    results = {}
    def consume(channel):
        results[channel] = queue.get_batch(channel=channel) if batch else queue.get(channel=channel)
        done[channel].set()
    workers = [threading.Thread(target=consume, args=(channel,)) for channel in done]
    for worker in workers:
        worker.start()
    try:
        queue.put("system", job("s"))
        assert done["system"].wait(1)
        assert not done["mic"].is_set()
        queue.put("mic", job("m", channel="mic"))
        assert done["mic"].wait(1)
        assert queue.snapshot()["pending"] == 0
    finally:
        queue.close(discard=True)
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()


def test_spilled_batches_keep_generation_and_isolated_revision_boundaries():
    queue = TranslationQueue(memory_limit=1)
    for name, fields in (("a", {}), ("b", {}), ("c", {"context_generation": 1}),
                         ("d", {"context_generation": 1}),
                         ("e", {"context_generation": 1, "provisional": True}),
                         ("f", {"context_generation": 1}),
                         ("g", {"context_generation": 1, "revision": 2}),
                         ("h", {"context_generation": 1})):
        queue.put("system", job(name, **fields))
    queue.close()
    batches = [queue.get_batch(channel="system") for _ in range(6)]
    assert [[entry.segment_id for entry in group] for group in batches] == [["a", "b"], ["c", "d"], ["e"], ["f"], ["g"], ["h"]]
    assert all(len({entry.context_generation for entry in group}) == 1 for group in batches)
    assert queue.spool_path is None


@pytest.mark.parametrize("batch", [False, True])
def test_filtered_wait_rejects_unknown_channel_and_does_not_consume_other_work(batch):
    queue = TranslationQueue()
    queue.put("mic", job("m", channel="mic"))
    take = queue.get_batch if batch else queue.get
    with pytest.raises(ValueError, match="Unknown translation channel"):
        take(channel="unknown", timeout=0)
    with pytest.raises(TimeoutError):
        take(channel="system", timeout=0)
    assert queue.snapshot()["pending"] == 1
    queue.close(discard=True)


def test_newer_revision_replaces_pending_job_at_original_fifo_position(monkeypatch):
    queue = TranslationQueue()
    monkeypatch.setattr("fanyiji.translation_queue.time.monotonic", lambda: 10)
    queue.put("system", job("a"))
    queue.put("system", job("b"))
    monkeypatch.setattr("fanyiji.translation_queue.time.monotonic", lambda: 20)
    updated = job("a", revision=2, text="彼女は今日は歌わない。", provisional=False)
    assert queue.put("system", updated)
    assert queue.snapshot()["oldest_age_seconds"] == 10
    assert queue.snapshot()["coalesced"] == 1
    assert queue.snapshot()["pending"] == 2
    results = drain(queue)
    assert results[0] is updated
    assert results[1].segment_id == "b"
    assert queue.snapshot()["oldest_age_seconds"] == 0


def test_older_or_duplicate_revision_cannot_replace_newer_pending_text():
    queue = TranslationQueue()
    expected = job("a", revision=3, text="新しい文。")
    queue.put("system", expected)
    queue.put("system", job("a", revision=1, text="古い文。"))
    queue.put("system", replace(expected))
    assert queue.snapshot()["coalesced"] == 2
    assert queue.snapshot()["stale_revisions"] == 2
    assert drain(queue) == [expected]


def test_revision_after_get_is_new_work_and_does_not_mutate_inflight_job():
    queue = TranslationQueue()
    original = job("a", text="別にそう。", provisional=True)
    queue.put("system", original)
    queue.put("system", job("b"))
    inflight = queue.get(timeout=0)
    corrected = replace(original, text="別にそういうことじゃない。", revision=2, provisional=False)
    queue.put("system", corrected)
    assert inflight is original and inflight.revision == 1
    assert inflight.text == "別にそう。"
    assert [item.segment_id for item in drain(queue)] == ["b", "a"]
    assert queue.snapshot()["coalesced"] == 0


def test_same_segment_identifier_is_independent_across_channels():
    queue = TranslationQueue(memory_limit=1)
    queue.put("system", job("a"))
    queue.put("mic", job("a", channel="mic", revision=2))
    assert len(drain(queue)) == 2


def test_spilled_revisions_keep_fifo_position_and_all_tuple_fields():
    queue = TranslationQueue(memory_limit=1)
    queue.put("system", job("a"))
    queue.put("system", job("b"))
    queue.put("system", job("c"))
    updated = job("b", revision=2, text="今日は歌わない。",
                  source_segment_ids=("b", "continued"), direction=("zh", "ja"))
    queue.put("system", updated)
    queue.put("system", job("b", revision=1))
    results = drain(queue)
    assert [item.segment_id for item in results] == ["a", "b", "c"]
    assert results[1] == updated
    assert type(results[1].direction) is tuple
    assert type(results[1].source_segment_ids) is tuple
    assert queue.snapshot()["coalesced"] == 2


def test_memory_reuse_does_not_overtake_older_spilled_jobs():
    queue = TranslationQueue(memory_limit=2)
    for name in ("a", "b", "c"):
        queue.put("system", job(name))
    assert queue.get().segment_id == "a"
    queue.put("system", job("d"))
    assert [item.segment_id for item in drain(queue)] == ["b", "c", "d"]


def test_large_queue_keeps_only_configured_number_of_jobs_in_memory():
    queue = TranslationQueue(memory_limit=8)
    for number in range(150):
        queue.put("system", job(number))
    snapshot = queue.snapshot()
    assert snapshot["pending"] == 150
    assert snapshot["in_memory"] == 8
    assert snapshot["spilled"] == snapshot["spilled_total"] == 142
    spool = queue.spool_path
    assert spool is not None and spool.is_file()
    assert spool.parent.name.startswith("fanyiji-mt-")
    assert [item.segment_id for item in drain(queue)] == list(map(str, range(150)))
    assert queue.spool_path is None and not spool.parent.exists()
    assert queue.snapshot()["spilled"] == 0
    assert queue.snapshot()["spilled_total"] == 142


def test_optional_constructor_factory_restores_spilled_job_without_pipeline_import():
    factory = Mock(side_effect=Job)
    queue = TranslationQueue(memory_limit=1, job_factory=factory)
    queue.put("system", job("a"))
    expected = job("b")
    queue.put("system", expected)
    results = drain(queue)
    assert results[1] == expected
    factory.assert_called_once()
    assert factory.call_args.kwargs["direction"] == ("ja", "zh")
    assert factory.call_args.kwargs["source_segment_ids"] == ("source-a", "source-b")


def test_batch_is_immediate_and_never_waits_to_fill(monkeypatch):
    queue = TranslationQueue()
    queue.put("system", job("a"))
    wait = Mock(side_effect=AssertionError("Must not wait for another sentence"))
    monkeypatch.setattr(queue._condition, "wait", wait)
    assert [item.segment_id for item in queue.get_batch()] == ["a"]
    wait.assert_not_called()
    queue.close()


def test_batches_are_fair_and_never_mix_sources_or_directions():
    queue = TranslationQueue(memory_limit=2)
    for name, direction in (("a", ("ja", "zh")), ("b", ("ja", "zh")), ("c", ("zh", "ja")),
                            ("d", ("ja", "zh"))):
        queue.put("system", job(name, direction=direction))
    for number in range(4):
        queue.put("mic", job(number, channel="mic", direction=("zh", "ja")))
    queue.close()
    batches = [queue.get_batch() for _ in range(5)]
    assert [[item.segment_id for item in batch] for batch in batches] == [["0", "1", "2"], ["a", "b"], ["3"], ["c"], ["d"]]
    assert all(len({(item.channel, item.direction) for item in batch}) == 1 for batch in batches)
    assert queue.get_batch() == []


def test_batch_character_cap_and_oversize_first_job_do_not_block_queue():
    queue = TranslationQueue(memory_limit=1)
    for name, text in (("a", "12345"), ("b", "123"), ("c", "1234"), ("d", "12")):
        queue.put("system", job(name, text=text))
    queue.close()
    assert [item.segment_id for item in queue.get_batch(max_characters=4)] == ["a"]
    assert [item.segment_id for item in queue.get_batch(max_characters=5)] == ["b"]
    assert [item.segment_id for item in queue.get_batch(max_characters=6)] == ["c", "d"]
    assert queue.get_batch() == []


def test_closed_queue_drains_then_returns_none_and_rejects_new_work():
    queue = TranslationQueue(memory_limit=1)
    queue.put("system", job("a"))
    queue.put("system", job("b"))
    spool = queue.spool_path
    queue.close()
    assert spool.exists()
    assert not queue.put("system", job("c"))
    assert queue.get().segment_id == "a"
    assert queue.get().segment_id == "b"
    assert queue.get(timeout=0) is None
    assert queue.get_batch(timeout=0) == []
    assert not spool.parent.exists()


def test_discard_is_idempotent_and_cleans_only_owned_temp_storage(tmp_path):
    sentinel = tmp_path / "user-model.txt"
    sentinel.write_text("keep")
    queue = TranslationQueue(memory_limit=1)
    queue.put("system", job("a"))
    queue.put("system", job("b"))
    spool = queue.spool_path
    queue.close(discard=True)
    queue.close(discard=True)
    assert queue.snapshot()["pending"] == 0
    assert queue.get() is None
    assert not spool.parent.exists()
    assert sentinel.read_text() == "keep"


def test_idle_timeout_is_distinct_from_closed_queue():
    queue = TranslationQueue()
    with pytest.raises(TimeoutError, match="No queued work"):
        queue.get(timeout=0)
    with pytest.raises(TimeoutError, match="No queued work"):
        queue.get_batch(timeout=0.001)
    queue.close()
    assert queue.get(timeout=0) is None


@pytest.mark.parametrize("batch", [False, True])
def test_close_wakes_waiting_consumer(batch):
    queue = TranslationQueue()
    started = threading.Event()
    results = []

    def consume():
        started.set()
        results.append(queue.get_batch() if batch else queue.get())

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert started.wait(1)
    queue.close(discard=True)
    consumer.join(2)
    assert not consumer.is_alive()
    assert results == ([[]] if batch else [None])


def test_spool_creation_error_is_visible_and_keeps_existing_job(monkeypatch):
    queue = TranslationQueue(memory_limit=1)
    queue.put("system", job("a"))
    monkeypatch.setattr("fanyiji.translation_queue.sqlite3.connect",
                        Mock(side_effect=sqlite3.OperationalError("disk unavailable")))
    with pytest.raises(sqlite3.OperationalError, match="disk unavailable"):
        queue.put("system", job("b"))
    assert queue.snapshot()["pending"] == 1
    assert queue.spool_path is None
    assert [item.segment_id for item in drain(queue)] == ["a"]


def test_failed_spool_insert_does_not_admit_or_drop_other_sentences():
    queue = TranslationQueue(memory_limit=1)
    queue.put("system", job("a"))
    queue.put("system", job("b"))
    queue._connection.execute("CREATE TRIGGER reject_insert BEFORE INSERT ON jobs "
                              "BEGIN SELECT RAISE(ABORT, 'storage rejected'); END")
    with pytest.raises(sqlite3.IntegrityError, match="storage rejected"):
        queue.put("system", job("c"))
    assert queue.snapshot()["pending"] == 2
    assert [item.segment_id for item in drain(queue)] == ["a", "b"]


def test_failed_spool_update_retains_previous_revision():
    queue = TranslationQueue(memory_limit=1)
    queue.put("system", job("a"))
    queue.put("system", job("b"))
    queue._connection.execute("CREATE TRIGGER reject_update BEFORE UPDATE ON jobs "
                              "BEGIN SELECT RAISE(ABORT, 'update rejected'); END")
    with pytest.raises(sqlite3.IntegrityError, match="update rejected"):
        queue.put("system", job("b", revision=2))
    results = drain(queue)
    assert len(results) == 2 and results[1].revision == 1
    assert queue.snapshot()["coalesced"] == 0


def test_batch_decode_error_does_not_remove_preceding_memory_job():
    queue = TranslationQueue(memory_limit=1)
    for name in ("a", "b", "c"):
        queue.put("system", job(name))
    original = queue._connection.execute("SELECT payload FROM jobs WHERE segment_id = 'b'").fetchone()[0]
    queue._connection.execute("UPDATE jobs SET payload = '{broken' WHERE segment_id = 'b'")
    queue._connection.commit()
    with pytest.raises(json.JSONDecodeError):
        queue.get_batch()
    assert queue.snapshot()["pending"] == 3
    queue._connection.execute("UPDATE jobs SET payload = ? WHERE segment_id = 'b'", (original,))
    queue._connection.commit()
    assert [item.segment_id for item in drain(queue)] == ["a", "b", "c"]


def test_batch_delete_failure_rolls_back_all_selected_spilled_jobs():
    queue = TranslationQueue(memory_limit=1)
    for name in ("a", "b", "c"):
        queue.put("system", job(name))
    queue._connection.execute("CREATE TRIGGER reject_delete BEFORE DELETE ON jobs "
                              "WHEN OLD.segment_id = 'c' BEGIN SELECT RAISE(ABORT, 'delete rejected'); END")
    with pytest.raises(sqlite3.IntegrityError, match="delete rejected"):
        queue.get_batch()
    assert queue.snapshot()["pending"] == 3
    assert queue._connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    queue._connection.execute("DROP TRIGGER reject_delete")
    assert [item.segment_id for item in drain(queue)] == ["a", "b", "c"]


def test_multiple_producers_and_consumers_deliver_each_independent_sentence_once():
    queue = TranslationQueue(memory_limit=4)
    barrier = threading.Barrier(5)
    results = []
    results_lock = threading.Lock()
    failures = []

    def produce(channel):
        try:
            barrier.wait(timeout=3)
            for number in range(100):
                assert queue.put(channel, job(number, channel=channel))
        except BaseException as exc:
            failures.append(exc)

    def consume():
        try:
            barrier.wait(timeout=3)
            while (item := queue.get()) is not None:
                with results_lock:
                    results.append((item.channel, item.segment_id))
        except BaseException as exc:
            failures.append(exc)

    producers = [threading.Thread(target=produce, args=(channel,)) for channel in ("mic", "system")]
    consumers = [threading.Thread(target=consume) for _ in range(2)]
    for thread in producers + consumers:
        thread.start()
    barrier.wait(timeout=3)
    for thread in producers:
        thread.join(5)
        assert not thread.is_alive()
    queue.close()
    for thread in consumers:
        thread.join(5)
        assert not thread.is_alive()
    assert failures == []
    assert Counter(results) == Counter((channel, str(number)) for channel in ("mic", "system") for number in range(100))
    assert queue.snapshot()["pending"] == 0 and queue.spool_path is None


@pytest.mark.parametrize("parameters", [{"limit": 0}, {"max_characters": 0}])
def test_invalid_batch_limits_are_rejected_before_dequeue(parameters):
    queue = TranslationQueue()
    queue.put("system", job("a"))
    with pytest.raises(ValueError, match="positive"):
        queue.get_batch(**parameters)
    assert queue.get().segment_id == "a"
    queue.close()


def test_invalid_channel_or_job_cannot_be_admitted():
    queue = TranslationQueue()
    with pytest.raises(ValueError, match="Unknown translation channel"):
        queue.put("other", job("a"))
    with pytest.raises(ValueError, match="differs"):
        queue.put("mic", job("a"))
    with pytest.raises(TypeError, match="dataclass"):
        queue.put("system", object())
    assert queue.snapshot()["pending"] == 0
    queue.close()


def test_invalid_memory_limit_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        TranslationQueue(memory_limit=0)


@pytest.mark.parametrize("batch", [False, True])
def test_final_dequeue_still_delivers_jobs_when_owned_directory_cleanup_fails(monkeypatch, caplog, batch):
    queue = TranslationQueue(memory_limit=1)
    queue.put("system", job("a"))
    queue.put("system", job("b"))
    directory = queue._spool_directory
    original_cleanup = directory.cleanup
    attempts = 0

    def cleanup():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("temporarily held by scanner")
        original_cleanup()

    monkeypatch.setattr(directory, "cleanup", cleanup)
    if batch:
        assert [item.segment_id for item in queue.get_batch()] == ["a", "b"]
    else:
        assert queue.get().segment_id == "a"
        assert queue.get().segment_id == "b"
    assert queue.snapshot()["pending"] == 0
    assert queue.snapshot()["pending_cleanup"] == 1
    assert "cleanup deferred: PermissionError" in caplog.text
    queue.close()
    assert attempts == 2
    assert queue.snapshot()["pending_cleanup"] == 0


def test_open_queue_can_spill_again_while_an_older_owned_directory_awaits_cleanup(monkeypatch):
    queue = TranslationQueue(memory_limit=1)
    queue.put("system", job("a"))
    queue.put("system", job("b"))
    old_directory = queue._spool_directory
    original_cleanup = old_directory.cleanup
    monkeypatch.setattr(old_directory, "cleanup", Mock(side_effect=PermissionError("busy")))
    assert [item.segment_id for item in queue.get_batch()] == ["a", "b"]
    assert queue.snapshot()["pending_cleanup"] == 1
    queue.put("system", job("c"))
    queue.put("system", job("d"))
    assert queue._spool_directory.name != old_directory.name
    assert [item.segment_id for item in queue.get_batch()] == ["c", "d"]
    assert queue.snapshot()["pending_cleanup"] == 1
    monkeypatch.setattr(old_directory, "cleanup", original_cleanup)
    queue.close()
    assert queue.snapshot()["pending_cleanup"] == 0


def test_final_dequeue_delivers_batch_when_sqlite_close_needs_retry():
    queue = TranslationQueue(memory_limit=1)
    queue.put("system", job("a"))
    queue.put("system", job("b"))
    connection = queue._connection

    class CloseOnceFailure:
        attempts = 0

        def __getattr__(self, name):
            return getattr(connection, name)

        def __enter__(self):
            return connection.__enter__()

        def __exit__(self, *args):
            return connection.__exit__(*args)

        def close(self):
            self.attempts += 1
            if self.attempts == 1:
                raise sqlite3.OperationalError("close temporarily failed")
            connection.close()

    queue._connection = CloseOnceFailure()
    spool = queue.spool_path
    assert [item.segment_id for item in queue.get_batch()] == ["a", "b"]
    assert queue.snapshot()["pending_cleanup"] == 1
    queue.close()
    assert queue.snapshot()["pending_cleanup"] == 0
    assert not spool.parent.exists()
