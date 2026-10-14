"""Fair text scheduling without making recognition wait for translation capacity."""
from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


@dataclass
class _Entry:
    sequence: int
    key: tuple[str, str]
    item: Any
    enqueued_at: float


class _ChannelView:
    """Expose pending counts for the previous queue's truthiness checks."""

    def __init__(self, owner: TranslationQueue, channel: str):
        self._owner = owner
        self._channel = channel

    def __len__(self) -> int:
        with self._owner._condition:
            return self._owner._counts[self._channel]

    def __bool__(self) -> bool:
        return len(self) != 0


class TranslationQueue:
    """Alternate sources, replacing only revisions that have not started.

    ``put`` never waits for capacity. Overflow is saved in an owned temporary
    SQLite database; errors are raised instead of silently losing sentences.
    ``job_factory`` accepts dataclass fields as keyword arguments and avoids a
    dependency on the pipeline's TranslationJob class.
    """

    def __init__(self, memory_limit: int = 64,
                 job_factory: Callable[..., Any] | None = None):
        if memory_limit < 1:
            raise ValueError("memory_limit must be positive")
        self.memory_limit = memory_limit
        self._job_factory = job_factory
        self._job_type: type | None = None
        self._condition = threading.Condition(threading.RLock())
        self._memory: dict[str, deque[_Entry]] = {"mic": deque(), "system": deque()}
        self._memory_index: dict[tuple[str, str], _Entry] = {}
        self._counts = {"mic": 0, "system": 0}
        self._spilled_counts = {"mic": 0, "system": 0}
        self._queues = {channel: _ChannelView(self, channel) for channel in self._counts}
        self._last = "system"
        self._sequence = 0
        self._closed = False
        self._coalesced = 0
        self._stale_revisions = 0
        self._spilled_total = 0
        self._spool_directory: tempfile.TemporaryDirectory | None = None
        self._connection: sqlite3.Connection | None = None
        self._pending_cleanup: list[tempfile.TemporaryDirectory] = []
        self._connection_cleanup_pending = False

    @property
    def spool_path(self) -> Path | None:
        """The program-owned spool location, or None when no file is needed."""
        with self._condition:
            if self._spool_directory is None:
                return None
            return Path(self._spool_directory.name) / "translations.sqlite3"

    def put(self, channel: str, item: Any, seconds: float = 0.0) -> bool:
        # ``seconds`` is accepted for FairQueue callers, but text throughput is
        # not meaningfully bounded by the duration of the original audio.
        del seconds
        with self._condition:
            if self._closed:
                return False
            if channel not in self._counts:
                raise ValueError(f"Unknown translation channel: {channel}")
            if not is_dataclass(item) or isinstance(item, type):
                raise TypeError("Translation jobs must be dataclass instances")
            if item.channel != channel:
                raise ValueError("Translation job channel differs from queue channel")
            if self._job_factory is None:
                if self._job_type is not None and type(item) is not self._job_type:
                    raise TypeError("Translation jobs must share one dataclass type")
                self._job_type = type(item)
            key = (channel, item.segment_id)
            revision = int(item.revision)
            existing = self._memory_index.get(key)
            if existing is not None:
                if revision > int(existing.item.revision):
                    existing.item = item
                else:
                    self._stale_revisions += 1
                self._coalesced += 1
                return True
            stored = None
            if self._spilled_counts[channel]:
                assert self._connection is not None
                stored = self._connection.execute(
                    "SELECT revision FROM jobs WHERE channel = ? AND segment_id = ?", key).fetchone()
            if stored is not None:
                if revision > stored[0]:
                    payload = self._serialize(item)
                    with self._connection:
                        self._connection.execute(
                            "UPDATE jobs SET revision = ?, payload = ? WHERE channel = ? AND segment_id = ?",
                            (revision, payload, *key))
                else:
                    self._stale_revisions += 1
                self._coalesced += 1
                return True

            sequence = self._sequence + 1
            enqueued_at = time.monotonic()
            if len(self._memory_index) < self.memory_limit:
                entry = _Entry(sequence, key, item, enqueued_at)
                self._memory[channel].append(entry)
                self._memory_index[key] = entry
            else:
                payload = self._serialize(item)
                self._ensure_spool()
                assert self._connection is not None
                with self._connection:
                    self._connection.execute(
                        "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?)",
                        (sequence, channel, item.segment_id, revision, enqueued_at, payload))
                self._spilled_counts[channel] += 1
                self._spilled_total += 1
            self._sequence = sequence
            self._counts[channel] += 1
            # Channel-bound consumers must all wake: notify() can select the
            # other source's idle worker and leave the owner asleep.
            self._condition.notify_all()
            return True

    def get(self, timeout: float | None = None, *, channel: str | None = None) -> Any | None:
        with self._condition:
            channel = self._wait_for_channel(timeout, channel)
            if channel is None:
                return None
            item = self._pop(channel)
            self._last = channel
            self._clean_if_drained()
            return item

    def get_batch(self, limit: int = 3, max_characters: int = 1200,
                  timeout: float | None = None, *, channel: str | None = None) -> list[Any]:
        """Take already available consecutive jobs from one source/direction.

        The first job is always returned even if it exceeds the character
        limit, so a long sentence cannot permanently block the queue.
        """
        if limit < 1 or max_characters < 1:
            raise ValueError("Batch limits must be positive")
        with self._condition:
            channel = self._wait_for_channel(timeout, channel)
            if channel is None:
                return []
            candidates = [(entry.sequence, entry) for entry in islice(self._memory[channel], limit)]
            if self._spilled_counts[channel]:
                assert self._connection is not None
                candidates.extend(self._connection.execute(
                    "SELECT sequence, payload FROM jobs WHERE channel = ? ORDER BY sequence LIMIT ?",
                    (channel, limit)).fetchall())
            selected: list[tuple[int, Any, _Entry | None]] = []
            characters = 0
            direction = None
            generation = None
            source_identity = None
            for sequence, candidate in sorted(candidates, key=lambda value: value[0]):
                entry = candidate if isinstance(candidate, _Entry) else None
                item = entry.item if entry is not None else self._deserialize(candidate)
                isolated = item.provisional or item.revision > 1
                if selected and (tuple(item.direction) != direction
                                 or getattr(item, "context_generation", 0) != generation or isolated
                                 or (getattr(item, "source_room", ""),
                                     getattr(item, "source_generation", 0)) != source_identity
                                 or characters + len(item.text) > max_characters):
                    break
                selected.append((sequence, item, entry))
                characters += len(item.text)
                direction = tuple(item.direction)
                generation = getattr(item, "context_generation", 0)
                source_identity = (getattr(item, "source_room", ""), getattr(item, "source_generation", 0))
                if len(selected) == limit or isolated:
                    break
            # Stage and decode everything before changing either storage tier.
            # A malformed record or SQLite error must not lose an earlier job.
            spilled = [(sequence,) for sequence, _, entry in selected if entry is None]
            if spilled:
                assert self._connection is not None
                with self._connection:
                    self._connection.executemany("DELETE FROM jobs WHERE sequence = ?", spilled)
            for _, _, entry in selected:
                if entry is not None:
                    self._memory[channel].popleft()
                    del self._memory_index[entry.key]
                else:
                    self._spilled_counts[channel] -= 1
                self._counts[channel] -= 1
            self._last = channel
            self._clean_if_drained()
            return [item for _, item, _ in selected]

    def close(self, discard: bool = False) -> None:
        with self._condition:
            self._closed = True
            if discard:
                for queue in self._memory.values():
                    queue.clear()
                self._memory_index.clear()
                for channel in self._counts:
                    self._counts[channel] = self._spilled_counts[channel] = 0
            self._condition.notify_all()
            self._clean_if_drained()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            oldest = [queue[0].enqueued_at for queue in self._memory.values() if queue]
            if any(self._spilled_counts.values()):
                assert self._connection is not None
                value = self._connection.execute("SELECT MIN(enqueued_at) FROM jobs").fetchone()[0]
                if value is not None:
                    oldest.append(value)
            return {"pending": sum(self._counts.values()),
                    "pending_by_channel": dict(self._counts),
                    "oldest_age_seconds": max(0.0, time.monotonic() - min(oldest)) if oldest else 0.0,
                    "coalesced": self._coalesced, "stale_revisions": self._stale_revisions,
                    "spilled": sum(self._spilled_counts.values()),
                    "spilled_total": self._spilled_total,
                    "in_memory": len(self._memory_index), "closed": self._closed,
                    "pending_cleanup": len(self._pending_cleanup) + int(self._connection_cleanup_pending)}

    def _wait_for_channel(self, timeout: float | None, channel: str | None = None) -> str | None:
        if channel is not None and channel not in self._counts:
            raise ValueError(f"Unknown translation channel: {channel}")
        deadline = None if timeout is None else time.monotonic() + timeout
        while not (self._counts[channel] if channel is not None else any(self._counts.values())):
            if self._closed:
                return None
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("No queued work")
            self._condition.wait(remaining)
        if channel is not None:
            return channel
        first = "system" if self._last == "mic" else "mic"
        return first if self._counts[first] else self._last

    def _head(self, channel: str) -> tuple[_Entry | None, tuple | None]:
        memory = self._memory[channel][0] if self._memory[channel] else None
        stored = None
        if self._spilled_counts[channel]:
            assert self._connection is not None
            stored = self._connection.execute(
                "SELECT sequence, payload FROM jobs WHERE channel = ? ORDER BY sequence LIMIT 1",
                (channel,)).fetchone()
        if memory is not None and (stored is None or memory.sequence < stored[0]):
            return memory, None
        return None, stored

    def _pop(self, channel: str) -> Any:
        memory, stored = self._head(channel)
        if memory is not None:
            item = memory.item
            self._memory[channel].popleft()
            del self._memory_index[memory.key]
        else:
            assert stored is not None and self._connection is not None
            item = self._deserialize(stored[1])
            with self._connection:
                self._connection.execute("DELETE FROM jobs WHERE sequence = ?", (stored[0],))
            self._spilled_counts[channel] -= 1
        self._counts[channel] -= 1
        return item

    @staticmethod
    def _serialize(item: Any) -> str:
        return json.dumps(asdict(item), ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    def _deserialize(self, payload: str) -> Any:
        fields = json.loads(payload)
        for name in ("direction", "source_segment_ids"):
            if name in fields:
                fields[name] = tuple(fields[name])
        factory = self._job_factory or self._job_type
        assert factory is not None
        return factory(**fields)

    def _ensure_spool(self) -> None:
        if self._connection is not None:
            return
        directory = tempfile.TemporaryDirectory(prefix="fanyiji-mt-")
        connection = None
        try:
            connection = sqlite3.connect(str(Path(directory.name) / "translations.sqlite3"),
                                         check_same_thread=False)
            connection.execute(
                "CREATE TABLE jobs (sequence INTEGER PRIMARY KEY, channel TEXT NOT NULL, "
                "segment_id TEXT NOT NULL, revision INTEGER NOT NULL, enqueued_at REAL NOT NULL, "
                "payload TEXT NOT NULL, UNIQUE(channel, segment_id))")
            connection.execute("CREATE INDEX channel_sequence ON jobs(channel, sequence)")
            connection.commit()
        except Exception:
            if connection is not None:
                try:
                    connection.close()
                except Exception as cleanup_error:
                    LOGGER.warning("Translation spool setup cleanup failed: %s", type(cleanup_error).__name__)
            try:
                directory.cleanup()
            except Exception as cleanup_error:
                self._pending_cleanup.append(directory)
                LOGGER.warning("Translation spool directory cleanup deferred: %s", type(cleanup_error).__name__)
            raise
        self._connection = connection
        self._spool_directory = directory

    def _clean_if_drained(self) -> None:
        if any(self._counts.values()):
            return
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception as cleanup_error:
                self._connection_cleanup_pending = True
                LOGGER.warning("Translation spool close deferred: %s", type(cleanup_error).__name__)
                return
            self._connection = None
            self._connection_cleanup_pending = False
        if self._spool_directory is not None:
            self._pending_cleanup.append(self._spool_directory)
            self._spool_directory = None
        remaining = []
        for directory in self._pending_cleanup:
            try:
                directory.cleanup()
            except Exception as cleanup_error:
                remaining.append(directory)
                # Cleanup follows dequeue. Report and retry later instead of
                # raising after the caller's sentence has already been taken.
                LOGGER.warning("Translation spool directory cleanup deferred: %s", type(cleanup_error).__name__)
        self._pending_cleanup = remaining
