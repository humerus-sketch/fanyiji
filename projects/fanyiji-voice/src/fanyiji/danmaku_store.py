"""Durable FIFO of final subtitles; browser credentials are never stored here."""
from __future__ import annotations

import hashlib
from contextlib import closing
import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .danmaku_protocol import (valid_danmaku_text, valid_room, MAX_DANMAKU_CHARACTERS,
                              danmaku_weighted_length, danmaku_body_length, MIN_DANMAKU_BODY_CHARACTERS, MAX_DANMAKU_WEIGHTED_LENGTH)
from .translation_output import is_chinese_output

MIN_SEND_CHARACTERS = 4  # Compatibility: two body characters plus the brackets.
SHORT_TEXT_CHARACTERS = MIN_SEND_CHARACTERS - 2
_CJK_ADJACENT_SPACES = re.compile(r"(?<=[^\x00-\x7f]) +| +(?=[^\x00-\x7f])")
_UNSET = object()
_ACTIVE = "state NOT IN ('page_visible','cancelled')"
DEDUP_WINDOW_SECONDS = 10.0


def _page_length(text: str) -> int:
    return sum(2 if ord(char) > 0xffff else 1 for char in text)


def _packet_fits(text: str, maximum: int, characters: int = MAX_DANMAKU_CHARACTERS) -> bool:
    return (_page_length(text) <= characters and danmaku_weighted_length(text) <= maximum
            and (not text[1:-1].strip() or valid_danmaku_text(text, normalized_only=True, max_characters=characters)))


def _pack_body(body: str, maximum: int, characters: int) -> list[str]:
    """Cut at the weighted cap in source order, without borrowing or rebalancing."""
    bodies, current = [], ""
    maximum = min(MAX_DANMAKU_WEIGHTED_LENGTH, maximum)
    for character in body:
        if _packet_fits("【" + current + character + "】", maximum, characters):
            current += character
        else:
            if not current:
                return []
            bodies.append(current)
            current = character
            if not _packet_fits("【" + current + "】", maximum, characters):
                return []
    if current:
        bodies.append(current)
    return ["【" + item + "】" for item in bodies if item.strip()]


def rebalance_short_parts(parts: list[str], max_weighted_length: int = MAX_DANMAKU_WEIGHTED_LENGTH,
                          max_characters: int = MAX_DANMAKU_CHARACTERS) -> list[str]:
    """Compatibility name: only cut each part, never move or merge characters."""
    if not parts or not all(valid_danmaku_text(part, normalized_only=True, max_characters=76) for part in parts):
        return []
    result = []
    for part in parts:
        cut = _pack_body(part[1:-1], max_weighted_length, max_characters)
        if not cut:
            return []
        result.extend(cut)
    return result


def content_digest(text: str) -> str:
    # The splitter removes spaces adjacent to non-ASCII text while holding
    # merges retain a readable space. Both are the same subtitle for dedup;
    # ASCII word spaces and the explicit multi-part newline remain meaningful.
    normalized = _CJK_ADJACENT_SPACES.sub("", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class DanmakuStore:
    def __init__(self, path: Path, wall_clock=None):
        self.path = Path(path)
        self._clock = wall_clock or time.time
        self.max_weighted_length = 80
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room TEXT NOT NULL, channel TEXT NOT NULL, segment_key TEXT NOT NULL,
                revision INTEGER NOT NULL, part INTEGER NOT NULL, text TEXT NOT NULL,
                created_at TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
                request_id TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS contents (
                room TEXT NOT NULL, digest TEXT NOT NULL, segment_key TEXT NOT NULL,
                PRIMARY KEY(room,digest)
            );
            CREATE TABLE IF NOT EXISTS segments (
                room TEXT NOT NULL, segment_key TEXT NOT NULL, revision INTEGER NOT NULL,
                PRIMARY KEY(room,segment_key)
            );
            CREATE TABLE IF NOT EXISTS metrics (
                name TEXT PRIMARY KEY, value INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS failed_handoffs (
                room TEXT NOT NULL, text TEXT NOT NULL, queue_id INTEGER NOT NULL,
                part INTEGER NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
                platform_code INTEGER,
                PRIMARY KEY(room,text,queue_id)
            );
            CREATE INDEX IF NOT EXISTS messages_pending ON messages(id) WHERE state!='page_visible';
            CREATE INDEX IF NOT EXISTS messages_state ON messages(state);
        """)
        self.connection.execute("CREATE INDEX IF NOT EXISTS messages_active ON messages(id) WHERE " + _ACTIVE)
        # A process may have died after POST reached the server. Never blindly
        # resend a durable 'sending' row when the application starts again.
        with self.connection:
            self.connection.execute("UPDATE messages SET state='verify', reason='restart_verify' WHERE state='sending'")
            if self._metadata("dedup_window_v1") is None:
                self._seed_recent_window()
                self._set_metadata("dedup_window_v1", "1")
            if self._metadata("public_digest_normalization") is None:
                self._set_metadata("public_digest_normalization", "1")

    def reset_for_restart(self) -> dict:
        """Start a new run without replaying old posts or recycling candidate IDs."""
        if self.connection.in_transaction:
            raise ValueError("restart_reset_requires_independent_transaction")
        counts = self.counts()
        backup = self.path.with_name("queue-before-last-restart.sqlite3")
        temporary = backup.with_suffix(".tmp")
        # SQLite backup includes committed WAL pages; copying the file does not.
        with closing(sqlite3.connect(temporary)) as destination:
            with destination:
                self.connection.backup(destination)
        os.replace(temporary, backup)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            # A failed filter-file write must not lose its sentence when the
            # runtime FIFO is cleared. Only the explicit handoff marker is
            # evidence of a failure; an interrupted POST/verification is not.
            for row in self.connection.execute(
                    "SELECT * FROM messages WHERE state IN ('blocked','verify') ORDER BY id").fetchall():
                if self.recovery_context(row["id"], row["part"]).get("handoff_pending") is not True:
                    continue
                self.connection.execute(
                    "INSERT OR IGNORE INTO failed_handoffs(room,text,queue_id,part,reason,created_at,platform_code) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (row["room"], row["text"], row["id"], row["part"], row["reason"] or "result_unknown",
                     row["created_at"], self.delivery_code(row["id"])))
            for table in ("messages", "contents", "segments", "metrics", "metadata"):
                self.connection.execute("DELETE FROM " + table)
            self._set_metadata("dedup_window_v1", "1")
            self._set_metadata("public_digest_normalization", "1")
        # sqlite_sequence is deliberately retained so historical failure IDs
        # cannot become associated with a different sentence in the new run.
        return counts

    def failed_handoffs(self) -> list[dict]:
        """Unfinished failure-library writes, never eligible for FIFO sending."""
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM failed_handoffs ORDER BY queue_id,part,rowid").fetchall()]

    def stage_failed_handoff(self, room: str, text: str, *, reason="non_chinese_translation",
                             queue_id: int | None = None, part: int = 0) -> int:
        """Save a rejected translation outside FIFO, before the filter-file write.

        Reserve a normal message identity without inserting a sendable row. The
        sequence survives restart, so this failure cannot attach to a later post.
        """
        if (not valid_room(room) or not isinstance(text, str) or not text.strip() or len(text) > 32768
                or not isinstance(reason, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", reason)
                or type(part) is not int or part < 0
                or (queue_id is not None and (type(queue_id) is not int or queue_id < 1))):
            raise ValueError("invalid_failed_handoff")
        if self.connection.in_transaction:
            raise ValueError("stage_requires_independent_transaction")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            if queue_id is None:
                row = self.connection.execute("SELECT seq FROM sqlite_sequence WHERE name='messages'").fetchone()
                queue_id = (row[0] if row else 0) + 1
                if row:
                    self.connection.execute("UPDATE sqlite_sequence SET seq=? WHERE name='messages'", (queue_id,))
                else:
                    self.connection.execute("INSERT INTO sqlite_sequence(name,seq) VALUES('messages',?)", (queue_id,))
            self.connection.execute(
                "INSERT OR IGNORE INTO failed_handoffs(room,text,queue_id,part,reason,created_at,platform_code) VALUES(?,?,?,?,?,?,NULL)",
                (room, text, queue_id, part, reason, datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")))
        return queue_id

    def acknowledge_failed_handoff(self, room: str, text: str, queue_id: int) -> bool:
        """Remove only the exact failure already saved in the independent library."""
        if (not isinstance(room, str) or not isinstance(text, str)
                or type(queue_id) is not int or queue_id < 1):
            raise ValueError("invalid_failed_handoff_identity")
        with self.connection:
            removed = self.connection.execute(
                "DELETE FROM failed_handoffs WHERE room=? AND text=? AND queue_id=?", (room, text, queue_id))
        return removed.rowcount == 1

    def _metadata(self, name: str) -> str | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    def _set_metadata(self, name: str, value: str):
        self.connection.execute("INSERT INTO metadata VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value", (name, value))

    @staticmethod
    def _timestamp(value: str | None) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and 0 <= number <= 10**12 else None

    def _seen_recently(self, room: str, digest: str, now: float) -> bool:
        timestamp = self._timestamp(self._metadata(f"dedup_seen_at:{room}:{digest}"))
        # A wall-clock rollback does not prematurely release a recent duplicate.
        return timestamp is not None and now - timestamp <= DEDUP_WINDOW_SECONDS

    def _remember(self, room: str, digest: str, segment_key: str, now: float):
        self.connection.execute("INSERT OR IGNORE INTO contents VALUES (?,?,?)", (room, digest, segment_key))
        self._set_metadata(f"dedup_seen_at:{room}:{digest}", repr(float(now)))
        self._set_metadata(f"dedup_seen_key:{room}:{digest}", segment_key)

    def _seed_recent_window(self):
        # Only one bounded lookup upgrades recent pre-window payloads. Never
        # scan the application's unlimited history on every browser restart.
        now = self._clock()
        groups = {}
        for row in self.connection.execute("SELECT * FROM messages ORDER BY id DESC LIMIT 256").fetchall():
            try:
                created = datetime.fromisoformat(row["created_at"]).timestamp()
            except (TypeError, ValueError, OverflowError, OSError):
                continue
            if not 0 <= now - created <= DEDUP_WINDOW_SECONDS:
                continue
            groups.setdefault((row["room"], row["segment_key"]), []).append(row)
        for (room, key), rows in groups.items():
            count = self.connection.execute("SELECT COUNT(*) FROM messages WHERE room=? AND segment_key=?", (room, key)).fetchone()[0]
            if len(rows) != count:
                continue
            rows.sort(key=lambda row: (row["part"], row["id"]))
            digest = content_digest("\n".join(row["text"] for row in rows))
            if self._metadata(f"dedup_seen_at:{room}:{digest}") is None:
                created = max(datetime.fromisoformat(row["created_at"]).timestamp() for row in rows)
                self._remember(room, digest, key, created)

    def enqueue(self, room: str, channel: str, segment_key: str, revision: int,
                translation: str, parts: list[str], *, filter_revision: str = "",
                filter_rule_signatures: list[str] | None = None, revise_unsubmitted: bool = False) -> bool:
        if revise_unsubmitted and channel != "chrome_caption":
            raise ValueError("caption_revision_requires_chrome_channel")
        parts = rebalance_short_parts(parts, self.max_weighted_length)
        # Deduplicate the normalized public payload, including bracket/space
        # normalization, rather than equivalent model formatting variants.
        digest = content_digest("\n".join(parts))
        if not parts:
            return False
        now = self._clock()
        with self.connection:
            if revise_unsubmitted:
                # Recheck POST/frozen state under SQLite's writer lock, including
                # another sender using this same database.
                self.connection.execute("BEGIN IMMEDIATE")
            previous = self.connection.execute(
                "SELECT revision FROM segments WHERE room=? AND segment_key=?", (room, segment_key)).fetchone()
            if previous is not None:
                if not revise_unsubmitted or revision <= previous["revision"]:
                    return False
                return self._revise_caption(room, segment_key, revision, translation, parts,
                    filter_revision, filter_rule_signatures)
            self.connection.execute("INSERT INTO segments VALUES (?,?,?)", (room, segment_key, revision))
            if self._seen_recently(room, digest, now):
                return False
            self._remember(room, digest, segment_key, now)
            created = datetime.fromtimestamp(now, timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")
            # Only the immediately preceding sub-two-character tail may absorb
            # the next utterance. Ordinary short sentences stay independent.
            waiting = self.connection.execute("SELECT * FROM messages WHERE " + _ACTIVE + " ORDER BY id DESC LIMIT 1").fetchone()
            if (waiting is not None and waiting["room"] == room and waiting["channel"] == channel
                    and waiting["state"] == "holding_short" and danmaku_body_length(waiting["text"]) < MIN_DANMAKU_BODY_CHARACTERS
                    and is_chinese_output(waiting["text"]) and is_chinese_output(translation)):
                previous_parts = [waiting["text"], *self.strict_remaining_parts(waiting["id"])]
                combined = _pack_body("".join(packet[1:-1] for packet in previous_parts) + " " +
                    "".join(packet[1:-1] for packet in parts), self.max_weighted_length, MAX_DANMAKU_CHARACTERS)
                if combined:
                    context = self.filter_context(waiting["id"])
                    combined_digest = content_digest("\n".join(combined))
                    if self._seen_recently(room, combined_digest, now):
                        self.connection.execute("UPDATE messages SET state='cancelled',reason='duplicate_skipped' WHERE id=?", (waiting["id"],))
                        self._set_metadata(f"cancelled_parts:{waiting['id']}", json.dumps(combined, ensure_ascii=False))
                        self._set_metadata(f"remaining_parts:{waiting['id']}", "[]")
                        self._save_filter_context(room, context["segment_key"], context["source"] + " " + translation, "", True)
                        return False
                    state = "holding_short" if danmaku_body_length(combined[0]) < MIN_DANMAKU_BODY_CHARACTERS else "pending"
                    self.connection.execute("UPDATE messages SET text=?,state=? WHERE id=?", (combined[0], state, waiting["id"]))
                    self._set_metadata(f"remaining_parts:{waiting['id']}", json.dumps(combined[1:], ensure_ascii=False))
                    self._save_filter_context(room, context["segment_key"], context["source"] + " " + translation, "", context["frozen"])
                    self._save_rule_signatures(room, context["segment_key"], filter_rule_signatures)
                    if revise_unsubmitted and not context["frozen"]:
                        members = self._caption_members(room, context["segment_key"])
                        if members is not None:
                            members.append({"key": segment_key, "revision": revision, "source": translation, "parts": parts})
                            self._save_caption_members(room, context["segment_key"], members)
                    self._remember(room, combined_digest, context["segment_key"], now)
                    return True
            self.connection.executemany(
                "INSERT INTO messages(room,channel,segment_key,revision,part,text,created_at,state) VALUES(?,?,?,?,?,?,?,?)",
                [(room, channel, segment_key, revision, index, part, created,
                  "holding_short" if danmaku_body_length(part) < MIN_DANMAKU_BODY_CHARACTERS else "pending")
                 for index, part in enumerate(parts)])
            self._save_filter_context(room, segment_key, translation, filter_revision, False)
            self._save_rule_signatures(room, segment_key, filter_rule_signatures)
            if revise_unsubmitted:
                self._save_caption_members(room, segment_key,
                    [{"key": segment_key, "revision": revision, "source": translation, "parts": parts}])
        return True

    def _caption_members(self, room: str, owner: str) -> list[dict] | None:
        encoded = self._metadata(f"caption_members:{room}:{owner}")
        if encoded is None:
            return None
        try:
            members = json.loads(encoded)
        except (TypeError, ValueError):
            raise ValueError("invalid_caption_members") from None
        if (not isinstance(members, list) or not 1 <= len(members) <= 4096
                or any(not isinstance(member, dict) or set(member) != {"key", "revision", "source", "parts"}
                    or not isinstance(member["key"], str) or not member["key"]
                    or type(member["revision"]) is not int or member["revision"] < 1
                    or not isinstance(member["source"], str) or not is_chinese_output(member["source"])
                    or not isinstance(member["parts"], list) or not member["parts"]
                    or not all(is_chinese_output(part) and valid_danmaku_text(part, normalized_only=True)
                               for part in member["parts"]) for member in members)
                or len({member["key"] for member in members}) != len(members)):
            raise ValueError("invalid_caption_members")
        return members

    def _save_caption_members(self, room: str, owner: str, members: list[dict]):
        self._set_metadata(f"caption_members:{room}:{owner}", json.dumps(members, ensure_ascii=False))
        for member in members:
            self._set_metadata(f"caption_owner:{room}:{member['key']}", owner)

    def caption_enqueue_state(self, room: str, segment_key: str) -> dict | None:
        segment = self.connection.execute("SELECT revision FROM segments WHERE room=? AND segment_key=?",
                                          (room, segment_key)).fetchone()
        if segment is None:
            return None
        owner = self._metadata(f"caption_owner:{room}:{segment_key}")
        rows = self.connection.execute("SELECT * FROM messages WHERE room=? AND segment_key=? ORDER BY id",
                                       (room, owner)).fetchall() if owner else []
        mutable = bool(rows and not self.filter_context(rows[0]["id"])["frozen"]
            and all(row["state"] in {"pending", "holding_short"} or (
                row["state"] == "cancelled" and row["reason"] == "filter_superseded") for row in rows))
        return {"revision": segment["revision"], "owner": owner, "mutable": mutable,
                "head_id": rows[0]["id"] if rows else None}

    def caption_member_source(self, room: str, segment_key: str) -> str | None:
        """Read this member's original source without conflating a merged group."""
        state = self.caption_enqueue_state(room, segment_key)
        if not state or not state["mutable"]:
            return None
        members = self._caption_members(room, state["owner"])
        return next((member["source"] for member in members or [] if member["key"] == segment_key), None)

    def _revise_caption(self, room, key, revision, translation, parts, filter_revision, signatures):
        state = self.caption_enqueue_state(room, key)
        if not state or not state["mutable"]:
            return False
        members = self._caption_members(room, state["owner"])
        if members is None or not any(member["key"] == key for member in members):
            return False
        for member in members:
            if member["key"] == key:
                member.update(revision=revision, source=translation, parts=parts)
        combined = _pack_body(" ".join("".join(part[1:-1] for part in member["parts"]) for member in members),
                              self.max_weighted_length, MAX_DANMAKU_CHARACTERS)
        if not combined:
            return False
        # A revised one-character tail can consume only its already queued
        # immediate successor; ordinary two-character sentences never merge.
        while danmaku_body_length(combined[-1]) < MIN_DANMAKU_BODY_CHARACTERS:
            following = self.connection.execute("SELECT * FROM messages WHERE " + _ACTIVE +
                " AND id>? AND NOT (room=? AND segment_key=?) ORDER BY id LIMIT 1",
                (state["head_id"], room, state["owner"])).fetchone()
            if (following is None or following["room"] != room or following["channel"] != "chrome_caption"
                    or following["state"] not in {"pending", "holding_short"}):
                break
            next_state = self.caption_enqueue_state(room, following["segment_key"])
            next_members = self._caption_members(room, next_state["owner"]) if next_state and next_state["mutable"] else None
            if not next_members or next_state["head_id"] != following["id"] or len(members) + len(next_members) > 4096:
                break
            next_body = " ".join("".join(packet[1:-1] for packet in member["parts"]) for member in next_members)
            continuation = _pack_body(combined[-1][1:-1] + " " + next_body,
                                      self.max_weighted_length, MAX_DANMAKU_CHARACTERS)
            if not continuation:
                break
            combined = [*combined[:-1], *continuation]
            members.extend(next_members)
            self.connection.execute("UPDATE messages SET state='cancelled',reason='caption_merged' WHERE room=? AND segment_key=? AND state IN ('pending','holding_short')",
                                    (room, next_state["owner"]))
        digest = content_digest("\n".join(combined))
        duplicate_owner = self._metadata(f"dedup_seen_key:{room}:{digest}")
        if duplicate_owner is None:
            recorded = self.connection.execute("SELECT segment_key FROM contents WHERE room=? AND digest=?",
                                               (room, digest)).fetchone()
            duplicate_owner = recorded["segment_key"] if recorded else None
        duplicate = (duplicate_owner not in {None, state["owner"]}
                     and self._seen_recently(room, digest, self._clock()))
        # Keep the first row's identity and continuations so following utterances
        # cannot overtake a growing/shrinking revised caption.
        identifier = state["head_id"]
        first_state = "cancelled" if duplicate else (
            "holding_short" if danmaku_body_length(combined[0]) < MIN_DANMAKU_BODY_CHARACTERS else "pending")
        self.connection.execute("UPDATE messages SET text=?,state=?,request_id='',reason='caption_revised',part=0 WHERE id=? AND state IN ('pending','holding_short')",
                                (combined[0], first_state, identifier))
        self.connection.execute("UPDATE messages SET state='cancelled',reason='filter_superseded' WHERE room=? AND segment_key=? AND id!=? AND state IN ('pending','holding_short')",
                                (room, state["owner"], identifier))
        self._set_metadata(f"remaining_parts:{identifier}", json.dumps(combined[1:], ensure_ascii=False))
        self._set_metadata(f"delivery_code:{identifier}", "none")
        self._set_metadata(f"delivery_baseline:{identifier}", "none")
        self._save_rule_signatures(room, state["owner"], signatures)
        if not duplicate:
            self._remember(room, digest, state["owner"], self._clock())
        self.connection.execute("UPDATE segments SET revision=? WHERE room=? AND segment_key=?", (revision, room, key))
        self.connection.execute("UPDATE messages SET revision=? WHERE room=? AND segment_key=? AND state IN ('pending','holding_short')",
                                (max(member["revision"] for member in members), room, state["owner"]))
        self._save_filter_context(room, state["owner"], " ".join(member["source"] for member in members),
                                  filter_revision if len(members) == 1 else "", duplicate)
        self._save_caption_members(room, state["owner"], members)
        return not duplicate

    @staticmethod
    def _valid_rule_signatures(value):
        return (isinstance(value, list) and len(value) <= 8192
                and all(isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) for item in value))

    def _save_rule_signatures(self, room, segment_key, signatures):
        if signatures is not None:
            if not self._valid_rule_signatures(signatures):
                raise ValueError("invalid_filter_rule_signatures")
            self._set_metadata(f"filter_rules:{room}:{segment_key}", json.dumps(sorted(set(signatures))))

    def applied_rule_signatures(self, identifier: int) -> list[str] | None:
        context = self.filter_context(identifier)
        value = self._metadata(f"f_filter_rules:{identifier}")
        if value is None:
            value = self._metadata(f"filter_rules:{context['room']}:{context['segment_key']}")
        if value is None:
            # Pre-filter legacy packets have no rule signatures to exclude.
            return [] if not context["revision"] else None
        try:
            signatures = json.loads(value)
        except (TypeError, ValueError):
            raise ValueError("invalid_filter_rule_signatures") from None
        if not self._valid_rule_signatures(signatures):
            raise ValueError("invalid_filter_rule_signatures")
        return signatures

    def _save_filter_context(self, room, segment_key, source, revision, frozen):
        self._set_metadata(f"content_filter:{room}:{segment_key}", json.dumps(
            {"source": source, "revision": revision, "frozen": bool(frozen)}, ensure_ascii=False))

    def filter_context(self, identifier: int) -> dict:
        row = self.connection.execute("SELECT room,segment_key FROM messages WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("missing_filter_head")
        rows = self.connection.execute("SELECT * FROM messages WHERE room=? AND segment_key=? ORDER BY id",
                                       (row["room"], row["segment_key"])).fetchall()
        value = self._metadata(f"content_filter:{row['room']}:{row['segment_key']}")
        if value is None:
            source = " ".join(packet[1:-1] for part in rows for packet in
                              [part["text"], *self.remaining_parts(part["id"])])
            stored = {"source": source, "revision": "", "frozen": False}
        else:
            try:
                stored = json.loads(value)
            except (TypeError, ValueError):
                raise ValueError("invalid_filter_context") from None
            if (not isinstance(stored, dict) or not isinstance(stored.get("source"), str)
                    or not isinstance(stored.get("revision"), str) or type(stored.get("frozen")) is not bool):
                raise ValueError("invalid_filter_context")
        # Historical delivered/cancelled packets prevent a whole original
        # utterance from being regenerated after partial submission/clearing.
        frozen = stored["frozen"] or any(part["state"] not in {"pending", "holding_short"}
            and not (part["state"] == "cancelled" and part["reason"] == "filter_superseded") for part in rows)
        return {**stored, "frozen": frozen, "room": row["room"], "segment_key": row["segment_key"],
                "head_id": rows[0]["id"]}

    def set_filter_frozen(self, identifier: int, frozen: bool):
        context = self.filter_context(identifier)
        with self.connection:
            self._save_filter_context(context["room"], context["segment_key"], context["source"], context["revision"], frozen)

    def unsubmitted_filter_heads(self) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM messages WHERE state IN ('pending','holding_short') ORDER BY id").fetchall()
        result, seen = [], set()
        for row in rows:
            group = (row["room"], row["segment_key"])
            if group not in seen:
                seen.add(group)
                result.append(dict(row))
        return result

    def prepare_filtered_parts(self, identifier: int, parts: list[str], revision: str,
                               rule_signatures: list[str] | None = None) -> bool:
        """Regenerate only an entire utterance whose first POST has not started."""
        parts = rebalance_short_parts(parts, self.max_weighted_length)
        if not parts or not all(valid_danmaku_text(part, normalized_only=True) for part in parts):
            return False
        context = self.filter_context(identifier)
        if context["frozen"] or context["head_id"] != identifier:
            return False
        with self.connection:
            state = "holding_short" if danmaku_body_length(parts[0]) < MIN_DANMAKU_BODY_CHARACTERS else "pending"
            self.connection.execute("UPDATE messages SET text=?,state=?,request_id='',reason='filter_adjusted',part=0 WHERE id=? AND state IN ('pending','holding_short')",
                                    (parts[0], state, identifier))
            self.connection.execute("UPDATE messages SET state='cancelled',reason='filter_superseded' WHERE room=? AND segment_key=? AND id!=? AND state IN ('pending','holding_short')",
                                    (context["room"], context["segment_key"], identifier))
            self._set_metadata(f"remaining_parts:{identifier}", json.dumps(parts[1:], ensure_ascii=False))
            self._set_metadata(f"delivery_code:{identifier}", "none")
            self._set_metadata(f"delivery_baseline:{identifier}", "none")
            self._save_filter_context(context["room"], context["segment_key"], context["source"], revision, False)
            self._save_rule_signatures(context["room"], context["segment_key"], rule_signatures)
            self._remember(context["room"], content_digest("\n".join(parts)), context["segment_key"], self._clock())
        return True

    def strict_remaining_parts(self, identifier: int) -> list[str]:
        value = self._metadata(f"remaining_parts:{identifier}")
        if value is None:
            return []
        try:
            parts = json.loads(value)
        except (TypeError, ValueError):
            raise ValueError("invalid_remaining_parts") from None
        if (not isinstance(parts, list) or len(parts) > 4096
                or not all(valid_danmaku_text(part, max_characters=76) for part in parts)):
            raise ValueError("invalid_remaining_parts")
        return parts

    def repair_f_rejected_head(self, expected: dict, expected_remaining: list[str],
                               baseline: int, parts: list[str], revision: str,
                               rule_signatures: list[str], *, recovery_context: dict | None = None) -> bool:
        """Apply user-confirmed rules only to this f head and its unsent tail."""
        if not parts or not all(valid_danmaku_text(part, normalized_only=True) for part in parts):
            return False
        parts = rebalance_short_parts(parts, self.max_weighted_length)
        if (type(baseline) is not int or not 0 <= baseline <= 10000 or not parts
                or len(parts) > 4096 or not self._valid_rule_signatures(rule_signatures)
                or not all(valid_danmaku_text(part, normalized_only=True) for part in parts)):
            return False
        if self.connection.in_transaction:
            raise ValueError("repair_requires_independent_transaction")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            current = self.head()
            if (current is None or any(current[key] != expected[key] for key in
                    ("id", "room", "text", "part", "request_id"))
                    or current["state"] != "verify" or current["reason"] != "platform_f"
                    or current["platform_code"] != 0 or not self.f_recovery(current["id"])["active"]
                    or self.delivery_baseline(current["id"]) != baseline
                    or self.strict_remaining_parts(current["id"]) != expected_remaining):
                self.connection.rollback()
                return False
            identifier = current["id"]
            recovery = self.f_recovery(identifier)
            recovery["active"] = False
            state = "holding_short" if danmaku_body_length(parts[0]) < MIN_DANMAKU_BODY_CHARACTERS else "pending"
            self.connection.execute("UPDATE messages SET text=?,state=?,request_id='',reason='f_head_rule_repaired' WHERE id=?",
                                    (parts[0], state, identifier))
            self._set_metadata(f"remaining_parts:{identifier}", json.dumps(parts[1:], ensure_ascii=False))
            self._set_metadata(f"delivery_code:{identifier}", "none")
            self._set_metadata(f"delivery_baseline:{identifier}", "none")
            self._set_metadata(f"f_recovery:{identifier}", json.dumps(recovery))
            # All rewritten continuation packets share these signatures; a
            # later unrelated rule edit must not cascade A -> B -> C.
            self._set_metadata(f"f_filter_rules:{identifier}", json.dumps(sorted(set(rule_signatures))))
            self._set_metadata(f"f_filter_repair:{identifier}", json.dumps({
                "revision": revision, "part": current["part"],
                "source_digest": content_digest(current["text"])}))
            if recovery_context is not None:
                encoded = json.dumps(recovery_context, ensure_ascii=False, allow_nan=False)
                if not isinstance(recovery_context, dict) or len(encoded.encode("utf-8")) > 65536:
                    raise ValueError("invalid_recovery_context")
                self._set_metadata(f"auto_recovery:{identifier}:{current['part']}", encoded)
            context = self.filter_context(identifier)
            self._save_filter_context(context["room"], context["segment_key"],
                                      context["source"], context["revision"], True)
            self._remember(current["room"], content_digest("\n".join(parts)), current["segment_key"], self._clock())
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return True

    def head(self) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM messages WHERE " + _ACTIVE + " ORDER BY id LIMIT 1").fetchone()
        if row is None:
            return None
        result = dict(row)
        result["platform_code"] = self.delivery_code(result["id"])
        return result

    def recovery_context(self, identifier: int, part: int) -> dict:
        try:
            value = json.loads(self._metadata(f"auto_recovery:{identifier}:{part}") or "{}")
        except (ValueError, TypeError):
            value = {}
        return value if isinstance(value, dict) else {}

    def save_recovery_context(self, identifier: int, part: int, value: dict):
        if type(identifier) is not int or identifier < 1 or type(part) is not int or part < 0 or not isinstance(value, dict):
            raise ValueError("invalid_recovery_context")
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > 65536:
            raise ValueError("recovery_context_too_large")
        with self.connection:
            self._set_metadata(f"auto_recovery:{identifier}:{part}", encoded)

    def delivery_code(self, identifier: int) -> int | None:
        value = self._metadata(f"delivery_code:{identifier}")
        if not isinstance(value, str) or re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value) is None or len(value) > 12:
            return None
        code = int(value)
        return code if -2**31 <= code < 2**31 else None

    def delivery_baseline(self, identifier: int) -> int | None:
        value = self._metadata(f"delivery_baseline:{identifier}")
        if not isinstance(value, str) or re.fullmatch(r"(?:0|[1-9][0-9]{0,4})", value) is None:
            return None
        count = int(value)
        return count if count <= 10000 else None

    def set_delivery_baseline(self, identifier: int, count):
        valid = type(count) is int and 0 <= count <= 10000
        with self.connection:
            self._set_metadata(f"delivery_baseline:{identifier}", str(count) if valid else "none")

    def f_recovery(self, identifier: int) -> dict:
        """Persist retry eligibility separately from read-only presence checks."""
        try:
            value = json.loads(self._metadata(f"f_recovery:{identifier}") or "{}")
        except (TypeError, ValueError):
            value = {}
        if not isinstance(value, dict):
            value = {}
        retries = value.get("retries", 0)
        last_f_at = self._timestamp(value.get("last_f_at"))
        return {"active": value.get("active") is True,
                "retries": retries if type(retries) is int and 0 <= retries <= 10**12 else 0,
                "last_f_at": last_f_at if last_f_at is not None and last_f_at <= 32503680000 else None}

    def has_f_recovery(self, identifier: int) -> bool:
        # An inactive record proves this version deliberately withdrew retry
        # eligibility. Old log lines must never resurrect it after a new POST.
        return self._metadata(f"f_recovery:{identifier}") is not None

    def activate_f_recovery(self, identifier: int, *, new_f=False):
        with self.connection:
            value = self.f_recovery(identifier)
            value["active"] = True
            if new_f:
                value["last_f_at"] = self._clock()
            self._set_metadata(f"f_recovery:{identifier}", json.dumps(value))

    def stop_f_recovery(self, identifier: int):
        with self.connection:
            value = self.f_recovery(identifier)
            value["active"] = False
            self._set_metadata(f"f_recovery:{identifier}", json.dumps(value))

    def begin_f_retry(self, identifier: int, request_id: str) -> int:
        # Clear eligibility before POST: an ambiguous result of this new POST
        # must not inherit permission to resend from the previous explicit f.
        with self.connection:
            value = self.f_recovery(identifier)
            value.update(active=False, retries=value["retries"] + 1)
            self._set_metadata(f"f_recovery:{identifier}", json.dumps(value))
            self.connection.execute("UPDATE messages SET state='sending',request_id=?,reason='' WHERE id=?", (request_id, identifier))
            self._set_metadata(f"delivery_code:{identifier}", "none")
            self._set_metadata(f"delivery_baseline:{identifier}", "none")
            self.connection.execute("INSERT INTO metrics(name,value) VALUES('f_retries',1) ON CONFLICT(name) DO UPDATE SET value=value+1")
        return value["retries"]

    def update(self, identifier: int, state: str, *, request_id: str = "", reason: str = "", platform_code=_UNSET):
        with self.connection:
            if platform_code is _UNSET:
                self.connection.execute("UPDATE messages SET state=?,request_id=?,reason=? WHERE id=?",
                                        (state, request_id, reason, identifier))
            else:
                code = platform_code if type(platform_code) is int and -2**31 <= platform_code < 2**31 else None
                self.connection.execute("UPDATE messages SET state=?,request_id=?,reason=? WHERE id=?",
                                        (state, request_id, reason, identifier))
                self._set_metadata(f"delivery_code:{identifier}", str(code) if code is not None else "none")

    def remaining_parts(self, identifier: int) -> list[str]:
        value = self._metadata(f"remaining_parts:{identifier}")
        if value is None:
            return []
        try:
            parts = json.loads(value)
        except (ValueError, TypeError):
            return []
        return parts if isinstance(parts, list) and len(parts) <= 4096 and all(
            valid_danmaku_text(part, max_characters=76) for part in parts) else []

    def release_holding(self, identifier: int):
        with self.connection:
            row = self.connection.execute("SELECT text FROM messages WHERE id=? AND state='holding_short'", (identifier,)).fetchone()
            if row is not None and danmaku_body_length(row["text"]) >= MIN_DANMAKU_BODY_CHARACTERS:
                self.connection.execute("UPDATE messages SET state='pending' WHERE id=? AND state='holding_short'", (identifier,))

    def prepare_parts(self, identifier: int, parts: list[str]) -> bool:
        """Repacketize only a proven unsubmitted/length-rejected FIFO head."""
        parts = rebalance_short_parts(parts, self.max_weighted_length)
        if not parts or not all(valid_danmaku_text(part, normalized_only=True) for part in parts):
            return False
        with self.connection:
            row = self.connection.execute("SELECT state,reason FROM messages WHERE id=?", (identifier,)).fetchone()
            if row is None or not (row["state"] == "pending" or (
                    row["state"] == "blocked" and row["reason"] == "platform_rejected"
                    and self.delivery_code(identifier) == 1003212)):
                return False
            state = "holding_short" if danmaku_body_length(parts[0]) < MIN_DANMAKU_BODY_CHARACTERS else "pending"
            self.connection.execute("UPDATE messages SET text=?,state=?,request_id='',reason='length_adjusted' WHERE id=?",
                                    (parts[0], state, identifier))
            self._set_metadata(f"remaining_parts:{identifier}", json.dumps(parts[1:], ensure_ascii=False))
            self._set_metadata(f"delivery_code:{identifier}", "none")
            self._set_metadata(f"delivery_baseline:{identifier}", "none")
        return True

    def mark_page_visible(self, identifier: int, request_id: str):
        """Advance this same FIFO ID; archive only the already verified packet."""
        with self.connection:
            row = self.connection.execute("SELECT * FROM messages WHERE id=?", (identifier,)).fetchone()
            if row is None or row["state"] in {"page_visible", "cancelled"}:
                return
            recovery = self.f_recovery(identifier)
            recovery["active"] = False
            self._set_metadata(f"f_recovery:{identifier}", json.dumps(recovery))
            remaining = self.remaining_parts(identifier)
            if remaining:
                cursor = self.connection.execute(
                    "INSERT INTO messages(room,channel,segment_key,revision,part,text,created_at,state,request_id,reason) VALUES(?,?,?,?,?,?,?,'page_visible',?,'')",
                    (row["room"], row["channel"], row["segment_key"], row["revision"], row["part"], row["text"], row["created_at"], request_id))
                self._set_metadata(f"delivery_code:{cursor.lastrowid}", self._metadata(f"delivery_code:{identifier}") or "none")
                state = "holding_short" if danmaku_body_length(remaining[0]) < MIN_DANMAKU_BODY_CHARACTERS else "pending"
                self.connection.execute("UPDATE messages SET text=?,state=?,request_id='',reason='',part=part+1 WHERE id=?",
                                        (remaining[0], state, identifier))
                self._set_metadata(f"remaining_parts:{identifier}", json.dumps(remaining[1:], ensure_ascii=False))
                self._set_metadata(f"delivery_code:{identifier}", "none")
                self._set_metadata(f"delivery_baseline:{identifier}", "none")
                self._set_metadata(f"f_recovery:{identifier}", "{}")
            else:
                self.connection.execute("UPDATE messages SET state='page_visible',request_id=?,reason='' WHERE id=?",
                                        (request_id, identifier))

    def defer_failed_head(self, expected: dict) -> bool:
        """Move a failed packet out of FIFO without claiming it was delivered.

        The caller must first persist its exact room/text/queue_id candidate in
        the independent filter store. Saving the candidate before advancing this
        transaction keeps a crash from losing the failed sentence. A stale
        callback must never remove a replacement packet or an unsent successor.
        """
        keys = ("id", "room", "text", "part", "request_id", "state", "reason", "platform_code")
        if (not isinstance(expected, dict) or not all(key in expected for key in keys)
                or type(expected["id"]) is not int or expected["id"] < 1
                or type(expected["part"]) is not int or expected["part"] < 0
                or not all(isinstance(expected[key], str) for key in
                           ("room", "text", "request_id", "state", "reason"))
                or (expected["platform_code"] is not None and type(expected["platform_code"]) is not int)
                or expected["state"] not in {"verify", "blocked"}):
            return False
        if self.connection.in_transaction:
            raise ValueError("defer_requires_independent_transaction")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            current = self.head()
            if current is None or any(current[key] != expected[key] for key in keys):
                self.connection.rollback()
                return False
            identifier = current["id"]
            # Corrupt continuation metadata must fail closed, not silently lose
            # the untranslated remainder while removing the failed head.
            remaining = self.strict_remaining_parts(identifier)
            context = self.filter_context(identifier)
            if remaining:
                self.connection.execute(
                    "INSERT INTO messages(room,channel,segment_key,revision,part,text,created_at,state,request_id,reason) VALUES(?,?,?,?,?,?,?,'cancelled',?,'deferred_analysis')",
                    (current["room"], current["channel"], current["segment_key"], current["revision"],
                     current["part"], current["text"], current["created_at"], current["request_id"]))
                state = "holding_short" if danmaku_body_length(remaining[0]) < MIN_DANMAKU_BODY_CHARACTERS else "pending"
                self.connection.execute(
                    "UPDATE messages SET text=?,state=?,request_id='',reason='',part=part+1 WHERE id=?",
                    (remaining[0], state, identifier))
                self._set_metadata(f"remaining_parts:{identifier}", json.dumps(remaining[1:], ensure_ascii=False))
            else:
                self.connection.execute(
                    "UPDATE messages SET state='cancelled',request_id='',reason='deferred_analysis' WHERE id=?",
                    (identifier,))
            self._set_metadata(f"delivery_code:{identifier}", "none")
            self._set_metadata(f"delivery_baseline:{identifier}", "none")
            self._set_metadata(f"f_recovery:{identifier}", "{}")
            self._save_filter_context(context["room"], context["segment_key"],
                                      context["source"], context["revision"], True)
            self.connection.execute(
                "INSERT INTO metrics(name,value) VALUES('deferred_analysis',1) ON CONFLICT(name) DO UPDATE SET value=value+1")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return True

    def _active_continuations(self) -> dict[int, list[str]]:
        identifiers = self.connection.execute("SELECT id FROM messages WHERE " + _ACTIVE).fetchall()
        return {row["id"]: parts for row in identifiers if (parts := self.remaining_parts(row["id"]))}

    def counts(self) -> dict:
        rows = self.connection.execute("SELECT state,COUNT(*) count FROM messages GROUP BY state").fetchall()
        result = {row["state"]: row["count"] for row in rows}
        result["queued"] = sum(value for key, value in result.items() if key not in {"page_visible", "cancelled"})
        result["queued"] += sum(map(len, self._active_continuations().values()))
        result["page_visible"] = result.get("page_visible", 0)
        result["holding"] = result.get("holding_short", 0)
        cleared = self._metadata("cleared_count")
        result["cleared"] = int(cleared) if isinstance(cleared, str) and cleared.isascii() and cleared.isdigit() else 0
        for name in ("accepted", "failures", "f_retries", "deferred_analysis"):
            value = self.connection.execute("SELECT value FROM metrics WHERE name=?", (name,)).fetchone()
            result[name] = value[0] if value is not None else 0
        return result

    def clearable_counts(self, protected_id: int | None = None) -> dict:
        clearable = len(self._clearable_ids(protected_id))
        total = self.connection.execute("SELECT COUNT(*) FROM messages WHERE " + _ACTIVE).fetchone()[0]
        return {"clearable": clearable + sum(map(len, self._active_continuations().values())), "retained_uncertain": total - clearable}

    def clear_pending(self, protected_id: int | None = None) -> dict:
        with self.connection:
            identifiers = self._clearable_ids(protected_id)
            continuations = self._active_continuations()
            # Future packets have never been submitted, even when their current
            # FIFO head must remain protected for a pending browser callback.
            for identifier, parts in continuations.items():
                self._set_metadata(f"cancelled_parts:{identifier}", json.dumps(parts, ensure_ascii=False))
                self._set_metadata(f"remaining_parts:{identifier}", "[]")
                context = self.filter_context(identifier)
                self._save_filter_context(context["room"], context["segment_key"], context["source"], context["revision"], True)
            removed = self.connection.executemany(
                "UPDATE messages SET state='cancelled' WHERE id=? AND state IN ('pending','holding_short','blocked')",
                [(identifier,) for identifier in identifiers]).rowcount if identifiers else 0
            removed += sum(map(len, continuations.values()))
            old_count = self._metadata("cleared_count")
            old_count = int(old_count) if isinstance(old_count, str) and old_count.isascii() and old_count.isdigit() else 0
            self._set_metadata("cleared_count", str(old_count + removed))
        retained = self.connection.execute("SELECT COUNT(*) FROM messages WHERE " + _ACTIVE).fetchone()[0]
        return {"removed": removed, "retained": retained,
                "retained_uncertain": self.clearable_counts(protected_id)["retained_uncertain"]}

    def _clearable_ids(self, protected_id: int | None = None) -> list[int]:
        result = []
        for row in self.connection.execute("SELECT id,state,reason FROM messages WHERE " + _ACTIVE).fetchall():
            if row["id"] == protected_id:
                continue
            if row["state"] in {"pending", "holding_short"}:
                result.append(row["id"])
            elif row["state"] == "blocked" and row["reason"] == "platform_rejected":
                code = self.delivery_code(row["id"])
                if code is not None and code != 0:
                    result.append(row["id"])
        return result

    def increment(self, name: str):
        if name not in {"accepted", "failures", "f_retries"}:
            raise ValueError("invalid_metric")
        with self.connection:
            self.connection.execute(
                "INSERT INTO metrics(name,value) VALUES(?,1) ON CONFLICT(name) DO UPDATE SET value=value+1", (name,))

    def close(self):
        self.connection.close()
