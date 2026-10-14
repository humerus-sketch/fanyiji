"""Bounded read-only observation of the installed application's real queue."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fanyiji.danmaku_protocol import BODY_FIELDS, danmaku_utf16_length, danmaku_weighted_length
from fanyiji.paths import data_root

FIELDS = {"time", "kind", "room", "operation", "queue_id", "request_id", "reason", "outcome",
          "elapsed_seconds", "platform_code", "http_status", "has_f", "revision",
          "request_started_epoch_ms", "request_elapsed_ms", "max_weighted_length",
          "weighted_length", "characters", "head_characters", "removed", "retained",
          "retained_uncertain", "parts", "packets", "remaining_parts", "before_count", "occurrence_count",
          "f_retries", "head_f_retries", "f_retry_active", "last_f_at",
          "filter_rule_count", "filter_candidate_count", "hit_count", "candidate_count", "error_type"}


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def queue_snapshot(path):
    if not path.exists():
        return {"state": "missing"}
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1) as connection:
            states = dict(connection.execute("SELECT state,COUNT(*) FROM messages GROUP BY state"))
            continuations = 0
            for (value,) in connection.execute("SELECT m.value FROM metadata m JOIN messages q "
                    "ON m.name='remaining_parts:' || q.id WHERE q.state NOT IN ('page_visible','cancelled')"):
                try:
                    parts = json.loads(value)
                except (ValueError, TypeError):
                    continue
                if isinstance(parts, list) and len(parts) <= 4096:
                    continuations += len(parts)
            row = connection.execute("SELECT id,room,state,reason,text FROM messages "
                "WHERE state NOT IN ('page_visible','cancelled') ORDER BY id LIMIT 1").fetchone()
            head = ({"id": row[0], "room": row[1], "state": row[2], "reason": row[3],
                     "characters": danmaku_utf16_length(row[4]),
                     "weighted_length": danmaku_weighted_length(row[4])} if row else None)
            retries = connection.execute("SELECT value FROM metrics WHERE name='f_retries'").fetchone()
            return {"states": states, "continuation_parts": continuations,
                    "f_retries": retries[0] if retries else 0,
                    "queued": continuations + sum(n for state, n in states.items()
                    if state not in {"page_visible", "cancelled"}), "head": head}
    except sqlite3.Error as error:
        return {"state": "unavailable", "error_type": type(error).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.duration <= 600:
        parser.error("duration must be in (0,600]")
    args.output.mkdir(parents=True, exist_ok=False)
    directory = data_root()
    queue_path, log_path = directory / "danmaku-queue.sqlite3", directory / "delivery.jsonl"
    offset = log_path.stat().st_size if log_path.exists() else 0
    report = {"started_at": timestamp(), "read_only": True,
              "body_field_names": list(BODY_FIELDS), "initial": queue_snapshot(queue_path)}
    events, carry = [], b""
    deadline = time.monotonic() + args.duration
    with (args.output / "events.jsonl").open("w", encoding="utf-8") as output:
        while time.monotonic() < deadline:
            try:
                with log_path.open("rb") as stream:
                    if stream.seek(0, 2) < offset:
                        offset, carry = 0, b""
                    stream.seek(offset)
                    chunk = stream.read()
                    offset = stream.tell()
                lines = (carry + chunk).split(b"\n")
                carry = lines.pop()
                for line in lines:
                    try:
                        raw = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(raw, dict):
                        continue
                    event = {key: value for key, value in raw.items() if key in FIELDS}
                    events.append(event)
                    output.write(json.dumps(event, ensure_ascii=False) + "\n")
                output.flush()
            except OSError:
                pass
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
    sends = [event for event in events if event.get("kind") == "result" and event.get("operation") == "send"]
    starts = [event["request_started_epoch_ms"] for event in sends
              if type(event.get("request_started_epoch_ms")) in (float, int)]
    gaps = [(right - left) / 1000 for left, right in zip(starts, starts[1:])]
    report.update(finished_at=timestamp(), final=queue_snapshot(queue_path),
                  posts=len(starts), send_results=len(sends),
                  accepted=sum(event.get("outcome") == "receipt_accepted" for event in sends),
                  page_confirmations=sum(event.get("kind") == "result" and event.get("operation") == "check"
                      and event.get("reason") == "sent_page_visible" for event in events),
                  platform_codes=dict(Counter(str(event.get("platform_code")) for event in sends)),
                  platform_f_results=sum(event.get("outcome") == "platform_f" for event in sends),
                  send_results_by_queue_id=dict(Counter(str(event.get("queue_id")) for event in sends)),
                  post_start_gaps_seconds=gaps, minimum_start_gap_seconds=min(gaps) if gaps else None,
                  duplicate_skipped=sum(event.get("kind") == "duplicate_skipped" for event in events))
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
