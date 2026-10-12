"""One browser, final subtitles only, durable FIFO and conservative delivery."""
from __future__ import annotations

import html
import json
import math
import re
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from .asr_protocol import contains_asr_protocol
from .browser_service import room_url
from .danmaku_store import DanmakuStore, SHORT_TEXT_CHARACTERS, MIN_SEND_CHARACTERS, rebalance_short_parts
from .danmaku_protocol import valid_danmaku_text, danmaku_weighted_length, sanitize_limits, is_danmaku_punctuation
from .danmaku_filter import DanmakuContentFilter
from .danmaku_recovery import SAFE_ERRORS, RecoveryDecision, failure_decision
from .paths import data_root
from .translation_output import is_chinese_output


MIN_INTERVAL_SECONDS = 3.5
MAX_MESSAGE_CHARACTERS = 40
_BOUNDARY = re.compile(r"(?<=[。！？!?；;，,、])|\s+")
_TAGS = re.compile(r"<\s*/?\s*[A-Za-z_][^<>]*>|<\|[^<>]*\|>")
_FAILURE = re.compile(r"^(?:\[|【)?(?:(?:此片段翻译失败|翻译缓存失败|本地暂译和\s*Codex\s*终稿均失败|识别控制标记|译文包含识别控制标记|translation failed)|(?:翻译中|等待后半句|translating)(?:[\]】…\.。\s]|$))", re.I)


def _page_length(text: str) -> int:
    # Chrome's String.length counts non-BMP emoji as two UTF-16 code units.
    return sum(2 if ord(char) > 0xffff else 1 for char in text)


def _non_chinese_decision() -> RecoveryDecision:
    return RecoveryDecision("recovering", "output_language", "analyze_with_codex",
                            "译文含非中文字符 已移交独立中文修正 后续队列继续发送")


def _finite_protocol_number(value: object, maximum: float) -> float | None:
    if type(value) not in {int, float} or not 0 <= value <= maximum or not math.isfinite(value):
        return None
    return round(float(value), 3)


def _take_page_chunk(text: str, limit: int, weighted_limit: int = MAX_MESSAGE_CHARACTERS * 2 - 4) -> tuple[str, str]:
    count = weight = 0
    for index, char in enumerate(text):
        count += 2 if ord(char) > 0xffff else 1
        weight += danmaku_weighted_length(char)
        if count > limit or weight > weighted_limit:
            return text[:index], text[index:]
    return text, ""


def split_danmaku(text: object, max_weighted_length: int = 80,
                  max_characters: int = MAX_MESSAGE_CHARACTERS) -> list[str]:
    """Wrap chunks within both the account's weighted and UTF-16 limits."""
    if (type(max_weighted_length) is not int or max_weighted_length < 6
            or type(max_characters) is not int or max_characters < 3):
        return []
    if not isinstance(text, str) or contains_asr_protocol(text):
        return []
    text = html.unescape(text).strip()
    if not text or _TAGS.search(text) or _FAILURE.search(text) or text.casefold() in {"none", "null"}:
        return []
    if any((ord(char) < 32 and not char.isspace()) or 0xd800 <= ord(char) <= 0xdfff for char in text):
        return []
    text = re.sub(r"\s+", " ", text).strip()
    # Existing presentation brackets must not turn into double brackets.
    if text.startswith("【") and text.endswith("】"):
        text = text[1:-1].strip()
    # Presentation punctuation becomes sentence spacing; ASR/protocol and
    # failure templates have already been rejected before this cleanup.
    text = "".join(" " if is_danmaku_punctuation(char) else char for char in text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    limit = min(MAX_MESSAGE_CHARACTERS, max_characters) - 2
    weighted_limit = min(80, max_weighted_length) - 4
    def fits(value):
        return _page_length(value) <= limit and danmaku_weighted_length(value) <= weighted_limit
    pieces, current = [], ""
    for token in _BOUNDARY.split(text):
        if not token:
            continue
        # A very long unpunctuated utterance still cannot exceed page limits.
        while not fits(token):
            if current:
                pieces.append(current)
                current = ""
            chunk, token = _take_page_chunk(token, limit, weighted_limit)
            if not chunk:
                return []
            pieces.append(chunk)
        separator = " " if current else ""
        if not fits(current + separator + token):
            pieces.append(current)
            current = token
        else:
            current += separator + token
    if current:
        pieces.append(current)
    result = ["【" + piece.strip() + "】" for piece in pieces if piece.strip()]
    return rebalance_short_parts(result, max_weighted_length=max_weighted_length,
        max_characters=min(MAX_MESSAGE_CHARACTERS, max_characters)) if all(
        valid_danmaku_text(piece, normalized_only=True) for piece in result) else []


class DanmakuService(QObject):
    status = Signal(dict)

    def __init__(self, browser_service, parent=None, store_path=None, filter_path=None, *, ai_analyzer=None, reset_on_start=False):
        super().__init__(parent)
        self.browser = browser_service
        path = Path(store_path) if store_path is not None else data_root() / "danmaku" / "queue.sqlite3"
        self.store = DanmakuStore(path)
        previous_run = self.store.reset_for_restart() if reset_on_start else None
        self.log_path = path.with_name("delivery.jsonl")
        self.filter_path = Path(filter_path) if filter_path is not None else (
            path.with_name("danmaku-content-filter.json") if store_path is not None else data_root() / "danmaku-content-filter.json")
        self.content_filter = None
        self._filter_error = ""
        self._filter_candidate_error = ""
        self._handoff_restore_error = ""
        self._filter_blocked = False
        self._filter_hits = 0
        self._ai_analyzer = ai_analyzer
        self._ai_settings = None
        self._ai_task = None
        self._ai_proposal = None
        self._ai_generation = 0
        self._recovery_decision = None
        self._diagnostic_f_cycle = {}
        self._historical_reviewed = set()
        self._review_attempts = {}
        self.validation_room = ""
        self._validator = None
        if ai_analyzer is not None:
            ai_analyzer.finished.connect(self._ai_finished)
        self._last_filter_revision = None
        try:
            self.content_filter = DanmakuContentFilter(self.filter_path)
        except Exception as error:
            self._filter_error = "读取过滤名单失败，请修复名单后重试"
            self._filter_blocked = True
        self._restore_legacy_rejection_code()
        self._restore_legacy_f_source()
        self.enabled, self.room, self.source = False, "", "system"
        self.interval_seconds = MIN_INTERVAL_SECONDS
        self.max_weighted_length = 80
        self.store.max_weighted_length = self.max_weighted_length
        self._clock = time.monotonic
        self._next_at = 0.0
        self._read_not_before = 0.0
        self._verify_deadline = None
        self._verify_queue_id = None
        self._verify_window_seconds = 15.0
        self._f_retry_queue_id = None
        self._inflight: dict | None = None
        self._session = None
        self._session_identity = uuid.uuid4().hex
        self._segments: OrderedDict = OrderedDict()
        self._closed = False
        self._login_room = ""
        self._latest_reason = "disabled"
        self._request_timeout = 45.0
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.tick)
        self.review_timer = QTimer(self)
        self.review_timer.setInterval(1000)
        self.review_timer.timeout.connect(self._review_history_tick)
        self.browser.danmaku_result.connect(self._on_result)
        if hasattr(self.browser, "event"):
            self.browser.event.connect(self._browser_event)
        if self.content_filter is not None:
            try:
                self.content_filter.analyze_candidates()
            except Exception as error:
                self._filter_candidate_error = "失败句检测结果保存失败，原队列保留"
                self._log("filter_candidate_error", error_type=type(error).__name__)
        if previous_run is not None:
            self._log("restart_reset", previous_queued=previous_run["queued"], statistics_reset=True)
        self._restore_failed_handoffs()

    def _restore_failed_handoffs(self):
        """Finish saved failure handoffs without restoring or submitting old FIFO rows."""
        if self._closed or self.content_filter is None:
            return False
        try:
            for pending in self.store.failed_handoffs():
                existing = next((row for row in self.content_filter.candidates(pending["room"])
                                 if row["text"] == pending["text"] and row["queue_id"] == pending["queue_id"]), None)
                if existing is None:
                    decision = (_non_chinese_decision() if pending["reason"] == "non_chinese_translation"
                                else failure_decision(pending["reason"], {"code": pending["platform_code"]}))
                    diagnosis = {"category": decision.category, "action": "transfer_failure",
                                 "message": "失败消息交由独立处理，继续后续队列"}
                    existing = self.content_filter.record_failure(pending["room"], pending["text"],
                        pending["reason"], queue_id=pending["queue_id"], diagnosis=diagnosis)
                else:
                    # Do not infer durability from an in-memory match. Saving
                    # the unchanged observation also covers a previous write
                    # whose acknowledgement was interrupted, without a recount.
                    self.content_filter.persist()
                # record_failure saves atomically before returning. If the
                # acknowledgement fails, an exact candidate prevents counting
                # this same observation again on the next startup/timer tick.
                self.store.acknowledge_failed_handoff(pending["room"], pending["text"], pending["queue_id"])
                self._log("failure_handoff_restored", queue_id=pending["queue_id"], part=pending["part"],
                          candidate_id=existing["id"])
            self._handoff_restore_error = ""
            return True
        except Exception as failure:
            # Keep the staging row and retry later; no browser request or
            # runtime queue mutation is needed to complete this write.
            message = "失败记录移交暂时失败 自动重试保存"
            if not self._handoff_restore_error:
                self._log("failure_handoff_restore_error", error_type=type(failure).__name__)
            self._handoff_restore_error = message
            return False

    def _restore_legacy_rejection_code(self):
        head = self.store.head()
        if (not head or head["state"] != "blocked" or head["reason"] != "platform_rejected"
                or head["platform_code"] is not None or not head["request_id"]):
            return
        try:
            with self.log_path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 1024 * 1024))
                lines = stream.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            return
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(record, dict):
                continue
            # Queue IDs alone are insufficient across database files. Require
            # this exact persisted request token and room from the same log.
            if (record.get("kind") != "result" or record.get("operation") != "send"
                    or type(record.get("queue_id")) is not int or record.get("queue_id") != head["id"] or record.get("room") != head["room"]
                    or record.get("request_id") != head["request_id"]):
                continue
            code = record.get("platform_code")
            if type(code) is int and -2**31 <= code < 2**31 and code != 0 and record.get("has_f") is False:
                self.store.update(head["id"], head["state"], request_id=head["request_id"],
                                  reason=head["reason"], platform_code=code)
            return

    def _restore_legacy_f_source(self):
        """Older manual checks overwrote f; only the last completed POST proves it."""
        head = self.store.head()
        if (not head or head["state"] not in {"blocked", "verify"}
                or head["reason"] not in {"result_unknown", "manual_verify", "receipt_unconfirmed"}
                or self.store.has_f_recovery(head["id"])):
            return
        try:
            with self.log_path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 1024 * 1024))
                lines = stream.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            return
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            if (not isinstance(record, dict) or record.get("operation") != "send"
                    or record.get("queue_id") != head["id"] or type(record.get("queue_id")) is not int
                    or record.get("room") != head["room"]):
                continue
            # A later dispatch/timeout, even without its result, supersedes f.
            if (record.get("kind") == "result" and record.get("outcome") == "platform_f"
                    and record.get("has_f") is True and record.get("platform_code") == 0
                    and type(record.get("platform_code")) is int and record.get("http_status") == 200
                    and record.get("characters") == _page_length(head["text"])
                    and record.get("weighted_length") == danmaku_weighted_length(head["text"])
                    and isinstance(record.get("request_id"), str) and record["request_id"]):
                self.store.activate_f_recovery(head["id"])
            return
    def configure(self, enabled, room, source="system", interval_seconds=MIN_INTERVAL_SECONDS, validation_room=""):
        if source not in {"system", "mic", "playback", "microphone", "all"}:
            raise ValueError("invalid_source")
        interval = float(interval_seconds)
        if not math.isfinite(interval):
            raise ValueError("invalid_interval")
        normalized = room_url(str(room)).rsplit("/", 1)[-1] if str(room).strip() else ""
        testing = room_url(str(validation_room)).rsplit("/", 1)[-1] if str(validation_room).strip() else ""
        if testing and normalized and int(testing) == int(normalized):
            raise ValueError("validation_room_matches_formal_room")
        if enabled and not normalized:
            raise ValueError("invalid_room")
        normalized_source = {"playback": "system", "microphone": "mic"}.get(source, source)
        if self.room != normalized or self.source != normalized_source:
            self._segments.clear()
            self._session_identity = uuid.uuid4().hex
            if self._ai_task and self._ai_analyzer:
                self._forget_cancelled_review(self._ai_task)
                self._ai_analyzer.cancel(self._ai_task["token"])
            self._ai_task = self._ai_proposal = None
            self._ai_generation += 1
        self.enabled, self.room = bool(enabled), normalized
        self.validation_room = testing
        if self._validator is not None:
            self._validator.set_rooms(testing, normalized)
        self.source = normalized_source
        self.interval_seconds = max(MIN_INTERVAL_SECONDS, interval)
        self._login_room = ""
        self._latest_reason = "waiting" if self.enabled else "disabled"
        head = self.store.head()
        if self.enabled and head and head["room"] == self.room:
            self._restore_f_recovery(head)
            head = self.store.head()
        if self.enabled and head and head["state"] == "blocked":
            self._latest_reason = head["reason"] or "result_unknown"
        if self.enabled:
            self.timer.start()
        elif not self._inflight:
            self.timer.stop()
        self._emit_status()

    def begin_session(self, session):
        self._session = session
        self._session_identity = uuid.uuid4().hex
        self._segments.clear()

    def handle_event(self, event):
        if self._closed or not self.enabled or self._session is None or event.get("session") != self._session:
            return
        channel = {"playback": "system", "microphone": "mic"}.get(event.get("channel"), event.get("channel"))
        if (channel not in {"mic", "system"} or (self.source != "all" and channel != self.source)
                or not self.room or "segment_id" not in event):
            return
        kind = event.get("kind")
        if kind not in {"original", "recognized", "subtitle_final", "translation", "translation_superseded"}:
            return
        try:
            revision = int(event.get("revision", 1))
        except (ValueError, TypeError):
            return
        if revision < 1:
            return
        key = (channel, str(event["segment_id"]))
        segment = self._segments.setdefault(key, {"revision": revision, "confirmed": False, "final_received": False, "translation": None,
                                                    "source_safe": False, "enqueued": False, "rejected_translation": None})
        if revision < segment["revision"]:
            return
        if revision > segment["revision"]:
            segment.update(revision=revision, confirmed=False, final_received=False, translation=None, source_safe=False,
                           rejected_translation=None)
        self._segments.move_to_end(key)
        while len(self._segments) > 8192:
            self._segments.popitem(last=False)
        if kind == "translation_superseded":
            segment["translation"] = None
            return
        if kind in {"original", "recognized"}:
            segment["source_safe"] = bool(split_danmaku(event.get("text"))) and not event.get("failed", False)
            segment["confirmed"] = segment["source_safe"] and (not event.get("provisional", False) or segment["final_received"])
        elif kind == "subtitle_final":
            segment["final_received"] = True
            segment["confirmed"] = segment["source_safe"]
        elif kind == "translation":
            if (event.get("translation_stage", "final") != "final" or event.get("failed", False)
                    or event.get("overall_failed", False)):
                return
            if not split_danmaku(event.get("text")):
                return
            segment["translation"] = event["text"]
            # provisional refers to ASR completeness, not to Codex quality.
            # An earlier subtitle_final can therefore validate this result.
        if segment["confirmed"] and segment["translation"] and not segment["enqueued"]:
            text = segment["translation"].strip()
            if not is_chinese_output(text):
                self._stage_non_chinese_translation(text, revision, segment)
                return
            self._apply_pending_filters()
            filtered, filter_revision = self._filter_translation(text, self.room)
            if not is_chinese_output(filtered):
                self._stage_non_chinese_translation(filtered, revision, segment)
                return
            try:
                accepted = self.store.enqueue(self.room, channel,
                    f"{self._session_identity}:{channel}:{key[1]}", revision, text, split_danmaku(filtered),
                    filter_revision=filter_revision,
                    filter_rule_signatures=self.content_filter.rule_signatures(self.room)
                        if self.content_filter is not None and filter_revision else None)
            except Exception:
                self._latest_reason = "storage_error"
                self._emit_status()
                return
            segment["enqueued"] = True
            self._log("enqueued" if accepted else "duplicate_skipped", revision=revision)
            self._emit_status()

    def _stage_non_chinese_translation(self, text, revision, segment):
        if segment["rejected_translation"] == text:
            return
        # Do not insert an unsendable sentence into FIFO, or strip its foreign
        # names. A handoff journal survives failed filter writes and restarts.
        try:
            queue_id = self.store.stage_failed_handoff(self.room, text,
                reason="non_chinese_translation")
        except Exception as failure:
            self._latest_reason = "storage_error"
            self._log("non_chinese_handoff_error", error_type=type(failure).__name__)
            self._emit_status()
            return
        segment["rejected_translation"] = text
        self._restore_failed_handoffs()
        self._latest_reason = "non_chinese_translation"
        self._log("non_chinese_translation_blocked", queue_id=queue_id,
                  revision=revision, characters=_page_length(text), submitted=False)
        self._emit_status()

    def _filter_translation(self, text, room):
        # An unusable dictionary must not discard a final subtitle. Persist
        # the original safe payload, then block its first POST until repaired.
        if self.content_filter is None:
            return text, ""
        try:
            filtered, hits = self.content_filter.apply(text, room)
            if not split_danmaku(filtered):
                raise ValueError("invalid_filtered_output")
            revision = self.content_filter.revision
        except Exception as error:
            self._filter_error = "过滤规则应用失败，待发消息保留，请修改规则后重试"
            self._filter_blocked = True
            self._log("filter_error", error_type=type(error).__name__)
            return text, ""
        self._filter_hits += hits
        if hits:
            self._log("filter_applied", hit_count=hits)
        return filtered, revision

    def _apply_pending_filters(self):
        if self._filter_blocked or self.content_filter is None:
            return False
        protected = self._inflight["row"]["id"] if self._inflight else None
        try:
            revision = self.content_filter.revision
            if revision == self._last_filter_revision:
                return True
            deferred = False
            for row in self.store.unsubmitted_filter_heads():
                if row["id"] == protected:
                    deferred = True
                    continue
                context = self.store.filter_context(row["id"])
                if context["frozen"] or context["revision"] == revision:
                    continue
                filtered, applied_revision = self._filter_translation(context["source"], row["room"])
                if not applied_revision:
                    return False
                # Account-specific packet splitting still happens immediately
                # before POST, after browser login has supplied its live cap.
                parts = split_danmaku(filtered)
                if not self.store.prepare_filtered_parts(row["id"], parts, applied_revision,
                        self.content_filter.rule_signatures(row["room"])):
                    raise ValueError("filter_repacketization_failed")
            if not deferred:
                self._last_filter_revision = revision
        except Exception as error:
            self._filter_error = "过滤名单处理失败，待发消息保留，请重试"
            self._filter_blocked = True
            self._log("filter_error", error_type=type(error).__name__)
            return False
        return True

    def refresh_content_filter(self):
        """Refresh unsent finals; f heads are repaired after a fresh page check."""
        if self._closed:
            return False
        if self.content_filter is None:
            try:
                self.content_filter = DanmakuContentFilter(self.filter_path)
            except Exception as error:
                self._filter_error = "读取过滤名单失败，请修复名单后重试"
                self._filter_blocked = True
                self._log("filter_error", error_type=type(error).__name__)
                self._emit_status()
                return False
        self._filter_error = self._filter_candidate_error = ""
        self._filter_blocked = False
        self._last_filter_revision = None
        result = self._apply_pending_filters()
        if not result:
            self._latest_reason = "filter_error"
        self._emit_status()
        return result

    def _repair_f_head_rules(self, head, baseline, occurrences, present):
        # A page check is an observation, not proof against delayed broadcast.
        # It permits editing only an explicit f head using confirmed local rules.
        if (self.content_filter is None or self._filter_blocked or baseline is None
                or type(occurrences) is not int or occurrences != baseline
                or type(present) is not bool or present != bool(occurrences)):
            return False
        try:
            applied = self.store.applied_rule_signatures(head["id"])
            signatures = self.content_filter.rule_signatures(head["room"])
            if applied is None:
                # Older already-filtered packets lack application history.
                # Deletion is idempotent; replacement would risk a cascade.
                applicable = [row for row in self.content_filter.rules() if row["enabled"]]
                if any(row["action"] != "remove" for row in applicable):
                    return False
                applied = []
            if set(signatures).issubset(applied):
                return False
            remaining = self.store.strict_remaining_parts(head["id"])
            packets, parts, hits = [head["text"], *remaining], [], 0
            for packet in packets:
                filtered, matched = self.content_filter.apply(packet, head["room"],
                    excluded_rule_signatures=applied)
                split = split_danmaku(filtered, max_weighted_length=self.max_weighted_length)
                if not split:
                    raise ValueError("invalid_filtered_output")
                parts.extend(split)
                hits += matched
            if not hits or packets == parts:
                return False
            if not self.store.repair_f_rejected_head(head, remaining, baseline, parts,
                    self.content_filter.revision, sorted(set(applied) | set(signatures))):
                return False
            self._filter_hits += hits
            self._f_retry_queue_id = None
            self._verify_queue_id = self._verify_deadline = None
            self._log("f_head_rule_repaired", queue_id=head["id"], hit_count=hits,
                      previous_characters=_page_length(head["text"]), characters=_page_length(parts[0]),
                      packets=len(parts), filter_revision=self.content_filter.revision)
            return True
        except Exception as error:
            self._filter_error = "失败消息过滤处理失败，待发消息保留，请修改规则后重试"
            self._filter_blocked = True
            self._log("filter_error", queue_id=head["id"], error_type=type(error).__name__)
            return False

    def configure_ai(self, settings):
        """Keep failure review isolated from the live subtitle conversation."""
        self._ai_generation += 1
        self._ai_settings = {name: getattr(settings, name, "") for name in (
            "codex_model", "codex_effort", "codex_speed")}
        if isinstance(settings, dict):
            self._ai_settings = {name: settings.get(name, "") for name in self._ai_settings}
        if not self._ai_settings["codex_speed"]:
            self._ai_settings["codex_speed"] = "standard"
        if self._ai_analyzer is None:
            from .danmaku_ai import DanmakuAIAnalyzer
            self._ai_analyzer = DanmakuAIAnalyzer(self)
            self._ai_analyzer.finished.connect(self._ai_finished)
        elif self._ai_task:
            self._forget_cancelled_review(self._ai_task)
            self._ai_analyzer.cancel(self._ai_task["token"])
        self._ai_task = self._ai_proposal = None
        self.review_timer.start()

    def _forget_cancelled_review(self, task):
        if task.get("historical"):
            candidate_id = task["candidate_id"]
            self._historical_reviewed.discard(candidate_id)
            self._review_attempts[candidate_id] = max(0, self._review_attempts.get(candidate_id, 1) - 1)

    def _review_history_tick(self):
        if self._closed or self.content_filter is None:
            return
        if not self._restore_failed_handoffs():
            self._emit_status()
            return
        if self._validator is None:
            from .danmaku_validation import DanmakuValidationRunner
            self._validator = DanmakuValidationRunner(self.browser, self.content_filter, self,
                log=self._log, clock=lambda: self._clock())
        if not self._inflight:
            try:
                self._validator.tick(self.validation_room, self.room, self.interval_seconds,
                                     earliest_post=self._next_at)
            except Exception as failure:
                self._log("validation_error", error_type=type(failure).__name__)
            self._emit_status()
        if self._ai_analyzer is None or self._ai_task or self._ai_analyzer.busy:
            return
        head = self.store.head()
        if head and self.enabled and self.store.f_recovery(head["id"])["active"]:
            return  # Live recovery is reviewed by its matched failure receipt.
        for candidate in self.content_filter.candidates(include_resolved=False):
            plan = self.content_filter.validation_plan(candidate["id"])
            if plan is not None and plan["state"] != "failed":
                continue
            if (candidate["id"] in self._historical_reviewed and plan is None) or self._review_attempts.get(candidate["id"], 0) >= 3:
                continue
            self._historical_reviewed.add(candidate["id"])
            self._review_attempts[candidate["id"]] = self._review_attempts.get(candidate["id"], 0) + 1
            self._submit_candidate_ai(candidate, historical=True)
            break

    def _submit_candidate_ai(self, candidate, *, historical=False, head=None, payload=None):
        token = uuid.uuid4().hex
        task = {"token": token, "candidate_id": candidate["id"], "historical": historical,
                "text": candidate["text"], "room": candidate["room"], "generation": self._ai_generation,
                "candidate_count": candidate["count"], "candidate_queue_id": candidate["queue_id"],
                "validation_id": (candidate.get("validation") or {}).get("validation_id")}
        if head is not None:
            task.update(id=head["id"], part=head["part"])
        detection = self.content_filter.detect(candidate["text"], candidate["room"])
        receipt = dict(payload or {})
        receipt.update(attempt_count=candidate["count"], characters=_page_length(candidate["text"]))
        data = {"text": candidate["text"], "receipt": receipt,
                "dictionary_hits": detection.get("hits", []), "confirmed_rules": detection.get("known_rules", [])}
        self._ai_task = task
        try:
            accepted = self._ai_analyzer.submit(token, data, self._ai_settings or {})
        except Exception:
            accepted = False
        if not accepted:
            self._ai_task = None
            self._ai_unavailable(candidate["id"])
            return False
        self.content_filter.set_candidate_status(candidate["id"], "recovering", diagnosis={
            "category": "ai_review", "action": "analyze_with_codex", "message": "正在请求 Codex 分析失败正文"})
        self._log("ai_review_started", candidate_id=candidate["id"], historical=historical)
        return True

    def _ai_unavailable(self, candidate_id):
        self.content_filter.set_candidate_status(candidate_id, "action_required", diagnosis={
            "category": "ai_review", "action": "review_on_restart", "message": "AI分析暂不可用 下次启动自动重试"})

    def _recovery_status(self, head, decision=None, *, status=None, message=None, action=None):
        if self.content_filter is None:
            return
        decision = decision or self._recovery_decision or failure_decision(head.get("reason", "result_unknown"))
        self._recovery_decision = decision
        diagnosis = dict(decision.diagnosis)
        if message is not None:
            diagnosis["message"] = message
        if action is not None:
            diagnosis["action"] = action
        try:
            ctx = self.store.recovery_context(head["id"], head["part"])
            for text in ctx.get("aliases", []) or [head["text"]]:
                self.content_filter.update_recovery(head["room"], text, head["id"],
                    status=status or decision.status, diagnosis=diagnosis)
        except Exception as error:
            self._filter_candidate_error = "失败句检测结果保存失败，原队列保留"
            self._log("recovery_record_error", queue_id=head["id"], error_type=type(error).__name__)

    def _record_failed_content(self, head, reason, payload=None):
        if self.content_filter is None:
            self._filter_candidate_error = "失败候选保存失败，请修复过滤名单"
            return
        try:
            decision = _non_chinese_decision() if reason == "non_chinese_translation" else failure_decision(reason, payload)
            self._recovery_decision = decision
            ctx = self.store.recovery_context(head["id"], head["part"])
            aliases = ctx.setdefault("aliases", [])
            if head["text"] not in aliases:
                aliases.append(head["text"])
            ctx["failure_count"] = ctx.get("failure_count", 0) + 1
            ctx["handoff_pending"] = True
            self.store.save_recovery_context(head["id"], head["part"], ctx)
            candidate = self.content_filter.record_failure(head["room"], head["text"], reason,
                queue_id=head["id"], diagnosis=decision.diagnosis)
            self.content_filter.update_recovery(head["room"], head["text"], head["id"],
                status=decision.status, diagnosis=decision.diagnosis)
            self._filter_candidate_error = ""
            analysis = candidate["analysis"]
            self._log("filter_candidate", queue_id=head["id"], failure_reason=reason,
                      candidate_count=len(self.content_filter.candidates(head["room"])),
                      detection_status=analysis["status"], detection_summary=analysis["summary_code"],
                      detection_hit_count=len(analysis["hits"]), dictionary_revision=analysis["dictionary_revision"])
        except Exception as error:
            self._filter_candidate_error = "失败候选保存失败，原队列保留"
            self._log("filter_candidate_error", queue_id=head["id"], error_type=type(error).__name__)

    def _transfer_to_analysis(self, head):
        """Save the independent recovery record before releasing the FIFO slot."""
        if self.content_filter is None or self._filter_candidate_error:
            return False
        candidate = next((row for row in self.content_filter.candidates(head["room"])
                          if row["text"] == head["text"] and row["queue_id"] == head["id"]), None)
        current = self.store.head()
        if candidate is None or current is None or any(current[key] != head[key] for key in ("id", "part", "room", "text")):
            return False
        try:
            if not self.store.defer_failed_head(current):
                return False
        except Exception as failure:
            self._filter_candidate_error = "失败记录移交暂时失败 自动重试保存"
            self._log("failure_transfer_error", queue_id=head["id"], error_type=type(failure).__name__)
            return False
        self._verify_queue_id = self._verify_deadline = self._f_retry_queue_id = None
        self._recovery_decision = None
        self._latest_reason = "failure_transferred"
        self._log("failure_transferred", queue_id=head["id"], part=head["part"], candidate_id=candidate["id"])
        # The AI task belongs to this immutable candidate, not the queue head
        # which may already have advanced to the next subtitle.
        if self._ai_analyzer is not None and not self._ai_task and not self._ai_analyzer.busy:
            plan = self.content_filter.validation_plan(candidate["id"])
            if plan is None:
                self._historical_reviewed.add(candidate["id"])
                self._review_attempts[candidate["id"]] = self._review_attempts.get(candidate["id"], 0) + 1
                self._submit_candidate_ai(candidate, historical=True)
        return True

    def _block_non_chinese_head(self, head):
        """Old/manual/filter-generated packets obey the same pre-POST gate."""
        current = self.store.head()
        if current is None or any(current[key] != head[key] for key in ("id", "part", "room", "text")):
            return
        self.store.stop_f_recovery(head["id"])
        self.store.update(head["id"], "blocked", reason="non_chinese_translation")
        blocked = self.store.head()
        self._record_failed_content(blocked, "non_chinese_translation")
        self._transfer_to_analysis(blocked)
        self._log("non_chinese_translation_blocked", queue_id=head["id"],
                  part=head["part"], characters=_page_length(head["text"]), submitted=False)
        self._emit_status()

    def _schedule_ai(self, head, payload):
        if self._ai_analyzer is None or self._ai_task or self._ai_proposal or self._ai_analyzer.busy:
            return
        ctx = self.store.recovery_context(head["id"], head["part"])
        if ctx.get("ai_attempts", 0) >= 3:
            return
        candidate = next((item for item in self.content_filter.candidates(head["room"])
                          if item["text"] == head["text"] and item["queue_id"] == head["id"]), None)
        if candidate is None:
            return
        plan = self.content_filter.validation_plan(candidate["id"])
        if plan is not None and plan["state"] != "failed":
            return
        if not self._submit_candidate_ai(candidate, head=head, payload=payload):
            return
        ctx["ai_attempts"] = ctx.get("ai_attempts", 0) + 1
        self.store.save_recovery_context(head["id"], head["part"], ctx)
        self._recovery_status(head, status="recovering", action="analyze_with_codex",
                              message="正在请求 Codex 分析失败正文")

    def _ai_finished(self, token, result, error):
        if self._closed or not self._ai_task or token != self._ai_task["token"]:
            return
        task, self._ai_task = self._ai_task, None
        head = self.store.head()
        if (task["generation"] != self._ai_generation or (not task.get("historical") and
                (not head or any(task[k] != head[k] for k in ("id", "part", "room", "text"))))):
            self._log("ai_review_stale", candidate_id=task.get("candidate_id"))
            return
        candidate = next((row for row in self.content_filter.candidates() if row["id"] == task["candidate_id"]), None)
        if candidate is None or candidate["text"] != task["text"] or candidate["room"] != task["room"]:
            return
        if candidate["status"] == "resolved":
            self._log("ai_review_stale", candidate_id=candidate["id"], reason="already_resolved")
            return
        if (candidate["count"] != task["candidate_count"] or candidate["queue_id"] != task["candidate_queue_id"]
                or (candidate.get("validation") or {}).get("validation_id") != task["validation_id"]):
            self._historical_reviewed.discard(candidate["id"])
            self._review_attempts[candidate["id"]] = max(0, self._review_attempts.get(candidate["id"], 1) - 1)
            self._log("ai_review_stale", candidate_id=candidate["id"], reason="candidate_changed")
            return
        try:
            from .danmaku_ai import validate_analysis_result
            if error:
                raise ValueError("ai_unavailable")
            result = validate_analysis_result(result, task["text"])
            if result["action"] != "rewrite":
                self.content_filter.set_candidate_status(candidate["id"], "action_required", diagnosis={
                    "category": "ai_review", "action": "review_on_restart", "message": "AI未提出可验证的局部修正 下次启动重新分析"})
            else:
                self.content_filter.save_validation_plan(candidate["id"], result["revised_text"], edits=result["edits"])
                # A structured result is advice. Only the separate test-room
                # runner may enable rules or mark this recovery complete.
                self._log("ai_plan_saved", candidate_id=candidate["id"], edits=len(result["edits"]), validated=False)
        except Exception as failure:
            self._ai_unavailable(candidate["id"])
            self._log("ai_review_error", candidate_id=candidate["id"], error_type=type(failure).__name__,
                      error_code=error if error in {"ai_unavailable", "ai_invalid_result", "ai_timeout", "ai_cancelled"} else "ai_validation_error")
        self._emit_status()

    def _resolve_failed_content(self, head):
        if self.content_filter is None:
            return
        ctx = self.store.recovery_context(head["id"], head["part"])
        learned = []
        try:
            for text in ctx.get("aliases", []) or [head["text"]]:
                candidate = next((row for row in self.content_filter.candidates(head["room"])
                                  if row["text"] == text and row["queue_id"] in {None, head["id"]}), None)
                if candidate is not None and self.content_filter.validation_plan(candidate["id"]) is not None:
                    continue  # Delivery of the original does not validate a correction.
                self.content_filter.resolve_success(head["room"], text, head["id"], resolution="page_observed")
            self._log("failure_record_resolved", queue_id=head["id"], learned_rules=len(learned),
                      observation="page_text_only", automatic_sender_confirmed=False)
        except Exception as error:
            self._filter_candidate_error = "失败句检测结果保存失败，原队列保留"
            self._log("recovery_record_error", queue_id=head["id"], error_type=type(error).__name__)
        if self._ai_task and self._ai_task.get("id") == head["id"]:
            self._ai_analyzer.cancel(self._ai_task["token"])
            self._ai_task = None
        self._ai_proposal = None
        self._recovery_decision = None

    def _browser_ready(self):
        return bool(getattr(self.browser, "is_browser_ready", False))

    def _apply_limits(self, payload):
        self.max_weighted_length = sanitize_limits(payload)["max_weighted_length"]
        self.store.max_weighted_length = self.max_weighted_length

    def _prepare_head(self, head):
        packets = [head["text"], *self.store.remaining_parts(head["id"])]
        parts = []
        for packet in packets:
            split = split_danmaku(packet, max_weighted_length=self.max_weighted_length)
            if not split:
                self._latest_reason = "text_too_long"
                return None
            parts.extend(split)
        if parts == packets:
            return head
        if not self.store.prepare_parts(head["id"], parts):
            return None
        self._log("length_adjusted", queue_id=head["id"], packets=len(parts),
                  characters=_page_length(head["text"]), weighted_length=danmaku_weighted_length(head["text"]),
                  max_weighted_length=self.max_weighted_length)
        return self.store.head()

    def _browser_event(self, event):
        if event.get("kind") in {"closed", "error", "starting"}:
            self._login_room = ""

    def _restore_f_recovery(self, head):
        recovery = self.store.f_recovery(head["id"])
        legacy_f = (head["state"] == "blocked" and head["reason"] == "platform_f"
                    and head["platform_code"] in {None, 0} and not self.store.has_f_recovery(head["id"]))
        if legacy_f and not recovery["active"]:
            self.store.activate_f_recovery(head["id"])
            recovery["active"] = True
        if not recovery["active"] or head["state"] not in {"blocked", "verify", "retry_pending"}:
            return False
        if self._f_retry_queue_id != head["id"]:
            # A new process cannot reuse a monotonic deadline. Wait at least
            # the configured interval after restoring, including legacy f rows.
            self._next_at = max(self._next_at, self._clock() + self.interval_seconds)
            self._f_retry_queue_id = head["id"]
            self._verify_queue_id = self._verify_deadline = None
            self.store.update(head["id"], "verify", request_id=head["request_id"], reason="platform_f")
            self._log("f_recovery_restored", queue_id=head["id"], retry_count=recovery["retries"],
                      wait_seconds=self.interval_seconds)
        self._latest_reason = "waiting_f_retry"
        return True

    def _continue_f_verification(self, head, request_id, now, *, ready=False):
        if ready and now >= self._next_at:
            self.store.update(head["id"], "retry_pending", request_id=request_id, reason="f_retry_pending")
            self._verify_queue_id = self._verify_deadline = None
            self._read_not_before = now
        else:
            self.store.update(head["id"], "verify", request_id=request_id, reason="platform_f")
            self._read_not_before = min(now + 0.5, self._next_at) if now < self._next_at else now + 0.5
        self._latest_reason = "waiting_f_retry"

    def tick(self):
        if self._closed:
            return
        if self._validator is not None and self._validator.busy:
            return
        now = self._clock()
        if self._inflight:
            if now - self._inflight["started"] >= self._request_timeout:
                flight, self._inflight = self._inflight, None
                if flight["operation"] == "send":
                    self._next_at = now + self.interval_seconds
                if flight["operation"] in {"send", "check"}:
                    if flight["operation"] == "check" and self.store.f_recovery(flight["row"]["id"])["active"]:
                        self._continue_f_verification(flight["row"], flight["request_id"], now)
                    else:
                        self.store.stop_f_recovery(flight["row"]["id"])
                        self.store.update(flight["row"]["id"], "blocked", request_id=flight["request_id"], reason="result_unknown")
                        self._latest_reason = "result_unknown"
                        if flight["operation"] == "send":
                            self._record_failed_content(flight["row"], "result_unknown")
                            self._transfer_to_analysis(flight["row"])
                        else:
                            self._record_failed_content(flight["row"], "receipt_unconfirmed")
                            self._transfer_to_analysis(flight["row"])
                else:
                    self._latest_reason = "waiting_connection"
                    self._read_not_before = now + self.interval_seconds
                self._log("request_timeout", operation=flight["operation"], queue_id=flight["row"]["id"])
                self._emit_status()
            return
        if not self.enabled or now < self._read_not_before:
            return
        self._apply_pending_filters()
        head = self.store.head()
        if head is None:
            self._latest_reason = "empty"
            self._emit_status()
            return
        if head["room"] != self.room:
            self._latest_reason = "queued_room_mismatch"
            self._emit_status()
            return
        if (head["state"] in {"blocked", "verify"}
                and self.store.recovery_context(head["id"], head["part"]).get("handoff_pending")):
            if self._filter_candidate_error:
                self._record_failed_content(head, head["reason"] or "result_unknown")
            if self._transfer_to_analysis(head):
                self._emit_status()
                return
            self._read_not_before = now + self.interval_seconds
            self._emit_status()
            return
        if head["state"] in {"pending", "holding_short", "retry_pending"} and not is_chinese_output(head["text"]):
            self._block_non_chinese_head(head)
            return
        if self._filter_blocked and head["state"] in {"pending", "holding_short"}:
            try:
                frozen = self.store.filter_context(head["id"])["frozen"]
            except Exception:
                frozen = False
            if not frozen:
                self._latest_reason = "filter_error"
                self._emit_status()
                return
        f_recovery = self._restore_f_recovery(head)
        if f_recovery:
            head = self.store.head()
            if self._filter_blocked:
                self._latest_reason = "filter_error"
                self._emit_status()
                return
        length_rejection = (head["state"] == "blocked" and head["reason"] == "platform_rejected"
                            and head["platform_code"] == 1003212)
        auto_verify = (head["state"] == "blocked" and head["reason"] in {
            "result_unknown", "receipt_unconfirmed", "manual_verify", "restart_verify"})
        if head["state"] == "blocked" and not length_rejection and not f_recovery and not auto_verify:
            self._latest_reason = head["reason"] or "result_unknown"
            self._recovery_status(head, failure_decision(head["reason"], {"code": head["platform_code"]}))
            self._emit_status()
            return
        # A short first utterance must still allow the read-only login probe.
        # Otherwise the conservative unknown account cap can prevent both its
        # merge and the request which discovers the actual account limit.
        if head["state"] == "holding_short" and self._login_room == self.room:
            normalized = split_danmaku(head["text"])
            if len(normalized) == 1 and _page_length(normalized[0]) >= MIN_SEND_CHARACTERS:
                self.store.release_holding(head["id"])
                head = self.store.head()
            else:
                self._latest_reason = "waiting_merge"
                self._emit_status()
                return
        if (head["state"] == "verify" and self._verify_queue_id == head["id"]
                and not f_recovery and self._verify_deadline is not None and now >= self._verify_deadline):
            self.store.update(head["id"], "blocked", request_id=head["request_id"], reason="result_unknown")
            self._latest_reason = "result_unknown"
            self._record_failed_content(head, "receipt_unconfirmed")
            self._transfer_to_analysis(head)
            self._verify_queue_id = self._verify_deadline = None
            self._read_not_before = now + self.interval_seconds
            self._emit_status()
            return
        if not self._browser_ready():
            self._latest_reason = "waiting_browser"
            self._emit_status()
            return
        # After three explicit failures, actively recheck authentication once
        # per cycle; do not reload the page or reset its observation baseline.
        if f_recovery:
            retries = self.store.f_recovery(head["id"])["retries"]
            cycle = retries // 3
            if cycle > self._diagnostic_f_cycle.get((head["id"], head["part"]), 0):
                self._diagnostic_f_cycle[(head["id"], head["part"])] = cycle
                self._login_room = ""
                self._log("recovery_environment_check", queue_id=head["id"], part=head["part"], retry_count=retries)
        if self._login_room != self.room:
            operation = "login"
        else:
            if head["state"] == "pending" or length_rejection:
                head = self._prepare_head(head)
                if head is None:
                    self._emit_status()
                    return
                if head["state"] == "blocked":
                    self._latest_reason = head["reason"]
                    self._emit_status()
                    return
            operation = "check" if head["state"] == "verify" or auto_verify else "send"
        send_at = max(self._next_at, self._validator.next_send_at if self._validator is not None else 0.0)
        if operation == "send" and now < send_at:
            self._latest_reason = "waiting_f_retry" if f_recovery else "cooldown"
            self._emit_status()
            return
        if operation == "send" and _page_length(head["text"]) < MIN_SEND_CHARACTERS:
            self.store.update(head["id"], "holding_short", reason="waiting_merge")
            self._latest_reason = "waiting_merge"
            self._emit_status()
            return
        if auto_verify:
            self._recovery_status(head, failure_decision("result_unknown"))
        self._dispatch(operation, head)

    def _dispatch(self, operation, head):
        if operation == "send" and not is_chinese_output(head["text"]):
            self._block_non_chinese_head(head)
            return
        request_id = uuid.uuid4().hex
        now = self._clock()
        retry_f = operation == "send" and self.store.f_recovery(head["id"])["active"]
        previous_baseline = self.store.delivery_baseline(head["id"])
        filter_was_frozen = self.store.filter_context(head["id"])["frozen"] if operation == "send" else True
        # Persist before handing POST to the browser. A crash between the two
        # operations becomes verification, which cannot duplicate a sent item.
        if operation == "send":
            self.store.set_filter_frozen(head["id"], True)
            if retry_f:
                self.store.begin_f_retry(head["id"], request_id)
            else:
                self.store.set_delivery_baseline(head["id"], None)
                self.store.update(head["id"], "sending", request_id=request_id, platform_code=None)
        elif operation == "check" and self._verify_queue_id != head["id"] and not self.store.f_recovery(head["id"])["active"]:
            self._verify_queue_id = head["id"]
            self._verify_deadline = now + self._verify_window_seconds
        self._inflight = {"operation": operation, "request_id": request_id, "row": head, "started": now,
                          "retry_f": retry_f, "previous_baseline": previous_baseline,
                          "filter_was_frozen": filter_was_frozen}
        text = head["text"] if operation in {"send", "check"} else ""
        try:
            accepted = self.browser.request_danmaku(operation, request_id, head["room"], text)
        except Exception:
            # An exception can happen after the child process received a POST.
            # Unlike an explicit false return, it is not proof of non-submission.
            self._inflight = None
            if operation in {"send", "check"}:
                if operation == "check" and self.store.f_recovery(head["id"])["active"]:
                    self._continue_f_verification(head, request_id, now)
                else:
                    self.store.stop_f_recovery(head["id"])
                    self.store.update(head["id"], "blocked", request_id=request_id, reason="result_unknown")
                    self._latest_reason = "result_unknown"
                    if operation == "send":
                        self._record_failed_content(head, "result_unknown")
                        self._transfer_to_analysis(head)
                    else:
                        self._record_failed_content(head, "receipt_unconfirmed")
                        self._transfer_to_analysis(head)
            else:
                self._login_room = ""
                self._latest_reason = "waiting_connection"
            self._read_not_before = now + self.interval_seconds
            if operation == "send":
                self._next_at = now + self.interval_seconds
            self._log("request_exception", operation=operation, queue_id=head["id"], request_id=request_id)
            self._emit_status()
            return
        if not accepted and self._inflight and self._inflight["request_id"] == request_id:
            self._inflight = None
            if operation == "send":
                if not filter_was_frozen:
                    self.store.set_filter_frozen(head["id"], False)
                    self._last_filter_revision = None
                if retry_f:
                    self.store.activate_f_recovery(head["id"])
                    self.store.set_delivery_baseline(head["id"], previous_baseline)
                self.store.update(head["id"], "retry_pending" if retry_f else "pending", reason="f_retry_pending" if retry_f else "not_submitted")
            self._login_room = ""
            self._latest_reason = "waiting_connection"
            self._read_not_before = now + self.interval_seconds
            if operation == "send":
                self._next_at = now + self.interval_seconds
        elif self._inflight:
            self._latest_reason = ("retrying_f" if retry_f else "waiting_f_retry" if self.store.f_recovery(head["id"])["active"] else
                                   "checking_login" if operation == "login" else "verifying" if operation == "check" else "sending")
        self._log("dispatch", operation=operation, queue_id=head["id"], request_id=request_id,
                  characters=_page_length(text), weighted_length=danmaku_weighted_length(text),
                  max_weighted_length=self.max_weighted_length, retry_f=retry_f,
                  retry_count=self.store.f_recovery(head["id"])["retries"])
        self._emit_status()

    def _on_result(self, result):
        if self._closed or not self._inflight or result.get("request_id") != self._inflight["request_id"]:
            return
        flight, self._inflight = self._inflight, None
        operation, head = flight["operation"], flight["row"]
        if result.get("operation", operation) != operation:
            self._inflight = flight
            return
        payload = result.get("result")
        if not isinstance(payload, dict):
            payload = {}
        state = payload.get("outcome", "")
        now = self._clock()
        # Only POST delivery starts the send cooldown. Read-only login/presence
        # requests remain serialized without doubling the subtitle interval.
        if operation == "send":
            self._next_at = now + self.interval_seconds
        self._read_not_before = now
        reason = ""
        code = payload.get("code")
        baseline = self.store.delivery_baseline(head["id"])
        occurrences = payload.get("occurrence_count")
        newly_present = (baseline is not None and type(occurrences) is int
                         and baseline < occurrences <= 10000)
        if operation == "send" and "max_weighted_length" in payload:
            self._apply_limits(payload)
        if operation == "send" and state != "not_sent":
            before_count = payload.get("before_count")
            if (_finite_protocol_number(payload.get("request_started_epoch_ms"), 10**15) is not None
                    and type(before_count) is int and 0 <= before_count <= 10000):
                self.store.set_delivery_baseline(head["id"], before_count)
        if operation == "login":
            if payload.get("is_login") is True and code in {None, 0} and payload.get("error") in {None, "", "none"}:
                self._apply_limits(payload)
                self._login_room = head["room"]
                reason = "waiting_f_retry" if self.store.f_recovery(head["id"])["active"] else "waiting"
            else:
                self._login_room = ""
                explicit_logout = (payload.get("error") in {None, "", "none"} and code == 0
                                   and payload.get("is_login") is False)
                reason = "login_required" if explicit_logout else "waiting_connection"
                if explicit_logout:
                    self._recovery_status(head, failure_decision("not_sent", {"error": "not_logged_in"}))
                self._read_not_before = now + self.interval_seconds
        elif (operation == "check" and payload.get("present") is True and payload.get("room_matches") is True
              and payload.get("chat_ready") is True and payload.get("error") in {None, "", "none"}
              and newly_present):
            self._resolve_failed_content(head)
            self.store.mark_page_visible(head["id"], flight["request_id"])
            self._verify_queue_id = self._verify_deadline = None
            reason = "sent_page_visible"
        elif operation == "send" and state == "not_sent":
            self._record_failed_content(head, "not_sent", payload)
            if not flight["filter_was_frozen"]:
                self.store.set_filter_frozen(head["id"], False)
                self._last_filter_revision = None
            retry_f = flight.get("retry_f", False)
            if retry_f:
                self.store.activate_f_recovery(head["id"])
                self.store.set_delivery_baseline(head["id"], flight["previous_baseline"])
            self.store.update(head["id"], "retry_pending" if retry_f else "pending", reason="f_retry_pending" if retry_f else "not_submitted")
            self._login_room = ""
            self._read_not_before = now + self.interval_seconds
            reason = "waiting_connection"
        elif (state in {"platform_f", "platform_rejected"}
              or (code is not None and code != 0)):
            rejection = "platform_f" if state == "platform_f" else "platform_rejected"
            retryable_f = (operation == "send" and state == "platform_f" and type(code) is int and code == 0
                           and payload.get("http_status") == 200 and payload.get("error") in {None, "", "none"})
            if retryable_f:
                self.store.activate_f_recovery(head["id"], new_f=True)
                self._f_retry_queue_id = head["id"]
                self._verify_queue_id = self._verify_deadline = None
            else:
                self.store.stop_f_recovery(head["id"])
                if rejection == "platform_f":
                    rejection = "platform_rejected" if code not in {None, 0} else "result_unknown"
            self.store.update(head["id"], "verify" if retryable_f else "blocked", request_id=flight["request_id"], reason=rejection, platform_code=code)
            if operation == "send":
                self.store.increment("failures")
                if code == 1003212 and rejection == "platform_rejected":
                    self._login_room = ""
            reason = "waiting_f_retry" if retryable_f else rejection
        elif operation == "send":
            self.store.stop_f_recovery(head["id"])
            if state == "receipt_accepted" and code == 0:
                self.store.increment("accepted")
            self.store.update(head["id"], "verify", request_id=flight["request_id"], reason="receipt_unconfirmed", platform_code=code)
            self._verify_queue_id = head["id"]
            self._verify_deadline = now + self._verify_window_seconds
            reason = "awaiting_verification"
        else:
            f_recovery = self.store.f_recovery(head["id"])["active"]
            if f_recovery:
                check_ready = (payload.get("room_matches") is True and payload.get("chat_ready") is True
                               and type(payload.get("present")) is bool and payload.get("error") in {None, "", "none"}
                               and type(occurrences) is int and 0 <= occurrences <= 10000)
                repaired = (check_ready and self._repair_f_head_rules(head, baseline,
                            occurrences, payload.get("present")))
                if not repaired:
                    self._continue_f_verification(head, flight["request_id"], now,
                                                  ready=check_ready and not self._filter_blocked)
                reason = "filter_error" if self._filter_blocked else "waiting_f_retry"
                if not check_ready:
                    self._login_room = ""
            # A check that cannot establish page visibility does not prove a
            # failure. Retain the exact queue head rather than POSTing it twice.
            elif self._verify_deadline is not None and now < self._verify_deadline:
                self.store.update(head["id"], "verify", request_id=flight["request_id"], reason="receipt_unconfirmed")
                self._read_not_before = now + 0.5
                reason = "awaiting_verification"
            else:
                self.store.update(head["id"], "blocked", request_id=flight["request_id"], reason="result_unknown")
                self._record_failed_content(head, "receipt_unconfirmed", payload)
                transferred = self._transfer_to_analysis(head)
                self._read_not_before = now + self.interval_seconds
                reason = "failure_transferred" if transferred else "result_unknown"
        self._latest_reason = reason
        if operation == "send" and state != "not_sent" and not (state == "receipt_accepted" and code == 0):
            failure_reason = "platform_f" if state == "platform_f" else "platform_rejected" if state == "platform_rejected" or code not in {None, 0} else "result_unknown"
            self._record_failed_content(head, failure_reason, payload)
            self._transfer_to_analysis(head)
        elif operation == "send" and state == "not_sent":
            self.store.update(head["id"], "blocked", request_id=flight["request_id"], reason="not_sent")
            self._transfer_to_analysis(head)
        timing_fields = {}
        if operation == "send":
            timing_fields = {
                "request_started_epoch_ms": _finite_protocol_number(payload.get("request_started_epoch_ms"), 10**15),
                "request_elapsed_ms": _finite_protocol_number(payload.get("request_elapsed_ms"), 120000)}
        self._log("result", operation=operation, queue_id=head["id"], request_id=flight["request_id"],
                  reason=reason, elapsed_seconds=round(now - flight["started"], 3),
                  outcome=state if state in {"receipt_accepted", "platform_f", "platform_rejected", "result_unknown", "not_sent"} else "",
                  characters=_page_length(head["text"]) if operation != "login" else 0,
                  weighted_length=danmaku_weighted_length(head["text"]) if operation != "login" else 0,
                  max_weighted_length=self.max_weighted_length,
                  before_count=payload.get("before_count") if operation == "send" and type(payload.get("before_count")) is int and 0 <= payload["before_count"] <= 10000 else None,
                  occurrence_count=occurrences if operation == "check" and type(occurrences) is int and 0 <= occurrences <= 10000 else None,
                  platform_code=code if type(code) is int and -2**31 <= code < 2**31 else None,
                  http_status=payload.get("http_status") if type(payload.get("http_status")) is int else None,
                  has_f=state == "platform_f", retry_f=flight.get("retry_f", False),
                  retry_count=self.store.f_recovery(head["id"])["retries"],
                  error=payload.get("error") if isinstance(payload.get("error"), str) and payload.get("error") in SAFE_ERRORS else "invalid_response", **timing_fields)
        if not self.enabled:
            self.timer.stop()
        self._emit_status()

    def verify_pending(self):
        """Explicitly recheck a blocked head; this operation never issues POST."""
        head = self.store.head()
        if not self._closed and head and head["state"] == "blocked" and not self._inflight and not self.store.f_recovery(head["id"])["active"]:
            self.store.update(head["id"], "verify", request_id=head["request_id"], reason="manual_verify")
            self._verify_queue_id = self._verify_deadline = None
            self._latest_reason = "awaiting_verification"
            self._emit_status()

    def resume(self):
        """User-requested recovery; uncertainty is checked, rejection reauthenticates."""
        head = self.store.head()
        if self._closed or not head or self._inflight or head["state"] != "blocked" or self.store.f_recovery(head["id"])["active"]:
            return
        if head["reason"] == "platform_rejected":
            self.store.update(head["id"], "pending", reason="manual_resume")
            self._login_room = ""
        else:
            self.store.update(head["id"], "verify", request_id=head["request_id"], reason="manual_verify")
            self._verify_queue_id = self._verify_deadline = None
        self._latest_reason = "awaiting_verification"
        if self.enabled:
            self.timer.start()
        self._emit_status()

    def snapshot(self):
        counts = self.store.counts()
        head = self.store.head()
        recovery = self.store.f_recovery(head["id"]) if head else {"active": False, "retries": 0, "last_f_at": None}
        f_active = recovery["active"] or bool(self._inflight and self._inflight.get("retry_f"))
        protected = self._inflight["row"]["id"] if self._inflight else None
        clearing = self.store.clearable_counts(protected)
        auto_verify = bool(head and head["state"] in {"blocked", "verify"} and head["reason"] in {
            "result_unknown", "receipt_unconfirmed", "manual_verify", "restart_verify"})
        state = self._latest_reason
        if self.enabled and self._ai_proposal:
            state = "ai_repair_ready"
        elif self.enabled and self._ai_task:
            state = "ai_analyzing"
        elif self.enabled and auto_verify and state in {"result_unknown", "verifying", "awaiting_verification"}:
            state = "auto_verifying"
        if self._ai_task and self._ai_task.get("historical"):
            state = "ai_analyzing"
        elif self._validator is not None and self._validator.state not in {"validation_idle", "waiting_validation"}:
            state = self._validator.state
        saving_failure = bool(self._handoff_restore_error or (head and
            self.store.recovery_context(head["id"], head["part"]).get("handoff_pending")))
        if saving_failure:
            state = "saving_failed_record"
            f_active = auto_verify = False
        decision = self._recovery_decision
        return {"enabled": self.enabled, "room": self.room, "source": self.source,
                "interval_seconds": self.interval_seconds, "queued": counts["queued"],
                "holding": counts["holding"],
                "page_visible": counts["page_visible"], "accepted": counts["accepted"], "failures": counts["failures"],
                "f_retries": counts["f_retries"], "head_f_retries": recovery["retries"], "f_retry_active": f_active,
                "deferred_analysis": counts.get("deferred_analysis", 0),
                "filter_error": self._filter_error, "filter_candidate_error": self._filter_candidate_error or self._handoff_restore_error,
                "filter_hits": self._filter_hits,
                "filter_rule_count": len(self.content_filter.rules()) if self.content_filter is not None else 0,
                "filter_candidate_count": len(self.content_filter.candidates(self.room or None, include_resolved=False)) if self.content_filter is not None else 0,
                "filter_resolved_count": self.content_filter.status_counts(self.room or None)["resolved"] if self.content_filter is not None else 0,
                "recovery_category": decision.category if decision else "",
                "recovery_action": ("retry_failure_storage" if saving_failure else "apply_local_edits" if self._ai_proposal else "analyze_with_codex" if self._ai_task else decision.action if decision else ""),
                "ai_enabled": self._ai_analyzer is not None,
                "validation_room": self.validation_room,
                "validation_state": self._validator.state if self._validator is not None else "waiting_validation",
                "filter_detection": self.content_filter.detection_info() if self.content_filter is not None else None,
                "last_f_at": datetime.fromtimestamp(recovery["last_f_at"], timezone(timedelta(hours=8))).isoformat(timespec="milliseconds") if recovery["last_f_at"] is not None else "",
                "cleared": counts["cleared"],
                "state": state,
                "blocked": bool(head and head["state"] == "blocked"),
                "automatic_recovery": bool(saving_failure or auto_verify or f_active or self._ai_task or self._ai_proposal),
                "queue_room": head["room"] if head else "",
                "queue_head_code": head["platform_code"] if head else None,
                "queue_head_length": _page_length(head["text"]) if head else 0,
                "last_platform_code": head["platform_code"] if head else None,
                "head_characters": _page_length(head["text"]) if head else 0,
                "head_weighted_length": danmaku_weighted_length(head["text"]) if head else 0,
                "max_weighted_length": self.max_weighted_length,
                **clearing,
                "inflight": self._inflight["operation"] if self._inflight else "",
                "in_flight": self._inflight["operation"] if self._inflight else "",
                "cooldown_remaining": max(0.0, self._next_at - self._clock()),
                "wait_seconds": max(0.0, self._next_at - self._clock())}

    def clear_pending(self):
        """Explicit cancellation preserves every uncertain or in-flight item."""
        if self._closed:
            return {"removed": 0, "retained": 0, "retained_uncertain": 0}
        protected = self._inflight["row"]["id"] if self._inflight else None
        result = self.store.clear_pending(protected)
        if not self._inflight:
            head = self.store.head()
            if not self.enabled:
                self._latest_reason = "disabled"
            elif head is None:
                self._latest_reason = "empty"
            elif head["room"] != self.room:
                self._latest_reason = "queued_room_mismatch"
            elif head["state"] == "blocked":
                self._latest_reason = head["reason"] or "result_unknown"
            else:
                self._latest_reason = "waiting_f_retry" if self.store.f_recovery(head["id"])["active"] else {"holding_short": "waiting_merge", "verify": "awaiting_verification"}.get(head["state"], "waiting")
        self._log("clear_pending", **result)
        self._emit_status()
        return result

    def _emit_status(self):
        self.status.emit(self.snapshot())

    def _log(self, kind, **fields):
        identifier = fields.get("queue_id")
        if type(identifier) is int:
            recovery = self.store.f_recovery(identifier)
            fields.update(f_retries=self.store.counts()["f_retries"], head_f_retries=recovery["retries"],
                          f_retry_active=recovery["active"] or bool(self._inflight and self._inflight.get("retry_f")),
                          last_f_at=datetime.fromtimestamp(recovery["last_f_at"], timezone(timedelta(hours=8))).isoformat(timespec="milliseconds") if recovery["last_f_at"] is not None else "")
        record = {"time": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds"),
                  "kind": kind, "room": self.room, **fields}
        try:
            with self.log_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def close(self):
        if not self._closed:
            self._closed = True
            self.timer.stop()
            self.review_timer.stop()
            if self._validator is not None:
                self._validator.close()
            if self._ai_analyzer is not None:
                self._ai_analyzer.close()
            self.store.close()
