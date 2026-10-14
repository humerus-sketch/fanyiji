"""Serialized, isolated posting checks for AI advice; no main-queue entries."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
import uuid

from PySide6.QtCore import QObject

from .danmaku_protocol import (danmaku_utf16_length, danmaku_weighted_length, valid_danmaku_text,
                              danmaku_body_length, MIN_DANMAKU_BODY_CHARACTERS, MAX_DANMAKU_WEIGHTED_LENGTH)
from .translation_output import is_chinese_output


class DanmakuValidationRunner(QObject):
    def __init__(self, browser, content_filter, parent=None, *, log=None, clock=None):
        super().__init__(parent)
        self.browser, self.content_filter = browser, content_filter
        self._clock = clock or time.monotonic
        self._log = log or (lambda *args, **kwargs: None)
        self._flight = None
        self._deferred_result = None
        self._next_at = 0.0
        self._login = ""
        self._closed = False
        self._enabled = True
        self.state = "waiting_validation"
        self._limits = 20
        self._rooms = ("", "")
        self._last_candidate_id = None
        self.browser.danmaku_result.connect(self._on_result)
        if hasattr(self.browser, "event"):
            self.browser.event.connect(self._event)

    @property
    def busy(self):
        return self._flight is not None or self._deferred_result is not None

    @property
    def next_send_at(self):
        return self._next_at

    def set_rooms(self, test_room, formal_room):
        rooms = (test_room, formal_room)
        if rooms != self._rooms:
            self._login = ""
        self._rooms = rooms

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        if not self._enabled:
            self.state = "validation_paused"

    def _event(self, event):
        if event.get("kind") in {"closed", "error", "starting"}:
            self._login = ""

    def tick(self, test_room, formal_room, interval, *, earliest_post=0.0):
        if self._closed or not self._enabled:
            return
        self.set_rooms(test_room, formal_room)
        now = self._clock()
        if self._deferred_result is not None:
            if now >= self._next_at:
                flight, result = self._deferred_result
                self._deferred_result = None
                self._flight = flight
                self._on_result(result)
            if self._deferred_result is not None:
                return
        if self._flight:
            if now - self._flight["started"] >= 45:
                flight, self._flight = self._flight, None
                self._login = ""
                self._next_at = now + interval
                # A lost POST result remains testing. A fresh page observation
                # cannot replace the absent receipt, so never blindly repost.
                self._log("validation_timeout", candidate_id=flight["candidate_id"], operation=flight["operation"])
            return
        if now < self._next_at:
            return
        plans = []
        for candidate in self.content_filter.candidates(include_resolved=False):
            plan = self.content_filter.validation_plan(candidate["id"])
            if plan is not None and plan["state"] != "passed":
                plans.append((candidate, plan))
        if not plans:
            self.state = "validation_idle"
            return
        if not test_room or (formal_room and int(test_room) == int(formal_room)):
            self.state = "waiting_validation"
            return
        # Failed plans are preserved for diagnosis/reanalysis. They must not
        # starve other independent corrections queued for validation.
        eligible = [(candidate, plan) for candidate, plan in plans if plan["state"] != "failed"]
        if not eligible:
            self.state = "validation_failed"
            return
        previous = next((index for index, (candidate, _) in enumerate(eligible)
                         if candidate["id"] == self._last_candidate_id), -1)
        candidate, plan = eligible[(previous + 1) % len(eligible)]
        if plan["test_room"] != test_room or plan["formal_room"] != formal_room:
            if plan["state"] == "testing":
                # Configuration changes cannot turn an outstanding POST into
                # permission to submit the same test sentence elsewhere.
                self.state = "validation_configuration_changed"
                self._last_candidate_id = candidate["id"]
                return
            self.content_filter.configure_validation(candidate["id"], test_room, formal_room=formal_room)
            plan = self.content_filter.validation_plan(candidate["id"])
        if not getattr(self.browser, "is_browser_ready", False):
            self.state = "waiting_validation_browser"
            return
        text = "【" + plan["revised_text"] + "】"
        if plan["state"] != "testing" and not is_chinese_output(text):
            self.content_filter.update_validation(candidate["id"], state="failed",
                message="修正方案仍含外语 必须先完整转换为中文")
            self.state = "validation_failed"
            return
        testing = plan["state"] == "testing"
        if (not valid_danmaku_text(text, normalized_only=not testing, max_characters=76 if testing else 40)
                or (not testing and (danmaku_body_length(text) < MIN_DANMAKU_BODY_CHARACTERS
                                     or danmaku_weighted_length(text) > MAX_DANMAKU_WEIGHTED_LENGTH))):
            self.content_filter.update_validation(candidate["id"], state="failed",
                message="修正方案须至少2个正文字符且最多40计数单位 含【】 中文2 英文数字空格1")
            self.state = "validation_failed"
            return
        if self._login != test_room:
            operation = "login"
        elif plan["state"] == "testing":
            operation = "check"
        else:
            if now < earliest_post:
                return
            if danmaku_weighted_length(text) > self._limits:
                self.content_filter.update_validation(candidate["id"], state="failed",
                    message="修正方案超过当前账号单条长度限制")
                self.state = "validation_failed"
                return
            operation = "send"
        self._dispatch(candidate, plan, operation, text, interval)

    def _dispatch(self, candidate, plan, operation, text, interval):
        if operation == "send" and not is_chinese_output(text):
            self.content_filter.update_validation(candidate["id"], state="failed",
                message="修正方案仍含外语 已阻止发送")
            self.state = "validation_failed"
            return
        if operation == "send" and (not valid_danmaku_text(text, normalized_only=True)
                or danmaku_body_length(text) < MIN_DANMAKU_BODY_CHARACTERS
                or danmaku_weighted_length(text) > min(MAX_DANMAKU_WEIGHTED_LENGTH, self._limits)):
            self.content_filter.update_validation(candidate["id"], state="failed",
                message="修正方案须至少2个正文字符且最多40计数单位 含【】 中文2 英文数字空格1")
            self.state = "validation_failed"
            return
        if operation != "login":
            self._last_candidate_id = candidate["id"]
        request_id = uuid.uuid4().hex
        flight = {"request_id": request_id, "operation": operation, "candidate_id": candidate["id"],
                  "validation_id": plan["validation_id"], "room": plan["test_room"], "text": text,
                  "started": self._clock(), "interval": max(3.5, interval)}
        if operation == "send":
            self.content_filter.update_validation(candidate["id"], state="testing", request_id=request_id)
        self._flight = flight
        try:
            accepted = self.browser.request_danmaku(operation, request_id, flight["room"],
                                                   "" if operation == "login" else text)
        except Exception:
            accepted = None  # It could have reached the browser; do not replay.
        if accepted is not True and self._flight:
            self._flight = None
            self._login = ""
            self._next_at = self._clock() + flight["interval"]
            if accepted is False and operation == "send":
                self.content_filter.update_validation(candidate["id"], state="ready", message="测试请求尚未提交")
        self.state = "validating"
        self._log("validation_dispatch", candidate_id=candidate["id"], validation_id=plan["validation_id"],
                  test_room=flight["room"], operation=operation, request_id=request_id,
                  characters=danmaku_utf16_length(text) if operation != "login" else 0)

    def _on_result(self, result):
        flight = self._flight
        try:
            self._process_result(result)
        except Exception as failure:
            if flight is not None and result.get("request_id") == flight["request_id"]:
                self._deferred_result = (flight, result)
                self._flight = None
                self._next_at = self._clock() + flight["interval"]
                self.state = "validation_storage_error"
                self._log("validation_storage_error", candidate_id=flight["candidate_id"],
                          operation=flight["operation"], error_type=type(failure).__name__)

    def _process_result(self, result):
        flight = self._flight
        if self._closed or flight is None or result.get("request_id") != flight["request_id"]:
            return
        if result.get("operation", flight["operation"]) != flight["operation"]:
            return
        self._flight = None
        payload = result.get("result", {})
        if not isinstance(payload, dict):
            payload = {}
        plan = self.content_filter.validation_plan(flight["candidate_id"])
        if (plan is None or plan["validation_id"] != flight["validation_id"] or plan["test_room"] != flight["room"]
                or (plan["test_room"], plan["formal_room"]) != self._rooms):
            return
        now, operation = self._clock(), flight["operation"]
        error = payload.get("error")
        clean = error in {None, "", "none"} if isinstance(error, (str, type(None))) else False
        self._next_at = now + flight["interval"]
        if operation == "login":
            if clean and payload.get("is_login") is True and payload.get("code") == 0:
                limit = payload.get("max_weighted_length")
                if type(limit) is int and 6 <= limit <= 80:
                    self._limits = min(MAX_DANMAKU_WEIGHTED_LENGTH, limit)
                self._login = flight["room"]
                self._next_at = now
            else:
                self._login = ""
                self.state = "waiting_validation_browser"
        elif operation == "send":
            outcome = payload.get("outcome")
            baseline = payload.get("before_count")
            if type(baseline) is not int or not 0 <= baseline <= 10000:
                baseline = None
            if outcome == "not_sent":
                self.content_filter.record_validation_receipt(flight["candidate_id"],
                    request_id=flight["request_id"], before_count=baseline,
                    receipt_accepted=False, code=payload.get("code"), http_status=payload.get("http_status"),
                    error="not_sent", request_started=False)
                self._login = ""
            elif outcome in {"platform_f", "platform_rejected"}:
                self.content_filter.record_validation_receipt(flight["candidate_id"],
                    request_id=flight["request_id"], before_count=baseline,
                    receipt_accepted=False, code=payload.get("code"), http_status=payload.get("http_status"),
                    error=outcome)
                self.content_filter.update_validation(flight["candidate_id"], state="failed",
                    message="测试房间未接受修正方案 不启用规则")
                self.state = "validation_failed"
            else:
                accepted = (outcome == "receipt_accepted" and payload.get("code") == 0
                            and payload.get("http_status") == 200 and clean)
                self.content_filter.record_validation_receipt(flight["candidate_id"],
                    request_id=flight["request_id"], before_count=baseline,
                    receipt_accepted=accepted, code=payload.get("code"),
                    http_status=payload.get("http_status"), error="none" if clean else "other")
        elif operation == "check":
            if not self._enabled:
                # A stop cannot undo a submitted request. Keep its receipt and
                # resume observation later, without creating a rule after stop.
                self.state = "validation_paused"
                return
            receipt = plan.get("receipt") or {}
            baseline, count = receipt.get("before_count"), payload.get("occurrence_count")
            if (receipt.get("receipt_accepted") is True and clean and payload.get("room_matches") is True
                    and payload.get("chat_ready") is True and payload.get("present") is True
                    and type(baseline) is int and type(count) is int and baseline < count <= 10000):
                self.content_filter.confirm_validation(flight["candidate_id"], evidence={
                    "test_room": flight["room"], "validation_id": plan["validation_id"],
                    "request_id": plan["request_id"], "observed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds"),
                    "receipt_accepted": True, "page_observed": True, "before_count": baseline,
                    "occurrence_count": count, "code": 0, "http_status": 200, "error": "none"})
                self.state = "validation_passed"
                self._log("validation_passed", candidate_id=flight["candidate_id"], test_room=flight["room"],
                          validation_id=plan["validation_id"], before_count=baseline, occurrence_count=count,
                          observation="page_text_only", confirmed_cause=False)
            elif not clean or payload.get("room_matches") is not True:
                self._login = ""
                self.state = "waiting_validation_browser"
        self._log("validation_result", candidate_id=flight["candidate_id"], operation=operation,
                  request_id=flight["request_id"], test_room=flight["room"],
                  receipt_accepted=payload.get("outcome") == "receipt_accepted",
                  code=payload.get("code") if type(payload.get("code")) is int else None)

    def close(self):
        self._closed = True
        self._flight = None
        self._deferred_result = None
