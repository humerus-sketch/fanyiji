"""Local text substitutions with provenance and durable failure observations."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timedelta, timezone
import unicodedata
import uuid

from .paths import data_root
from .danmaku_wordlist import DanmakuWordlist, DanmakuWordlistError, MAX_HITS
from .translation_output import is_chinese_output


MAX_RULE_CHARACTERS = 80
MAX_CANDIDATE_CHARACTERS = 32768
DEFAULT_REPLACEMENT = "已过滤"
CANDIDATE_STATUSES = ("active", "recovering", "waiting_confirmation", "resolved", "action_required", "waiting_validation")
MAX_LEARNED_TERM_CHARACTERS = 12
_UNSET = object()


class DanmakuFilterError(ValueError):
    """Readable validation/storage failure; callers must surface this error."""


def _length(text: str) -> int:
    return sum(2 if ord(character) > 0xFFFF else 1 for character in text)


def _text(value: object, label: str, maximum: int, *, empty=False) -> str:
    if not isinstance(value, str):
        raise DanmakuFilterError(f"{label}必须是文字")
    value = value.strip()
    if not value and not empty:
        raise DanmakuFilterError(f"{label}不能为空")
    if _length(value) > maximum:
        raise DanmakuFilterError(f"{label}不能超过{maximum}个字符（按 UTF-16 计数）")
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value):
        raise DanmakuFilterError(f"{label}不能包含控制字符或隐藏字符")
    return value


def _room(value: object) -> str:
    if value is None or value == "*":
        return ""
    if not isinstance(value, str):
        raise DanmakuFilterError("直播间必须是数字房间号；留空表示全部直播间")
    value = value.strip()
    if value and not re.fullmatch(r"[1-9][0-9]{0,19}", value):
        raise DanmakuFilterError("直播间必须是数字房间号；留空表示全部直播间")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise DanmakuFilterError(f"{label}必须是开启或关闭")
    return value


def _action(value: object) -> str:
    if not isinstance(value, str) or value not in {"replace", "remove"}:
        raise DanmakuFilterError("过滤动作必须是 replace 或 remove")
    return value


def _replacement(value: object, action: str) -> str:
    text = _text(value, "替换内容", MAX_RULE_CHARACTERS, empty=True)
    return "" if action == "remove" else text or DEFAULT_REPLACEMENT


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise DanmakuFilterError("本地过滤名单的记录 ID 无效")
    return value


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def _time(value: object) -> str:
    if not isinstance(value, str):
        raise DanmakuFilterError("本地过滤名单的时间格式无效")
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise DanmakuFilterError("本地过滤名单的时间格式无效") from None
    if parsed.utcoffset() != timedelta(hours=8):
        raise DanmakuFilterError("本地过滤名单的时间必须使用 UTC+8")
    return value


def _queue_id(value: object) -> int | None:
    if value is not None and (type(value) is not int or not 0 < value <= 2**63 - 1):
        raise DanmakuFilterError("失败候选的队列 ID 必须是正整数")
    return value


def _candidate_status(value: object) -> str:
    if not isinstance(value, str) or value not in CANDIDATE_STATUSES:
        raise DanmakuFilterError("失败候选的处理状态无效")
    return value


def _diagnosis_code(value: object, label: str, *, empty=False) -> str:
    if value == "" and empty:
        return ""
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise DanmakuFilterError(f"{label}必须是小写英文状态代码")
    return value


def _diagnosis(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"category", "action", "message"}:
        raise DanmakuFilterError("失败候选的诊断格式无效")
    return {"category": _diagnosis_code(value["category"], "诊断类别"),
            "action": _diagnosis_code(value["action"], "诊断动作"),
            "message": _text(value["message"], "诊断说明", 512, empty=True)}


def _resolution(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"at", "resolution", "queue_id"}:
        raise DanmakuFilterError("失败候选的恢复记录格式无效")
    return {"at": _time(value["at"]), "resolution": _diagnosis_code(value["resolution"], "恢复结果"),
            "queue_id": _queue_id(value["queue_id"])}


def _rule_evidence(value: object, origin: str) -> dict | None:
    if origin == "user":
        if value is not None:
            raise DanmakuFilterError("人工规则不能携带自动学习证据")
        return None
    if origin == "ai_analysis":
        keys = {"candidate_id", "source_digest", "learned_at", "confirmed_cause"}
        if not isinstance(value, dict) or set(value) != keys:
            raise DanmakuFilterError("AI 建议规则的分析证据格式无效")
        digest = value["source_digest"]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise DanmakuFilterError("AI 建议规则的原文摘要无效")
        if value["confirmed_cause"] is not False:
            raise DanmakuFilterError("AI 建议不能宣称已证实平台拒绝原因")
        return {"candidate_id": _identifier(value["candidate_id"]), "source_digest": digest,
                "learned_at": _time(value["learned_at"]), "confirmed_cause": False}
    keys = {"queue_id", "failed_digest", "revised_digest", "failure_count", "learned_at",
            "observed_at", "confirmed_cause"}
    if not isinstance(value, dict) or set(value) != keys:
        raise DanmakuFilterError("经验规则的观察证据格式无效")
    queue_id = _queue_id(value["queue_id"])
    if queue_id is None or type(value["failure_count"]) is not int or not 1 <= value["failure_count"] <= 2**63 - 1:
        raise DanmakuFilterError("经验规则缺少有效失败次数或队列关联")
    digests = (value["failed_digest"], value["revised_digest"])
    if any(not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest) for digest in digests) or digests[0] == digests[1]:
        raise DanmakuFilterError("经验规则必须关联两个不同的正文摘要")
    if value["confirmed_cause"] is not False:
        raise DanmakuFilterError("经验规则不能宣称已证实平台拒绝原因")
    return {"queue_id": queue_id, "failed_digest": digests[0], "revised_digest": digests[1],
            "failure_count": value["failure_count"], "learned_at": _time(value["learned_at"]),
            "observed_at": _time(value["observed_at"]), "confirmed_cause": False}


def _canonical_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _unwrapped_body(text: str) -> str:
    text = _canonical_whitespace(text)
    return text[1:-1].strip() if text.startswith("【") and text.endswith("】") else text


def _replay_local_edits(original: str, edits: list[dict]) -> tuple[dict[str, str], str]:
    if not isinstance(edits, list) or not 1 <= len(edits) <= 8:
        raise DanmakuFilterError("自动学习只接受一至八个局部编辑")
    replacements = {}
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"term", "replacement"}:
            raise DanmakuFilterError("自动学习的局部编辑格式无效")
        term = _text(edit["term"], "经验原词", MAX_LEARNED_TERM_CHARACTERS)
        replacement = _text(edit["replacement"], "经验替换", MAX_LEARNED_TERM_CHARACTERS, empty=True)
        if (_length(term) < 2 or len(term) < 2 or term == original.strip("【】 ")
                or term not in original or term == replacement or term in replacements
                or any(character in "【】" for character in term + replacement)):
            raise DanmakuFilterError("自动学习只能保存实际修改的短词，不能保存整句或重复编辑")
        replacements[term] = replacement
    used = set()
    pattern = re.compile("|".join(re.escape(term) for term in sorted(replacements, key=lambda term: (-len(term), term))))
    def replay(match):
        used.add(match.group(0))
        return replacements[match.group(0)]
    revised = pattern.sub(replay, original)
    if used != set(replacements):
        raise DanmakuFilterError("局部编辑单次重放与实际观察的修正正文不一致")
    if (_canonical_whitespace(revised) == _canonical_whitespace(original)
            or not _canonical_whitespace(revised).strip("【】 ")):
        raise DanmakuFilterError("自动学习必须有有效局部修改且不能删除全部正文")
    return replacements, revised


def _digest(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise DanmakuFilterError("验证正文摘要无效")
    return value


def _validated_proof(value: object) -> dict | None:
    if value is None:
        return None
    keys = {"test_room", "formal_room", "validation_id", "request_id", "observed_at", "before_count",
            "occurrence_count", "source_digest", "revised_digest", "confirmed_cause", "receipt_accepted",
            "page_observed", "code", "http_status", "error"}
    if not isinstance(value, dict) or set(value) != keys:
        raise DanmakuFilterError("专用测试房间验证证据格式无效")
    test_room, formal_room = _room(value["test_room"]), _room(value["formal_room"])
    before, after = value["before_count"], value["occurrence_count"]
    if (not test_room or test_room == formal_room or type(before) is not int or type(after) is not int
            or not 0 <= before < after <= 10000 or value["confirmed_cause"] is not False
            or value["receipt_accepted"] is not True or value["page_observed"] is not True
            or type(value["code"]) is not int or value["code"] != 0
            or type(value["http_status"]) is not int or value["http_status"] != 200
            or not isinstance(value["error"], str) or value["error"] not in {"", "none"}):
        raise DanmakuFilterError("必须有独立测试房间正常回执及新增页面观察")
    return {**value, "test_room": test_room, "formal_room": formal_room,
            "validation_id": _identifier(value["validation_id"]), "request_id": _identifier(value["request_id"]),
            "observed_at": _time(value["observed_at"]), "source_digest": _digest(value["source_digest"]),
            "revised_digest": _digest(value["revised_digest"])}


def _validation_receipt(value: object) -> dict | None:
    if value is None:
        return None
    keys = {"request_id", "before_count", "receipt_accepted", "code", "http_status", "error", "request_started",
            "recorded_at", "test_room", "validation_id"}
    if not isinstance(value, dict) or set(value) != keys:
        raise DanmakuFilterError("验证请求回执格式无效")
    before = value["before_count"]
    if before is not None and (type(before) is not int or not 0 <= before <= 10000):
        raise DanmakuFilterError("验证请求的页面基线无效")
    for key in ("code", "http_status"):
        if value[key] is not None and type(value[key]) is not int:
            raise DanmakuFilterError("验证请求的回执状态无效")
    accepted, started = _boolean(value["receipt_accepted"], "验证回执接受标记"), _boolean(value["request_started"], "验证请求提交标记")
    error = _text(value["error"], "验证回执错误", 128, empty=True)
    if accepted and (not started or value["code"] != 0 or value["http_status"] != 200 or error not in {"", "none"}):
        raise DanmakuFilterError("验证接受回执与提交状态不一致")
    test_room = _room(value["test_room"])
    if not test_room:
        raise DanmakuFilterError("验证回执缺少专用测试房间")
    return {**value, "request_id": _identifier(value["request_id"]), "before_count": before,
            "receipt_accepted": accepted, "request_started": started, "recorded_at": _time(value["recorded_at"]),
            "test_room": test_room, "validation_id": _identifier(value["validation_id"]), "error": error}


def _explicit_validation_rejection(plan: dict) -> bool:
    receipt = plan.get("receipt")
    if (plan["state"] != "failed" or not receipt or not receipt["request_started"]
            or receipt["receipt_accepted"]):
        return False
    if receipt["error"] == "platform_f":
        return receipt["code"] == 0 and receipt["http_status"] == 200
    return receipt["error"] == "platform_rejected" and (
        (type(receipt["code"]) is int and receipt["code"] != 0)
        or (type(receipt["http_status"]) is int and 400 <= receipt["http_status"] <= 599))


def _validation_can_revise(plan: dict) -> bool:
    receipt = plan.get("receipt")
    submitted = ((receipt is not None and receipt["request_started"])
                 or (receipt is None and bool(plan["request_id"])))
    return plan["state"] != "testing" and (not submitted or _explicit_validation_rejection(plan))


def _validation_plan(value: object, room: str, text: str) -> dict | None:
    if value is None:
        return None
    keys = {"validation_id", "state", "source_room", "formal_room", "test_room", "original_text", "revised_text",
            "edits", "attempts", "request_id", "message", "created_at", "updated_at", "evidence", "validated_rule_ids", "receipt"}
    if not isinstance(value, dict) or set(value) != keys or value.get("state") not in {"waiting_room", "ready", "testing", "failed", "passed"}:
        raise DanmakuFilterError("AI 验证方案格式无效")
    original = _text(value["original_text"], "验证原文", MAX_CANDIDATE_CHARACTERS)
    replacements, replayed = _replay_local_edits(original, value["edits"])
    revised = _text(value["revised_text"], "验证修正正文", MAX_CANDIDATE_CHARACTERS)
    test_room, formal_room = _room(value["test_room"]), _room(value["formal_room"])
    attempts, request_id = value["attempts"], value["request_id"]
    identifiers = value["validated_rule_ids"]
    proof = _validated_proof(value["evidence"])
    receipt = _validation_receipt(value["receipt"])
    if (original != text or _room(value["source_room"]) != room or revised != _unwrapped_body(replayed)
            or (test_room and test_room == formal_room) or type(attempts) is not int or not 0 <= attempts <= 2**63 - 1
            or not isinstance(identifiers, list) or len(identifiers) > 8
            or (value["state"] in {"ready", "testing", "passed"} and not test_room)
            or (value["state"] in {"testing", "passed"} and not request_id)
            or (value["state"] == "passed") != (proof is not None)):
        raise DanmakuFilterError("AI 验证方案与候选或验证结果不一致")
    result = {**value, "validation_id": _identifier(value["validation_id"]), "source_room": room,
              "formal_room": formal_room, "test_room": test_room, "original_text": original, "revised_text": revised,
              "edits": [{"term": term, "replacement": replacement} for term, replacement in replacements.items()],
              "request_id": _identifier(request_id) if request_id else "", "message": _text(value["message"], "验证说明", 512, empty=True),
              "created_at": _time(value["created_at"]), "updated_at": _time(value["updated_at"]),
              "evidence": proof, "receipt": receipt, "validated_rule_ids": [_identifier(identifier) for identifier in identifiers]}
    if receipt is not None and (receipt["request_id"] != request_id or receipt["test_room"] != test_room
            or receipt["validation_id"] != result["validation_id"]):
        raise DanmakuFilterError("验证请求回执与当前方案不一致")
    if proof is not None and (proof["test_room"] != test_room or proof["formal_room"] != formal_room
            or proof["validation_id"] != result["validation_id"] or proof["request_id"] != request_id
            or proof["source_digest"] != hashlib.sha256(original.encode("utf-8")).hexdigest()
            or proof["revised_digest"] != hashlib.sha256(revised.encode("utf-8")).hexdigest()
            or receipt is None or not receipt["receipt_accepted"] or not receipt["request_started"]
            or proof["before_count"] != receipt["before_count"] or proof["code"] != receipt["code"]
            or proof["http_status"] != receipt["http_status"]):
        raise DanmakuFilterError("验证成功证据与当前方案不一致")
    return result


def _validation_history(value: object, room: str, text: str) -> list[dict]:
    if not isinstance(value, list):
        raise DanmakuFilterError("修正验证历史必须是列表")
    history, identifiers = [], set()
    for item in value:
        plan = _validation_plan(item, room, text)
        if plan is None or plan["validation_id"] in identifiers:
            raise DanmakuFilterError("修正验证历史包含无效或重复方案")
        identifiers.add(plan["validation_id"])
        history.append(plan)
    return history


def _archive_validation(candidate: dict, plan: dict | None):
    if plan is not None and not any(item["validation_id"] == plan["validation_id"]
                                   for item in candidate.setdefault("repair_history", [])):
        candidate["repair_history"].append(copy.deepcopy(plan))


def _analysis(value: object, text: str) -> dict:
    """Validate persisted observations without trusting imported analysis fields."""
    if not isinstance(value, dict) or value.get("status") not in {"checked", "unavailable", "error"}:
        raise DanmakuFilterError("失败句检测结果格式无效")
    if value.get("engine") != "local-wordlist":
        raise DanmakuFilterError("失败句检测引擎无效")
    hits, rules, notes = (value.get(key) for key in ("hits", "known_rules", "notes"))
    if (not isinstance(hits, list) or len(hits) > MAX_HITS or not isinstance(rules, list)
            or len(rules) > MAX_HITS or not isinstance(notes, list) or len(notes) > 8):
        raise DanmakuFilterError("失败句检测结果数量无效")
    normalized_hits = []
    for hit in hits:
        if not isinstance(hit, dict) or hit.get("match_type") not in {"literal", "normalized"}:
            raise DanmakuFilterError("失败句检测命中格式无效")
        start, end = hit.get("start"), hit.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise DanmakuFilterError("失败句检测位置无效")
        matched = _text(hit.get("matched"), "检测原文", MAX_CANDIDATE_CHARACTERS)
        if matched != text[start:end]:
            raise DanmakuFilterError("失败句检测位置与原文不一致")
        normalized_hits.append({"term": _text(hit.get("term"), "检测词", 128),
            "category": _text(hit.get("category"), "检测词分类", 128),
            "source_id": _text(hit.get("source_id"), "检测词来源", 4096),
            "room": _room(hit.get("room", "")),
            "matched": matched, "start": start, "end": end, "match_type": hit["match_type"]})
    if any(not isinstance(row, dict) for row in rules):
        raise DanmakuFilterError("失败句已知规则格式无效")
    if value.get("summary_code") not in {"no_hits", "advisory_hits", "no_wordlist", "dictionary_error"}:
        raise DanmakuFilterError("失败句检测结论无效")
    return {"status": value["status"], "engine": "local-wordlist", "checked_at": _time(value.get("checked_at")),
        "dictionary_revision": _text(value.get("dictionary_revision"), "词库版本", 256, empty=True),
        "rule_revision": _text(value.get("rule_revision"), "规则版本", 256),
        "hits": normalized_hits,
        "known_rules": [{"term": _text(row.get("term"), "过滤词", MAX_RULE_CHARACTERS),
                         "action": _action(row.get("action", "replace")),
                         "replacement": _replacement(row.get("replacement"), _action(row.get("action", "replace")))}
                        for row in rules],
        "notes": [_text(note, "检测说明", 128) for note in notes],
        "summary_code": value["summary_code"], "error": _text(value.get("error"), "检测错误", 256, empty=True),
        "truncated": _boolean(value.get("truncated"), "检测截断标记")}


class DanmakuContentFilter:
    def __init__(self, path: Path | str | None = None, detector_path: Path | str | None = None):
        self.path = Path(path) if path is not None else data_root() / "danmaku-content-filter.json"
        self.detector_path = Path(detector_path) if detector_path is not None else self.path.with_name("danmaku-wordlist.json")
        self._detector = DanmakuWordlist(self.detector_path)
        self.last_error = ""
        self._state = {"version": 1, "enabled": True, "rules": [], "candidates": []}
        try:
            source = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except (OSError, UnicodeError) as error:
            self._raise(f"无法读取本地过滤名单：{error}")
        try:
            self._state = self._validate(json.loads(source))
        except (ValueError, TypeError, OverflowError) as error:
            self._raise(f"本地过滤名单文件损坏或格式无效：{error}")

    @property
    def enabled(self) -> bool:
        return self._state["enabled"]

    @property
    def filter_error(self) -> str:
        return self.last_error

    @property
    def revision(self) -> str:
        rules = [{key: row[key] for key in ("term", "replacement", "room", "enabled", "action")}
                 for row in self._state["rules"]]
        rules.sort(key=lambda row: (row["room"], row["term"], row["replacement"], row["enabled"], row["action"]))
        value = {"enabled": self.enabled, "rules": rules}
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()

    def _raise(self, message: str):
        self.last_error = message
        raise DanmakuFilterError(message)

    @staticmethod
    def _validated_rule(row: object) -> dict:
        if not isinstance(row, dict):
            raise DanmakuFilterError("过滤规则格式无效")
        identifier = _identifier(row.get("id"))
        term = _text(row.get("term"), "过滤词", MAX_RULE_CHARACTERS)
        room = _room(row.get("room"))
        action = _action(row.get("action", "replace"))
        replacement = _replacement(row.get("replacement"), action)
        origin = row.get("origin", "user")
        if not isinstance(origin, str) or origin not in {"user", "ai_success_observed", "ai_analysis"}:
            raise DanmakuFilterError("过滤规则来源无效")
        if origin != "user" and (not 2 <= _length(term) <= MAX_LEARNED_TERM_CHARACTERS
                or len(term) < 2 or _length(replacement) > MAX_LEARNED_TERM_CHARACTERS):
            raise DanmakuFilterError("经验规则必须是局部短词编辑")
        audit = _rule_evidence(row.get("evidence"), origin)
        validation = _validated_proof(row.get("validation"))
        if validation is not None and origin == "user":
            raise DanmakuFilterError("人工规则不能携带 AI 验证证据")
        if validation is not None and validation["source_digest"] != audit.get("source_digest", audit.get("failed_digest")):
            raise DanmakuFilterError("规则验证证据与原文摘要不一致")
        result = {"id": identifier, "term": term, "action": action, "replacement": replacement,
            "room": room, "enabled": _boolean(row.get("enabled"), "规则开关") and (origin == "user" or validation is not None),
            "created_at": _time(row.get("created_at")), "updated_at": _time(row.get("updated_at")),
            "origin": origin, "evidence": audit, "validation": validation}
        aliases = row.get("signature_aliases", [])
        if (not isinstance(aliases, list) or any(not isinstance(alias, str)
                or not re.fullmatch(r"[a-f0-9]{64}", alias) for alias in aliases)):
            raise DanmakuFilterError("历史规则签名必须是 SHA256 列表")
        if aliases:
            result["signature_aliases"] = sorted(set(aliases))
        return result

    @staticmethod
    def _validate(value: object) -> dict:
        if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
            raise DanmakuFilterError("不支持的本地过滤名单版本")
        enabled = _boolean(value.get("enabled"), "过滤名单开关")
        rules, candidates = value.get("rules"), value.get("candidates")
        if not isinstance(rules, list) or not isinstance(candidates, list):
            raise DanmakuFilterError("规则和候选必须是列表")
        normalized_rules, normalized_candidates = [], []
        rule_ids, scopes, candidate_ids, failures = set(), set(), set(), set()
        for row in rules:
            row = DanmakuContentFilter._validated_rule(row)
            identifier, term, room = row["id"], row["term"], row["room"]
            if identifier in rule_ids or (room, term) in scopes:
                raise DanmakuFilterError("过滤名单包含重复规则")
            rule_ids.add(identifier)
            scopes.add((room, term))
            normalized_rules.append(row)
        migration = value.get("scope_migration")
        if migration is not None:
            if (not isinstance(migration, dict) or migration.get("version") != 1
                    or type(migration.get("version")) is not int or not isinstance(migration.get("legacy_rules"), list)):
                raise DanmakuFilterError("全局规则迁移记录格式无效")
            for row in migration["legacy_rules"]:
                DanmakuContentFilter._validated_rule(row)
            migration = {"version": 1, "migrated_at": _time(migration.get("migrated_at")),
                         "legacy_rules": copy.deepcopy(migration["legacy_rules"])}
        if any(row["room"] for row in normalized_rules):
            if migration is None:
                migration = {"version": 1, "migrated_at": _now(), "legacy_rules": []}
            snapshots = {json.dumps(row, ensure_ascii=False, sort_keys=True) for row in migration["legacy_rules"]}
            for row in rules:
                snapshot = json.dumps(row, ensure_ascii=False, sort_keys=True)
                if snapshot not in snapshots:
                    migration["legacy_rules"].append(copy.deepcopy(row))
                    snapshots.add(snapshot)
            groups = {}
            for row in normalized_rules:
                groups.setdefault(row["term"], []).append(row)
            normalized_rules = []
            for group in groups.values():
                # Preserve deliberate manual/global choices before using recency
                # to resolve legacy room conflicts. Every original stays in the audit.
                winner = max(group, key=lambda row: (row["origin"] == "user", not row["room"],
                    datetime.fromisoformat(row["updated_at"]), row["enabled"], row["id"]))
                aliases = set(winner.get("signature_aliases", []))
                semantic_keys = ("term", "replacement", "action", "enabled")
                for row in group:
                    if all(row[key] == winner[key] for key in semantic_keys):
                        aliases.add(DanmakuContentFilter._rule_signature(row))
                        aliases.update(row.get("signature_aliases", []))
                winner = {**winner, "room": ""}
                aliases.discard(DanmakuContentFilter._rule_signature(winner))
                if aliases:
                    winner["signature_aliases"] = sorted(aliases)
                else:
                    winner.pop("signature_aliases", None)
                normalized_rules.append(winner)
        for row in candidates:
            if not isinstance(row, dict):
                raise DanmakuFilterError("失败候选格式无效")
            identifier = _identifier(row.get("id"))
            room = _room(row.get("room"))
            text = _text(row.get("text"), "失败内容", MAX_CANDIDATE_CHARACTERS)
            count = row.get("count")
            if type(count) is not int or count < 1:
                raise DanmakuFilterError("失败候选次数必须是正整数")
            if identifier in candidate_ids or (room, text) in failures:
                raise DanmakuFilterError("过滤名单包含重复失败候选")
            candidate_ids.add(identifier)
            failures.add((room, text))
            candidate = {"id": identifier, "room": room, "text": text,
                "reason": _text(row.get("reason"), "失败原因", 256), "count": count,
                "first_time": _time(row.get("first_time")), "last_time": _time(row.get("last_time")),
                "status": _candidate_status(row.get("status", "active")),
                "queue_id": _queue_id(row.get("queue_id")), "diagnosis": _diagnosis(row.get("diagnosis")),
                "resolved_at": _time(row["resolved_at"]) if row.get("resolved_at") is not None else None,
                "resolution": _diagnosis_code(row.get("resolution", ""), "恢复结果", empty=True),
                "last_resolution": _resolution(row.get("last_resolution")),
                "validation": _validation_plan(row.get("validation"), room, text),
                "last_validation": _validation_plan(row.get("last_validation"), room, text),
                "repair_history": _validation_history(row.get("repair_history", []), room, text),
                "analysis_completed_at": _time(row["analysis_completed_at"]) if row.get("analysis_completed_at") is not None else None,
                "analysis_resolution": _diagnosis_code(row.get("analysis_resolution", ""), "分析结果", empty=True)}
            if candidate["resolution"] in {"ai_rules_saved", "ai_no_local_edit"} and not (
                    candidate["validation"] and candidate["validation"]["state"] == "passed"):
                candidate.update(status="waiting_validation", resolved_at=None, resolution="",
                                 analysis_completed_at=candidate["analysis_completed_at"] or row.get("resolved_at"),
                                 analysis_resolution=candidate["analysis_resolution"] or row["resolution"])
            if ((candidate["status"] == "resolved") !=
                    (candidate["resolved_at"] is not None and bool(candidate["resolution"]))
                    or (candidate["resolved_at"] is None) != (candidate["resolution"] == "")):
                raise DanmakuFilterError("失败候选的处理状态与恢复结果不一致")
            if "analysis" in row:
                candidate["analysis"] = _analysis(row["analysis"], text)
            normalized_candidates.append(candidate)
        state = {"version": 1, "enabled": enabled, "rules": normalized_rules, "candidates": normalized_candidates}
        if migration is not None:
            state["scope_migration"] = migration
        return state

    def _save(self, state: dict):
        state = self._validate(state)
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                    prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent, delete=False) as output:
                temporary = Path(output.name)
                json.dump(state, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        except (OSError, UnicodeError) as error:
            self._raise(f"无法保存本地过滤名单；原规则未修改：{error}")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        self._state = state
        self.last_error = ""

    def set_enabled(self, enabled: bool):
        state = copy.deepcopy(self._state)
        state["enabled"] = _boolean(enabled, "过滤名单开关")
        self._save(state)

    def persist(self):
        """Atomically save current records without adding a failure observation."""
        self._save(copy.deepcopy(self._state))

    def rules(self) -> list[dict]:
        return copy.deepcopy(self._state["rules"])

    def list_rules(self) -> list[dict]:
        return self.rules()

    def candidates(self, room: str | None = None, *, include_resolved: bool = True) -> list[dict]:
        include_resolved = _boolean(include_resolved, "显示已恢复记录")
        rows = self._state["candidates"]
        if room is not None:
            scope = _room(room)
            rows = [row for row in rows if row["room"] == scope]
        if not include_resolved:
            rows = [row for row in rows if row["status"] != "resolved"]
        return copy.deepcopy(rows)

    def list_candidates(self, room: str | None = None, *, include_resolved: bool = True) -> list[dict]:
        return self.candidates(room, include_resolved=include_resolved)

    def status_counts(self, room: str | None = None) -> dict[str, int]:
        counts = dict.fromkeys(CANDIDATE_STATUSES, 0)
        for row in self.candidates(room):
            counts[row["status"]] += 1
        return counts

    def detection_info(self) -> dict:
        return self._detector.info()

    def detector_entries(self) -> list[dict]:
        return self._detector.entries()

    def detector_sources(self) -> list[dict]:
        return self._detector.sources()

    def get_allowlist(self) -> list[str]:
        return self._detector.get_allowlist()

    def set_allowlist(self, terms: list[str]):
        try:
            self._detector.set_allowlist(terms)
        except DanmakuWordlistError as error:
            raise DanmakuFilterError(str(error)) from None
        return self.analyze_candidates()

    def reload_detector(self) -> dict:
        return self._detector.reload()

    def detect(self, text: str, room: str = "") -> dict:
        scope = _room(room)
        result = self._detector.detect(text, scope)
        result.update(checked_at=_now(), rule_revision=self.revision,
                      known_rules=[{"term": row["term"], "replacement": row["replacement"], "action": row["action"]}
                          for row in self._state["rules"] if self.enabled and row["enabled"]
                          and isinstance(text, str) and row["term"] in text][:MAX_HITS])
        return result

    def _analyze_row(self, row: dict, *, force=False) -> bool:
        previous = row.get("analysis")
        if (not force and isinstance(previous, dict) and previous.get("engine") == "local-wordlist"
                and previous.get("dictionary_revision") == self._detector.revision
                and previous.get("rule_revision") == self.revision
                and previous.get("status") == ("checked" if self._detector.status == "ready"
                        else "unavailable" if self._detector.status == "no_wordlist" else "error")):
            return False
        row["analysis"] = self.detect(row["text"], row["room"])
        return True

    def analyze_candidate(self, candidate_id: str, force: bool = False) -> dict:
        candidate_id = _identifier(candidate_id)
        state = copy.deepcopy(self._state)
        row = next((row for row in state["candidates"] if row["id"] == candidate_id), None)
        if row is None:
            raise DanmakuFilterError("要检测的失败候选不存在")
        if self._analyze_row(row, force=force):
            self._save(state)
        return copy.deepcopy(row["analysis"])

    def analyze_candidates(self, force: bool = False) -> list[dict]:
        state = copy.deepcopy(self._state)
        changed = False
        for row in state["candidates"]:
            changed = self._analyze_row(row, force=force) or changed
        if changed:
            self._save(state)
        return copy.deepcopy(state["candidates"])

    @staticmethod
    def _rule_signature(row: dict) -> str:
        semantic = {key: row[key] for key in ("term", "replacement", "action", "room", "enabled")}
        return hashlib.sha256(json.dumps(semantic, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()

    def _selected_rules(self, scope: str) -> list[dict]:
        if not self.enabled:
            return []
        applicable = [row for row in self._state["rules"] if row["enabled"]]
        applicable.sort(key=lambda row: (-len(row["term"]), row["id"]))
        selected = {}
        for row in applicable:
            selected.setdefault(row["term"], row)
        return list(selected.values())

    def rule_signatures(self, room: str = "") -> list[str]:
        return sorted(self._rule_signature(row) for row in self._selected_rules(_room(room)))

    def apply(self, text: str, room: str = "", *, excluded_rule_signatures=()) -> tuple[str, int]:
        if not isinstance(text, str):
            raise DanmakuFilterError("待过滤内容必须是文字")
        selected = self._selected_rules(_room(room))
        if not isinstance(excluded_rule_signatures, (list, tuple, set, frozenset)) or any(
                not isinstance(signature, str) or not re.fullmatch(r"[a-f0-9]{64}", signature)
                for signature in excluded_rule_signatures):
            raise DanmakuFilterError("已应用规则签名必须是 SHA256 列表")
        excluded = set(excluded_rule_signatures)
        replacements = {row["term"]: row["replacement"] for row in selected
                        if not excluded.intersection({self._rule_signature(row), *row.get("signature_aliases", [])})}
        if not replacements:
            return text, 0
        pattern = re.compile("|".join(re.escape(term) for term in replacements))
        # re.sub scans only the original text. Replacement text never re-enters
        # matching, so rules such as A -> B and B -> A cannot loop.
        return pattern.subn(lambda match: replacements[match.group(0)], text)

    def add_rule(self, term: str, replacement: str = DEFAULT_REPLACEMENT, room: str = "", enabled: bool = True,
                 *, action: str = "replace") -> dict:
        term = _text(term, "过滤词", MAX_RULE_CHARACTERS)
        action = _action(action)
        replacement = _replacement(replacement, action)
        _room(room)
        room, enabled = "", _boolean(enabled, "规则开关")
        if any(row["term"] == term for row in self._state["rules"]):
            raise DanmakuFilterError("已存在相同过滤词，请编辑原规则")
        now = _now()
        row = {"id": uuid.uuid4().hex, "term": term, "replacement": replacement, "action": action, "room": room,
               "enabled": enabled, "created_at": now, "updated_at": now, "origin": "user", "evidence": None, "validation": None}
        state = copy.deepcopy(self._state)
        state["rules"].append(row)
        self._save(state)
        return copy.deepcopy(row)

    def update_rule(self, rule_id: str, *, term=_UNSET, replacement=_UNSET, room=_UNSET, enabled=_UNSET,
                    action=_UNSET) -> dict:
        rule_id = _identifier(rule_id)
        state = copy.deepcopy(self._state)
        row = next((row for row in state["rules"] if row["id"] == rule_id), None)
        if row is None:
            raise DanmakuFilterError("要编辑的过滤规则不存在")
        previous_semantics = {key: row[key] for key in ("term", "replacement", "room", "action")}
        previous_enabled = row["enabled"]
        if term is not _UNSET:
            row["term"] = _text(term, "过滤词", MAX_RULE_CHARACTERS)
        if action is not _UNSET:
            row["action"] = _action(action)
        if replacement is not _UNSET:
            row["replacement"] = _replacement(replacement, row["action"])
        else:
            row["replacement"] = _replacement(row["replacement"], row["action"])
        if room is not _UNSET:
            _room(room)
        row["room"] = ""
        if enabled is not _UNSET:
            row["enabled"] = _boolean(enabled, "规则开关")
        if any(other["id"] != rule_id and other["term"] == row["term"] for other in state["rules"]):
            raise DanmakuFilterError("已存在相同过滤词，请编辑原规则")
        if any(row[key] != value for key, value in previous_semantics.items()):
            row.update(origin="user", evidence=None, validation=None)
            row.pop("signature_aliases", None)
        elif row["enabled"] != previous_enabled:
            row.pop("signature_aliases", None)
        if row["enabled"] and row["origin"] != "user" and row["validation"] is None:
            raise DanmakuFilterError("AI 规则尚未通过专用测试房间验证，不能启用")
        row["updated_at"] = _now()
        self._save(state)
        return copy.deepcopy(row)

    def learn_successful_edits(self, room: str, edits: list[dict], *, evidence: dict) -> list[dict]:
        """Retain source-room observations as plans; only dedicated testing writes new rules."""
        room = _room(room)
        if not room:
            raise DanmakuFilterError("经验规则只能保存到实测直播间")
        evidence_keys = {"queue_id", "original_text", "current_text", "failure_count", "observed_at",
                         "receipt_accepted", "page_observed"}
        if not isinstance(evidence, dict) or set(evidence) != evidence_keys:
            raise DanmakuFilterError("自动学习必须提供完整提交与页面观察证据")
        queue_id = _queue_id(evidence["queue_id"])
        failure_count = evidence["failure_count"]
        if queue_id is None or type(failure_count) is not int or not 1 <= failure_count <= 2**63 - 1:
            raise DanmakuFilterError("自动学习必须关联实际失败队列和次数")
        if evidence["receipt_accepted"] is not True or evidence["page_observed"] is not True:
            raise DanmakuFilterError("自动学习必须有修正后新请求的正常回执与新增页面观察")
        _time(evidence["observed_at"])
        original = _text(evidence["original_text"], "学习前正文", MAX_CANDIDATE_CHARACTERS)
        revised = _text(evidence["current_text"], "学习后正文", MAX_CANDIDATE_CHARACTERS)
        associated = [row for row in self._state["candidates"] if row["queue_id"] == queue_id]
        if associated and not any(row["room"] == room and row["text"] == original for row in associated):
            raise DanmakuFilterError("学习证据与已关联的失败房间或原正文不一致")
        if original == revised or not revised.strip("【】 "):
            raise DanmakuFilterError("自动学习必须有有效局部修改且不能删除全部正文")
        _, replayed = _replay_local_edits(original, edits)
        if _canonical_whitespace(replayed) != _canonical_whitespace(revised):
            raise DanmakuFilterError("局部编辑单次重放与实际观察的修正正文不一致")
        candidate = next((row for row in associated if row["room"] == room and row["text"] == original), None)
        if candidate is not None:
            self.save_ai_suggestions(room, original, edits, candidate_id=candidate["id"])
        return []

    def save_ai_suggestions(self, room: str, original_text: str, edits: list[dict], *,
                            candidate_id: str) -> list[dict]:
        """Persist an AI sentence plan; this compatibility entry point never creates rules."""
        room, original = _room(room), _text(original_text, "分析原文", MAX_CANDIDATE_CHARACTERS)
        candidate_id = _identifier(candidate_id)
        candidate = next((row for row in self._state["candidates"] if row["id"] == candidate_id), None)
        if not room or candidate is None or candidate["room"] != room or candidate["text"] != original:
            raise DanmakuFilterError("AI 建议与失败候选的房间或原文不一致")
        if edits == []:
            return []
        replacements, replayed = _replay_local_edits(original, edits)
        previous_plan = candidate["validation"]
        prior_plans = candidate["repair_history"] + [candidate["last_validation"], previous_plan]
        # Changing the representation of edits cannot turn an old candidate
        # sentence into an endlessly new test request.
        if any(plan is not None and plan["revised_text"] == _unwrapped_body(replayed)
               for plan in prior_plans):
            return []
        if previous_plan is not None:
            if not _validation_can_revise(previous_plan):
                raise DanmakuFilterError("已提交的验证方案只能继续核验，不能替换")
        state = copy.deepcopy(self._state)
        now = _now()
        candidate = next(row for row in state["candidates"] if row["id"] == candidate_id)
        candidate.update(status="waiting_validation", resolved_at=None, resolution="",
                         analysis_completed_at=now, analysis_resolution="ai_plan_saved")
        if previous_plan is not None:
            # Old version files only kept last_validation. Preserve it before
            # replacing that compatibility pointer with the current plan.
            _archive_validation(candidate, candidate["last_validation"])
            _archive_validation(candidate, previous_plan)
            candidate["last_validation"] = copy.deepcopy(previous_plan)
        candidate["validation"] = {"validation_id": uuid.uuid4().hex, "state": "waiting_room", "source_room": room,
            "formal_room": room, "test_room": "", "original_text": original,
            "revised_text": _unwrapped_body(replayed),
            "edits": [{"term": term, "replacement": replacement} for term, replacement in replacements.items()],
            "attempts": 0, "request_id": "", "message": "", "created_at": now, "updated_at": now,
            "evidence": None, "validated_rule_ids": [], "receipt": None}
        self._save(state)
        return []

    def save_validation_plan(self, candidate_id: str, revised_text: str, *, edits: list[dict]) -> dict:
        candidate_id = _identifier(candidate_id)
        row = next((row for row in self._state["candidates"] if row["id"] == candidate_id), None)
        if row is None:
            raise DanmakuFilterError("要验证的失败候选不存在")
        revised = _text(revised_text, "AI 修正正文", MAX_CANDIDATE_CHARACTERS)
        _, replayed = _replay_local_edits(row["text"], edits)
        if _unwrapped_body(revised) != _unwrapped_body(replayed):
            raise DanmakuFilterError("AI 修正句与局部编辑重放不一致")
        self.save_ai_suggestions(row["room"], row["text"], edits, candidate_id=candidate_id)
        return next(row for row in self.candidates() if row["id"] == candidate_id)

    def validation_plan(self, candidate_id: str) -> dict | None:
        candidate_id = _identifier(candidate_id)
        row = next((row for row in self._state["candidates"] if row["id"] == candidate_id), None)
        if row is None:
            raise DanmakuFilterError("要验证的失败候选不存在")
        return copy.deepcopy(row["validation"])

    def failed_validation_plans(self, candidate_id: str) -> list[dict]:
        """Return durable failed wording checks that permit a different correction."""
        candidate_id = _identifier(candidate_id)
        row = next((row for row in self._state["candidates"] if row["id"] == candidate_id), None)
        if row is None:
            raise DanmakuFilterError("要验证的失败候选不存在")
        result, identifiers = [], set()
        for plan in row["repair_history"] + [row["last_validation"], row["validation"]]:
            if (plan is not None and plan["state"] == "failed" and _validation_can_revise(plan)
                    and plan["validation_id"] not in identifiers):
                identifiers.add(plan["validation_id"])
                result.append(copy.deepcopy(plan))
        return result

    def configure_validation(self, candidate_id: str, test_room: str, formal_room: str = "") -> dict:
        candidate_id, test_room, formal_room = _identifier(candidate_id), _room(test_room), _room(formal_room)
        if not test_room or test_room == formal_room:
            raise DanmakuFilterError("专用测试房间必须与当前正式发送房间分开")
        state = copy.deepcopy(self._state)
        row = next((row for row in state["candidates"] if row["id"] == candidate_id), None)
        if row is None or row["validation"] is None:
            raise DanmakuFilterError("失败候选尚无可验证的局部修正方案")
        plan = row["validation"]
        if plan["test_room"] == test_room and plan["formal_room"] == formal_room and plan["state"] != "waiting_room":
            return copy.deepcopy(row)
        if plan["state"] == "testing":
            raise DanmakuFilterError("验证请求结果尚未确定，不能更换测试房间")
        plan.update(validation_id=uuid.uuid4().hex, state="ready", test_room=test_room, formal_room=formal_room,
                    request_id="", message="", evidence=None, receipt=None, validated_rule_ids=[], updated_at=_now())
        self._set_status(row, "waiting_validation")
        for rule in state["rules"]:
            # A new sentence test cannot revoke a previously verified global
            # rule. Only an unverified draft remains disabled while testing.
            if (rule["origin"] == "ai_analysis" and rule["evidence"]["candidate_id"] == candidate_id
                    and rule["validation"] is None):
                rule["enabled"] = False
        self._save(state)
        return copy.deepcopy(row)

    def update_validation(self, candidate_id: str, *, state: str, message: str = "", request_id: str = "") -> dict:
        candidate_id = _identifier(candidate_id)
        if not isinstance(state, str) or state not in {"waiting_room", "ready", "testing", "failed"}:
            raise DanmakuFilterError("验证状态只能由正常回执与页面观察完成")
        message = _text(message, "验证说明", 512, empty=True)
        request_id = _identifier(request_id) if request_id else ""
        document = copy.deepcopy(self._state)
        row = next((row for row in document["candidates"] if row["id"] == candidate_id), None)
        if row is None or row["validation"] is None:
            raise DanmakuFilterError("失败候选尚无可验证的局部修正方案")
        plan = row["validation"]
        if plan["state"] == "passed":
            return copy.deepcopy(row)
        if state in {"ready", "testing"} and not plan["test_room"]:
            raise DanmakuFilterError("尚未配置专用测试房间")
        if state == "testing":
            if not request_id:
                raise DanmakuFilterError("提交验证前必须记录请求 ID")
            if plan["state"] == "testing" and plan["request_id"] != request_id:
                raise DanmakuFilterError("旧验证结果尚未确定，不能重新提交")
            if plan["receipt"] and plan["receipt"]["request_started"] and plan["request_id"] != request_id:
                raise DanmakuFilterError("已提交的验证请求需要继续只读核验")
            plan.update(request_id=request_id, receipt=plan["receipt"] if plan["request_id"] == request_id else None)
        plan.update(state=state, message=message, updated_at=_now())
        self._set_status(row, "waiting_validation")
        self._save(document)
        return copy.deepcopy(row)

    def record_validation_receipt(self, candidate_id: str, *, request_id: str, before_count: int | None,
                                  receipt_accepted: bool, code=None, http_status=None, error="none",
                                  request_started: bool = True) -> dict:
        candidate_id, request_id = _identifier(candidate_id), _identifier(request_id)
        document = copy.deepcopy(self._state)
        row = next((row for row in document["candidates"] if row["id"] == candidate_id), None)
        if row is None or row["validation"] is None:
            raise DanmakuFilterError("失败候选尚无可验证的局部修正方案")
        plan = row["validation"]
        if plan["request_id"] != request_id or plan["state"] not in {"testing", "failed"}:
            raise DanmakuFilterError("验证回执不属于当前待核验请求")
        receipt = _validation_receipt({"request_id": request_id, "before_count": before_count,
            "receipt_accepted": receipt_accepted, "code": code, "http_status": http_status, "error": error,
            "request_started": request_started, "recorded_at": _now(), "test_room": plan["test_room"],
            "validation_id": plan["validation_id"]})
        previous = plan["receipt"]
        if previous is not None:
            if previous["request_started"] and not receipt["request_started"]:
                raise DanmakuFilterError("已提交验证请求不能改成未提交")
            if previous["receipt_accepted"] and (not receipt["receipt_accepted"] or previous["before_count"] != receipt["before_count"]):
                raise DanmakuFilterError("已接受验证回执的页面基线不能改写")
        if request_started and (previous is None or not previous["request_started"]):
            plan["attempts"] += 1
        plan.update(receipt=receipt, updated_at=_now())
        if not request_started:
            plan["state"] = "ready"
        self._save(document)
        return copy.deepcopy(row)

    def confirm_validation(self, candidate_id: str, *, evidence: dict) -> dict:
        candidate_id = _identifier(candidate_id)
        required = {"test_room", "validation_id", "request_id", "observed_at", "receipt_accepted", "page_observed",
                    "before_count", "occurrence_count", "code", "http_status", "error"}
        if not isinstance(evidence, dict) or set(evidence) != required:
            raise DanmakuFilterError("验证完成必须提供完整当前请求证据")
        document = copy.deepcopy(self._state)
        row = next((row for row in document["candidates"] if row["id"] == candidate_id), None)
        if row is None or row["validation"] is None:
            raise DanmakuFilterError("失败候选尚无可验证的局部修正方案")
        plan = row["validation"]
        proof = _validated_proof({**evidence, "formal_room": plan["formal_room"], "confirmed_cause": False,
            "source_digest": hashlib.sha256(plan["original_text"].encode("utf-8")).hexdigest(),
            "revised_digest": hashlib.sha256(plan["revised_text"].encode("utf-8")).hexdigest()})
        receipt = plan["receipt"]
        if (plan["state"] not in {"testing", "failed", "passed"} or proof["test_room"] != plan["test_room"]
                or proof["request_id"] != plan["request_id"] or proof["validation_id"] != plan["validation_id"]
                or receipt is None or not receipt["request_started"] or not receipt["receipt_accepted"]
                or receipt["before_count"] != proof["before_count"] or receipt["code"] != proof["code"]
                or receipt["http_status"] != proof["http_status"]):
            raise DanmakuFilterError("验证完成证据必须匹配持久化回执及页面基线")
        if plan["state"] == "passed":
            return copy.deepcopy(row)
        # Derive the actual lexical difference only after independent testing.
        # Persisted analysis proposals are never treated as tested replacement facts.
        from .danmaku_ai import DanmakuAIValidationError, validate_analysis_result
        if not is_chinese_output(plan["revised_text"]):
            raise DanmakuFilterError("验证修正正文必须为全中文 不能学习带外语的规则")
        try:
            checked = validate_analysis_result({"revised_text": plan["revised_text"]}, plan["original_text"])
        except DanmakuAIValidationError:
            raise DanmakuFilterError("测试后句子差异未通过局部与语义安全检查") from None
        if checked["action"] != "rewrite" or checked["edits"] != plan["edits"]:
            raise DanmakuFilterError("测试后重新计算的局部差异与持久化方案不一致")
        edits = {edit["term"]: edit["replacement"] for edit in checked["edits"]}
        validated = []
        for rule in document["rules"]:
            audit = rule["evidence"]
            belongs = (rule["origin"] == "ai_analysis" and audit["candidate_id"] == candidate_id)
            if rule["origin"] == "ai_success_observed":
                belongs = audit["queue_id"] == row["queue_id"] and audit["failed_digest"] == proof["source_digest"]
            if (belongs and rule["term"] in edits
                    and rule["replacement"] == edits[rule["term"]]):
                if not rule["enabled"]:
                    rule.pop("signature_aliases", None)
                rule.update(enabled=True, validation=proof, updated_at=_now())
                validated.append(rule["id"])
        existing_terms = {rule["term"] for rule in document["rules"]}
        now = _now()
        audit = {"candidate_id": candidate_id, "source_digest": proof["source_digest"],
                 "learned_at": now, "confirmed_cause": False}
        for term, replacement in edits.items():
            if term in existing_terms:
                continue
            rule = {"id": uuid.uuid4().hex, "term": term, "replacement": replacement,
                    "action": "replace" if replacement else "remove", "room": "", "enabled": True,
                    "created_at": now, "updated_at": now, "origin": "ai_analysis", "evidence": audit, "validation": proof}
            document["rules"].append(rule)
            validated.append(rule["id"])
        plan.update(state="passed", evidence=proof, validated_rule_ids=validated, message="", updated_at=_now())
        self._set_status(row, "resolved", resolution="test_room_validated")
        self._save(document)
        return copy.deepcopy(row)

    def remove_rule(self, rule_id: str) -> bool:
        rule_id = _identifier(rule_id)
        state = copy.deepcopy(self._state)
        state["rules"] = [row for row in state["rules"] if row["id"] != rule_id]
        if len(state["rules"]) == len(self._state["rules"]):
            return False
        self._save(state)
        return True

    def record_failure(self, room: str, text: str, reason: str, *, queue_id: int | None = None,
                       diagnosis: dict | None = None) -> dict:
        room = _room(room)
        text = _text(text, "失败内容", MAX_CANDIDATE_CHARACTERS)
        reason = _text(reason, "失败原因", 256)
        queue_id, diagnosis = _queue_id(queue_id), _diagnosis(diagnosis)
        state = copy.deepcopy(self._state)
        now = _now()
        row = next((row for row in state["candidates"] if row["room"] == room and row["text"] == text), None)
        if row is None:
            row = {"id": uuid.uuid4().hex, "room": room, "text": text, "reason": reason,
                   "count": 1, "first_time": now, "last_time": now}
            state["candidates"].append(row)
        else:
            row.update(reason=reason, count=row["count"] + 1, last_time=now)
            if row["validation"] is not None and row["validation"]["state"] == "passed":
                # A later actual failure is new evidence. Keep the successful
                # audit and learned rules, but do not let the old passed plan
                # prevent this occurrence from being analyzed again.
                _archive_validation(row, row["last_validation"])
                _archive_validation(row, row["validation"])
                row["last_validation"] = copy.deepcopy(row["validation"])
                row["validation"] = None
        row.update(status="active", queue_id=queue_id if queue_id is not None else row.get("queue_id"),
                   diagnosis=diagnosis, resolved_at=None, resolution="", last_resolution=row.get("last_resolution"),
                   validation=row.get("validation"), last_validation=row.get("last_validation"),
                   repair_history=row.get("repair_history", []),
                   analysis_completed_at=row.get("analysis_completed_at"),
                   analysis_resolution=row.get("analysis_resolution", ""))
        # Cache by both immutable observations and dictionary/rule revisions.
        # Continued f retries only update their counter, not repeat the scan.
        self._analyze_row(row)
        self._save(state)
        return copy.deepcopy(row)

    @staticmethod
    def _set_status(row: dict, status: str, diagnosis: dict | None = None, *, resolution="page_observed"):
        row["status"] = status
        if diagnosis is not None:
            row["diagnosis"] = diagnosis
        if status == "resolved":
            row.update(resolved_at=_now(), resolution=resolution)
            row["last_resolution"] = {"at": row["resolved_at"], "resolution": resolution,
                                      "queue_id": row["queue_id"]}
        else:
            row.update(resolved_at=None, resolution="")

    def set_candidate_status(self, candidate_id: str, status: str, *, diagnosis: dict | None = None,
                             resolution: str = "page_observed") -> dict:
        candidate_id, status = _identifier(candidate_id), _candidate_status(status)
        diagnosis = _diagnosis(diagnosis)
        resolution = _diagnosis_code(resolution, "恢复结果")
        state = copy.deepcopy(self._state)
        row = next((row for row in state["candidates"] if row["id"] == candidate_id), None)
        if row is None:
            raise DanmakuFilterError("要更新的失败候选不存在")
        if status == "resolved" and resolution in {"ai_no_local_edit", "ai_rules_saved"}:
            status = "waiting_validation"
            row.update(analysis_completed_at=row["analysis_completed_at"] if row["analysis_resolution"] == resolution
                       else _now(), analysis_resolution=resolution)
        if status == "resolved" and resolution == "test_room_validated" and not (
                row["validation"] and row["validation"]["state"] == "passed"):
            raise DanmakuFilterError("AI 修正方案尚未通过专用测试房间验证")
        if (row["status"] == status and (diagnosis is None or row["diagnosis"] == diagnosis)
                and (status != "resolved" or row["resolution"] == resolution)):
            return copy.deepcopy(row)
        self._set_status(row, status, diagnosis, resolution=resolution)
        self._save(state)
        return copy.deepcopy(row)

    def complete_ai_review(self, candidate_id: str, resolution: str, *, diagnosis: dict | None = None) -> dict:
        if not isinstance(resolution, str) or resolution not in {"ai_no_local_edit", "ai_rules_saved"}:
            raise DanmakuFilterError("AI 分析完成状态无效")
        return self.set_candidate_status(candidate_id, "resolved", diagnosis=diagnosis, resolution=resolution)

    @staticmethod
    def _matching_candidate(state: dict, room: str, text: str, queue_id: int | None) -> dict | None:
        # Legacy observations have no queue ID; matching may attach them to the
        # current exact text, but a different linked queue item cannot be archived.
        return next((row for row in state["candidates"] if row["room"] == room and row["text"] == text
                     and row["queue_id"] in ({None, queue_id} if queue_id is not None else {None})), None)

    def update_recovery(self, room: str, text: str, queue_id: int | None = None, *, status="recovering",
                        diagnosis: dict | None = None) -> dict | None:
        room, text = _room(room), _text(text, "失败内容", MAX_CANDIDATE_CHARACTERS)
        queue_id, status, diagnosis = _queue_id(queue_id), _candidate_status(status), _diagnosis(diagnosis)
        state = copy.deepcopy(self._state)
        row = self._matching_candidate(state, room, text, queue_id)
        if row is None:
            return None
        if row["status"] == "resolved" and status != "resolved":
            # A read-only status poll must not reopen a recovery already recorded;
            # only a newly observed failure starts another active lifecycle.
            return copy.deepcopy(row)
        if (row["status"] == status and (queue_id is None or row["queue_id"] == queue_id)
                and (diagnosis is None or row["diagnosis"] == diagnosis)):
            return copy.deepcopy(row)
        if queue_id is not None:
            row["queue_id"] = queue_id
        self._set_status(row, status, diagnosis)
        self._save(state)
        return copy.deepcopy(row)

    def resolve_success(self, room: str, text: str, queue_id: int | None = None, *,
                        resolution="page_observed") -> dict | None:
        """Archive an exact observation; page presence alone does not prove its sender."""
        room, text = _room(room), _text(text, "失败内容", MAX_CANDIDATE_CHARACTERS)
        queue_id, resolution = _queue_id(queue_id), _diagnosis_code(resolution, "恢复结果")
        state = copy.deepcopy(self._state)
        row = self._matching_candidate(state, room, text, queue_id)
        if row is None:
            return None
        if row["status"] == "resolved" and row["resolution"] == resolution:
            return copy.deepcopy(row)
        if queue_id is not None:
            row["queue_id"] = queue_id
        self._set_status(row, "resolved", resolution=resolution)
        self._save(state)
        return copy.deepcopy(row)

    def remove_candidate(self, candidate_id: str) -> bool:
        candidate_id = _identifier(candidate_id)
        state = copy.deepcopy(self._state)
        state["candidates"] = [row for row in state["candidates"] if row["id"] != candidate_id]
        if len(state["candidates"]) == len(self._state["candidates"]):
            return False
        self._save(state)
        return True

    def promote_candidate(self, candidate_id: str, term: str, replacement: str = DEFAULT_REPLACEMENT,
                          room: str | None = None, *, enabled: bool = True, action: str = "replace") -> dict:
        candidate_id = _identifier(candidate_id)
        candidate = next((row for row in self._state["candidates"] if row["id"] == candidate_id), None)
        if candidate is None:
            raise DanmakuFilterError("要确认的失败候选不存在")
        term = _text(term, "过滤词", MAX_RULE_CHARACTERS)
        if term not in candidate["text"]:
            raise DanmakuFilterError("确认的过滤词必须来自所选失败内容")
        # Keep the original observation: confirming a local rule is no proof
        # that the platform rejected this particular word.
        return self.add_rule(term, replacement, candidate["room"] if room is None else room,
                             enabled=enabled, action=action)
