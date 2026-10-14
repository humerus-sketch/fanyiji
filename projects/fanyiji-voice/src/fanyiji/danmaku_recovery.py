"""Evidence-based recovery decisions; no inference of platform banned words."""
from __future__ import annotations

from dataclasses import dataclass


SAFE_ERRORS = frozenset({"none", "wrong_room", "not_logged_in", "csrf_missing", "csrf_invalid",
    "request_timeout", "network_unknown", "invalid_response", "script_unavailable", "script_timeout",
    "window_closed", "browser_closed", "navigation_changed", "queue_full", "browser_unavailable",
    "feature_unavailable", "text_too_long", "chat_unavailable"})


@dataclass(frozen=True)
class RecoveryDecision:
    status: str
    category: str
    action: str
    message: str

    @property
    def diagnosis(self):
        return {"category": self.category, "action": self.action, "message": self.message}


def failure_decision(reason: str, payload: dict | None = None) -> RecoveryDecision:
    payload = payload or {}
    error = payload.get("error")
    if not isinstance(error, str):
        error = ""
    if reason == "platform_f":
        return RecoveryDecision("recovering", "platform_response", "retry_current",
                                "平台返回 f 原因未确定 正在自动分析并保留顺序重试")
    if reason in {"result_unknown", "receipt_unconfirmed", "restart_verify", "manual_verify"}:
        return RecoveryDecision("waiting_confirmation", "uncertain_delivery", "verify_only",
                                "发送结果未知 正在持续只读核验 不重复提交")
    if reason == "platform_rejected":
        if payload.get("code") == 1003212:
            return RecoveryDecision("recovering", "length", "repacketize", "正在按账号长度上限重新拆分")
        return RecoveryDecision("action_required", "platform_rejection", "check_environment",
                                "平台明确拒绝 原因未确定 正在自动检查登录和页面状态")
    if error == "not_logged_in":
        return RecoveryDecision("action_required", "authentication", "wait_for_login",
                                "账号未登录 完成登录后自动恢复当前消息")
    if error in {"csrf_missing", "csrf_invalid"}:
        return RecoveryDecision("recovering", "authentication", "recheck_login", "登录凭证检查失败 正在自动复检")
    if error == "text_too_long":
        return RecoveryDecision("recovering", "length", "repacketize", "正在按账号长度上限重新拆分")
    if error == "feature_unavailable":
        return RecoveryDecision("action_required", "local_feature", "repair_local_file", "本机弹幕功能不可用 消息保留")
    return RecoveryDecision("recovering", "connection", "recheck_login", "请求尚未提交 正在自动检查连接和页面")
