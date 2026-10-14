"""CodeBuddy 2.164 account markers and isolated, tool-free JSON translation."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
import time


_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
_ACCOUNT_FILE = "Tencent-Cloud.coding-copilot.info"
_KEY_OVERRIDES = {"CODEBUDDY_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
_OPAQUE_OVERRIDES = {"CODEBUDDY_AUTH_TOKEN", "AUTH_TOKEN", "ACC_PRODUCT_CONFIG_V3", "ACC_PRODUCT_CONFIG_V2",
                     "CODEBUDDY_CREDENTIALS_IN_MEMORY", "CODEBUDDY_CUSTOM_HEADERS"}


def _error(code):
    from .domestic_cli_engine import DomesticCliError
    return DomesticCliError(code)


def _account_marker_path() -> Path:
    # Public 2.164.0 bundle: FilePathService.sharedDataPath/auth/
    # resolveCustomAuthId(product.authentication.id).info. For the official
    # default product, resolveCustomAuthId preserves the ID rather than hashing it.
    home = Path.home()
    if os.name == "nt":
        base = home / "AppData/Local/CodeBuddyExtension"
    elif sys.platform == "darwin":
        base = home / "Library/Application Support/CodeBuddyExtension"
    else:
        base = home / ".local/share/CodeBuddyExtension"
    return base / "Data/Public/auth" / _ACCOUNT_FILE


def _probe_result(status, message, *, account_auth=False):
    return {"status": status, "message": message,
            "auth_mode": "oauth" if account_auth else "api_key" if status == "api_key_blocked" else "not_verified", "models": [],
            "capabilities": {"protocol": "codebuddy_json", "account_auth": account_auth, "tool_free": True}}


def probe_account(*, cancellation_event=None) -> dict:
    """Read only non-secret account/type/expiry markers; never decrypt credentials."""
    if cancellation_event is not None and cancellation_event.is_set():
        return _probe_result("cancelled", "CodeBuddy 账号检查已取消。")
    from .domestic_cli_runtime import safe_cli_environment
    environment = safe_cli_environment("codebuddy")
    if any(key in environment for key in _KEY_OVERRIDES):
        return _probe_result("api_key_blocked", "检测到 API Key 环境配置，本模式仅接受官方账号登录。")
    if any(key in environment for key in _OPAQUE_OVERRIDES):
        return _probe_result("auth_unverified", "检测到自定义认证环境，无法确认使用官方账号登录。")
    path = _account_marker_path()
    try:
        if Path(str(path) + ".logged-out").exists() or not path.is_file():
            return _probe_result("not_logged_in", "未找到 CodeBuddy 官方账号登录状态，请打开官方 CLI 登录。")
        if path.stat().st_size > 1024 * 1024:
            return _probe_result("auth_unverified", "CodeBuddy 账号记录格式无法确认。")
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            return _probe_result("auth_unverified", "CodeBuddy 账号记录格式无法确认。")
        auth, account = raw.get("auth"), raw.get("account")
        if not isinstance(auth, dict) or not isinstance(account, dict):
            return _probe_result("auth_unverified", "CodeBuddy 账号记录已保护或格式未知，请在官方 CLI 核验。")
        kind = auth.get("tokenType")
        if kind == "ApiKey":
            return _probe_result("api_key_blocked", "CodeBuddy 当前记录为 API Key，本模式仅接受官方账号登录。")
        # Bearer by itself is insufficient: the official CustomToken storage also
        # emits Bearer. Browser-account sessions carry a future refresh expiry;
        # CustomToken/API-key storage uses the LONG_TIME_AGO sentinel instead.
        expires = auth.get("refreshExpiresAt")
        credential_fields_present = "accessToken" in auth and "refreshToken" in auth
        account_present = isinstance(account.get("uid"), str) and bool(account["uid"])
        future_refresh = (type(expires) in (int, float) and math.isfinite(expires)
                          and expires > time.time() * 1000)
        if kind != "Bearer" or not account_present or not credential_fields_present:
            return _probe_result("auth_unverified", "CodeBuddy 账号认证方式无法确认，请在官方 CLI 核验。")
        if not future_refresh:
            return _probe_result("auth_unverified", "CodeBuddy 刷新期限已过或记录不足，请在官方 CLI 重新登录。")
        if cancellation_event is not None and cancellation_event.is_set():
            return _probe_result("cancelled", "CodeBuddy 账号检查已取消。")
        return _probe_result("ready", "检测到 CodeBuddy 官方账号及有效刷新期限；实际请求仍由官方 CLI 核验。", account_auth=True)
    except (OSError, ValueError, UnicodeError):
        return _probe_result("auth_unverified", "无法确认 CodeBuddy 本地账号记录，请在官方 CLI 核验。")


def _has_tool_request(value, depth=0):
    if depth > 20:
        raise _error("invalid_response")
    if isinstance(value, dict):
        if value.get("type") in ("tool_use", "tool_call", "function_call") or value.get("stop_reason") == "tool_use":
            return True
        if value.get("tool_calls") or value.get("function_call"):
            return True
        return any(_has_tool_request(child, depth + 1) for child in value.values() if isinstance(child, (dict, list)))
    if isinstance(value, list):
        return any(_has_tool_request(child, depth + 1) for child in value)
    return False


def _parse_result(stdout: str) -> tuple[str, str]:
    try:
        result = json.loads(stdout)
    except (TypeError, ValueError, RecursionError):
        raise _error("invalid_response") from None
    if not isinstance(result, dict):
        raise _error("invalid_response")
    if _has_tool_request(result):
        raise _error("tool_request")
    if result.get("type") != "result" or result.get("subtype") != "success" or result.get("is_error") is not False:
        raise _error("process_failed")
    content = result.get("result")
    if not isinstance(content, str) or not content.strip():
        raise _error("invalid_response")
    actual_model = ""
    usage = result.get("modelUsage")
    if isinstance(usage, dict):
        identifiers = [identifier for identifier, value in usage.items()
                       if isinstance(identifier, str) and _MODEL_ID.fullmatch(identifier) and isinstance(value, dict)
                       and identifier.lower() not in ("auto", "default", "automatic")
                       and type(value.get("outputTokens")) in (int, float) and value["outputTokens"] > 0]
        if len(identifiers) == 1:
            actual_model = identifiers[0]
    return content.strip(), actual_model


def invoke_codebuddy(transport, argv_prefix: list[str], environment: dict, messages: list[dict],
                    requested_model: str, cancellation_event=None, *, account_probe=None) -> tuple[str, str]:
    if cancellation_event is not None and cancellation_event.is_set():
        raise _error("cancelled")
    if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str) for message in messages):
        raise _error("invalid_response")
    if not isinstance(requested_model, str) or (requested_model and not _MODEL_ID.fullmatch(requested_model)):
        raise _error("model_mismatch")
    if any(key in environment for key in _KEY_OVERRIDES | _OPAQUE_OVERRIDES):
        raise _error("api_key_blocked")
    # Each print invocation re-reads the vendor's login record. A once-verified
    # engine must not fall back to a newly selected API-Key route after logout.
    account = (account_probe or probe_account)(cancellation_event=cancellation_event)
    if not isinstance(account, dict) or account.get("status") != "ready":
        raise _error(account.get("status", "auth_unverified") if isinstance(account, dict) else "auth_unverified")
    capabilities = account.get("capabilities")
    if (account.get("auth_mode") != "oauth" or not isinstance(capabilities, dict)
            or capabilities.get("account_auth") is not True):
        raise _error("auth_unverified")
    if capabilities.get("tool_free") is not True or capabilities.get("protocol") != "codebuddy_json":
        raise _error("tool_request")
    # In 2.164 an empty setting-sources string falls back to default scopes.
    # The shipped SettingsSourcePolicy explicitly accepts "none" for isolation.
    argv = [*argv_prefix, "--tools", "", "--strict-mcp-config", "--setting-sources", "none",
            "--no-session-persistence", "--output-format", "json", "-p"]
    if requested_model:
        argv.extend(["--model", requested_model])
    # Every instruction, reference and source is delivered on stdin, never in
    # command arguments, a user-facing command line, or a prompt file.
    # Print mode consumes plain text, not API-style role messages. The app's
    # static translation instructions precede the quoted request data on stdin;
    # do not misrepresent these as native CodeBuddy system-role messages.
    instructions = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    reference = "\n\n".join(message["content"] for message in messages if message["role"] != "system")
    payload = instructions + "\n\n引用输入 JSON：\n" + reference
    if cancellation_event is not None and cancellation_event.is_set():
        raise _error("cancelled")
    stdout = transport.request(argv, payload, environment, cancellation_event=cancellation_event)
    return _parse_result(stdout)
