"""Kimi Code 2.1.1 account-only print mode with an enforced empty tool set.

The official CLI requires ``-p <prompt>`` and does not read a print prompt from
stdin. A fixed Node bootstrap transfers our pipe into its in-memory argv before
importing the official entrypoint. Source text never enters OS argv or a file.
The CLI remains responsible for OAuth refresh and all cloud communication.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import tempfile
import time
import tomllib

from .paths import data_root


_ALIAS = re.compile(r"kimi-code/[A-Za-z0-9][A-Za-z0-9._:/-]{0,117}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
_MANAGED_PROVIDER = "managed:kimi-code"
_OFFICIAL_BASES = {"https://api.kimi.com/coding/v1", "https://api.kimi.ai/coding/v1"}
_MAX_PROFILE_BYTES = 256 * 1024
_MAX_TEXT_BYTES = 128 * 1024
_BOOTSTRAP = r"""import {pathToFileURL} from 'node:url';
const [entry,agent,skills,model]=process.argv.slice(1);
let prompt='';
process.stdin.setEncoding('utf8');
for await (const chunk of process.stdin) {
  prompt+=chunk.toString('utf8');
  if (Buffer.byteLength(prompt,'utf8')>131072) throw new Error('invalid_prompt');
}
if (!prompt.trim()) throw new Error('invalid_prompt');
process.argv=[process.execPath,entry,'--agent-file',agent,'--skills-dir',skills,
              '--output-format','stream-json','-m',model,'-p',prompt];
await import(pathToFileURL(entry).href);
"""


def _error(code):
    from .domestic_cli_engine import DomesticCliError
    return DomesticCliError(code)


def _cancelled(event):
    return event is not None and event.is_set()


def _status(code, *, models=(), configured_model="") -> dict:
    messages = {"ready": "已保存 Kimi 官方 OAuth 登录，令牌有效性和额度由官方客户端核验。",
                "not_logged_in": "请先在国内 CLI 管理中完成 Kimi 账号登录。",
                "api_key_blocked": "Kimi 翻译只使用官方账号登录，不接受 API Key。",
                "auth_unverified": "无法确认 Kimi 使用官方账号配置，请检查登录状态。",
                "unsupported": "Kimi 配置含额外执行扩展，不能用于无工具翻译。",
                "cancelled": "Kimi 账号检查已取消。"}
    return {"status": code, "message": messages[code],
            "auth_mode": "oauth" if code == "ready" else "api_key" if code == "api_key_blocked" else "not_verified",
            "models": list(models), "configured_model": configured_model,
            "capabilities": {"protocol": "kimi_stream_json", "account_auth": code == "ready",
                             "tool_free": code == "ready"}}


def _read_owned(path: Path, home: Path, parser):
    if not path.resolve().is_relative_to(home) or path.stat().st_size > _MAX_PROFILE_BYTES:
        raise ValueError("invalid_profile")
    value = parser(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid_profile")
    return value


def _has_manual_credentials(value):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).replace("_", "").lower()
            if normalized in {"apikey", "apikeyenv"} and child not in (None, ""):
                return True
            if normalized in {"customheaders", "headers"} and isinstance(child, dict) and child:
                return True
            if key == "env" and isinstance(child, dict) and child:
                return True
            if _has_manual_credentials(child):
                return True
    elif isinstance(value, list):
        return any(_has_manual_credentials(child) for child in value)
    return False


def probe_account(*, cancellation_event=None) -> dict:
    """Read only the app-owned profile; never open OS-home Kimi credentials."""
    if _cancelled(cancellation_event):
        return _status("cancelled")
    from .domestic_cli_runtime import safe_cli_environment
    environment = safe_cli_environment("kimi")
    root = data_root().resolve()
    expected = root / "cli-accounts" / "kimi"
    supplied = environment.get("KIMI_CODE_HOME")
    if not supplied:
        return _status("auth_unverified")
    home = Path(supplied).resolve()
    if home != expected.resolve() or not home.is_relative_to(root):
        return _status("auth_unverified")
    config_file, token_file = home / "config.toml", home / "credentials" / "kimi-code.json"
    if not config_file.is_file() or not token_file.is_file():
        return _status("not_logged_in")
    try:
        config = _read_owned(config_file, home, tomllib.loads)
        if _has_manual_credentials(config):
            return _status("api_key_blocked")
        if any(config.get(key) for key in ("hooks", "plugins", "mcp", "mcp_servers", "mcpServers")):
            return _status("unsupported")
        # Extensions are absent from a fresh owned login profile. Do not launch
        # independently configured hook/MCP/plugin processes during translation.
        if any((home / name).exists() for name in ("mcp.json", "hooks.json", "plugins.json")):
            return _status("unsupported")
        providers = config.get("providers")
        provider = providers.get(_MANAGED_PROVIDER) if isinstance(providers, dict) else None
        if not isinstance(provider, dict) or provider.get("type") != "kimi":
            return _status("auth_unverified")
        base = provider.get("base_url", provider.get("baseUrl", ""))
        oauth = provider.get("oauth")
        if (not isinstance(base, str) or base.rstrip("/") not in _OFFICIAL_BASES
                or not isinstance(oauth, dict) or oauth.get("key") != "oauth/kimi-code"
                or oauth.get("storage", "file") != "file"
                or oauth.get("oauth_host", oauth.get("oauthHost", "https://auth.kimi.com")) != "https://auth.kimi.com"):
            return _status("auth_unverified")
        models = []
        raw_models = config.get("models")
        if isinstance(raw_models, dict):
            for alias, value in raw_models.items():
                if (isinstance(alias, str) and _ALIAS.fullmatch(alias) and isinstance(value, dict)
                        and value.get("provider") == _MANAGED_PROVIDER and isinstance(value.get("model"), str)
                        and _MODEL.fullmatch(value["model"])):
                    # Aliases are the official CLI's selectable ids, not an
                    # assertion about the server's returned inference model.
                    models.append({"id": alias, "name": alias})
        configured = config.get("default_model", config.get("defaultModel", ""))
        if not isinstance(configured, str) or configured not in {item["id"] for item in models}:
            return _status("auth_unverified")
        credentials = _read_owned(token_file, home, json.loads)
        access, refresh = credentials.get("access_token"), credentials.get("refresh_token")
        if not isinstance(access, str) or not access.strip():
            return _status("not_logged_in")
        expires = credentials.get("expires_at", 0)
        if type(expires) not in (int, float) or not math.isfinite(expires) or expires < 0:
            return _status("auth_unverified")
        if expires and expires <= time.time() * 1000 and (not isinstance(refresh, str) or not refresh.strip()):
            return _status("not_logged_in")
    except (OSError, UnicodeError, ValueError, TypeError):
        return _status("auth_unverified")
    if _cancelled(cancellation_event):
        return _status("cancelled")
    return _status("ready", models=models, configured_model=configured)


def _decode_output(value: str) -> tuple[str, str]:
    """Consume exactly one final Assistant message; never print raw JSONL."""
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 256 * 1024:
        raise _error("invalid_response")
    assistants = []
    actual_model = ""
    for line in value.splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            raise _error("invalid_response") from None
        if not isinstance(item, dict):
            raise _error("invalid_response")
        role = item.get("role")
        if role == "tool" or item.get("tool_calls") or item.get("function_call"):
            raise _error("tool_request")
        if role == "meta":
            if item.get("type") not in {"system.version", "session.resume_hint", "turn.step.retrying"}:
                raise _error("invalid_response")
            continue
        if role != "assistant" or not isinstance(item.get("content"), str) or not item["content"].strip():
            raise _error("invalid_response")
        assistants.append(item["content"].strip())
        if "model" in item:
            model = item["model"]
            if not isinstance(model, str) or not _MODEL.fullmatch(model):
                raise _error("model_unconfirmed")
            actual_model = model
    if len(assistants) != 1:
        raise _error("invalid_response")
    return assistants[0], actual_model


def invoke_kimi(transport, argv_prefix: list[str], environment: dict, messages: list[dict],
                requested_model: str, cancellation_event=None) -> tuple[str, str]:
    """Run the official print CLI via the shared owned/cancelable runner."""
    if _cancelled(cancellation_event):
        raise _error("cancelled")
    if (not isinstance(argv_prefix, list) or len(argv_prefix) != 2
            or not all(isinstance(item, str) and item for item in argv_prefix)
            or Path(argv_prefix[0]).name.lower() not in {"node", "node.exe"}
            or Path(argv_prefix[1]).name != "main.mjs"):
        raise _error("unsupported")
    if (not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(item, dict) or set(item) != {"role", "content"}
                   or not isinstance(item["content"], str) or not item["content"].strip() for item in messages)
            or [item["role"] for item in messages] != ["system", "user"]):
        raise _error("invalid_response")
    if len(messages[1]["content"].encode("utf-8")) > _MAX_TEXT_BYTES:
        raise _error("invalid_response")
    account = probe_account(cancellation_event=cancellation_event)
    if account.get("status") != "ready":
        raise _error(account.get("status", "auth_unverified"))
    models = {item["id"] for item in account["models"]}
    selected = requested_model or account["configured_model"]
    if selected not in models:
        raise _error("model_mismatch")
    from .domestic_cli_runtime import safe_cli_environment
    expected_home = safe_cli_environment("kimi").get("KIMI_CODE_HOME")
    if environment.get("KIMI_CODE_HOME") != expected_home:
        raise _error("auth_unverified")
    with tempfile.TemporaryDirectory(prefix="fanyiji-kimi-agent-") as temporary:
        directory = Path(temporary)
        agent, skills = directory / "subtitle-translator.md", directory / "empty-skills"
        skills.mkdir()
        agent.write_text("---\nname: subtitle-translator\ndescription: Translate one subtitle without tools\n"
                         "tools: []\nsubagents: []\ndisallowedTools:\n  - select_tools\n---\n\n"
                         + messages[0]["content"] + "\n", encoding="utf-8")
        # process.argv is assigned inside this fixed bootstrap only after it
        # reads stdin. Windows process command lines therefore contain no ASR.
        argv = [argv_prefix[0], "--input-type=module", "-e", _BOOTSTRAP,
                argv_prefix[1], str(agent), str(skills), selected]
        child_environment = dict(environment)
        child_environment.update({"KIMI_DISABLE_TELEMETRY": "1", "KIMI_DISABLE_CRON": "1",
                                  "KIMI_CODE_NO_AUTO_UPDATE": "1", "KIMI_LOG_LEVEL": "off",
                                  "KIMI_CODE_BACKGROUND_KEEP_ALIVE_ON_EXIT": "0",
                                  "KIMI_CODE_BACKGROUND_PRINT_BACKGROUND_MODE": "exit"})
        output = transport.request(argv, messages[1]["content"], child_environment,
                                   cancellation_event=cancellation_event)
    if _cancelled(cancellation_event):
        raise _error("cancelled")
    return _decode_output(output)
