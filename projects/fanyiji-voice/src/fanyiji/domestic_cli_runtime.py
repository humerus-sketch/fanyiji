"""Locate official clients and inspect account mode without taking their credentials."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit

from .domestic_cli_catalog import BLOCKED_PROVIDERS, validate_provider
from .paths import data_root


# These versions were checked against the vendor's shipped argument/parser code.
# A newer client must be verified before its protocol is treated as usable.
PACKAGES = {
    "minimax": ("mmx-cli", "dist/mmx.mjs", "1.0.28"),
    "minimax_code": ("@minimax-ai/code", "cli.js", "0.6.9"),
    "kimi": ("@moonshot-ai/kimi-code", "dist/main.mjs", "2.1.1"),
    "codebuddy": ("@tencent-ai/codebuddy-code", "bin/codebuddy", "2.164.0"),
}
_BIN_NAMES = {"minimax": "mmx", "minimax_code": "mcode", "kimi": "kimi",
              "codebuddy": "codebuddy", "trae": "traecli", "qwen": "qwen"}
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")


def _node() -> Path | None:
    executable = shutil.which("node.exe" if os.name == "nt" else "node")
    return Path(executable).resolve() if executable else None


def find_cli(provider: str) -> Path | None:
    provider = validate_provider(provider)
    if provider in PACKAGES:
        package, entry, _ = PACKAGES[provider]
        paths = [data_root() / "cli-runtime" / provider / "node_modules" / package / entry]
        launcher = shutil.which(_BIN_NAMES[provider])
        if launcher:
            # npm wrappers are never passed to a shell: resolve the known JS entry.
            parent = Path(launcher).resolve().parent
            paths += [parent / "node_modules" / package / entry,
                      parent.parent / "lib" / "node_modules" / package / entry]
        for path in paths:
            if path.is_file() and _node() is not None:
                return path.resolve()
        return None
    found = shutil.which(_BIN_NAMES[provider])
    if found and Path(found).suffix.lower() not in (".bat", ".cmd", ".ps1"):
        return Path(found).resolve()
    return None


def cli_argv(provider: str) -> list[str]:
    path = find_cli(provider)
    if path is None:
        raise RuntimeError("未找到官方 CLI，请先安装")
    if provider in PACKAGES:
        node = _node()
        if node is None:
            raise RuntimeError("未找到 Node.js 运行环境")
        return [str(node), str(path)]
    return [str(path)]


def safe_cli_environment(provider: str) -> dict[str, str]:
    validate_provider(provider)
    # Pass OS, locale and transport settings; exclude arbitrary Node injection,
    # API Keys, token overrides and vendor endpoint/provider overrides.
    names = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
             "TMPDIR", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA",
             "LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA",
             "LANG", "LC_ALL", "LC_CTYPE", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
             "NO_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS"}
    environment = {key: value for key, value in os.environ.items() if key.upper() in names}
    environment.update({"NO_COLOR": "1", "DO_NOT_TRACK": "1"})
    # Respect a vendor's explicit account-storage location, not an auth token.
    for name in {"minimax": ("MMX_CONFIG_DIR",),
                 "minimax_code": ("MINIMAX_DATA_DIR", "MAVIS_DATA_DIR"),
                 "kimi": ("KIMI_CODE_HOME",),
                 "codebuddy": ("CODEBUDDY_CONFIG_DIR",),
                 "trae": ("TRAECLI_HOME",)}.get(provider, ()):
        if os.environ.get(name):
            environment[name] = os.environ[name]
    if provider == "kimi":
        environment["KIMI_CODE_HOME"] = str((data_root() / "cli-accounts" / "kimi").resolve())
    return environment


def _cancelled(cancellation_event) -> bool:
    return cancellation_event is not None and cancellation_event.is_set()


def run_cli_checked(provider: str, arguments: list[str], *, cancellation_event=None) -> str:
    from .domestic_cli_engine import _OwnedCliProcessTransport
    transport = _OwnedCliProcessTransport(timeout=20)
    try:
        return transport.request(cli_argv(provider) + arguments, "", safe_cli_environment(provider),
                                 cancellation_event=cancellation_event)
    finally:
        transport.close()


def _read_json(path: Path) -> dict:
    try:
        if path.stat().st_size > 256 * 1024:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, ValueError):
        return {}


def _package_version(provider: str, entry: Path) -> str:
    package, relative, _ = PACKAGES[provider]
    root = entry
    for _ in Path(relative).parts:
        root = root.parent
    value = _read_json(root / "package.json")
    version = value.get("version")
    return version if value.get("name") == package and isinstance(version, str) and len(version) <= 64 else ""


def _result(provider: str, status: str, message: str, **fields) -> dict:
    return {"provider": provider, "status": status, "message": message, "installed": False,
            "version": "", "auth_mode": "not_verified", "models": [],
            "capabilities": {"protocol": "", "account_auth": False, "tool_free": False}, **fields}


def _official_minimax_url(value) -> bool:
    if not value:
        return True
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    return (parsed.scheme == "https" and parsed.hostname in ("api.minimax.io", "api.minimax.cn")
            and parsed.port in (None, 443) and parsed.username is None and parsed.password is None
            and not parsed.query and not parsed.fragment and parsed.path in ("", "/"))


def _valid_mmx_oauth(value) -> bool:
    # Match the vendor's parseOAuth exactly: a partially edited record is
    # discarded by mmx, which would otherwise fall back to config.api_key.
    return (isinstance(value, dict) and isinstance(value.get("access_token"), str)
            and bool(value["access_token"]) and isinstance(value.get("refresh_token"), str)
            and isinstance(value.get("expires_at"), str))


def _probe_mmx(*, cancellation_event=None) -> dict:
    path = Path(os.environ.get("MMX_CONFIG_DIR", str(Path.home() / ".mmx"))) / "config.json"
    config = _read_json(path)
    oauth = config.get("oauth")
    if not _valid_mmx_oauth(oauth):
        return {"status": "api_key_blocked" if config.get("api_key") else "not_logged_in",
                "message": "请通过官方账号登录；本模式不使用 API Key。"}
    if not _official_minimax_url(config.get("base_url")) or not _official_minimax_url(oauth.get("resource_url")):
        return {"status": "auth_unverified", "message": "CLI 配置了自定义服务地址，请恢复官方账号服务后再检查。"}
    # Only the official command refreshes/validates its own OAuth credentials.
    result = json.loads(run_cli_checked("minimax", ["auth", "status", "--output", "json"],
                                        cancellation_event=cancellation_event))
    if not isinstance(result, dict) or result.get("method") != "oauth":
        return {"status": "not_logged_in", "message": "官方 CLI 未确认账号登录。"}
    default = config.get("default_text_model")
    models = [{"id": default, "name": default}] if isinstance(default, str) and _MODEL.fullmatch(default) else []
    return {"status": "ready", "message": "MiniMax 官方 OAuth 账号已登录，可进行文字翻译；额度由账号方案决定。",
            "auth_mode": "oauth", "models": models,
            "capabilities": {"protocol": "mmx_json", "account_auth": True, "tool_free": True}}


def _probe_mcode(*, cancellation_event=None) -> dict:
    directory = Path(os.environ.get("MINIMAX_DATA_DIR") or os.environ.get("MAVIS_DATA_DIR") or (Path.home() / ".minimax"))
    states = [directory / "auth" / "prod" / region / "mcode-public" / "auth-state.json"
              for region in ("cn", "en")]
    authenticated = any((value := _read_json(path)).get("status") == "authenticated"
                        and value.get("clientId") == "mcode-public" and value.get("audience") == "agent-backend"
                        for path in states)
    if not authenticated:
        return {"status": "not_logged_in", "message": "请先使用 MiniMax Code 官方账号登录。"}
    try:
        assert_account_route("minimax_code", cancellation_event=cancellation_event)
    except Exception:
        return {"status": "auth_unverified", "message": "MiniMax Code 账号路由已变化，请恢复官方 Token Plan 账号配置后检查。"}
    result = json.loads(run_cli_checked("minimax_code", ["provider", "list", "--json"],
                                        cancellation_event=cancellation_event))
    if not isinstance(result, dict) or result.get("minimaxModelSource") != "token_plan":
        return {"status": "api_key_blocked", "message": "MiniMax Code 当前未使用 Token Plan 账号路线；API Key 路线已禁用。"}
    models = []
    for provider in result.get("providers", []):
        if (not isinstance(provider, dict) or provider.get("kind") != "minimax-oauth"
                or provider.get("hasApiKey") is not False or provider.get("active") is not True
                or provider.get("enabled") is not True):
            continue
        for item in provider.get("models", []):
            if isinstance(item, str):
                identifier, name = item, item
            elif isinstance(item, dict):
                identifier = item.get("modelId") or item.get("id")
                name = item.get("name") or identifier
            else:
                continue
            if isinstance(identifier, str) and _MODEL.fullmatch(identifier):
                models.append({"id": identifier, "name": name if isinstance(name, str) and len(name) <= 128 else identifier})
    if not models:
        return {"status": "auth_unverified", "message": "账号已登录，但官方客户端尚未提供可用的账号模型；请检查账号权限和额度。", "auth_mode": "oauth"}
    return {"status": "ready", "message": "MiniMax Code 官方账号已登录，使用无工具 lightweight 翻译模式。",
            "auth_mode": "oauth", "models": models, "default_model": models[0]["id"],
            "capabilities": {"protocol": "mcode_json", "account_auth": True, "tool_free": True}}


def assert_account_route(provider: str, *, cancellation_event=None) -> None:
    """Cheap per-request check: a login/settings change cannot enable a Key fallback."""
    from .domestic_cli_engine import DomesticCliError
    if _cancelled(cancellation_event):
        raise DomesticCliError("cancelled")
    if provider == "minimax":
        config = _read_json(Path(os.environ.get("MMX_CONFIG_DIR", str(Path.home() / ".mmx"))) / "config.json")
        oauth = config.get("oauth")
        if not _valid_mmx_oauth(oauth):
            raise DomesticCliError("api_key_blocked" if config.get("api_key") else "not_logged_in")
        if not _official_minimax_url(config.get("base_url")) or not _official_minimax_url(oauth.get("resource_url")):
            raise DomesticCliError("auth_unverified")
    elif provider == "minimax_code":
        directory = Path(os.environ.get("MINIMAX_DATA_DIR") or os.environ.get("MAVIS_DATA_DIR") or (Path.home() / ".minimax"))
        authenticated = any((value := _read_json(directory / "auth" / "prod" / region / "mcode-public" / "auth-state.json")).get("status") == "authenticated"
                            and value.get("clientId") == "mcode-public" and value.get("audience") == "agent-backend"
                            for region in ("cn", "en"))
        if not authenticated:
            raise DomesticCliError("not_logged_in")
        config_path = directory / "config.yaml"
        if config_path.exists():
            try:
                import yaml
                if config_path.stat().st_size > 256 * 1024:
                    raise ValueError("oversize_config")
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                if not isinstance(config, dict):
                    raise ValueError("invalid_config")
            except Exception:
                raise DomesticCliError("auth_unverified") from None
            if config.get("minimaxModelSource") not in (None, "token_plan"):
                raise DomesticCliError("api_key_blocked")
            providers = config.get("provider", {})
            builtin = providers.get("minimax", {}) if isinstance(providers, dict) else None
            options = builtin.get("options", {}) if isinstance(builtin, dict) else None
            if not isinstance(options, dict):
                raise DomesticCliError("auth_unverified")
            if options.get("authMode") not in (None, "managed-login"):
                raise DomesticCliError("api_key_blocked")
            base = options.get("baseURL")
            if base:
                try:
                    url = urlsplit(base)
                    allowed = (url.scheme == "https" and url.hostname in
                               ("agent.minimax.cn", "agent.minimax.io", "agent.minimaxi.com")
                               and url.port in (None, 443) and url.username is None and url.password is None
                               and not url.query and not url.fragment)
                except (TypeError, ValueError):
                    allowed = False
                if not allowed:
                    raise DomesticCliError("auth_unverified")
    else:
        raise DomesticCliError("unsupported")


def probe_cli(provider: str, *, cancellation_event=None) -> dict:
    provider = validate_provider(provider)
    if _cancelled(cancellation_event):
        return _result(provider, "cancelled", "检查已取消")
    if provider in BLOCKED_PROVIDERS:
        return _result(provider, "unsupported", BLOCKED_PROVIDERS[provider])
    entry = find_cli(provider)
    if entry is None:
        return _result(provider, "not_installed", "未安装官方 CLI，请通过官方下载与说明安装。")
    version = _package_version(provider, entry) if provider in PACKAGES else ""
    base = {"installed": True, "version": version}
    if provider not in PACKAGES or version != PACKAGES[provider][2]:
        return _result(provider, "unsupported", "此 CLI 版本尚未验证，不能确认其账号与翻译协议。", **base)
    try:
        if provider == "minimax":
            value = _probe_mmx(cancellation_event=cancellation_event)
        elif provider == "minimax_code":
            value = _probe_mcode(cancellation_event=cancellation_event)
        elif provider == "kimi":
            from .domestic_cli_kimi import probe_account
            value = probe_account(cancellation_event=cancellation_event)
        else:
            from .domestic_cli_codebuddy import probe_account
            value = probe_account(cancellation_event=cancellation_event)
        if _cancelled(cancellation_event):
            return _result(provider, "cancelled", "检查已取消", **base)
        fields = {key: value[key] for key in ("auth_mode", "models", "capabilities", "configured_model", "default_model")
                  if key in value}
        return _result(provider, value["status"], value["message"], **base, **fields)
    except Exception:
        # CLI stderr, OAuth account fields and exception payloads are private.
        status = "cancelled" if _cancelled(cancellation_event) else "error"
        return _result(provider, status, "检查已取消" if status == "cancelled" else "无法完成官方 CLI 账号检查，请在官方客户端核对登录。", **base)


def login_cli(provider: str, *, cancellation_event=None) -> None:
    provider = validate_provider(provider)
    if provider in BLOCKED_PROVIDERS or _cancelled(cancellation_event):
        raise RuntimeError("该服务不支持本模式或操作已取消")
    arguments = {"minimax": ["auth", "login", "--recommend", "--region", "cn"],
                 "minimax_code": ["login"], "kimi": ["login"], "codebuddy": [],
                 "trae": ["login"]}[provider]
    # A real console is intentional: official device/OAuth flows need user input.
    options = {"env": safe_cli_environment(provider), "shell": False,
               "cwd": str(data_root() / "cli-runtime" / provider)}
    Path(options["cwd"]).mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    process = subprocess.Popen(cli_argv(provider) + arguments, **options)
    deadline = time.monotonic() + 900
    try:
        while process.poll() is None:
            if _cancelled(cancellation_event) or time.monotonic() >= deadline:
                raise RuntimeError("账号登录已取消或超时")
            threading.Event().wait(0.1)
        if _cancelled(cancellation_event):
            raise RuntimeError("账号登录已取消")
        if process.returncode != 0 and provider == "codebuddy":
            # Closing the interactive TUI often exits nonzero. A fresh official
            # browser-account marker can still confirm the completed login.
            from .domestic_cli_codebuddy import probe_account
            if probe_account(cancellation_event=cancellation_event).get("status") == "ready":
                return
        if process.returncode != 0:
            raise RuntimeError("官方账号登录未完成")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
