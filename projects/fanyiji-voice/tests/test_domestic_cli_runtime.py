import json
from pathlib import Path
import threading

import pytest

from fanyiji import domestic_cli_runtime as runtime


@pytest.fixture
def account_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("MMX_CONFIG_DIR", raising=False)
    monkeypatch.delenv("MINIMAX_DATA_DIR", raising=False)
    monkeypatch.delenv("MAVIS_DATA_DIR", raising=False)
    return tmp_path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_environment_rejects_key_endpoint_and_node_injection(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "data_root", lambda: tmp_path)
    for key in ("MINIMAX_API_KEY", "CODEBUDDY_API_KEY", "OPENAI_API_KEY", "KIMI_API_KEY",
                "CODEBUDDY_AUTH_TOKEN", "NODE_OPTIONS", "MINIMAX_BASE_URL", "KIMI_CODE_HOME"):
        monkeypatch.setenv(key, "synthetic-private-value")
    for provider in runtime.PACKAGES:
        env = runtime.safe_cli_environment(provider)
        assert "synthetic-private-value" not in env.values()
        assert not any("API_KEY" in key or key == "NODE_OPTIONS" for key in env)
    assert runtime.safe_cli_environment("kimi")["KIMI_CODE_HOME"] == str((tmp_path / "cli-accounts" / "kimi").resolve())


def test_mmx_no_account_or_key_never_invokes_cloud(account_home, monkeypatch):
    monkeypatch.setattr(runtime, "run_cli_checked", lambda *args, **kwargs: pytest.fail("must not contact account service"))
    assert runtime._probe_mmx()["status"] == "not_logged_in"
    write_json(account_home / ".mmx" / "config.json", {"api_key": "synthetic-private-value"})
    assert runtime._probe_mmx()["status"] == "api_key_blocked"


@pytest.mark.parametrize("endpoint", ["https://third-party.example", "http://api.minimax.io",
                                       "https://user:password@api.minimax.cn", "https://api.minimax.io/?x=y"])
def test_mmx_custom_endpoint_blocks_before_cli(account_home, monkeypatch, endpoint):
    write_json(account_home / ".mmx" / "config.json", {"base_url": endpoint,
               "oauth": {"access_token": "synthetic-private-token", "refresh_token": "synthetic-private-refresh", "expires_at": "2099-01-01T00:00:00Z"}})
    monkeypatch.setattr(runtime, "run_cli_checked", lambda *args, **kwargs: pytest.fail("unsafe endpoint"))
    assert runtime._probe_mmx()["status"] == "auth_unverified"


def test_mmx_official_status_sanitized(account_home, monkeypatch):
    write_json(account_home / ".mmx" / "config.json", {"oauth": {
        "access_token": "synthetic-private-token", "refresh_token": "synthetic-private-refresh", "expires_at": "2099-01-01T00:00:00Z"},
        "default_text_model": "MiniMax-example"})
    def run(provider, arguments, **kwargs):
        assert provider == "minimax" and arguments == ["auth", "status", "--output", "json"]
        return json.dumps({"method": "oauth", "account": "synthetic-private-account", "token_expires": 1234})
    monkeypatch.setattr(runtime, "run_cli_checked", run)
    result = runtime._probe_mmx()
    assert result["status"] == "ready" and result["auth_mode"] == "oauth"
    assert "synthetic-private" not in json.dumps(result)


def mcode_state(home):
    write_json(home / ".minimax" / "auth" / "prod" / "cn" / "mcode-public" / "auth-state.json",
               {"status": "authenticated", "clientId": "mcode-public", "audience": "agent-backend"})


def test_mcode_model_list_excludes_api_providers(account_home, monkeypatch):
    mcode_state(account_home)
    monkeypatch.setattr(runtime, "run_cli_checked", lambda *args, **kwargs: json.dumps({
        "minimaxModelSource": "token_plan", "providers": [
            {"kind": "minimax-api-key", "active": True, "enabled": True, "hasApiKey": True,
             "models": ["private-api-model"]},
            {"kind": "minimax-oauth", "active": True, "enabled": True, "hasApiKey": False,
             "models": [{"modelId": "MiniMax-example", "name": "Example"}]}]}))
    result = runtime._probe_mcode()
    assert result["status"] == "ready"
    assert result["models"] == [{"id": "MiniMax-example", "name": "Example"}]


@pytest.mark.parametrize("active,enabled", [(False, True), (True, False)])
def test_mcode_inactive_oauth_cannot_enter_pool(account_home, monkeypatch, active, enabled):
    mcode_state(account_home)
    monkeypatch.setattr(runtime, "run_cli_checked", lambda *args, **kwargs: json.dumps({
        "minimaxModelSource": "token_plan", "providers": [{"kind": "minimax-oauth", "hasApiKey": False,
             "active": active, "enabled": enabled, "models": ["MiniMax-example"]}]}))
    assert runtime._probe_mcode()["status"] == "auth_unverified"


def test_blocked_and_cancelled_never_start_client(monkeypatch):
    monkeypatch.setattr(runtime, "find_cli", lambda provider: pytest.fail("must not inspect or invoke"))
    assert runtime.probe_cli("qwen")["status"] == "unsupported"
    cancelled = threading.Event()
    cancelled.set()
    assert runtime.probe_cli("minimax", cancellation_event=cancelled)["status"] == "cancelled"


def test_unverified_version_cannot_look_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "find_cli", lambda provider: tmp_path / "official-entry.mjs")
    monkeypatch.setattr(runtime, "_package_version", lambda *args: "999.0.0")
    monkeypatch.setattr(runtime, "_probe_mmx", lambda **kwargs: pytest.fail("unverified protocol"))
    result = runtime.probe_cli("minimax")
    assert result["status"] == "unsupported" and result["installed"] is True


def test_provider_cannot_inject_command():
    with pytest.raises(ValueError):
        runtime.cli_argv("minimax; Write-Output unsafe")


@pytest.mark.parametrize("missing", ["refresh_token", "expires_at"])
def test_partial_mmx_oauth_cannot_fall_back_to_file_key(account_home, monkeypatch, missing):
    oauth = {"access_token": "synthetic-private-token", "refresh_token": "synthetic-private-refresh", "expires_at": "2099-01-01T00:00:00Z"}
    del oauth[missing]
    write_json(account_home / ".mmx" / "config.json", {"oauth": oauth, "api_key": "synthetic-private-key"})
    monkeypatch.setattr(runtime, "run_cli_checked", lambda *args, **kwargs: pytest.fail("must not query Key quota"))
    assert runtime._probe_mmx()["status"] == "api_key_blocked"
    from fanyiji.domestic_cli_engine import DomesticCliError
    with pytest.raises(DomesticCliError) as error:
        runtime.assert_account_route("minimax")
    assert error.value.code == "api_key_blocked"


@pytest.mark.parametrize("options,code", [({"authMode": "api-key"}, "api_key_blocked"),
    ({"authMode": "managed-login", "baseURL": "https://custom.example/v1"}, "auth_unverified"),
    ({"baseURL": "https://user:password@agent.minimax.cn/v1"}, "auth_unverified")])
def test_mcode_changed_provider_options_stop_before_client(account_home, monkeypatch, options, code):
    import yaml
    from fanyiji.domestic_cli_engine import DomesticCliError
    mcode_state(account_home)
    (account_home / ".minimax" / "config.yaml").write_text(yaml.safe_dump({"provider": {"minimax": {"options": options}}}), encoding="utf-8")
    with pytest.raises(DomesticCliError) as error:
        runtime.assert_account_route("minimax_code")
    assert error.value.code == code
    monkeypatch.setattr(runtime, "run_cli_checked", lambda *args, **kwargs: pytest.fail("unverified auth route"))
    assert runtime._probe_mcode()["status"] == "auth_unverified"


def test_mcode_official_managed_preset_placeholder_is_allowed(account_home):
    import yaml
    mcode_state(account_home)
    (account_home / ".minimax" / "config.yaml").write_text(yaml.safe_dump({"provider": {"minimax": {"options": {
        "authMode": "managed-login", "baseURL": "https://agent.minimax.cn/mavis/api/v1/llm/v1", "apiKey": "sk-xxx"}}}}), encoding="utf-8")
    runtime.assert_account_route("minimax_code")


def test_completed_codebuddy_login_survives_nonzero_tui_exit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fanyiji import domestic_cli_codebuddy as adapter
    monkeypatch.setattr(runtime, "data_root", lambda: tmp_path)
    monkeypatch.setattr(runtime, "cli_argv", lambda provider: ["official-node", "official-entry"])
    process = SimpleNamespace(poll=lambda: 130, returncode=130)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(adapter, "probe_account", lambda **kwargs: {"status": "ready"})
    runtime.login_cli("codebuddy")


def test_unlogged_codebuddy_nonzero_exit_is_not_success(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fanyiji import domestic_cli_codebuddy as adapter
    monkeypatch.setattr(runtime, "data_root", lambda: tmp_path)
    monkeypatch.setattr(runtime, "cli_argv", lambda provider: ["official-node", "official-entry"])
    process = SimpleNamespace(poll=lambda: 130, returncode=130)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(adapter, "probe_account", lambda **kwargs: {"status": "not_logged_in"})
    with pytest.raises(RuntimeError):
        runtime.login_cli("codebuddy")
