"""Official Kimi no-tool print protocol using synthetic owned OAuth profiles."""
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from unittest.mock import Mock

import pytest

from fanyiji import domestic_cli_kimi as kimi
from fanyiji import domestic_cli_runtime as runtime
from fanyiji.domestic_cli_engine import DomesticCliError


MODEL = "kimi-code/kimi-for-coding"
CONTENT = '{"id":"request-1","translation":"你好"}'


@pytest.fixture
def owned(tmp_path, monkeypatch):
    root = tmp_path / "app-data"
    home = root / "cli-accounts" / "kimi"
    home.mkdir(parents=True)
    (home / "credentials").mkdir()
    config = (f'default_model = "{MODEL}"\n'
              '[providers."managed:kimi-code"]\n'
              'type = "kimi"\nbase_url = "https://api.kimi.com/coding/v1"\napi_key = ""\n'
              '[providers."managed:kimi-code".oauth]\nkey = "oauth/kimi-code"\nstorage = "file"\n'
              f'[models."{MODEL}"]\nprovider = "managed:kimi-code"\nmodel = "kimi-for-coding"\n')
    (home / "config.toml").write_text(config, encoding="utf-8")
    token = {"access_token": "synthetic-access-value", "refresh_token": "synthetic-refresh-value",
             "token_type": "Bearer", "expires_at": time.time() * 1000 + 120000}
    (home / "credentials" / "kimi-code.json").write_text(json.dumps(token), encoding="utf-8")
    monkeypatch.setattr(kimi, "data_root", lambda: root)
    monkeypatch.setattr(runtime, "safe_cli_environment", lambda _provider: {"KIMI_CODE_HOME": str(home.resolve())})
    return home, token, config


def output(content=CONTENT, **extra):
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in [
        {"role": "meta", "type": "system.version", "version": "2.1.1"},
        {"role": "assistant", "content": content, **extra},
        {"role": "meta", "type": "session.resume_hint", "session_id": "old-session", "command": "kimi -r old-session"},
    ])


def messages():
    return [{"role": "system", "content": "严格翻译，只输出当前 id 和 translation。"},
            {"role": "user", "content": '{"current_text":"こんにちは、世界。","id":"request-1"}'}]


def test_probe_reports_only_safe_owned_account_and_configured_model_fields(owned):
    home, token, _config = owned
    before = {path: path.read_bytes() for path in home.rglob("*") if path.is_file()}
    status = kimi.probe_account()
    assert status["status"] == "ready" and status["auth_mode"] == "oauth"
    assert status["configured_model"] == MODEL
    assert status["models"] == [{"id": MODEL, "name": MODEL}]
    assert status["capabilities"] == {"protocol": "kimi_stream_json", "account_auth": True, "tool_free": True}
    assert token["access_token"] not in repr(status) and token["refresh_token"] not in repr(status)
    assert before == {path: path.read_bytes() for path in home.rglob("*") if path.is_file()}


def test_probe_never_reads_an_os_home_profile_or_follows_environment_override(owned, monkeypatch, tmp_path):
    external = tmp_path / "other-profile"
    external.mkdir()
    monkeypatch.setattr(runtime, "safe_cli_environment", lambda _: {"KIMI_CODE_HOME": str(external)})
    assert kimi.probe_account()["status"] == "auth_unverified"
    assert not list(external.iterdir())


def test_probe_without_owned_login_returns_not_logged_in_without_creating_files(owned):
    home, _token, _config = owned
    (home / "credentials" / "kimi-code.json").unlink()
    assert kimi.probe_account()["status"] == "not_logged_in"
    assert not (home / "credentials" / "kimi-code.json").exists()


@pytest.mark.parametrize("addition", ['api_key = "synthetic-api-key"', 'api_key_env = "OTHER_MODEL_KEY"'])
def test_probe_rejects_api_key_config_without_returning_secret_values(owned, addition):
    home, _token, config = owned
    config = config.replace('api_key = ""', addition)
    (home / "config.toml").write_text(config, encoding="utf-8")
    status = kimi.probe_account()
    assert status["status"] == "api_key_blocked" and status["auth_mode"] == "api_key"
    assert "synthetic-api-key" not in repr(status) and "OTHER_MODEL_KEY" not in repr(status)


@pytest.mark.parametrize("replacement", [
    ('https://api.kimi.com/coding/v1', 'https://example.invalid/v1'),
    ('key = "oauth/kimi-code"', 'key = "different-credential-slot"'),
    ('provider = "managed:kimi-code"', 'provider = "openai"'),
])
def test_probe_rejects_custom_endpoint_credential_slot_or_non_account_models(owned, replacement):
    home, _token, config = owned
    (home / "config.toml").write_text(config.replace(*replacement), encoding="utf-8")
    assert kimi.probe_account()["status"] == "auth_unverified"


@pytest.mark.parametrize("name", ["mcp.json", "hooks.json", "plugins.json"])
def test_probe_does_not_launch_with_additional_execution_configuration(owned, name):
    home, _token, _config = owned
    (home / name).write_text("{}", encoding="utf-8")
    assert kimi.probe_account()["status"] == "unsupported"


def test_expired_oauth_without_refresh_is_not_logged_in_but_cli_can_refresh_saved_account(owned):
    home, token, _config = owned
    token["expires_at"] = 1
    path = home / "credentials" / "kimi-code.json"
    token["refresh_token"] = ""
    path.write_text(json.dumps(token), encoding="utf-8")
    assert kimi.probe_account()["status"] == "not_logged_in"
    token["refresh_token"] = "synthetic-refresh-value"
    path.write_text(json.dumps(token), encoding="utf-8")
    assert kimi.probe_account()["status"] == "ready"


@pytest.mark.parametrize("expiry", [float("nan"), float("inf"), -1, True])
def test_probe_does_not_treat_corrupt_expiry_as_confirmed_oauth(owned, expiry):
    home, token, _config = owned
    token["expires_at"] = expiry
    (home / "credentials" / "kimi-code.json").write_text(json.dumps(token), encoding="utf-8")
    assert kimi.probe_account()["status"] == "auth_unverified"


def test_probe_stops_before_any_profile_read_when_canceled(owned, monkeypatch):
    event = threading.Event()
    event.set()
    environment = Mock(side_effect=AssertionError("should not look up profile"))
    monkeypatch.setattr(runtime, "safe_cli_environment", environment)
    assert kimi.probe_account(cancellation_event=event)["status"] == "cancelled"
    environment.assert_not_called()


def test_print_json_ignores_only_official_metadata_and_does_not_invent_actual_model():
    assert kimi._decode_output(output()) == (CONTENT, "")
    assert kimi._decode_output(output(model="kimi-native-model")) == (CONTENT, "kimi-native-model")


@pytest.mark.parametrize("record", [
    {"role": "tool", "content": "ran something"},
    {"role": "assistant", "content": CONTENT, "tool_calls": [{"function": {"name": "Bash"}}]},
    {"role": "assistant", "content": CONTENT, "function_call": {"name": "Bash"}},
])
def test_tool_records_and_tool_requests_cannot_become_a_translation(record):
    with pytest.raises(DomesticCliError) as error:
        kimi._decode_output(json.dumps(record))
    assert error.value.code == "tool_request"


@pytest.mark.parametrize("value", [
    "not JSON", "{}", json.dumps({"role": "user", "content": CONTENT}),
    json.dumps({"role": "meta", "type": "system.version"}),
    output() + "\n" + json.dumps({"role": "assistant", "content": "second reply"}),
    json.dumps({"role": "meta", "type": "unknown.event"}) + "\n" + output(),
])
def test_partial_unknown_or_multiple_final_messages_are_rejected(value):
    with pytest.raises(DomesticCliError) as error:
        kimi._decode_output(value)
    assert error.value.code == "invalid_response"


def test_invoke_uses_fixed_bootstrap_empty_tools_agent_and_only_pipe_for_source(owned):
    home, _token, _config = owned
    seen = []
    def request(argv, payload, environment, *, cancellation_event=None):
        seen.append((argv, payload, environment, cancellation_event))
        agent = Path(argv[-3])
        skills = Path(argv[-2])
        markdown = agent.read_text(encoding="utf-8")
        assert "tools: []\nsubagents: []" in markdown
        assert "- select_tools" in markdown
        assert messages()[0]["content"] in markdown
        assert "こんにちは" not in markdown
        assert skills.is_dir() and not list(skills.iterdir())
        assert "こんにちは" not in " ".join(argv)
        assert payload == messages()[1]["content"]
        assert environment["KIMI_CODE_HOME"] == str(home.resolve())
        assert environment["KIMI_LOG_LEVEL"] == "off"
        assert environment["KIMI_CODE_NO_AUTO_UPDATE"] == "1"
        assert environment["KIMI_DISABLE_CRON"] == "1"
        assert "process.stdin.setEncoding('utf8')" in argv[3]
        assert "process.argv=" in argv[3] and "await import" in argv[3]
        return output()
    event = threading.Event()
    result = kimi.invoke_kimi(SimpleTransport(request), ["node.exe", "official/main.mjs"],
                             {"KIMI_CODE_HOME": str(home.resolve())}, messages(), "", event)
    assert result == (CONTENT, "") and len(seen) == 1 and seen[0][3] is event
    assert seen[0][0][-1] == MODEL
    assert not Path(seen[0][0][-3]).exists()


class SimpleTransport:
    def __init__(self, request):
        self.request = request


def test_requested_model_is_a_verified_account_alias_and_never_an_api_key_provider(owned):
    home, _token, _config = owned
    transport = Mock()
    with pytest.raises(DomesticCliError) as error:
        kimi.invoke_kimi(transport, ["node", "main.mjs"], {"KIMI_CODE_HOME": str(home.resolve())},
                         messages(), "openai/gpt-anything")
    assert error.value.code == "model_mismatch"
    transport.request.assert_not_called()


@pytest.mark.parametrize("argv", [["kimi.exe"], ["cmd.exe", "main.mjs"], ["node.exe", "different-script.js"]])
def test_invoke_does_not_fall_back_to_shell_or_unsupported_binary(owned, argv):
    transport = Mock()
    with pytest.raises(DomesticCliError) as error:
        kimi.invoke_kimi(transport, argv, {}, messages(), "")
    assert error.value.code == "unsupported"
    transport.request.assert_not_called()


def test_cancel_during_invocation_discards_even_valid_final_json(owned):
    home, _token, _config = owned
    event = threading.Event()
    def request(*_args, **_options):
        event.set()
        return output()
    with pytest.raises(DomesticCliError) as error:
        kimi.invoke_kimi(SimpleTransport(request), ["node", "main.mjs"],
                         {"KIMI_CODE_HOME": str(home.resolve())}, messages(), "", event)
    assert error.value.code == "cancelled"


def test_fixed_node_bootstrap_transfers_multibyte_stdin_to_official_cli_argument_shape(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable; this fixture never runs an actual vendor CLI")
    entry = tmp_path / "main.mjs"
    entry.write_text("process.stdout.write(JSON.stringify({flags:process.argv.slice(2,-1),"
                     "prompt:process.argv.at(-1)}));", encoding="utf-8")
    source = "主播正在讲话 こんにちは 世界 " * 1800
    result = subprocess.run([node, "--input-type=module", "-e", kimi._BOOTSTRAP,
                             str(entry), "owned-agent.md", "empty-skills", MODEL],
                            input=source.encode("utf-8"), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=tmp_path, timeout=10, check=True)
    data = json.loads(result.stdout)
    assert data["prompt"] == source
    assert data["flags"] == ["--agent-file", "owned-agent.md", "--skills-dir", "empty-skills",
                             "--output-format", "stream-json", "-m", MODEL, "-p"]
