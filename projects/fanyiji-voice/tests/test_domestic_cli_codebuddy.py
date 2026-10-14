"""CodeBuddy adapter tests: public protocol shapes and synthetic account markers."""
import json
import threading
from unittest.mock import Mock

import pytest

from fanyiji import domestic_cli_codebuddy as adapter
from fanyiji.domestic_cli_engine import DomesticCliError


def output(**changes):
    return {"type": "result", "subtype": "success", "is_error": False,
            "result": '{"id":"test-id","translation":"测试译文"}', **changes}


@pytest.fixture
def account_marker(tmp_path, monkeypatch):
    for key in adapter._KEY_OVERRIDES | adapter._OPAQUE_OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "auth/Tencent-Cloud.coding-copilot.info"
    path.parent.mkdir()
    monkeypatch.setattr(adapter, "_account_marker_path", lambda: path)
    monkeypatch.setattr(adapter.time, "time", lambda: 100)
    monkeypatch.setattr("fanyiji.domestic_cli_runtime.safe_cli_environment", lambda provider: {})
    return path


def write_marker(path, **changes):
    data = {"auth": {"tokenType": "Bearer", "refreshExpiresAt": 200_000,
                     "accessToken": "PRIVATE-ACCESS", "refreshToken": "PRIVATE-REFRESH"},
            "account": {"uid": "PRIVATE-UID", "nickname": "PRIVATE-NAME"}, **changes}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_probe_account_reads_only_markers_and_never_returns_credentials(account_marker):
    write_marker(account_marker)
    result = adapter.probe_account()
    assert result["status"] == "ready" and result["auth_mode"] == "oauth"
    assert result["capabilities"] == {"protocol": "codebuddy_json", "account_auth": True, "tool_free": True}
    assert "PRIVATE" not in json.dumps(result)


def test_missing_and_logged_out_account_are_not_authenticated(account_marker):
    assert adapter.probe_account()["status"] == "not_logged_in"
    write_marker(account_marker)
    account_marker.with_name(account_marker.name + ".logged-out").write_text("logout", encoding="utf-8")
    assert adapter.probe_account()["status"] == "not_logged_in"


@pytest.mark.parametrize("auth,status", [
    ({"tokenType": "ApiKey", "refreshExpiresAt": 200_000}, "api_key_blocked"),
    ({"tokenType": "Bearer", "refreshExpiresAt": 1}, "auth_unverified"),
    ({"tokenType": "Bearer", "refreshExpiresAt": True}, "auth_unverified"),
    ({"tokenType": "Bearer"}, "auth_unverified"),
    ({"tokenType": "Unknown", "refreshExpiresAt": 200_000}, "auth_unverified"),
])
def test_bearer_alone_and_api_key_never_count_as_official_account(account_marker, auth, status):
    write_marker(account_marker, auth=auth)
    assert adapter.probe_account()["status"] == status


@pytest.mark.parametrize("key", sorted(adapter._KEY_OVERRIDES | adapter._OPAQUE_OVERRIDES))
def test_custom_env_overrides_are_rejected_without_exposing_values(account_marker, monkeypatch, key):
    write_marker(account_marker)
    monkeypatch.setattr("fanyiji.domestic_cli_runtime.safe_cli_environment", lambda provider: {key: "PRIVATE-OVERRIDE"})
    result = adapter.probe_account()
    assert result["status"] == ("api_key_blocked" if key in adapter._KEY_OVERRIDES else "auth_unverified")
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("key", sorted(adapter._KEY_OVERRIDES | adapter._OPAQUE_OVERRIDES))
def test_unrelated_parent_keys_do_not_block_account_when_not_passed_to_cli(account_marker, monkeypatch, key):
    write_marker(account_marker)
    monkeypatch.setenv(key, "PRIVATE-PARENT")
    assert adapter.probe_account()["status"] == "ready"


def test_adapter_probe_merges_into_runtime_without_duplicate_provider_keyword(account_marker):
    from fanyiji.domestic_cli_runtime import _result
    write_marker(account_marker)
    value = adapter.probe_account()
    result = _result("codebuddy", value.pop("status"), value.pop("message"), installed=True, version="2.164.0", **value)
    assert result["provider"] == "codebuddy" and result["status"] == "ready"


def test_account_markers_without_credential_fields_do_not_claim_authenticated(account_marker):
    write_marker(account_marker, auth={"tokenType": "Bearer", "refreshExpiresAt": 200_000})
    assert adapter.probe_account()["status"] == "auth_unverified"


@pytest.mark.parametrize("data", ["bad json", "[]", '{"protected":"opaque"}', '{"auth":{},"account":[]}'])
def test_unknown_or_encrypted_whole_record_does_not_assume_logged_in(account_marker, data):
    account_marker.write_text(data, encoding="utf-8")
    assert adapter.probe_account()["status"] == "auth_unverified"


def test_cancelled_probe_does_not_open_marker(account_marker, monkeypatch):
    cancelled = threading.Event()
    cancelled.set()
    monkeypatch.setattr(adapter, "_account_marker_path", Mock(side_effect=AssertionError("must not read")))
    assert adapter.probe_account(cancellation_event=cancelled)["status"] == "cancelled"


def test_account_file_path_uses_public_product_id_and_shared_data_not_a_guessed_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter.Path, "home", lambda: tmp_path)
    path = adapter._account_marker_path()
    assert path.name == "Tencent-Cloud.coding-copilot.info"
    assert path.parts[-4:-1] == ("Data", "Public", "auth")


def test_invocation_sends_all_source_and_system_only_over_stdin_and_disables_tools():
    transport = Mock()
    transport.request.return_value = json.dumps(output(modelUsage={"actual-model": {"outputTokens": 12}}))
    messages = [{"role": "system", "content": "STATIC RULES"}, {"role": "user", "content": "PRIVATE-SOURCE"}]
    event = threading.Event()
    account_probe = Mock(return_value=adapter._probe_result("ready", "test account", account_auth=True))
    content, model = adapter.invoke_codebuddy(transport, ["node", "codebuddy.js"], {}, messages, "chosen-model", event,
                                             account_probe=account_probe)
    args, payload, environment = transport.request.call_args.args
    assert args == ["node", "codebuddy.js", "--tools", "", "--strict-mcp-config", "--setting-sources", "none",
                    "--no-session-persistence", "--output-format", "json", "-p", "--model", "chosen-model"]
    assert "PRIVATE-SOURCE" not in str(args) and "STATIC RULES" not in str(args)
    assert payload == "STATIC RULES\n\n引用输入 JSON：\nPRIVATE-SOURCE" and environment == {}
    assert transport.request.call_args.kwargs["cancellation_event"] is event
    assert json.loads(content)["translation"] == "测试译文" and model == "actual-model"


@pytest.mark.parametrize("usage", [None, {}, {"auto": {"outputTokens": 12}},
    {"one": {"outputTokens": 12}, "two": {"outputTokens": 5}}, {"one": {"outputTokens": 0}},
    {"one": {"outputTokens": True}}])
def test_unconfirmed_actual_model_remains_empty(usage):
    content, model = adapter._parse_result(json.dumps(output(modelUsage=usage)))
    assert content and model == ""


@pytest.mark.parametrize("value,code", [
    ([], "invalid_response"),
    (output(is_error=True), "process_failed"),
    (output(subtype="error_during_execution"), "process_failed"),
    (output(result=""), "invalid_response"),
    (output(result={"translation": "test"}), "invalid_response"),
    (output(tool_calls=[{"name": "Bash"}]), "tool_request"),
    (output(message={"content": [{"type": "tool_use", "name": "Read"}]}), "tool_request"),
])
def test_invalid_results_and_tool_requests_are_rejected_without_private_output(value, code):
    with pytest.raises(DomesticCliError) as failure:
        adapter._parse_result(json.dumps(value))
    assert failure.value.code == code


def test_malformed_cli_output_is_not_reported_as_translation():
    with pytest.raises(DomesticCliError) as failure:
        adapter._parse_result("private logs\n{}");
    assert failure.value.code == "invalid_response"
    assert "private logs" not in str(failure.value)


@pytest.mark.parametrize("model", [None, "bad model", "--flag", "x" * 129])
def test_invalid_model_does_not_launch_process(model):
    transport = Mock()
    with pytest.raises(DomesticCliError):
        adapter.invoke_codebuddy(transport, ["node", "script"], {}, [{"role": "user", "content": "test"}], model)
    transport.request.assert_not_called()


@pytest.mark.parametrize("status", ["api_key_blocked", "not_logged_in", "auth_unverified"])
def test_cached_ready_cannot_submit_new_source_after_account_route_changes(status):
    transport = Mock()
    transport.request.return_value = json.dumps(output())
    account_probe = Mock(side_effect=[adapter._probe_result("ready", "test account", account_auth=True),
                                      adapter._probe_result(status, "changed")])
    args = (transport, ["node", "entry"], {}, [{"role": "system", "content": "static"},
                                               {"role": "user", "content": "private source"}], "")
    adapter.invoke_codebuddy(*args, account_probe=account_probe)
    with pytest.raises(DomesticCliError) as failure:
        adapter.invoke_codebuddy(*args, account_probe=account_probe)
    assert failure.value.code == status
    assert transport.request.call_count == 1 and account_probe.call_count == 2


def test_cancel_after_account_check_still_prevents_transport_request():
    cancellation = threading.Event()
    def account_probe(**kwargs):
        cancellation.set()
        return adapter._probe_result("ready", "test account", account_auth=True)
    transport = Mock()
    with pytest.raises(DomesticCliError) as failure:
        adapter.invoke_codebuddy(transport, ["node", "entry"], {}, [{"role": "user", "content": "private"}], "",
                                 cancellation, account_probe=account_probe)
    assert failure.value.code == "cancelled"
    transport.request.assert_not_called()
