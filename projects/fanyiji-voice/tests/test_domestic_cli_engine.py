import io
import json
from pathlib import Path
import threading
from types import ModuleType
import sys

import pytest

from fanyiji.domestic_cli_engine import DomesticCliError, DomesticCliTranslationEngine, _OwnedCliProcessTransport


def ready(provider, *, cancellation_event=None):
    return {"provider": provider, "status": "ready", "installed": True, "version": "1.0.0",
            "auth_mode": "oauth", "models": [{"id": "MiniMax-M3", "name": "MiniMax M3"}],
            "capabilities": {"account_auth": True, "tool_free": True,
                             "protocol": {"minimax": "mmx_json", "minimax_code": "mcode_json",
                                          "kimi": "kimi_stream_json", "codebuddy": "codebuddy_json"}.get(provider)}}


class FakeTransport:
    def __init__(self, variant=None):
        self.calls = []
        self.variant = variant
        self.cancelled = self.closed = 0

    def request(self, argv, payload, environment, *, cancellation_event=None):
        self.calls.append((argv, payload, environment, cancellation_event))
        if "text" in argv and "chat" in argv:
            source = json.loads(json.loads(payload)[1]["content"])
            response = {"role": "assistant", "model": "MiniMax-M3", "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": json.dumps({"id": source["id"], "translation": "你好"})}]}
        else:
            source = json.loads(payload.split("引用输入 JSON：\n", 1)[1])
            response = {"schemaVersion": 1, "type": "exec.result", "status": "succeeded",
                        "output": {"id": source["id"], "translation": "你好"},
                        "model": {"providerId": "minimax", "modelId": "MiniMax-M3"}}
        if self.variant is not None:
            response = self.variant(response)
        return json.dumps(response, ensure_ascii=False)

    def cancel_pending(self):
        self.cancelled += 1

    def close(self):
        self.closed += 1


def engine(provider="minimax", **options):
    return DomesticCliTranslationEngine(provider, command_builder=lambda _p: ["node.exe", "official.js"],
                                        environment_factory=lambda _p: {"PATH": "safe"}, probe=ready,
                                        route_checker=options.pop("route_checker", lambda *_a, **_kw: None), **options)


def test_native_mmx_stdin_contract_model_confirmation_and_context_budget():
    transport = FakeTransport()
    translator = engine(transport=transport)
    translator.preload()
    assert translator.backend_info["model_confirmed"] is False
    assert translator.translate("ナイス", "ja", "zh", context="旧原文" * 200) == "你好"
    argv, payload, environment, _cancel = transport.calls[0]
    assert argv[:8] == ["node.exe", "official.js", "text", "chat", "--messages-file", "-", "--output", "json"]
    assert "--tool" not in argv
    assert "ナイス" not in str(argv)
    messages = json.loads(payload)
    data = json.loads(messages[1]["content"])
    assert len(data["reference_context"]) <= 384
    assert data["current_text"] == "ナイス"
    assert data["term_hints"]
    assert environment == {"PATH": "safe"}
    assert translator.backend_info["model_id"] == "MiniMax-M3"
    assert translator.backend_info["model_confirmed"] is True
    assert set(translator.last_request_metrics) == {"translation_seconds"}


def test_mcode_uses_lightweight_native_oauth_model_and_schema():
    transport = FakeTransport()
    translator = engine("minimax_code", transport=transport)
    assert translator.translate("こんにちは", "ja", "zh") == "你好"
    argv, payload, _env, _cancel = transport.calls[0]
    assert argv[2:7] == ["exec", "--mode", "lightweight", "--input", "-"]
    assert argv[argv.index("--model") + 1] == "minimax/MiniMax-M3"
    assert argv[argv.index("--permission") + 1] == "off"
    assert "--continue" not in argv and "--session" not in argv
    assert "こんにちは" not in str(argv)
    assert "こんにちは" in payload
    schema = json.loads(argv[argv.index("--output-schema") + 1])
    assert schema["additionalProperties"] is False
    assert translator.backend_info["model_id"] == "minimax/MiniMax-M3"
    assert translator.backend_info["configured_model"] == "minimax/MiniMax-M3"


@pytest.mark.parametrize("provider", ["minimax", "minimax_code"])
def test_requested_model_mismatch_never_returns_valid_translation(provider):
    translator = engine(provider, model="Other-Model", transport=FakeTransport())
    with pytest.raises(DomesticCliError) as error:
        translator.translate("こんにちは", "ja", "zh")
    assert error.value.code == "model_mismatch"
    assert translator.backend_info["model_confirmed"] is False


@pytest.mark.parametrize("status", ["not_installed", "not_logged_in", "api_key_blocked", "auth_unverified", "unsupported"])
def test_probe_gates_account_and_protocol_before_any_source_submission(status):
    transport = FakeTransport()
    translator = DomesticCliTranslationEngine(probe=lambda _p, **_kw: {"status": status}, transport=transport)
    with pytest.raises(DomesticCliError) as error:
        translator.translate("こんにちは", "ja", "zh")
    assert error.value.code == status
    assert not transport.calls


def test_ready_probe_still_cannot_enable_api_key_auth_or_tools():
    for change, expected in (({"auth_mode": "api_key"}, "api_key_blocked"),
                              ({"capabilities": {"protocol": "mmx_json", "account_auth": True, "tool_free": False}}, "tool_request")):
        translator = DomesticCliTranslationEngine(probe=lambda p, **_kw: {**ready(p), **change}, transport=FakeTransport())
        with pytest.raises(DomesticCliError) as error:
            translator.preload()
        assert error.value.code == expected


def test_blocked_qwen_never_probes_or_launches():
    translator = DomesticCliTranslationEngine("qwen", probe=lambda *_a, **_kw: pytest.fail("blocked"))
    with pytest.raises(DomesticCliError) as error:
        translator.preload()
    assert error.value.code == "unsupported"


@pytest.mark.parametrize("variant,code", [
    (lambda r: {**r, "model": None}, "model_unconfirmed"),
    (lambda r: {**r, "stop_reason": "max_tokens"}, "invalid_response"),
    (lambda r: {**r, "stop_reason": "tool_use"}, "tool_request"),
    (lambda r: {**r, "content": [{"type": "tool_use", "name": "shell"}]}, "tool_request"),
    (lambda r: {**r, "content": [{"type": "text", "text": '{"id":"wrong","translation":"你好"}'}]}, "invalid_response"),
])
def test_malformed_native_response_does_not_escape_guard(variant, code):
    translator = engine(transport=FakeTransport(variant))
    with pytest.raises(DomesticCliError) as error:
        translator.translate("こんにちは", "ja", "zh")
    assert error.value.code == code


def test_no_source_file_created_or_process_called_for_empty_text():
    transport = FakeTransport()
    translator = engine(transport=transport)
    assert translator.translate(" ", "ja", "zh") == ""
    assert translator.translate("你好", "zh", "zh") == "你好"
    assert not transport.calls


def test_cancel_after_engine_check_prevents_stale_native_request(monkeypatch):
    transport = FakeTransport()
    translator = engine(transport=transport)
    original = translator._check
    def cancel(generation, cancellation=None):
        original(generation, cancellation)
        translator.cancel_pending()
    monkeypatch.setattr(translator, "_check", cancel)
    with pytest.raises(DomesticCliError) as error:
        translator.translate("こんにちは", "ja", "zh")
    assert error.value.code == "cancelled"
    assert not transport.calls


def test_late_success_after_cancel_is_not_published():
    transport = FakeTransport()
    translator = engine(transport=transport)
    request = transport.request
    def late(*args, **kwargs):
        value = request(*args, **kwargs)
        translator.cancel_pending()
        return value
    transport.request = late
    with pytest.raises(DomesticCliError) as error:
        translator.translate("こんにちは", "ja", "zh")
    assert error.value.code == "cancelled"
    assert not translator.backend_info["model_confirmed"]


def test_close_is_idempotent_and_never_restarts():
    transport = FakeTransport()
    translator = engine(transport=transport)
    translator.close()
    translator.close()
    assert transport.closed == 1
    with pytest.raises(DomesticCliError) as error:
        translator.preload()
    assert error.value.code == "closed"


@pytest.mark.parametrize("provider,function", [("kimi", "invoke_kimi"), ("codebuddy", "invoke_codebuddy")])
def test_account_adapters_can_return_valid_translation_without_claiming_unreported_model(monkeypatch, provider, function):
    received = []
    module = ModuleType("fanyiji.domestic_cli_" + provider)
    def invoke(transport, argv, environment, messages, model, *, cancellation_event=None):
        source = json.loads(messages[1]["content"])
        received.append((argv, environment, model, cancellation_event))
        return json.dumps({"id": source["id"], "translation": "你好"}), ""
    setattr(module, function, invoke)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    translator = engine(provider, model="selected-model", transport=FakeTransport())
    assert translator.translate("こんにちは", "ja", "zh") == "你好"
    assert translator.backend_info["requested_model"] == "selected-model"
    assert translator.backend_info["model_id"] == ""
    assert translator.backend_info["model_name"] == "未确认"
    assert translator.backend_info["model_confirmed"] is False
    assert received[0][0] == ["node.exe", "official.js"]


def test_mcode_never_selects_custom_provider_or_missing_official_catalog():
    translator = engine("minimax_code", model="custom_provider:legacy/M3", transport=FakeTransport())
    with pytest.raises(DomesticCliError) as error:
        translator.translate("こんにちは", "ja", "zh")
    assert error.value.code == "api_key_blocked"
    translator = DomesticCliTranslationEngine("minimax_code", probe=lambda p, **_kw: {**ready(p), "models": []},
                                              transport=FakeTransport())
    with pytest.raises(DomesticCliError) as error:
        translator.preload()
    assert error.value.code == "model_unconfirmed"


@pytest.mark.parametrize("alias", ["default-model", "fast-model", "balanced-model", "primary-model", "deep-model"])
def test_codebuddy_official_alias_uses_actual_model_without_false_mismatch(monkeypatch, alias):
    module = ModuleType("fanyiji.domestic_cli_codebuddy")
    def invoke(_transport, _argv, _environment, messages, _model, *, cancellation_event=None):
        source = json.loads(messages[1]["content"])
        return json.dumps({"id": source["id"], "translation": "你好"}), "Actual-Native-Model"
    module.invoke_codebuddy = invoke
    monkeypatch.setitem(sys.modules, module.__name__, module)
    translator = engine("codebuddy", model=alias, transport=FakeTransport())
    assert translator.translate("こんにちは", "ja", "zh") == "你好"
    assert translator.backend_info["requested_model"] == alias
    assert translator.backend_info["model_id"] == "Actual-Native-Model"


def test_cached_preload_never_skips_account_route_check_after_oauth_is_removed():
    transport = FakeTransport()
    states = iter([{"status": "ready"}, {"status": "api_key_blocked"}])
    translator = engine(transport=transport, route_checker=lambda *_a, **_kw: next(states))
    translator.preload()
    assert translator.translate("こんにちは", "ja", "zh") == "你好"
    with pytest.raises(DomesticCliError) as error:
        translator.translate("こんばんは", "ja", "zh")
    assert error.value.code == "api_key_blocked"
    assert len(transport.calls) == 1


class FakeProcess:
    def __init__(self, output=b"{}", returncode=0, blocked=False):
        self.stdin, self.stdout = io.BytesIO(), io.BytesIO(output)
        self.returncode = None if blocked else returncode
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed, self.returncode = True, -1

    def wait(self, **_kwargs):
        return self.returncode


def test_owned_runner_has_empty_workspace_shell_off_and_no_stderr(tmp_path):
    launches = []
    process = FakeProcess(b'{"result":true}')
    def factory(argv, **options):
        launches.append((argv, options))
        assert Path(options["cwd"]).is_dir()
        assert not list(Path(options["cwd"]).iterdir())
        return process
    transport = _OwnedCliProcessTransport(process_factory=factory)
    assert transport.request(["node.exe", "official.js"], "私密原文", {"PATH": "safe"}) == '{"result":true}'
    assert launches[0][1]["shell"] is False
    assert launches[0][1]["stderr"] == -3
    assert not Path(launches[0][1]["cwd"]).exists()
    transport.close()


def test_owned_runner_cancellation_kills_only_owned_process():
    process = FakeProcess(blocked=True)
    transport = _OwnedCliProcessTransport(process_factory=lambda *_a, **_kw: process)
    cancel = threading.Event()
    timer = threading.Timer(0.05, cancel.set)
    timer.start()
    try:
        with pytest.raises(DomesticCliError) as error:
            transport.request(["node.exe", "official.js"], "source", {}, cancellation_event=cancel)
        assert error.value.code == "cancelled"
        assert process.killed
    finally:
        timer.cancel()
        transport.close()


def test_owned_runner_timeout_and_limit_are_bounded_and_redacted():
    process = FakeProcess(blocked=True)
    transport = _OwnedCliProcessTransport(process_factory=lambda *_a, **_kw: process, timeout=0.03)
    with pytest.raises(DomesticCliError) as error:
        transport.request(["node.exe", "official.js"], "source", {})
    assert error.value.code == "timeout"
    assert process.killed
    excessive = FakeProcess(b"private" * 50000)
    transport = _OwnedCliProcessTransport(process_factory=lambda *_a, **_kw: excessive)
    with pytest.raises(DomesticCliError) as error:
        transport.request(["node.exe", "official.js"], "source", {})
    assert error.value.code == "invalid_response"
    assert "private" not in str(error.value)
