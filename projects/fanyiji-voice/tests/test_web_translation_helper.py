import io
import json
import threading
from types import SimpleNamespace

import pytest

from fanyiji import web_translation_helper as helper
from fanyiji.web_translation_protocol import provider_url


def command():
    return {"op": "translate", "id": "1" * 32, "text": "こんにちは", "source_lang": "ja", "target_lang": "zh"}


def host(source="", provider="google_web"):
    calls = []
    core = SimpleNamespace(Environment=SimpleNamespace(BrowserVersionString="146.0.3856.84"),
                           Navigate=lambda url: calls.append(("navigate", url)))
    native = SimpleNamespace(IsDisposed=False, webview=SimpleNamespace(CoreWebView2=core),
                             Close=lambda: calls.append(("close",)), Show=lambda: calls.append(("show",)),
                             Activate=lambda: calls.append(("activate",)))
    native.BeginInvoke = lambda action: action()
    window = SimpleNamespace(native=native)
    sink = io.StringIO()
    instance = helper._Host(window, io.StringIO(source), sink, lambda fn: fn,
                            SimpleNamespace(GetAvailableBrowserVersionString=lambda: "146.0.3856.84"), provider)
    instance.native_ready.set()
    instance.initialized = True
    return instance, sink, calls


def test_show_opens_only_fixed_provider_page_and_does_not_touch_live_browser():
    instance, sink, calls = host(provider="youdao_web")
    instance.execute({"op": "show"})
    assert calls == [("navigate", provider_url("youdao_web")), ("show",), ("activate",)]
    assert not sink.getvalue()


def test_closed_host_does_not_reopen_on_late_show_or_translate():
    instance, sink, calls = host()
    instance.execute({"op": "close"})
    instance.execute({"op": "show"})
    instance.execute(command())
    assert calls == [("close",)]
    assert not sink.getvalue()


def test_reader_rejects_script_url_cookie_fields_without_echoing():
    instance, sink, calls = host('{"op":"eval","script":"secret-cookie"}\n{"op":"show"}\n')
    instance.read_commands()
    assert "secret-cookie" not in sink.getvalue()
    assert json.loads(sink.getvalue()) == {"kind": "error", "code": "invalid_command"}
    assert calls[-1] == ("close",)


def test_navigation_allowlist_blocks_other_sites():
    instance, sink, calls = host()
    args = SimpleNamespace(Uri="https://evil.example/?cookie=secret", Cancel=False)
    instance.navigation_starting(None, args)
    assert args.Cancel is True
    assert json.loads(sink.getvalue()) == {"kind": "error", "code": "navigation_blocked"}
    assert "secret" not in sink.getvalue()
    assert calls == [("show",), ("activate",)]


def test_dom_state_pending_then_matching_success(monkeypatch):
    instance, sink, calls = host()
    item = command()
    instance._translation_url = provider_url("google_web", "ja", "zh")
    instance.page_ready.set()
    monkeypatch.setattr(helper, "prepare_script", lambda *_a: "prepare")
    monkeypatch.setattr(helper, "poll_script", lambda *_a: "poll")
    values = iter([
        {"id": item["id"], "status": "pending"}, {"id": item["id"], "status": "pending"},
        {"id": item["id"], "status": "pending"}, {"id": item["id"], "status": "ok", "text": "你好"},
    ])
    scripts = []
    def evaluate(script):
        scripts.append(script)
        return next(values)
    monkeypatch.setattr(instance, "evaluate", evaluate)
    instance.process_translation(item)
    assert scripts == ["prepare", "poll", "prepare", "poll"]
    assert json.loads(sink.getvalue()) == {
        "kind": "result", "id": item["id"], "result": {"id": item["id"], "status": "ok", "text": "你好"}}
    assert calls == [("show",)]


def test_interaction_requirement_exposes_window_and_returns_fixed_error(monkeypatch):
    instance, sink, calls = host()
    item = command()
    instance._translation_url = provider_url("google_web", "ja", "zh")
    instance.page_ready.set()
    monkeypatch.setattr(instance, "evaluate", lambda *_a: {
        "id": item["id"], "status": "error", "code": "interaction_required"})
    instance.process_translation(item)
    assert json.loads(sink.getvalue())["result"]["code"] == "interaction_required"
    assert calls == [("show",), ("show",), ("activate",)]


def test_mismatched_result_id_never_becomes_translation(monkeypatch):
    instance, sink, calls = host()
    item = command()
    instance._translation_url = provider_url("google_web", "ja", "zh")
    instance.page_ready.set()
    monkeypatch.setattr(instance, "evaluate", lambda *_a: {
        "id": "2" * 32, "status": "ok", "text": "旧译文"})
    instance.process_translation(item)
    value = json.loads(sink.getvalue())
    assert value["result"]["status"] == "error"
    assert "旧译文" not in sink.getvalue()


def test_failed_page_load_is_not_treated_as_a_translation(monkeypatch):
    instance, sink, calls = host()
    item = command()
    instance._translation_url = provider_url("google_web", "ja", "zh")
    instance._navigation_failed = True
    instance.page_ready.set()
    navigate = instance.navigate
    def fail_again(url):
        navigate(url)
        instance._navigation_failed = True
        instance.page_ready.set()
        return True
    monkeypatch.setattr(instance, "navigate", fail_again)
    monkeypatch.setattr(instance, "evaluate", lambda *_a: pytest.fail("must not read a failed document"))
    instance.process_translation(item)
    assert json.loads(sink.getvalue())["result"]["code"] == "page_not_ready"
    assert calls == [("show",), ("navigate", provider_url("google_web", "ja", "zh")), ("show",), ("activate",)]


def test_manual_evaluation_exception_does_not_leave_diagnostic_text():
    instance, sink, _calls = host()
    def fail(_script):
        raise RuntimeError("credentials=secret")
    instance.window.evaluate_js = fail
    assert instance.evaluate("fixed local test") is None
    assert not sink.getvalue()


def test_runtime_overrides_stop_before_profile_creation(tmp_path, monkeypatch):
    sink = io.StringIO()
    monkeypatch.setattr(helper.sys, "stdin", io.StringIO())
    monkeypatch.setattr(helper.sys, "stdout", sink)
    monkeypatch.setenv("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "--unsafe")
    if helper.os.name != "nt":
        pytest.skip("Windows runtime override check")
    assert helper.main(["--provider", "google_web", "--profile", str(tmp_path / "unused")]) == 1
    assert json.loads(sink.getvalue())["code"] == "runtime_override_blocked"
    assert not (tmp_path / "unused").exists()


def test_google_native_input_uses_only_the_request_text(monkeypatch):
    instance, sink, _calls = host()
    calls = []
    monkeypatch.setattr(instance, "evaluate", lambda _script: True)
    monkeypatch.setattr(instance, "native_call", lambda method, parameters: calls.append((method, parameters)) or True)
    assert instance.native_input(command(), "clear")
    assert len(calls) == 6
    assert all(method == "Input.dispatchKeyEvent" for method, _parameters in calls)
    assert instance.native_input(command(), "insert")
    assert calls[-1] == ("Input.insertText", {"text": command()["text"]})
    assert not sink.getvalue()


def test_native_input_never_targets_youdao_or_wrong_focus(monkeypatch):
    instance, _sink, _calls = host(provider="youdao_web")
    monkeypatch.setattr(instance, "evaluate", lambda _script: pytest.fail("Youdao stays on validated DOM path"))
    assert not instance.native_input(command(), "insert")
    instance.provider = "google_web"
    monkeypatch.setattr(instance, "evaluate", lambda _script: False)
    monkeypatch.setattr(instance, "native_call", lambda *_args: pytest.fail("not focused"))
    assert not instance.native_input(command(), "insert")


def test_native_commands_have_a_fixed_allowlist():
    instance, _sink, _calls = host()
    assert not instance.native_call("Network.getAllCookies", {})
    assert not instance.native_call("Runtime.evaluate", {"expression": "arbitrary"})


def test_prepare_native_action_completes_before_poll(monkeypatch):
    instance, sink, _calls = host()
    item = command()
    instance._translation_url = provider_url("google_web", "ja", "zh")
    instance.page_ready.set()
    monkeypatch.setattr(helper, "prepare_script", lambda *_a: "prepare")
    monkeypatch.setattr(helper, "poll_script", lambda *_a: "poll")
    values = iter([{"id": item["id"], "status": "input", "action": "insert"},
                   {"id": item["id"], "status": "ok", "text": "你好"}])
    calls = []
    def evaluate(script):
        calls.append(script)
        return next(values)
    monkeypatch.setattr(instance, "evaluate", evaluate)
    monkeypatch.setattr(instance, "native_input", lambda command, action: calls.append(action) or True)
    instance.process_translation(item)
    assert calls == ["prepare", "insert", "poll"]
    assert json.loads(sink.getvalue())["result"]["text"] == "你好"


def test_failed_native_action_never_polls_or_returns_old_output(monkeypatch):
    instance, sink, calls = host()
    item = command()
    instance._translation_url = provider_url("google_web", "ja", "zh")
    instance.page_ready.set()
    monkeypatch.setattr(instance, "evaluate", lambda *_a: {"id": item["id"], "status": "input", "action": "clear"})
    monkeypatch.setattr(instance, "native_input", lambda *_a: False)
    monkeypatch.setattr(helper, "poll_script", lambda *_a: pytest.fail("must not poll after native error"))
    instance.process_translation(item)
    assert json.loads(sink.getvalue())["result"]["code"] == "input_unavailable"
    assert calls == [("show",), ("show",), ("activate",)]


def test_show_then_translate_waits_for_same_navigation_without_restart(monkeypatch):
    instance, sink, calls = host()
    item = command()
    url = provider_url("google_web", "ja", "zh")
    instance.execute({"op": "show"})
    entered = threading.Event()
    original_wait = instance.page_ready.wait
    def wait(timeout=None):
        entered.set()
        return original_wait(timeout)
    monkeypatch.setattr(instance.page_ready, "wait", wait)
    values = iter([{"id": item["id"], "status": "pending"},
                   {"id": item["id"], "status": "ok", "text": "你好"}])
    monkeypatch.setattr(instance, "evaluate", lambda *_a: next(values))
    thread = threading.Thread(target=instance.process_translation, args=(item,))
    thread.start()
    try:
        assert entered.wait(1)
        assert [entry for entry in calls if entry[0] == "navigate"] == [("navigate", url)]
        instance.navigation_starting(None, SimpleNamespace(Uri=url, NavigationId=7, Cancel=False))
        instance.navigation_completed(None, SimpleNamespace(IsSuccess=False, NavigationId=6))
        assert not instance.page_ready.is_set()
        instance.navigation_completed(None, SimpleNamespace(IsSuccess=True, NavigationId=7))
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert json.loads(sink.getvalue())["result"]["status"] == "ok"
        instance.navigation_completed(None, SimpleNamespace(IsSuccess=False, NavigationId=6))
        assert not instance._navigation_failed
    finally:
        instance.close()
        thread.join(timeout=1)


def test_completion_before_new_navigation_start_does_not_mark_new_page_ready():
    instance, _sink, _calls = host()
    instance.navigate(provider_url("google_web"))
    instance.navigation_completed(None, SimpleNamespace(IsSuccess=False, NavigationId=5))
    assert not instance.page_ready.is_set()
    assert not instance._navigation_failed


def test_navigation_test_double_without_ids_remains_supported():
    instance, _sink, _calls = host()
    instance.navigate(provider_url("google_web"))
    instance.navigation_completed(None, SimpleNamespace(IsSuccess=True))
    assert instance.page_ready.is_set()
    assert not instance._navigation_failed


def test_diagnostic_text_cannot_replace_template_provider_token():
    item = command()
    item["text"] = "__GOOGLE__"
    script = helper._diagnostic_script(item, "google_web")
    assert '"text": "__GOOGLE__"' in script
    assert "google=true" in script


@pytest.mark.parametrize("provider,expected", [("google_web", [("show",)]), ("youdao_web", [])])
def test_google_first_translation_shows_without_repeated_activation(provider, expected):
    instance, _sink, calls = host(provider=provider)
    instance.show_for_translation()
    instance.show_for_translation()
    assert calls == expected
    assert not any(entry[0] == "activate" for entry in calls)


def test_preloaded_host_does_not_show_until_translation_or_manual_show():
    instance, _sink, calls = host()
    assert not instance._shown
    assert not calls
    instance.execute({"op": "show"})
    instance.show_for_translation()
    assert calls == [("navigate", provider_url("google_web")), ("show",), ("activate",)]
