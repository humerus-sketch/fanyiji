import json
from pathlib import Path
import queue
import threading

import pytest

from fanyiji.web_translation_engine import (
    GoogleWebTranslationEngine, YoudaoWebTranslationEngine, WebTranslationError, WebTranslationTransport,
)


class FakeTransport:
    def __init__(self, result=None):
        self.result = result
        self.commands = []
        self.preloaded = self.shown = self.cancelled = self.closed = 0

    def preload(self, **_kwargs):
        self.preloaded += 1

    def request(self, command, **_kwargs):
        self.commands.append(command)
        if callable(self.result):
            return self.result(command)
        return {"id": command["id"], "status": "ok", "text": "你好"} if self.result is None else self.result

    def show_window(self, **_kwargs):
        self.shown += 1

    def cancel_pending(self):
        self.cancelled += 1

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("engine_type,provider", [(GoogleWebTranslationEngine, "google_web"),
                                                   (YoudaoWebTranslationEngine, "youdao_web")])
def test_web_backend_is_distinct_from_local_and_codex(engine_type, provider):
    transport = FakeTransport()
    engine = engine_type(transport=transport)
    assert engine.backend_info["provider"] == provider
    assert engine.backend_info["model_id"] == provider
    assert engine.backend_info["backend"] == "webview2"
    assert engine.backend_info["network_required"] is True
    assert not engine.backend_info["supports_source_context"]
    engine.preload()
    engine.show_window()
    assert transport.preloaded == transport.shown == 1
    assert engine.translate("こんにちは", "ja", "zh") == "你好"
    assert set(transport.commands[0]) == {"op", "id", "text", "source_lang", "target_lang"}
    assert set(engine.last_request_metrics) == {"translation_seconds"}


def test_empty_or_same_language_never_submits_to_website():
    transport = FakeTransport()
    engine = GoogleWebTranslationEngine(transport=transport)
    assert engine.translate("  ", "ja", "zh") == ""
    assert engine.translate("你好", "zh", "zh") == "你好"
    assert not transport.commands


def test_cancellation_discards_even_a_late_success_from_transport():
    transport = FakeTransport()
    engine = GoogleWebTranslationEngine(transport=transport)
    def late(command):
        engine.cancel_pending()
        return {"id": command["id"], "status": "ok", "text": "过期译文"}
    transport.result = late
    with pytest.raises(WebTranslationError) as error:
        engine.translate("こんにちは", "ja", "zh")
    assert error.value.code == "cancelled"
    assert transport.cancelled == 1


def test_cancelled_waiter_does_not_start_a_new_browser():
    transport = FakeTransport()
    engine = GoogleWebTranslationEngine(transport=transport)
    event = threading.Event()
    event.set()
    with pytest.raises(WebTranslationError):
        engine.translate("こんにちは", "ja", "zh", cancellation_event=event)
    assert not transport.commands
    engine.close()
    engine.close()
    assert transport.closed == 1
    with pytest.raises(WebTranslationError) as error:
        engine.preload()
    assert error.value.code == "closed"


def test_error_messages_do_not_echo_website_content():
    error = WebTranslationError({"cookie": "secret"})
    assert error.code == "browser_unavailable"
    assert "secret" not in str(error)
    transport = FakeTransport(lambda command: {"id": command["id"], "status": "error",
                                              "code": "interaction_required"})
    with pytest.raises(WebTranslationError) as caught:
        YoudaoWebTranslationEngine(transport=transport).translate("こんにちは", "ja", "zh")
    assert caught.value.code == "interaction_required"


def test_chinese_output_policy_is_not_bypassed_by_webpage():
    from fanyiji.translation_output import ChineseOutputError
    transport = FakeTransport(lambda command: {"id": command["id"], "status": "ok", "text": "こんにちは"})
    with pytest.raises(ChineseOutputError):
        GoogleWebTranslationEngine(transport=transport).translate("hello", "ja", "zh")


class ReadPipe:
    def __init__(self):
        self.lines = queue.Queue()
        self.closed = False

    def emit(self, value):
        self.lines.put(json.dumps(value) + "\n")

    def readline(self, _limit):
        return self.lines.get(timeout=5)

    def close(self):
        self.closed = True
        self.lines.put("")


class WritePipe:
    def __init__(self, callback):
        self.callback = callback
        self.buffer = ""
        self.closed = False

    def write(self, value):
        if self.closed:
            raise ValueError("closed")
        self.buffer += value

    def flush(self):
        for line in self.buffer.splitlines():
            self.callback(json.loads(line))
        self.buffer = ""

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, *, respond=True, ready=True):
        self.stdout = ReadPipe()
        self.stdin = WritePipe(self.command)
        self.alive, self.respond = True, respond
        self.killed = False
        self.commands = []
        self.sent = threading.Event()
        self.diagnostic = None
        if ready:
            self.stdout.emit({"kind": "ready", "provider": "google_web"})

    def command(self, command):
        self.commands.append(command)
        self.sent.set()
        if command["op"] == "translate" and self.respond:
            if self.diagnostic is not None:
                self.stdout.emit({"kind": "diagnostic", **self.diagnostic})
            self.stdout.emit({"kind": "result", "id": "0" * 32,
                              "result": {"id": "0" * 32, "status": "ok", "text": "旧译文"}})
            self.stdout.emit({"kind": "result", "id": command["id"],
                              "result": {"id": command["id"], "status": "ok", "text": "你好"}})

    def poll(self):
        return None if self.alive else 1

    def kill(self):
        self.killed, self.alive = True, False
        self.stdout.close()

    def wait(self, **_kwargs):
        return 1


def command(identifier="1" * 32):
    return {"op": "translate", "id": identifier, "text": "こんにちは", "source_lang": "ja", "target_lang": "zh"}


def test_owned_transport_reuses_helper_and_fences_stale_result(tmp_path):
    processes, launches = [], []
    def factory(args, **options):
        launches.append((args, options))
        process = FakeProcess()
        processes.append(process)
        return process
    transport = WebTranslationTransport("google_web", profile=tmp_path / "google", process_factory=factory)
    transport.preload()
    assert transport.request(command())["text"] == "你好"
    assert transport.request(command("2" * 32))["text"] == "你好"
    transport.show_window()
    assert len(processes) == 1
    args, options = launches[0]
    assert "fanyiji.web_translation_helper" in args
    assert "--browser-helper" not in args
    assert options["stderr"] == -3  # DEVNULL
    assert "text" not in args
    transport.close()
    assert processes[0].killed


def test_timeout_kills_helper_then_allows_a_fresh_request(tmp_path):
    processes = []
    def factory(*_args, **_options):
        process = FakeProcess(respond=bool(processes))
        processes.append(process)
        return process
    transport = WebTranslationTransport("google_web", profile=tmp_path, process_factory=factory,
                                        request_timeout=0.05)
    with pytest.raises(WebTranslationError) as error:
        transport.request(command())
    assert error.value.code == "translation_timeout"
    assert processes[0].killed
    assert transport.request(command("2" * 32))["text"] == "你好"
    assert len(processes) == 2
    transport.close()


def test_cancel_event_aborts_blocked_owned_pipe(tmp_path):
    process = FakeProcess(respond=False)
    transport = WebTranslationTransport("google_web", profile=tmp_path, process_factory=lambda *_a, **_k: process)
    cancel = threading.Event()
    timer = threading.Timer(0.05, cancel.set)
    timer.start()
    try:
        with pytest.raises(WebTranslationError) as error:
            transport.request(command(), cancellation_event=cancel)
        assert error.value.code == "cancelled"
        assert process.killed
    finally:
        timer.cancel()
        transport.close()


def test_frozen_launcher_uses_entrypoint_dispatch(tmp_path, monkeypatch):
    import sys
    process, launches = FakeProcess(), []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    def factory(args, **kwargs):
        launches.append(args)
        return process
    transport = WebTranslationTransport("youdao_web", profile=tmp_path, process_factory=factory)
    transport.preload()
    assert launches[0][1] == "--web-translation-helper"
    assert "-m" not in launches[0]
    transport.close()


def test_show_does_not_wait_for_an_inflight_translation(tmp_path):
    process = FakeProcess(respond=False)
    transport = WebTranslationTransport("google_web", profile=tmp_path, process_factory=lambda *_a, **_k: process)
    transport.preload()
    result = []
    def translate():
        try:
            transport.request(command())
        except WebTranslationError as error:
            result.append(error.code)
    thread = threading.Thread(target=translate)
    thread.start()
    try:
        assert process.sent.wait(1)
        assert process.commands[0]["op"] == "translate"
        transport.show_window()
        assert process.commands[-1] == {"op": "show"}
    finally:
        transport.cancel_pending()
        thread.join(timeout=2)
        transport.close()
    assert result == ["cancelled"]


def test_global_helper_error_fences_the_active_dom(tmp_path):
    process = FakeProcess(respond=False)
    transport = WebTranslationTransport("google_web", profile=tmp_path, process_factory=lambda *_a, **_k: process)
    transport.preload()
    process.stdout.emit({"kind": "error", "code": "navigation_blocked", "diagnostic": "cookie=secret"})
    with pytest.raises(WebTranslationError) as error:
        transport.request(command())
    assert error.value.code == "navigation_blocked"
    assert "secret" not in str(error.value)
    assert process.killed
    transport.close()


def test_cancel_between_engine_check_and_transport_show_never_launches(tmp_path, monkeypatch):
    launches = []
    engine = GoogleWebTranslationEngine(profile=tmp_path, process_factory=lambda *_a, **_k: launches.append(True))
    original_check = engine._check
    def check_then_cancel(generation, cancellation_event=None):
        original_check(generation, cancellation_event)
        engine.cancel_pending()
    monkeypatch.setattr(engine, "_check", check_then_cancel)
    with pytest.raises(WebTranslationError) as error:
        engine.show_window()
    assert error.value.code == "cancelled"
    assert not launches
    engine.close()


def test_external_cancelled_show_never_creates_profile(tmp_path):
    cancel = threading.Event()
    cancel.set()
    profile = tmp_path / "must-not-exist"
    transport = WebTranslationTransport("google_web", profile=profile, process_factory=lambda *_a, **_k: pytest.fail("no launch"))
    with pytest.raises(WebTranslationError):
        transport.show_window(cancellation_event=cancel)
    assert not profile.exists()
    transport.close()


def test_safe_diagnostics_drop_all_unapproved_fields(tmp_path):
    process = FakeProcess()
    transport = WebTranslationTransport("google_web", profile=tmp_path, process_factory=lambda *_a, **_k: process)
    transport.preload()
    process.diagnostic = {"input_present": True, "output_present": False,
                          "source_matches": True, "language_pair_selected": True, "phase": "waiting",
                          "ready_state": "complete", "cookie": "secret", "text": "private"}
    transport.request(command())
    assert transport.last_diagnostic == {
        "input_present": True, "output_present": False, "source_matches": True,
        "language_pair_selected": True, "phase": "waiting", "ready_state": "complete"}
    assert "secret" not in str(transport.last_diagnostic)
    assert "private" not in str(transport.last_diagnostic)
    transport.close()
