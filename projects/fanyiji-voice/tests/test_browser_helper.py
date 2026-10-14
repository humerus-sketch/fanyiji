import io
import json
from types import SimpleNamespace

import pytest

from fanyiji.browser_helper import _Host, allowed_navigation, stable_version, validate_command


@pytest.mark.parametrize("version", ["146.0.3856.84", "154.0.4258.31"])
def test_stable_runtime_versions(version):
    assert stable_version(version)


@pytest.mark.parametrize("version", ["", None, "0", "0.0.0.0", "154.0.4258.31 beta", "155.0.4283.13 dev"])
def test_preview_or_invalid_runtime_is_rejected(version):
    assert not stable_version(version)


@pytest.mark.parametrize("url", [
    "https://passport.bilibili.com/login", "https://live.bilibili.com/1698225",
    "https://www.bilibili.com/", "https://bilibili.com/",
])
def test_top_level_navigation_stays_on_bilibili_https(url):
    assert allowed_navigation(url)


@pytest.mark.parametrize("url", [
    "http://live.bilibili.com/1698225", "file:///C:/Users/test", "javascript:alert(1)",
    "https://bilibili.com.evil.test/", "https://evilbilibili.com/", "https://user:pass@live.bilibili.com/1",
    "https://live.bilibili.com:1234/1", "https://live.bilibili.com:wrong/1", "https://live.bilibili.com/1\n",
    None, "https://localhost/", "https://live.bilibili.com\\@evil.test/1",
])
def test_unsafe_top_level_navigation_is_rejected(url):
    assert not allowed_navigation(url)


@pytest.mark.parametrize("command", [
    {"op": "show"}, {"op": "refresh"}, {"op": "close"}, {"op": "version"},
    {"op": "navigate", "url": "https://passport.bilibili.com/login"},
    {"op": "navigate", "url": "https://live.bilibili.com/1698225"},
])
def test_small_command_allowlist(command):
    assert validate_command(command) == command


@pytest.mark.parametrize("command", [
    [], {"op": "eval", "script": "alert(1)"}, {"op": "version", "extra": 1},
    {"op": "navigate", "url": "https://www.bilibili.com/"},
    {"op": "navigate", "url": "https://live.bilibili.com/0"},
    {"op": "navigate", "url": "https://live.bilibili.com/1?csrf=secret"},
    {"op": "navigate", "url": "https://live.bilibili.com/1#anything"},
    {"op": "navigate", "url": "https://passport.bilibili.com/login", "cookie": "secret"},
])
def test_unknown_fields_scripts_and_arbitrary_urls_are_rejected(command):
    with pytest.raises(ValueError, match="invalid_command"):
        validate_command(command)


def make_host(source="", version="146.0.3856.84", available="154.0.4258.31"):
    calls = []
    core = SimpleNamespace(Environment=SimpleNamespace(BrowserVersionString=version), BrowserProcessId=64242,
                           Navigate=lambda url: calls.append(("navigate", url)),
                           Reload=lambda: calls.append(("refresh",)))
    native = SimpleNamespace(IsDisposed=False, webview=SimpleNamespace(CoreWebView2=core),
                             Close=lambda: calls.append(("close",)), Show=lambda: calls.append(("show",)),
                             Activate=lambda: calls.append(("activate",)))
    native.BeginInvoke = lambda action: action()
    window = SimpleNamespace(native=native)
    environment_type = SimpleNamespace(GetAvailableBrowserVersionString=lambda: available)
    sink = io.StringIO()
    host = _Host(window, io.StringIO(source), sink, lambda callback: callback, environment_type, False)
    host.native_ready.set()
    return host, sink, calls


def test_reader_dispatches_valid_commands_and_closes_on_parent_eof():
    host, sink, calls = make_host('{"op":"refresh"}\n{"op":"version"}\n')
    host.read_commands()
    assert calls == [("refresh",), ("close",)]
    assert json.loads(sink.getvalue()) == {
        "kind": "version", "active_version": "146.0.3856.84", "available_version": "154.0.4258.31",
    }


def test_invalid_command_does_not_echo_credentials_or_close_parent_queue():
    host, sink, calls = make_host('{"op":"navigate","url":"https://evil.test/?cookie=secret"}\n{"op":"show"}\n')
    host.read_commands()
    assert calls == [("show",), ("activate",), ("close",)]
    assert sink.getvalue() == '{"kind": "error", "code": "invalid_command"}\n'
    assert not host.failure


def test_actual_preview_runtime_is_closed_instead_of_reported_ready():
    host, sink, calls = make_host(version="155.0.4283.13 canary")
    host.versions("ready")
    assert json.loads(sink.getvalue()) == {"kind": "error", "code": "unsupported_runtime_channel"}
    assert calls == [("close",)]
    assert host.failure


def test_native_command_exception_is_redacted():
    host, sink, calls = make_host()
    def fail():
        raise RuntimeError("cookie=private")
    host.window.native.webview.CoreWebView2.Reload = fail
    host.execute({"op": "refresh"})
    assert sink.getvalue() == '{"kind": "error", "code": "command_failed"}\n'


def test_close_prevents_late_navigation_or_window_reactivation():
    host, sink, calls = make_host()
    host.execute({"op": "close"})
    host.execute({"op": "navigate", "url": "https://live.bilibili.com/1698225"})
    host.execute({"op": "show"})
    host.execute({"op": "refresh"})
    assert host.stopped.is_set()
    assert calls == [("close",)]
    assert not sink.getvalue()


def test_failed_webview_initialization_closes_and_reports_only_type_code():
    host, sink, calls = make_host()
    host.initialization_completed(None, SimpleNamespace(IsSuccess=False))
    assert json.loads(sink.getvalue()) == {"kind": "error", "code": "browser_initialization_failed"}
    assert calls == [("close",)]


def test_unknown_native_update_failure_does_not_crash_or_leak(monkeypatch):
    host, sink, calls = make_host()
    def fail(*args):
        raise RuntimeError("cookie=private")
    monkeypatch.setattr(host, "versions", fail)
    host.update_available()
    assert sink.getvalue() == '{"kind": "error", "code": "version_probe_failed"}\n'
    assert calls == []


def live_source_events(sink):
    return [json.loads(line) for line in sink.getvalue().splitlines()
            if json.loads(line).get("kind") == "live_source"]


def begin_live_navigation(host, url="https://live.bilibili.com/1698225", navigation_id=11):
    args = SimpleNamespace(Uri=url, NavigationId=navigation_id, Cancel=False)
    host.navigation_starting(None, args)
    return args


def finish_live_navigation(host, url="https://live.bilibili.com/1698225",
                           navigation_id=11, success=True):
    host.window.native.webview.CoreWebView2.Source = url
    host.navigation_completed(None, SimpleNamespace(NavigationId=navigation_id, IsSuccess=success))


@pytest.mark.parametrize("url", [
    "https://www.bilibili.com/video/BV1xx411c7mD", "https://www.bilibili.com/video/av123",
])
def test_reference_video_navigation_command_uses_same_native_window(url):
    host, sink, calls = make_host()
    command = {"op": "navigate", "url": url}
    assert validate_command(command) == command
    host.execute(command)
    assert calls == [("navigate", url)] and not sink.getvalue()


@pytest.mark.parametrize("url", [
    "https://www.bilibili.com/video/BV1xx411c7mD/?cookie=private",
    "https://www.bilibili.com/video/av00123", "https://www.bilibili.com/video/av123#private",
    "https://www.bilibili.com/video/av123/", "https://www.bilibili.com/video/av0",
    "https://www.bilibili.com/video/BV1xx411c7mD/extra",
])
def test_reference_video_navigation_command_requires_canonical_url(url):
    with pytest.raises(ValueError, match="invalid_command"):
        validate_command({"op": "navigate", "url": url})


@pytest.mark.parametrize("url,canonical", [
    ("https://www.bilibili.com/video/BV1xx411c7mD", "https://www.bilibili.com/video/BV1xx411c7mD"),
    ("https://www.bilibili.com/video/BV1xx411c7mD/?cookie=private#reply", "https://www.bilibili.com/video/BV1xx411c7mD"),
    ("https://www.bilibili.com/video/av00123?p=2", "https://www.bilibili.com/video/av123"),
])
def test_reference_video_source_has_native_pid_but_no_live_room(url, canonical):
    host, sink, _calls = make_host()
    begin_live_navigation(host, url)
    finish_live_navigation(host, url)
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "ready", "room": "", "generation": 1,
        "browser_process_id": 64242, "source_kind": "video", "source_url": canonical}
    assert "cookie" not in sink.getvalue() and "private" not in sink.getvalue()


@pytest.mark.parametrize("url,success", [
    ("https://www.bilibili.com/video/BV1xx411c7mD", False),
    ("https://www.bilibili.com/video/BV1xx411c7mD/extra", True),
    ("https://www.bilibili.com/video/av0", True),
    ("https://www.bilibili.com/video/%61v123", True),
    ("https://user:private@www.bilibili.com/video/av123", True),
    ("https://www.bilibili.com.evil/video/av123", True),
])
def test_unsafe_or_failed_video_page_cannot_become_reference_source(url, success):
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    finish_live_navigation(host, url, success=success)
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "unavailable", "room": "", "generation": 1}
    assert "private" not in sink.getvalue()


def test_reference_video_switch_invalidates_metadata_and_drops_old_navigation():
    host, sink, _calls = make_host()
    begin_live_navigation(host, "https://www.bilibili.com/video/av123", navigation_id=11)
    finish_live_navigation(host, "https://www.bilibili.com/video/av123", navigation_id=11)
    begin_live_navigation(host, navigation_id=12)
    before = live_source_events(sink)
    assert before[-1] == {"kind": "live_source", "state": "navigating", "room": "", "generation": 2}
    finish_live_navigation(host, "https://www.bilibili.com/video/av123", navigation_id=11)
    assert live_source_events(sink) == before
    host.window.native.webview.CoreWebView2.BrowserProcessId = 65432
    finish_live_navigation(host, navigation_id=12)
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "ready", "room": "1698225", "generation": 2,
        "browser_process_id": 65432}


def test_reference_video_cannot_use_invalid_browser_pid_or_publish_after_close():
    host, sink, _calls = make_host()
    begin_live_navigation(host, "https://www.bilibili.com/video/av123")
    host.window.native.webview.CoreWebView2.BrowserProcessId = True
    finish_live_navigation(host, "https://www.bilibili.com/video/av123")
    assert live_source_events(sink)[-1]["state"] == "unavailable"
    host.on_closed()
    before = live_source_events(sink)
    begin_live_navigation(host, "https://www.bilibili.com/video/av456", navigation_id=12)
    finish_live_navigation(host, "https://www.bilibili.com/video/av456", navigation_id=12)
    assert live_source_events(sink) == before


def change_native_source(host, url, *, new_document=False):
    host.window.native.webview.CoreWebView2.Source = url
    host.source_changed(None, SimpleNamespace(IsNewDocument=new_document))


def test_native_ready_subscribes_source_changed_once():
    class NativeEvent:
        def __init__(self):
            self.handlers = []
        def __iadd__(self, handler):
            self.handlers.append(handler)
            return self
    host, sink, _calls = make_host()
    control = host.window.native.webview
    core = control.CoreWebView2
    core.Environment.NewBrowserVersionAvailable = NativeEvent()
    core.SourceChanged = NativeEvent()
    control.NavigationStarting = NativeEvent()
    control.NavigationCompleted = NativeEvent()
    host.ready()
    host.ready()
    assert core.SourceChanged.handlers == [host.source_changed]
    assert control.NavigationStarting.handlers == [host.navigation_starting]
    assert control.NavigationCompleted.handlers == [host.navigation_completed]


def test_video_spa_switch_fences_before_native_pid_validation_and_rejects_old_completion():
    host, sink, _calls = make_host()
    begin_live_navigation(host, "https://www.bilibili.com/video/av123")
    finish_live_navigation(host, "https://www.bilibili.com/video/av123")
    class NativeSource:
        Source = "https://www.bilibili.com/video/BV1xx411c7mD?cookie=private"
        @property
        def BrowserProcessId(self):
            assert live_source_events(sink)[-1] == {
                "kind": "live_source", "state": "navigating", "room": "", "generation": 2}
            return 65432
    host.window.native.webview.CoreWebView2 = NativeSource()
    host.source_changed(None, SimpleNamespace(IsNewDocument=False))
    events = live_source_events(sink)
    assert events[-1] == {
        "kind": "live_source", "state": "ready", "room": "", "generation": 2,
        "browser_process_id": 65432, "source_kind": "video",
        "source_url": "https://www.bilibili.com/video/BV1xx411c7mD"}
    assert "private" not in sink.getvalue()
    finish_live_navigation(host, "https://www.bilibili.com/video/av123", navigation_id=11)
    assert live_source_events(sink) == events


@pytest.mark.parametrize("url", [
    "https://www.bilibili.com/video/av123?cookie=private#reply",
    "https://www.bilibili.com/video/av123/#other",
    "https://www.bilibili.com/video/av00123?p=2",
])
def test_query_and_fragment_changes_keep_same_canonical_reference_identity(url):
    host, sink, _calls = make_host()
    begin_live_navigation(host, "https://www.bilibili.com/video/av123")
    finish_live_navigation(host, "https://www.bilibili.com/video/av123")
    before = live_source_events(sink)
    change_native_source(host, url)
    assert live_source_events(sink) == before
    assert host.navigation_generation == 1


def test_source_changed_during_full_navigation_waits_for_matching_completed():
    host, sink, _calls = make_host()
    begin_live_navigation(host, "https://www.bilibili.com/video/av123")
    finish_live_navigation(host, "https://www.bilibili.com/video/av123")
    begin_live_navigation(host, "https://www.bilibili.com/video/av456", navigation_id=12)
    before = live_source_events(sink)
    change_native_source(host, "https://www.bilibili.com/video/av456", new_document=True)
    change_native_source(host, "https://www.bilibili.com/video/av789", new_document=False)
    assert live_source_events(sink) == before
    finish_live_navigation(host, "https://www.bilibili.com/video/av123", navigation_id=11)
    assert live_source_events(sink) == before
    finish_live_navigation(host, "https://www.bilibili.com/video/av456", navigation_id=12)
    assert live_source_events(sink)[-1]["source_url"] == "https://www.bilibili.com/video/av456"
    assert live_source_events(sink)[-1]["generation"] == 2


@pytest.mark.parametrize("url", [
    "https://www.bilibili.com/", "https://www.bilibili.com/video/av0",
    "https://www.bilibili.com/video/av123/extra", "https://user:private@www.bilibili.com/video/av456",
    "https://evil.test/video/av456", "about:blank",
])
def test_spa_switch_to_non_reference_source_clears_previous_video_identity(url):
    host, sink, _calls = make_host()
    begin_live_navigation(host, "https://www.bilibili.com/video/av123")
    finish_live_navigation(host, "https://www.bilibili.com/video/av123")
    change_native_source(host, url)
    assert live_source_events(sink)[-2:] == [
        {"kind": "live_source", "state": "navigating", "room": "", "generation": 2},
        {"kind": "live_source", "state": "unavailable", "room": "", "generation": 2}]
    assert host._source_url == "" and "private" not in sink.getvalue()


def test_spa_switch_between_video_and_live_preserves_original_live_schema():
    host, sink, _calls = make_host()
    begin_live_navigation(host, "https://www.bilibili.com/video/av123")
    finish_live_navigation(host, "https://www.bilibili.com/video/av123")
    change_native_source(host, "https://live.bilibili.com/1698225?from=home")
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "ready", "room": "1698225", "generation": 2,
        "browser_process_id": 64242}
    before = live_source_events(sink)
    change_native_source(host, "https://live.bilibili.com/1698225/#other")
    assert live_source_events(sink) == before
    change_native_source(host, "https://www.bilibili.com/video/av456")
    assert live_source_events(sink)[-1]["room"] == ""
    assert live_source_events(sink)[-1]["source_url"] == "https://www.bilibili.com/video/av456"
    assert live_source_events(sink)[-1]["generation"] == 3


def test_spa_switch_invalid_native_pid_cannot_bind_new_reference_video():
    host, sink, _calls = make_host()
    begin_live_navigation(host, "https://www.bilibili.com/video/av123")
    finish_live_navigation(host, "https://www.bilibili.com/video/av123")
    host.window.native.webview.CoreWebView2.BrowserProcessId = 0
    change_native_source(host, "https://www.bilibili.com/video/av456")
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "unavailable", "room": "", "generation": 2}


@pytest.mark.parametrize("mode", ["closed", "smoke"])
def test_source_changed_after_close_or_in_smoke_mode_emits_no_audio_source(mode):
    host, sink, _calls = make_host()
    begin_live_navigation(host, "https://www.bilibili.com/video/av123")
    finish_live_navigation(host, "https://www.bilibili.com/video/av123")
    if mode == "closed":
        host.on_closed()
    else:
        host.smoke = True
    before = live_source_events(sink)
    change_native_source(host, "https://www.bilibili.com/video/av456")
    assert live_source_events(sink) == before


def test_new_document_without_navigation_started_fails_closed_until_real_navigation():
    host, sink, _calls = make_host()
    begin_live_navigation(host, "https://www.bilibili.com/video/av123")
    finish_live_navigation(host, "https://www.bilibili.com/video/av123")
    change_native_source(host, "https://www.bilibili.com/video/av456", new_document=True)
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "navigating", "room": "", "generation": 2}
    before = live_source_events(sink)
    finish_live_navigation(host, "https://www.bilibili.com/video/av456", navigation_id=11)
    assert live_source_events(sink) == before
    begin_live_navigation(host, "https://www.bilibili.com/video/av456", navigation_id=12)
    finish_live_navigation(host, "https://www.bilibili.com/video/av456", navigation_id=12)
    assert live_source_events(sink)[-1]["state"] == "ready"
    assert live_source_events(sink)[-1]["generation"] == 3


def test_live_source_invalidates_before_navigation_and_ready_has_only_room_identity():
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    assert live_source_events(sink) == [
        {"kind": "live_source", "state": "navigating", "room": "", "generation": 1}]
    finish_live_navigation(host)
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "ready", "room": "1698225", "generation": 1,
        "browser_process_id": 64242}
    assert all(set(row) == ({"kind", "state", "room", "generation", "browser_process_id"}
                           if row["state"] == "ready" else {"kind", "state", "room", "generation"})
               for row in live_source_events(sink))


@pytest.mark.parametrize("url,success", [
    ("https://live.bilibili.com/1698225", False),
    ("https://passport.bilibili.com/login", True),
    ("https://www.bilibili.com/1698225", True),
    ("https://live.bilibili.com/", True),
    ("https://live.bilibili.com/0", True),
    ("https://live.bilibili.com/1698225/extra", True),
    ("https://live.bilibili.com/1１２３", True),
    ("https://live.bilibili.com/1698225.evil", True),
    ("http://live.bilibili.com/1698225", True),
    ("https://live.bilibili.com.evil/1698225", True),
    ("https://live.bilibili.com:1234/1698225", True),
    ("https://user:secret@live.bilibili.com/1698225", True),
])
def test_live_source_non_room_or_failed_page_is_unavailable(url, success):
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    finish_live_navigation(host, url=url, success=success)
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "unavailable", "room": "", "generation": 1}
    assert "secret" not in sink.getvalue()


@pytest.mark.parametrize("url", [
    "https://live.bilibili.com/1698225/", "https://live.bilibili.com/1698225?from=home",
])
def test_live_source_safe_same_room_url_normalizes_to_room_without_url_query(url):
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    finish_live_navigation(host, url=url)
    assert live_source_events(sink)[-1]["room"] == "1698225"
    assert live_source_events(sink)[-1]["state"] == "ready"
    assert url not in sink.getvalue()


def test_live_source_old_navigation_completion_cannot_bind_new_source_to_old_room():
    host, sink, _calls = make_host()
    begin_live_navigation(host, navigation_id=11)
    begin_live_navigation(host, url="https://live.bilibili.com/21399689", navigation_id=12)
    before = live_source_events(sink)
    finish_live_navigation(host, navigation_id=11)
    assert live_source_events(sink) == before
    finish_live_navigation(host, url="https://live.bilibili.com/21399689", navigation_id=12)
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "ready", "room": "21399689", "generation": 2,
        "browser_process_id": 64242}


def test_live_source_blank_navigation_invalidates_previous_room():
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    finish_live_navigation(host)
    begin_live_navigation(host, url="about:blank", navigation_id=12)
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "navigating", "room": "", "generation": 2}
    finish_live_navigation(host, url="about:blank", navigation_id=12)
    assert live_source_events(sink)[-1] == {
        "kind": "live_source", "state": "unavailable", "room": "", "generation": 2}


def test_live_source_close_prevents_late_navigation_start_and_completion():
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    host.on_closed()
    before = live_source_events(sink)
    begin_live_navigation(host, navigation_id=12)
    finish_live_navigation(host, navigation_id=12)
    assert live_source_events(sink) == before
    assert host.navigation_generation == 1


def test_live_source_blocked_navigation_does_not_cancel_still_valid_old_room():
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    finish_live_navigation(host)
    before = live_source_events(sink)
    args = begin_live_navigation(host, url="https://evil.example/?cookie=secret", navigation_id=12)
    assert args.Cancel is True
    finish_live_navigation(host, navigation_id=12)
    assert live_source_events(sink) == before
    assert "secret" not in sink.getvalue()


def test_live_source_native_source_failure_is_unavailable_without_exception_text():
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    class DisposedCore:
        @property
        def Source(self):
            raise RuntimeError("cookie=private")
    host.window.native.webview.CoreWebView2 = DisposedCore()
    host.navigation_completed(None, SimpleNamespace(NavigationId=11, IsSuccess=True))
    assert live_source_events(sink)[-1]["state"] == "unavailable"
    assert "private" not in sink.getvalue()


@pytest.mark.parametrize("pid", [0, -1, 2**32, None, "64242", True, 64242.0, b"64242"])
def test_live_source_invalid_native_browser_pid_is_unavailable(pid):
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    host.window.native.webview.CoreWebView2.BrowserProcessId = pid
    finish_live_navigation(host)
    assert live_source_events(sink)[-1] == {
        "kind":"live_source", "state":"unavailable", "room":"", "generation":1}


def test_live_source_native_browser_pid_exception_is_redacted_and_never_ready():
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    class BrokenPidCore:
        Source = "https://live.bilibili.com/1698225?token=private"
        @property
        def BrowserProcessId(self):
            raise RuntimeError("cookie=private")
    host.window.native.webview.CoreWebView2 = BrokenPidCore()
    host.navigation_completed(None, SimpleNamespace(NavigationId=11, IsSuccess=True))
    assert live_source_events(sink)[-1]["state"] == "unavailable"
    assert "private" not in sink.getvalue() and "BrowserProcessId" not in sink.getvalue()


def test_live_source_missing_native_browser_pid_does_not_fall_back_to_host():
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    del host.window.native.webview.CoreWebView2.BrowserProcessId
    finish_live_navigation(host)
    assert live_source_events(sink)[-1] == {
        "kind":"live_source", "state":"unavailable", "room":"", "generation":1}


def test_live_source_page_query_cannot_supply_native_browser_identity():
    host, sink, _calls = make_host()
    begin_live_navigation(host)
    finish_live_navigation(host, url="https://live.bilibili.com/1698225?browser_process_id=99")
    assert live_source_events(sink)[-1]["browser_process_id"] == 64242
    assert "browser_process_id=99" not in sink.getvalue()


def test_live_source_smoke_navigation_does_not_create_live_capture_source():
    host, sink, _calls = make_host()
    host.smoke = True
    begin_live_navigation(host)
    finish_live_navigation(host)
    assert live_source_events(sink) == []


def test_pipe_keeps_provided_stream_unchanged():
    from fanyiji.browser_helper import _pipe
    stream = io.StringIO()
    assert _pipe(stream, -11, "w") is stream


@pytest.mark.parametrize("override", [
    "WEBVIEW2_BROWSER_EXECUTABLE_FOLDER", "WEBVIEW2_RELEASE_CHANNEL_PREFERENCE",
    "WEBVIEW2_CHANNEL_SEARCH_KIND", "WEBVIEW2_RELEASE_CHANNELS", "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
])
def test_external_runtime_overrides_are_refused_before_gui_or_profile_creation(tmp_path, monkeypatch, override):
    from fanyiji import browser_helper
    if browser_helper.os.name != "nt":
        pytest.skip("Windows helper")
    sink = io.StringIO()
    monkeypatch.setattr(browser_helper.sys, "stdin", io.StringIO())
    monkeypatch.setattr(browser_helper.sys, "stdout", sink)
    monkeypatch.setenv(override, "external-runtime-setting")
    profile = tmp_path / "uncreated-profile"
    assert browser_helper.main(["--profile", str(profile)]) == 1
    assert json.loads(sink.getvalue()) == {"kind": "error", "code": "runtime_override_blocked"}
    assert not profile.exists()


def danmaku_command(op="danmaku_send", request_id="request_1", room="1698225"):
    command = {"op": op, "request_id": request_id, "room": room}
    if op != "danmaku_login":
        command["text"] = "【测试译文】"
    return command


@pytest.mark.parametrize("op", ["danmaku_send", "danmaku_login", "danmaku_check"])
def test_restricted_danmaku_commands_are_part_of_pipe_allowlist(op):
    assert validate_command(danmaku_command(op)) == danmaku_command(op)


def test_disabled_local_feature_blocks_commands_before_worker_start(monkeypatch):
    from fanyiji import local_features
    monkeypatch.setattr(local_features, "danmaku_available", lambda: False)
    host, sink, _calls = make_host()
    host.enqueue_danmaku(danmaku_command())
    result = json.loads(sink.getvalue())
    assert result["kind"] == "danmaku_result"
    assert result["request_id"] == "request_1"
    assert result["result"]["outcome"] == "not_sent"
    assert result["result"]["error"] == "feature_unavailable"
    assert host.danmaku_thread is None


def test_native_wrong_room_never_evaluates_script(monkeypatch):
    host, sink, _calls = make_host()
    host.window.native.webview.CoreWebView2.Source = "https://live.bilibili.com/1"
    monkeypatch.setattr(host, "evaluate_danmaku", lambda *_args, **_kwargs: pytest.fail("must not evaluate"))
    host.process_danmaku(danmaku_command())
    result = json.loads(sink.getvalue())["result"]
    assert result["outcome"] == "not_sent"
    assert result["error"] == "wrong_room"


def accepted_danmaku():
    return {"http_status": 200, "code": 0, "msg_class": "empty", "message_class": "empty", "error": "none",
            "request_started_epoch_ms": 1700000000000, "request_elapsed_ms": 250, "before_count": 0, "cookie": "secret"}


def test_worker_evaluates_promises_outside_native_ui_thread_and_closes_cleanly():
    import threading
    host, sink, _calls = make_host()
    host.window.native.webview.CoreWebView2.Source = "https://live.bilibili.com/1698225"
    evaluated = []
    complete = threading.Event()
    def evaluate(script, callback=None):
        evaluated.append((threading.current_thread().name, callback is not None))
        callback(accepted_danmaku())
        complete.set()
        return True
    host.window.evaluate_js = evaluate
    host.enqueue_danmaku(danmaku_command())
    assert complete.wait(2)
    host.danmaku_queue.join()
    host.on_closed()
    host.join_worker()
    assert not host.danmaku_thread.is_alive()
    assert evaluated == [("embedded-danmaku-js", True)]
    result = json.loads(sink.getvalue())
    assert result["operation"] == "send"
    assert result["result"]["outcome"] == "receipt_accepted"
    assert result["result"]["before_count"] == 0
    assert "secret" not in sink.getvalue()


def test_host_spacing_survives_reload_and_ignores_wall_clock(monkeypatch):
    from fanyiji import browser_helper
    host, sink, _calls = make_host()
    now = [0.0]
    class ClockStop:
        def is_set(self):
            return False
        def wait(self, duration):
            now[0] += duration
            return False
    host.stopped = ClockStop()
    starts = []
    monkeypatch.setattr(browser_helper.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(host, "native_room_matches", lambda room: True)
    def evaluate(*_args, **_kwargs):
        starts.append(now[0])
        now[0] += 0.25
        return accepted_danmaku()
    monkeypatch.setattr(host, "evaluate_danmaku", evaluate)
    host.process_danmaku(danmaku_command())
    host.navigation_starting(None, SimpleNamespace(Uri="https://live.bilibili.com/1698225", Cancel=False))
    host.process_danmaku(danmaku_command(request_id="request_2"))
    assert starts == pytest.approx([0, 3.75])
    events = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert sum(event["kind"] == "danmaku_result" for event in events) == 2


def test_unfinished_dispatch_blocks_followup_send_and_never_retries(monkeypatch):
    host, sink, _calls = make_host()
    monkeypatch.setattr(host, "native_room_matches", lambda room: True)
    count = []
    def evaluate(*_args, **_kwargs):
        count.append(1)
        host.transport_uncertain = True
        return {"error": "script_timeout", "raw": "secret"}
    monkeypatch.setattr(host, "evaluate_danmaku", evaluate)
    host.process_danmaku(danmaku_command())
    host.process_danmaku(danmaku_command(request_id="request_2"))
    assert len(count) == 1
    assert host.transport_uncertain
    receipts = [json.loads(line)["result"] for line in sink.getvalue().splitlines()]
    assert all(result["outcome"] == "result_unknown" for result in receipts)
    assert "secret" not in sink.getvalue()


def test_tainted_transport_still_allows_readonly_login_and_presence_check(monkeypatch):
    host, sink, _calls = make_host()
    host.transport_uncertain = True
    monkeypatch.setattr(host, "native_room_matches", lambda room: True)
    scripts = []
    def evaluate(script, **_kwargs):
        scripts.append(script)
        if "#chat-history-list" in script:
            return {"error": "none", "room_matches": True, "chat_ready": True, "present": True,
                    "occurrence_count": 1}
        return {"error": "none", "code": 0, "is_login": True}
    monkeypatch.setattr(host, "evaluate_danmaku", evaluate)
    host.process_danmaku(danmaku_command("danmaku_check"))
    host.process_danmaku(danmaku_command("danmaku_login", request_id="request_2"))
    assert len(scripts) == 2
    assert all("method:'POST'" not in script for script in scripts)
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert responses[0]["result"]["present"]
    assert responses[0]["result"]["occurrence_count"] == 1
    assert responses[1]["result"]["is_login"]
    assert host.transport_uncertain


def test_legacy_long_punctuated_unknown_message_remains_readonly_checkable(monkeypatch):
    host, sink, _calls = make_host()
    monkeypatch.setattr(host, "native_room_matches", lambda room: True)
    legacy = "【" + "中" * 25 + "，" + "中" * 28 + "】"
    evaluated = []
    def evaluate(script, **_kwargs):
        evaluated.append(script)
        return {"error":"none", "room_matches":True, "chat_ready":True,
                "present":True, "occurrence_count":2}
    monkeypatch.setattr(host, "evaluate_danmaku", evaluate)
    host.process_danmaku({**danmaku_command("danmaku_check"), "text":legacy})
    assert len(evaluated) == 1 and "method:'POST'" not in evaluated[0]
    assert json.loads(sink.getvalue())["result"]["occurrence_count"] == 2
    assert legacy not in sink.getvalue()


def test_settled_network_unknown_can_be_readonly_verified_without_poisoning_transport(monkeypatch):
    host, sink, _calls = make_host()
    monkeypatch.setattr(host, "native_room_matches", lambda room: True)
    monkeypatch.setattr(host, "evaluate_danmaku", lambda *_args, **_kwargs: {"error": "network_unknown"})
    host.process_danmaku(danmaku_command())
    assert json.loads(sink.getvalue())["result"]["outcome"] == "result_unknown"
    assert not host.transport_uncertain


def test_navigation_during_send_is_unknown_instead_of_not_sent(monkeypatch):
    host, sink, _calls = make_host()
    matches = iter([True, False])
    monkeypatch.setattr(host, "native_room_matches", lambda room: next(matches))
    monkeypatch.setattr(host, "evaluate_danmaku", lambda *_args, **_kwargs: accepted_danmaku())
    host.process_danmaku(danmaku_command())
    result = json.loads(sink.getvalue())["result"]
    assert result["error"] == "navigation_changed"
    assert result["outcome"] == "result_unknown"
    assert host.transport_uncertain


def test_close_interrupts_wait_and_late_callback_does_not_emit_duplicate():
    import threading
    host, sink, _calls = make_host()
    callback_holder = []
    entered = threading.Event()
    def evaluate(script, callback=None):
        callback_holder.append(callback)
        entered.set()
        return True
    host.window.evaluate_js = evaluate
    outputs = []
    thread = threading.Thread(target=lambda: outputs.append(host.evaluate_danmaku("Promise.resolve({})", promise=True, generation=0)))
    thread.start()
    assert entered.wait(1)
    host.on_closed()
    thread.join(2)
    assert not thread.is_alive()
    assert outputs == [{"error": "window_closed"}]
    callback_holder[0](accepted_danmaku())
    assert outputs == [{"error": "window_closed"}]
    assert sink.getvalue() == ""


def test_pipe_queue_full_has_no_raw_command_payload(monkeypatch):
    import queue
    from fanyiji import local_features
    monkeypatch.setattr(local_features, "danmaku_available", lambda: True)
    host, sink, _calls = make_host()
    host.danmaku_thread = object()
    host.danmaku_queue = queue.Queue(maxsize=1)
    host.danmaku_queue.put(danmaku_command())
    host.enqueue_danmaku(danmaku_command(request_id="request_2"))
    result = json.loads(sink.getvalue())["result"]
    assert result["error"] == "queue_full"
    assert result["outcome"] == "not_sent"
    assert "测试译文" not in sink.getvalue()


@pytest.mark.parametrize("operation", ["danmaku_login", "danmaku_check"])
@pytest.mark.parametrize("failure", ["navigation", "timeout", "dispatch_exception"])
def test_readonly_interruption_never_poisons_subsequent_real_send(monkeypatch, operation, failure):
    """Run the real worker evaluator with native dispatch fakes, then perform POST JS."""
    import threading
    from fanyiji import browser_helper
    host, sink, _calls = make_host()
    monkeypatch.setattr(host, "native_room_matches", lambda room: True)
    clock = [0.0]
    monkeypatch.setattr(browser_helper.time, "monotonic", lambda: clock[0])
    scripts = []
    held_dispatch = threading.Event()
    release_dispatch = threading.Event()
    def evaluate(script, callback=None):
        scripts.append(script)
        if len(scripts) == 1:
            if failure == "dispatch_exception":
                raise RuntimeError("cookie=secret")
            if failure == "navigation":
                host.navigation_generation += 1
            elif failure == "timeout":
                clock[0] = 19.0
            # A synchronous check can block in the native dispatcher, while a
            # login Promise may simply never callback. Both must be bounded.
            if callback is None:
                held_dispatch.set()
                assert release_dispatch.wait(2)
                return None
            return True
        assert "method:'POST'" in script
        callback(accepted_danmaku())
        release_dispatch.set()
        return True
    host.window.evaluate_js = evaluate
    host.process_danmaku(danmaku_command(operation))
    assert not host.transport_uncertain
    first = json.loads(sink.getvalue().splitlines()[0])["result"]
    assert first["error"] == {"navigation": "navigation_changed", "timeout": "script_timeout",
                              "dispatch_exception": "script_unavailable"}[failure]
    host.process_danmaku(danmaku_command(request_id="actual_send"))
    release_dispatch.set()
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(responses) == 2
    assert responses[-1]["request_id"] == "actual_send"
    assert responses[-1]["result"]["outcome"] == "receipt_accepted"
    assert len(scripts) == 2
    assert not host.transport_uncertain
    assert "secret" not in sink.getvalue()
