"""Browser controls are isolated from audio and credentials; no real websites."""
import json
import os
import threading
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication
from fanyiji.browser_service import BrowserService, room_url
from fanyiji.browser_dialog import BrowserDialog


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def service(app, tmp_path):
    with patch("fanyiji.browser_service.data_root", return_value=tmp_path):
        result = BrowserService()
    yield result
    result.close()


@pytest.mark.parametrize("value,expected", [("1698225", "1698225"), ("00023", "23"),
    ("https://live.bilibili.com/1698225/?from=home", "1698225")])
def test_room_url(value, expected):
    assert room_url(value) == "https://live.bilibili.com/" + expected


@pytest.mark.parametrize("value", ["", "0", "-1", "１２３", "1/../../foo", "1234567890123",
    "https://evil.example/1698225", "https://live.bilibili.com.evil.example/123", "23#fragment"])
def test_room_url_rejects_invalid_input(value):
    with pytest.raises(ValueError):
        room_url(value)


def test_creating_service_does_not_open_browser_or_check_network(service):
    assert service.process.state() == QProcess.ProcessState.NotRunning
    assert not service.timer.isActive()
    assert not service._checking
    assert not service.profile.exists()


def test_public_frozen_service_is_available_but_requires_explicit_sender_settings(app, tmp_path, monkeypatch):
    from fanyiji import local_features
    monkeypatch.setattr(local_features.sys, "frozen", True, raising=False)
    monkeypatch.setattr(local_features, "data_root", lambda: tmp_path)
    with patch("fanyiji.browser_service.data_root", return_value=tmp_path), \
            patch.object(BrowserService, "open") as open_browser, \
            patch.object(BrowserService, "_send") as send:
        public = BrowserService()
        try:
            assert public.danmaku_available is True
            assert public.danmaku_enabled is False
            assert public.danmaku_repair_enabled is False
            assert public.danmaku_interval_seconds == 3.5
            assert not public.preferences_path.exists()
            assert not (tmp_path / "local-features.json").exists()
            public.save_preferences("1698225", False, danmaku_enabled=True)
            assert public.danmaku_enabled is True
            assert json.loads(public.preferences_path.read_text())["danmaku_enabled"] is True
            assert public.process.state() == QProcess.ProcessState.NotRunning
            assert not public.profile.exists()
            open_browser.assert_not_called()
            send.assert_not_called()
        finally:
            public.close()


def test_public_frozen_service_restores_saved_options_without_local_grant_or_browser_open(app, tmp_path, monkeypatch):
    from fanyiji import local_features
    monkeypatch.setattr(local_features.sys, "frozen", True, raising=False)
    monkeypatch.setattr(local_features, "data_root", lambda: tmp_path)
    preferences = tmp_path / "browser-preferences.json"
    preferences.write_text(json.dumps({"room": "21399689", "auto_check": False,
        "danmaku_enabled": True, "danmaku_repair_enabled": True,
        "danmaku_interval_seconds": 1, "danmaku_validation_room": "1698225"}), encoding="utf-8")
    original = preferences.read_bytes()
    with patch("fanyiji.browser_service.data_root", return_value=tmp_path), \
            patch.object(BrowserService, "open") as open_browser:
        public = BrowserService()
        try:
            assert public.danmaku_available is True
            assert public.danmaku_enabled is True
            assert public.danmaku_repair_enabled is True
            assert public.danmaku_interval_seconds == 3.5
            assert public.danmaku_validation_room == "1698225"
            assert preferences.read_bytes() == original
            assert public.process.state() == QProcess.ProcessState.NotRunning
            assert not public.profile.exists()
            assert not (tmp_path / "local-features.json").exists()
            open_browser.assert_not_called()
        finally:
            public.close()


@pytest.mark.parametrize("available,closed,expected", [
    (True, False, 1), (False, False, 0), (True, True, 0)])
def test_clear_queue_request_is_local_and_requires_available_live_service(service, available, closed, expected):
    calls = []
    service.danmaku_clear_requested.connect(lambda: calls.append(True))
    service.danmaku_available, service._closed = available, closed
    with patch.object(service, "_send") as send:
        service.clear_danmaku_queue()
    assert len(calls) == expected
    send.assert_not_called()
    service._closed = False


@pytest.mark.parametrize("available,closed,expected", [
    (True, False, 1), (False, False, 0), (True, True, 0)])
def test_filter_manager_request_needs_no_browser_and_sends_no_external_command(
        service, available, closed, expected):
    calls = []
    service.content_filter_requested.connect(lambda: calls.append(True))
    service.danmaku_available, service._closed = available, closed
    with patch.object(service, "_send") as send:
        service.request_content_filter()
    assert len(calls) == expected
    send.assert_not_called()
    assert service.process.state() == QProcess.ProcessState.NotRunning
    service._closed = False


def test_preferences_are_separate_and_only_contain_non_sensitive_fields(service):
    service.save_preferences("123", False)
    assert json.loads(service.preferences_path.read_text()) == {"room": "123", "auto_check": False}
    assert not service.timer.isActive()
    with patch("fanyiji.browser_service.data_root", return_value=service.preferences_path.parent):
        second = BrowserService()
    try:
        assert second.room == "123" and second.auto_check is False
        assert not second.profile.exists()
    finally:
        second.close()


def test_invalid_room_does_not_replace_saved_preferences(service):
    service.save_preferences("123", False)
    before = service.preferences_path.read_bytes()
    with pytest.raises(ValueError):
        service.save_preferences("https://other.example/23", True)
    assert service.preferences_path.read_bytes() == before


def test_no_commands_sent_before_native_ready_except_close(service):
    actual = service.process
    try:
        service.process = Mock()
        service.process.state.return_value = QProcess.ProcessState.Running
        service.refresh()
        service.process.write.assert_not_called()
        service.stop_browser()
        assert json.loads(service.process.write.call_args.args[0]) == {"op": "close"}
    finally:
        service.process = actual


def test_pending_navigation_is_sent_only_after_ready_event(service):
    actual = service.process
    events = []
    service.event.connect(events.append)
    try:
        service.process = Mock()
        service.process.state.return_value = QProcess.ProcessState.Running
        service._pending_url = room_url("123")
        service.process.readAllStandardOutput.return_value = b'{"kind":"ready","active_version":"146.0.1.2"}\n'
        service._read_events()
        assert service.active_version == "146.0.1.2"
        assert json.loads(service.process.write.call_args.args[0]) == {"op":"navigate","url":room_url("123")}
        assert service._pending_url is None
        assert events[-1]["kind"] == "ready"
    finally:
        service.process = actual


def test_disabled_automatic_checks_do_not_request_network(service):
    service.auto_check = False
    with patch.object(service, "check_updates") as check:
        service.activate_checks()
    check.assert_not_called()


def test_dialog_invalid_room_never_navigates(service):
    service.auto_check = False
    dialog = BrowserDialog(service)
    dialog.room.setText("https://other.example/1")
    with patch.object(service, "open") as opened:
        dialog.open_room()
    opened.assert_not_called()
    assert dialog.status.text() == "直播间格式无效"
    dialog.close()


def test_offline_check_not_reported_as_current(service):
    service.auto_check = False
    dialog = BrowserDialog(service)
    dialog.on_snapshot({"state": "check_unavailable", "installed_version": "146.0.1.2"})
    assert "不能确认" in dialog.status.text()
    assert "未确认" in dialog.versions.text()
    dialog.close()


def test_closed_event_clears_running_version_label(service):
    service.auto_check = False
    dialog = BrowserDialog(service)
    dialog.on_event({"kind":"ready", "active_version":"146.0.1.2", "available_version":"146.0.1.2"})
    dialog.on_event({"kind":"closed"})
    assert "正在运行：未启动" in dialog.versions.text()
    assert "146.0.1.2" in dialog.versions.text()
    dialog.close()


def test_late_network_result_recomputes_state_after_browser_restart(service):
    service.active_version = "147.0.1.2"
    snapshots = []
    service.snapshot.connect(snapshots.append)
    with patch("fanyiji.browser_runtime.installed_runtime_version", return_value="147.0.1.2"):
        service._on_check_finished({"active_version":"146.0.1.2", "catalog_version":"147.0.1.2",
                                   "state":"restart_required"})
    assert snapshots[-1]["active_version"] == "147.0.1.2"
    assert snapshots[-1]["state"] == "current"


@pytest.mark.parametrize("same_process", [True, False])
def test_startup_timeout_kills_only_the_timed_out_process(service, same_process):
    actual = service.process
    try:
        service.process = Mock()
        service.process.state.return_value = QProcess.ProcessState.Running
        service.process.processId.return_value = 42
        callbacks = []
        with patch("fanyiji.browser_service.QTimer.singleShot", side_effect=lambda delay, callback: callbacks.append(callback)):
            service._startup_timeout()
        service.process.terminate.assert_called_once()
        service.process.processId.return_value = 42 if same_process else 43
        callbacks[0]()
        assert service.process.kill.called is same_process
    finally:
        service.process = actual


def test_profile_failure_does_not_start_helper_and_is_visible(service):
    actual = service.profile
    events = []
    service.event.connect(events.append)
    try:
        service.profile = Mock()
        service.profile.mkdir.side_effect = PermissionError()
        with patch.object(service.process, "start") as start:
            service.open("https://passport.bilibili.com/login")
        start.assert_not_called()
        assert events[-1] == {"kind":"error", "code":"profile_unavailable"}
    finally:
        service.profile = actual


@pytest.fixture
def fake_process(service):
    actual = service.process
    process = Mock()
    process.state.return_value = QProcess.ProcessState.NotRunning
    process.processId.return_value = 4242
    process.readAllStandardOutput.return_value = b""
    service.process = process
    yield process
    service._pending_url = None
    service._ready = service._launching = service._stopping = False
    service.process = actual


def feed_event(service, process, event):
    process.readAllStandardOutput.return_value = json.dumps(event).encode() + b"\n"
    service._read_events()


def written_commands(process):
    return [json.loads(call.args[0]) for call in process.write.call_args_list]


def live_source_event(state="ready", room="1698225", generation=1, browser_process_id=64242):
    result = {"kind": "live_source", "state": state, "room": room, "generation": generation}
    if state == "ready":
        result["browser_process_id"] = browser_process_id
    return result


def prepare_live_source(service, process):
    process.state.return_value = QProcess.ProcessState.Running
    service._ready = True
    service._generation = 3
    events = []
    service.live_source_changed.connect(events.append)
    return events


def reference_video_event(url="https://www.bilibili.com/video/BV1xx411c7mD", generation=1, pid=64242):
    return {**live_source_event("ready", "", generation, pid),
            "source_kind": "video", "source_url": url}


def test_reference_video_source_is_separate_from_actual_live_room_and_preferences(service, fake_process):
    service.save_preferences("1698225", False)
    before = service.preferences_path.read_bytes()
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, reference_video_event(generation=7))
    expected = {"state": "ready", "room": "", "process_id": 64242,
                "generation": 7, "browser_generation": 3,
                "source_kind": "video", "source_url": "https://www.bilibili.com/video/BV1xx411c7mD"}
    assert received == [expected] and service.live_source_snapshot() == expected
    assert service.room == "1698225" and service.preferences_path.read_bytes() == before
    fake_process.processId.assert_not_called()


@pytest.mark.parametrize("change", [
    {"room": "1698225"}, {"source_kind": "live"}, {"source_kind": "unknown"},
    {"source_kind": None}, {"source_kind": []}, {"source_url": None},
    {"source_url": "https://live.bilibili.com/1698225"},
    {"source_url": "https://www.bilibili.com.evil/video/av123"},
    {"source_url": "https://user:private@www.bilibili.com/video/av123"},
    {"source_url": "https://www.bilibili.com/video/av123?cookie=private"},
    {"source_url": "https://www.bilibili.com/video/av123#private"},
    {"source_url": "https://www.bilibili.com/video/av00123"},
    {"source_url": "https://www.bilibili.com/video/av0"},
    {"browser_process_id": 0}, {"browser_process_id": True},
    {"generation": True}, {"generation": -1},
    {"state": "navigating"}, {"state": "unavailable"}, {"cookie": "private"},
])
def test_reference_video_malformed_identity_does_not_replace_live_source(service, fake_process, change):
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, live_source_event())
    before = service.live_source_snapshot()
    feed_event(service, fake_process, {**reference_video_event(), **change})
    assert received == [before] and service.live_source_snapshot() == before


def test_reference_video_partial_metadata_is_rejected(service, fake_process):
    received = prepare_live_source(service, fake_process)
    for missing in ("source_kind", "source_url"):
        event = reference_video_event()
        event.pop(missing)
        feed_event(service, fake_process, event)
    assert received == []


def test_reference_video_old_generation_cannot_replace_source_and_live_switch_clears_metadata(service, fake_process):
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, reference_video_event(generation=7))
    before = service.live_source_snapshot()
    feed_event(service, fake_process, reference_video_event("https://www.bilibili.com/video/av123", 6, 65432))
    assert received == [before]
    feed_event(service, fake_process, live_source_event("navigating", "", 8))
    assert "source_kind" not in service.live_source_snapshot() and "source_url" not in service.live_source_snapshot()
    feed_event(service, fake_process, live_source_event(generation=8, browser_process_id=65432))
    assert received[-1] == {"state": "ready", "room": "1698225", "process_id": 65432,
                            "generation": 8, "browser_generation": 3}


def test_reference_video_browser_restart_invalidates_metadata_and_uses_new_native_pid(service, fake_process):
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, reference_video_event(generation=7))
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(0, QProcess.ExitStatus.NormalExit)
    assert received[-1]["process_id"] == 0 and "source_url" not in received[-1]
    service.open_reference_video("https://www.bilibili.com/video/av00123/?p=2")
    fake_process.state.return_value = QProcess.ProcessState.Running
    feed_event(service, fake_process, {"kind": "ready"})
    feed_event(service, fake_process, reference_video_event("https://www.bilibili.com/video/av123", 1, 65432))
    assert received[-1]["browser_generation"] == 4 and received[-1]["process_id"] == 65432
    assert received[-1]["source_url"] == "https://www.bilibili.com/video/av123"


def test_reference_video_open_reuses_ready_browser_and_never_changes_live_preferences(service, fake_process):
    service.save_preferences("1698225", False)
    before = service.preferences_path.read_bytes()
    fake_process.state.return_value = QProcess.ProcessState.Running
    service._ready = True
    service.open_reference_video("https://www.bilibili.com/video/av00123/?p=2&cookie=private")
    service.open_reference_video("https://www.bilibili.com/video/BV1xx411c7mD#reply")
    fake_process.start.assert_not_called()
    assert written_commands(fake_process) == [
        {"op": "navigate", "url": "https://www.bilibili.com/video/av123"}, {"op": "show"},
        {"op": "navigate", "url": "https://www.bilibili.com/video/BV1xx411c7mD"}, {"op": "show"}]
    assert service.room == "1698225" and service.preferences_path.read_bytes() == before


def test_reference_video_invalid_open_does_not_launch_or_mutate_pending_navigation(service, fake_process):
    service._pending_url = room_url("1698225")
    with pytest.raises(ValueError, match="invalid_reference_video_url"):
        service.open_reference_video("https://live.bilibili.com/21399689")
    fake_process.start.assert_not_called()
    assert service._pending_url == room_url("1698225")


def test_reference_video_open_during_browser_close_waits_for_exit(service, fake_process):
    fake_process.state.return_value = QProcess.ProcessState.Running
    service._ready = True
    service.stop_browser()
    service.open_reference_video("https://www.bilibili.com/video/av123", "ja")
    fake_process.start.assert_not_called()
    assert not any(command["op"] == "navigate" for command in written_commands(fake_process))
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(0, QProcess.ExitStatus.NormalExit)
    fake_process.start.assert_called_once()
    assert service._pending_url == "https://www.bilibili.com/video/av123"
    assert fake_process.setArguments.call_args.args[0][-1] == "ja"


def test_live_source_ready_uses_native_browser_pid_not_helper_pid_and_both_generations(service, fake_process):
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, live_source_event(generation=7))
    expected = {"state": "ready", "room": "1698225", "process_id": 64242,
                "generation": 7, "browser_generation": 3}
    assert received == [expected]
    assert service.live_source_snapshot() == expected
    fake_process.processId.assert_not_called()
    copy = service.live_source_snapshot()
    copy["room"] = "21399689"
    assert service.live_source_snapshot() == expected


@pytest.mark.parametrize("event", [
    {"kind": "live_source", "state": "ready", "room": "1698225", "generation": 1},
    {"kind": "live_source", "state": "ready", "room": "1698225"},
    {"kind": "live_source", "state": "ready", "generation": 1},
    {"kind": "live_source", "room": "1698225", "generation": 1},
    {**live_source_event(), "process_id": 99},
    {**live_source_event(), "cookie": "secret"},
    live_source_event(state="invalid"), live_source_event(state=[]), live_source_event(state={}),
    live_source_event(state=None), live_source_event(generation=-1),
    live_source_event(generation=True), live_source_event(generation=1.0),
    live_source_event(room=None), live_source_event(room=[]), live_source_event(room="0"),
    live_source_event(room="001698225"), live_source_event(room="1698225 "),
    live_source_event(room="https://live.bilibili.com/1698225"),
    live_source_event(room="１２３"), live_source_event(state="navigating", room="1698225"),
    live_source_event(state="unavailable", room="1698225"),
])
def test_live_source_malformed_schema_never_changes_capture_identity(service, fake_process, event):
    received = prepare_live_source(service, fake_process)
    before = service.live_source_snapshot()
    feed_event(service, fake_process, event)
    assert received == []
    assert service.live_source_snapshot() == before


@pytest.mark.parametrize("pid", [0, -1, None, "not-a-pid", "64242", True, 1.0, 2**32, [], {}])
def test_live_source_ready_requires_valid_native_browser_process_id(service, fake_process, pid):
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, live_source_event(browser_process_id=pid))
    assert received == []
    assert service.live_source_snapshot()["process_id"] == 0


@pytest.mark.parametrize("state", ["navigating", "unavailable"])
def test_live_source_non_ready_zero_pid_is_allowed_but_nonzero_is_rejected(service, fake_process, state):
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, live_source_event())
    active = service.live_source_snapshot()
    feed_event(service, fake_process, {**live_source_event(state, "", 2), "browser_process_id": 64242})
    assert service.live_source_snapshot() == active
    feed_event(service, fake_process, {**live_source_event(state, "", 2), "browser_process_id": 0})
    assert received[-1] == {"state":state, "room":"", "process_id":0,
                            "generation":2, "browser_generation":3}


def test_live_source_cannot_fall_back_to_alive_helper_without_native_pid(service, fake_process):
    received = prepare_live_source(service, fake_process)
    fake_process.processId.return_value = 4242
    feed_event(service, fake_process, {"kind":"live_source", "state":"ready", "room":"1698225", "generation":1})
    assert received == [] and service.live_source_snapshot()["state"] == "unavailable"
    fake_process.processId.assert_not_called()


@pytest.mark.parametrize("field", ["_ready", "_stopping", "_closed"])
def test_live_source_after_stop_or_before_ready_is_ignored(service, fake_process, field):
    received = prepare_live_source(service, fake_process)
    setattr(service, field, field != "_ready")
    feed_event(service, fake_process, live_source_event())
    assert received == []
    assert service.live_source_snapshot()["process_id"] == 0
    service._closed = False


def test_live_source_older_audio_navigation_generation_is_ignored(service, fake_process):
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, live_source_event(generation=7))
    before = service.live_source_snapshot()
    for event in (live_source_event(generation=6), live_source_event("navigating", "", 5),
                  live_source_event("unavailable", "", 4)):
        feed_event(service, fake_process, event)
    assert received == [before]
    assert service.live_source_snapshot() == before


def test_live_source_navigation_invalidates_pid_then_ready_binds_new_room(service, fake_process):
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, live_source_event(generation=1))
    feed_event(service, fake_process, live_source_event("navigating", "", 2))
    assert service.live_source_snapshot() == {
        "state": "navigating", "room": "", "process_id": 0, "generation": 2, "browser_generation": 3}
    feed_event(service, fake_process, live_source_event("ready", "21399689", 2))
    assert received[-1] == {
        "state": "ready", "room": "21399689", "process_id": 64242,
        "generation": 2, "browser_generation": 3}


@pytest.mark.parametrize("operation", ["stop", "close_event", "crash", "failed_start", "timeout", "close"])
def test_live_source_process_lifecycle_invalidates_room_and_pid(service, fake_process, operation):
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, live_source_event())
    fake_process.waitForFinished.return_value = True
    if operation == "stop":
        service.stop_browser()
    elif operation == "close_event":
        feed_event(service, fake_process, {"kind": "closed"})
    elif operation == "crash":
        fake_process.state.return_value = QProcess.ProcessState.NotRunning
        service._finished(1, QProcess.ExitStatus.CrashExit)
    elif operation == "failed_start":
        service._process_error(QProcess.ProcessError.FailedToStart)
    elif operation == "timeout":
        service._ready = False
        service._startup_timeout()
    else:
        service.close()
    assert received[-1] == {
        "state": "unavailable", "room": "", "process_id": 0,
        "generation": 0, "browser_generation": 3}
    assert service.live_source_snapshot() == received[-1]


def test_live_source_restart_resets_audio_generation_and_binds_new_pid(service, fake_process):
    received = prepare_live_source(service, fake_process)
    feed_event(service, fake_process, live_source_event(generation=7))
    # Any partial line from the dead helper is discarded at finished/restart.
    service._buffer = b'{"kind":"live_source","state":"ready","room":"old'
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(0, QProcess.ExitStatus.NormalExit)
    assert not service._buffer
    service.open("21399689")
    assert service._generation == 4
    fake_process.processId.return_value = 5252
    fake_process.state.return_value = QProcess.ProcessState.Running
    feed_event(service, fake_process, {"kind": "ready"})
    feed_event(service, fake_process, live_source_event(room="21399689", generation=1, browser_process_id=65252))
    assert received[-1] == {
        "state": "ready", "room": "21399689", "process_id": 65252,
        "generation": 1, "browser_generation": 4}


def test_rapid_open_while_starting_launches_only_one_helper(service, fake_process):
    # Keep NotRunning until after start() to exercise the launch reservation,
    # including reentrant open calls before QProcess publishes its new state.
    fake_process.start.side_effect = lambda: service.open("123")
    service.open("https://passport.bilibili.com/login")
    service.open("456")
    fake_process.start.assert_called_once()
    assert service._pending_url == room_url("456")
    fake_process.state.return_value = QProcess.ProcessState.Running
    feed_event(service, fake_process, {"kind": "ready"})
    assert written_commands(fake_process) == [{"op": "navigate", "url": room_url("456")}]


def test_concurrent_open_requests_are_serialized_on_the_service_thread(app, service, fake_process):
    calls = []
    fake_process.start.side_effect = lambda: calls.append(threading.get_ident())
    workers = [threading.Thread(target=service.open, args=("123",)) for _ in range(20)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    fake_process.start.assert_not_called()
    app.processEvents()
    fake_process.start.assert_called_once()
    assert calls == [threading.get_ident()]
    assert service._pending_url == room_url("123")


def test_repeated_open_reuses_ready_window(service, fake_process):
    fake_process.state.return_value = QProcess.ProcessState.Running
    service._ready = True
    service.open("123")
    service.open("456")
    fake_process.start.assert_not_called()
    assert written_commands(fake_process) == [
        {"op": "navigate", "url": room_url("123")}, {"op": "show"},
        {"op": "navigate", "url": room_url("456")}, {"op": "show"},
    ]


@pytest.mark.parametrize("state", [QProcess.ProcessState.Starting, QProcess.ProcessState.Running])
def test_open_during_close_waits_for_exit_and_uses_latest_request(service, fake_process, state):
    fake_process.state.return_value = state
    service._launching = state == QProcess.ProcessState.Starting
    service._ready = state == QProcess.ProcessState.Running
    service.stop_browser()
    service.open("123", "ja")
    service.open("456", "en")
    fake_process.start.assert_not_called()
    assert not any(command["op"] == "navigate" for command in written_commands(fake_process))
    fake_process.state.return_value = QProcess.ProcessState.Running
    service._process_started()
    feed_event(service, fake_process, {"kind": "ready", "active_version": "146.0.1.2"})
    assert not service._ready and service.active_version is None
    assert all(command == {"op": "close"} for command in written_commands(fake_process))
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(0, QProcess.ExitStatus.NormalExit)
    fake_process.start.assert_called_once()
    assert fake_process.setArguments.call_args.args[0][-1] == "en"
    assert service._pending_url == room_url("456")
    fake_process.state.return_value = QProcess.ProcessState.Running
    feed_event(service, fake_process, {"kind": "ready"})
    assert written_commands(fake_process)[-1] == {"op": "navigate", "url": room_url("456")}


def test_native_close_event_prevents_reuse_before_process_exit(service, fake_process):
    fake_process.state.return_value = QProcess.ProcessState.Running
    service._ready = True
    feed_event(service, fake_process, {"kind": "closed"})
    service.open("123")
    fake_process.start.assert_not_called()
    assert not service._ready
    assert service._stopping
    assert not any(command["op"] == "navigate" for command in written_commands(fake_process))
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(0, QProcess.ExitStatus.NormalExit)
    fake_process.start.assert_called_once()


def test_repeat_close_cancels_a_queued_reopen(service, fake_process):
    fake_process.state.return_value = QProcess.ProcessState.Running
    service.stop_browser()
    service.open("123")
    service.stop_browser()
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(0, QProcess.ExitStatus.NormalExit)
    fake_process.start.assert_not_called()
    assert service._pending_url is None


def test_stuck_close_terminates_then_kills_without_starting_another_helper(service, fake_process):
    fake_process.state.return_value = QProcess.ProcessState.Running
    callbacks = []
    with patch("fanyiji.browser_service.QTimer.singleShot", side_effect=lambda delay, cb: callbacks.append((delay, cb))):
        service.stop_browser()
        service.open("123")
        assert callbacks[0][0] == 3000
        callbacks[0][1]()
        fake_process.terminate.assert_called_once()
        assert callbacks[1][0] == 1000
        callbacks[1][1]()
    fake_process.kill.assert_called_once()
    fake_process.start.assert_not_called()


def test_old_close_timeout_does_not_terminate_a_new_helper(service, fake_process):
    fake_process.state.return_value = QProcess.ProcessState.Running
    callbacks = []
    with patch("fanyiji.browser_service.QTimer.singleShot", side_effect=lambda delay, cb: callbacks.append(cb)):
        service.stop_browser()
    service.open("123")
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(0, QProcess.ExitStatus.NormalExit)
    fake_process.state.return_value = QProcess.ProcessState.Running
    # Even another close must not revive the previous generation's timer.
    service._stopping = True
    callbacks[0]()
    fake_process.terminate.assert_not_called()
    fake_process.kill.assert_not_called()


def test_shutdown_cancels_pending_restart(service, fake_process):
    fake_process.state.return_value = QProcess.ProcessState.Running
    service.stop_browser()
    service.open("123")
    fake_process.waitForFinished.return_value = True
    service.close()
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(0, QProcess.ExitStatus.NormalExit)
    fake_process.start.assert_not_called()


def test_failed_start_releases_launch_reservation_for_next_open(service, fake_process):
    service.open("123")
    service._process_error(QProcess.ProcessError.FailedToStart)
    assert not service._launching
    assert not service.startup_timer.isActive()
    assert service._pending_url is None
    service.open("456")
    assert fake_process.start.call_count == 2


def test_close_after_start_failure_never_reopens_stale_navigation(service, fake_process):
    service.open("123")
    service._process_error(QProcess.ProcessError.FailedToStart)
    service.stop_browser()
    fake_process.start.assert_called_once()
    assert not service._stopping
    assert service._pending_url is None


def test_unexpected_helper_crash_does_not_restart_automatically(service, fake_process):
    service.open("123")
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(1, QProcess.ExitStatus.CrashExit)
    fake_process.start.assert_called_once()
    assert service._pending_url is None


def test_sender_preferences_persist_but_cannot_bypass_minimum_interval(service):
    service.save_preferences("1698225", False, danmaku_enabled=True,
                             danmaku_source="system", danmaku_interval_seconds=3.5)
    before = service.preferences_path.read_bytes()
    for invalid in (0, 3.49, float("nan"), True):
        with pytest.raises(ValueError):
            service.save_preferences("1698225", False, danmaku_interval_seconds=invalid)
        assert service.preferences_path.read_bytes() == before
    with patch("fanyiji.browser_service.data_root", return_value=service.preferences_path.parent):
        second = BrowserService()
    try:
        assert second.danmaku_preferences() == {
            "enabled": True, "room": "1698225", "source": "system", "interval_seconds": 3.5,
            "validation_room": "", "repair_enabled": False}
    finally:
        second.close()


def test_validation_room_is_persistent_separate_and_never_opens_browser(service):
    service.save_preferences("21399689", False, danmaku_enabled=True,
                             danmaku_validation_room="https://live.bilibili.com/001698225/?from=test")
    preferences = json.loads(service.preferences_path.read_text())
    assert preferences["room"] == "21399689"
    assert preferences["danmaku_validation_room"] == "1698225"
    assert service.danmaku_preferences()["validation_room"] == "1698225"
    assert service.process.state() == QProcess.ProcessState.NotRunning
    assert service._pending_url is None
    with patch("fanyiji.browser_service.data_root", return_value=service.preferences_path.parent):
        second = BrowserService()
    try:
        assert second.room == "21399689"
        assert second.danmaku_validation_room == "1698225"
        assert not second.profile.exists()
    finally:
        second.close()
    service.save_preferences("21399689", False)
    assert service.danmaku_validation_room == "1698225"
    assert json.loads(service.preferences_path.read_text())["danmaku_validation_room"] == "1698225"


@pytest.mark.parametrize("invalid", ["21399689", "0021399689", "https://live.bilibili.com/21399689", "0", "bad", True, []])
def test_invalid_or_same_validation_room_never_overwrites_good_preferences(service, invalid):
    service.save_preferences("21399689", False, danmaku_enabled=True, danmaku_validation_room="1698225")
    before = service.preferences_path.read_bytes()
    with pytest.raises(ValueError):
        service.save_preferences("21399689", True, danmaku_validation_room=invalid)
    assert service.preferences_path.read_bytes() == before
    assert service.room == "21399689"
    assert service.danmaku_validation_room == "1698225"
    assert service.auto_check is False


@pytest.mark.parametrize("validation", ["21399689", "bad", True])
def test_invalid_loaded_validation_room_does_not_disable_other_saved_settings(service, validation):
    service.preferences_path.write_text(json.dumps({"room": "21399689", "auto_check": False,
        "danmaku_enabled": True, "danmaku_validation_room": validation}), encoding="utf-8")
    with patch("fanyiji.browser_service.data_root", return_value=service.preferences_path.parent):
        second = BrowserService()
    try:
        assert second.room == "21399689"
        assert second.danmaku_validation_room == ""
        assert second.danmaku_enabled is True
        assert second.auto_check is False
    finally:
        second.close()


def test_unavailable_service_cannot_enable_sender(service):
    service.danmaku_available = False
    with pytest.raises(ValueError):
        service.save_preferences("1698225", False, danmaku_enabled=True)
    assert not service.request_danmaku("send", "req", "1698225", "【有效测试正文内容】")


def test_repair_switch_preserves_other_saved_settings_and_only_start_opens_test_room(service):
    service.danmaku_available = True
    service.save_preferences("21399689", False, danmaku_enabled=True,
        danmaku_source="mic", danmaku_interval_seconds=8.5, danmaku_validation_room="1698225")
    before = json.loads(service.preferences_path.read_text())
    events = []
    service.prefs_changed.connect(events.append)
    with patch.object(service, "open") as open_browser, patch.object(service, "_send") as send:
        service.set_danmaku_repair_enabled(True)
        saved = json.loads(service.preferences_path.read_text())
        assert saved == {**before, "danmaku_repair_enabled": True}
        assert events[-1]["repair_enabled"] is True
        assert service.danmaku_enabled is True
        with patch("fanyiji.browser_service.data_root", return_value=service.preferences_path.parent), \
                patch("fanyiji.local_features.danmaku_available", return_value=True):
            reloaded = BrowserService()
        try:
            assert reloaded.danmaku_repair_enabled is True
            assert reloaded.danmaku_source == "mic"
            assert reloaded.danmaku_validation_room == "1698225"
        finally:
            reloaded.close()
        service.set_danmaku_repair_enabled(False)
        assert json.loads(service.preferences_path.read_text()) == before
        assert events[-1]["repair_enabled"] is False
        open_browser.assert_called_once_with("https://live.bilibili.com/1698225", "zh-CN")
        send.assert_not_called()
    assert service.process.state() == QProcess.ProcessState.NotRunning
    assert not service.profile.exists()


@pytest.mark.parametrize("saved", [None, 1, "true", [], False])
def test_legacy_or_invalid_repair_setting_loads_disabled(service, saved):
    preferences = {"room": "21399689", "danmaku_enabled": True,
                   "danmaku_validation_room": "1698225"}
    if saved is not None:
        preferences["danmaku_repair_enabled"] = saved
    service.preferences_path.write_text(json.dumps(preferences), encoding="utf-8")
    with patch("fanyiji.browser_service.data_root", return_value=service.preferences_path.parent), \
            patch("fanyiji.local_features.danmaku_available", return_value=True):
        reloaded = BrowserService()
    try:
        assert reloaded.danmaku_repair_enabled is False
        assert reloaded.danmaku_enabled is True
    finally:
        reloaded.close()


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", [], {}])
def test_repair_setter_rejects_non_boolean_without_changing_settings(service, invalid):
    service.save_preferences("21399689", False)
    before = service.preferences_path.read_bytes()
    with pytest.raises(ValueError):
        service.set_danmaku_repair_enabled(invalid)
    assert service.preferences_path.read_bytes() == before
    assert service.danmaku_repair_enabled is False


def test_unavailable_service_cannot_enable_or_restore_saved_repair_switch(service):
    service.preferences_path.write_text(json.dumps({"room": "21399689",
        "danmaku_repair_enabled": True, "danmaku_enabled": True}), encoding="utf-8")
    with patch("fanyiji.browser_service.data_root", return_value=service.preferences_path.parent), \
            patch("fanyiji.local_features.danmaku_available", return_value=False):
        public = BrowserService()
    try:
        assert public.danmaku_repair_enabled is False
        before = public.preferences_path.read_bytes()
        with pytest.raises(ValueError):
            public.set_danmaku_repair_enabled(True)
        assert public.preferences_path.read_bytes() == before
        assert public.process.state() == QProcess.ProcessState.NotRunning
    finally:
        public.close()


def test_repair_switch_write_failure_keeps_previous_state(service):
    service.danmaku_available = True
    service.save_preferences("21399689", False, danmaku_validation_room="1698225")
    before = service.preferences_path.read_bytes()
    events = []
    service.prefs_changed.connect(events.append)
    with patch("fanyiji.browser_service.os.replace", side_effect=OSError("disk failed")):
        with pytest.raises(OSError):
            service.set_danmaku_repair_enabled(True)
    assert service.preferences_path.read_bytes() == before
    assert service.danmaku_repair_enabled is False
    assert events == []


@pytest.mark.parametrize("validation", ["", "21399689", "bad"])
def test_repair_start_requires_saved_distinct_valid_test_room(service, validation):
    service.danmaku_available = True
    service.save_preferences("21399689", False)
    before = service.preferences_path.read_bytes()
    service.danmaku_validation_room = validation
    with patch.object(service, "open") as open_browser, pytest.raises(ValueError):
        service.set_danmaku_repair_enabled(True)
    open_browser.assert_not_called()
    assert service.preferences_path.read_bytes() == before
    assert service.danmaku_repair_enabled is False


def test_loading_or_saving_enabled_repair_does_not_open_test_room(service):
    service.danmaku_available = True
    with patch.object(service, "open") as open_browser:
        service.save_preferences("21399689", False, danmaku_validation_room="1698225",
                                 danmaku_repair_enabled=True)
        open_browser.assert_not_called()
    with patch("fanyiji.browser_service.data_root", return_value=service.preferences_path.parent), \
            patch("fanyiji.local_features.danmaku_available", return_value=True), \
            patch.object(BrowserService, "open") as open_browser:
        reloaded = BrowserService()
    try:
        assert reloaded.danmaku_repair_enabled is True
        open_browser.assert_not_called()
        assert reloaded.process.state() == QProcess.ProcessState.NotRunning
    finally:
        reloaded.close()


def test_loaded_repair_preference_without_sender_key_survives_browser_setting_save(service):
    service.preferences_path.write_text(json.dumps({"room": "21399689",
        "danmaku_validation_room": "1698225", "danmaku_repair_enabled": True}), encoding="utf-8")
    with patch("fanyiji.browser_service.data_root", return_value=service.preferences_path.parent), \
            patch("fanyiji.local_features.danmaku_available", return_value=True):
        reloaded = BrowserService()
    try:
        reloaded.save_preferences("21399689", False)
        assert json.loads(reloaded.preferences_path.read_text())["danmaku_repair_enabled"] is True
        assert reloaded.process.state() == QProcess.ProcessState.NotRunning
    finally:
        reloaded.close()


def test_sender_result_requires_matching_request_and_operation(service, fake_process):
    fake_process.state.return_value = QProcess.ProcessState.Running
    fake_process.write.side_effect = len
    service._ready = True
    received = []
    service.danmaku_result.connect(received.append)
    assert service.request_danmaku("login", "login1", "1698225")
    feed_event(service, fake_process, {"kind": "danmaku_result", "request_id": "wrong",
        "operation": "login", "result": {"is_login": True}})
    feed_event(service, fake_process, {"kind": "danmaku_result", "request_id": "login1",
        "operation": "send", "result": {"outcome": "receipt_accepted"}})
    assert not received
    proper = {"kind": "danmaku_result", "request_id": "login1", "operation": "login",
              "result": {"is_login": True, "error": "none", "code": 0}}
    feed_event(service, fake_process, proper)
    feed_event(service, fake_process, proper)
    assert received == [proper]


def test_browser_disconnect_preserves_uncertain_sender_result(service, fake_process):
    fake_process.state.return_value = QProcess.ProcessState.Running
    fake_process.write.side_effect = len
    service._ready = True
    received = []
    service.danmaku_result.connect(received.append)
    assert service.request_danmaku("send", "send1", "1698225", "【这是有效的测试译文正文】")
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(1, QProcess.ExitStatus.CrashExit)
    assert received[0]["request_id"] == "send1"
    assert received[0]["result"]["outcome"] == "result_unknown"
    assert not service._danmaku_requests
