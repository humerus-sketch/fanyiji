"""Optional WebView2 child process; never initializes the browser during ASR startup."""
from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, QTimer, Signal, Slot

from .browser_runtime import runtime_snapshot
from .paths import data_root
from .live_audio_protocol import normalize_reference_video_url


def room_url(value: str) -> str:
    value = value.strip()
    prefix = "https://live.bilibili.com/"
    if value.startswith(prefix):
        value = value[len(prefix):].split("?", 1)[0].rstrip("/")
    if not value.isascii() or not value.isdigit() or not 1 <= len(value) <= 12 or int(value) <= 0:
        raise ValueError("invalid_room")
    return prefix + str(int(value))


class BrowserService(QObject):
    event = Signal(dict)
    snapshot = Signal(dict)
    prefs_changed = Signal(dict)
    danmaku_result = Signal(dict)
    danmaku_status = Signal(dict)
    danmaku_resume_requested = Signal()
    danmaku_clear_requested = Signal()
    content_filter_requested = Signal()
    live_source_changed = Signal(dict)
    _check_finished = Signal(dict)
    _open_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.profile = data_root() / "browser" / "default"
        self.preferences_path = data_root() / "browser-preferences.json"
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_events)
        # Third-party URLs and exceptions can contain private data. Do not log stderr.
        self.process.readyReadStandardError.connect(lambda: self.process.readAllStandardError())
        self.process.errorOccurred.connect(self._process_error)
        self.process.started.connect(self._process_started)
        self.process.finished.connect(self._finished)
        self._buffer = b""
        self._ready = False
        self._launching = False
        self._stopping = False
        self._generation = 0
        self._live_source = {"state": "unavailable", "room": "", "process_id": 0,
                             "generation": 0, "browser_generation": 0}
        self.active_version = None
        self._pending_url = None
        self._current_url = ""
        self._pending_language = "zh-CN"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-update-check")
        self._checking = False
        self._closed = False
        self._check_finished.connect(self._on_check_finished)
        self._open_requested.connect(self.open)
        self.auto_check = True
        self.room = ""
        from .local_features import danmaku_available
        self.danmaku_available = danmaku_available()
        self.danmaku_enabled = False
        self.danmaku_repair_enabled = False
        self.danmaku_source = "system"
        self.danmaku_interval_seconds = 3.5
        self.danmaku_validation_room = ""
        self.danmaku_snapshot = {}
        self._has_danmaku_preferences = False
        self._danmaku_requests = {}
        self._danmaku_commands = {}
        self._unknown_submissions = {}
        self._retired_submissions = set()
        self.danmaku_status.connect(self._cache_danmaku_status)
        try:
            preferences = json.loads(self.preferences_path.read_text(encoding="utf-8"))
            self.auto_check = preferences.get("auto_check", True) is not False
            if isinstance(preferences.get("room"), str) and preferences["room"]:
                self.room = room_url(preferences["room"]).rsplit("/", 1)[-1]
            if preferences.get("danmaku_source") in ("system", "mic", "all"):
                self.danmaku_source = preferences["danmaku_source"]
            interval = preferences.get("danmaku_interval_seconds", 3.5)
            if type(interval) in (int, float) and math.isfinite(interval):
                self.danmaku_interval_seconds = min(60.0, max(3.5, float(interval)))
            self.danmaku_enabled = preferences.get("danmaku_enabled") is True and self.danmaku_available
            self.danmaku_repair_enabled = preferences.get("danmaku_repair_enabled") is True and self.danmaku_available
            self._has_danmaku_preferences = ("danmaku_enabled" in preferences
                                            or "danmaku_repair_enabled" in preferences)
            validation_room = preferences.get("danmaku_validation_room", "")
            if isinstance(validation_room, str) and validation_room.strip():
                try:
                    normalized_validation = room_url(validation_room).rsplit("/", 1)[-1]
                    if normalized_validation != self.room:
                        self.danmaku_validation_room = normalized_validation
                except ValueError:
                    pass
        except (OSError, ValueError, AttributeError):
            pass
        self.timer = QTimer(self)
        self.timer.setInterval(60 * 60 * 1000)
        self.timer.timeout.connect(self.check_updates)
        self.startup_timer = QTimer(self)
        self.startup_timer.setSingleShot(True)
        self.startup_timer.setInterval(45000)
        self.startup_timer.timeout.connect(self._startup_timeout)

    def save_preferences(self, room: str, auto_check: bool, *, danmaku_enabled=None,
                         danmaku_source=None, danmaku_interval_seconds=None,
                         danmaku_validation_room=None, danmaku_repair_enabled=None):
        normalized_room = room_url(room).rsplit("/", 1)[-1] if room.strip() else ""
        enabled = self.danmaku_enabled if danmaku_enabled is None else danmaku_enabled
        repair_enabled = self.danmaku_repair_enabled if danmaku_repair_enabled is None else danmaku_repair_enabled
        source = self.danmaku_source if danmaku_source is None else danmaku_source
        interval = self.danmaku_interval_seconds if danmaku_interval_seconds is None else danmaku_interval_seconds
        validation_room = self.danmaku_validation_room if danmaku_validation_room is None else danmaku_validation_room
        if not isinstance(validation_room, str):
            raise ValueError("invalid_validation_room")
        normalized_validation = room_url(validation_room).rsplit("/", 1)[-1] if validation_room.strip() else ""
        if normalized_validation and normalized_validation == normalized_room:
            raise ValueError("same_validation_room")
        if (type(enabled) is not bool or type(repair_enabled) is not bool or source not in ("system", "mic", "all")
                or type(interval) not in (int, float) or not math.isfinite(interval)
                or not 3.5 <= interval <= 60):
            raise ValueError("invalid_danmaku_settings")
        if enabled and (not normalized_room or not self.danmaku_available):
            raise ValueError("danmaku_unavailable")
        if repair_enabled and not self.danmaku_available:
            raise ValueError("danmaku_unavailable")
        persist_danmaku = (self._has_danmaku_preferences or danmaku_enabled is not None
                          or danmaku_validation_room is not None or danmaku_repair_enabled is not None)
        value = {"room": normalized_room, "auto_check": bool(auto_check)}
        if persist_danmaku:
            value.update(danmaku_enabled=enabled, danmaku_source=source,
                         danmaku_interval_seconds=float(interval),
                         danmaku_validation_room=normalized_validation,
                         danmaku_repair_enabled=repair_enabled)
        self.preferences_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.preferences_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.preferences_path)
        self.room, self.auto_check = normalized_room, bool(auto_check)
        self.danmaku_enabled, self.danmaku_source = enabled, source
        self.danmaku_repair_enabled = repair_enabled
        self.danmaku_interval_seconds = float(interval)
        self.danmaku_validation_room = normalized_validation
        self._has_danmaku_preferences = persist_danmaku
        self.timer.start() if self.auto_check else self.timer.stop()
        self.prefs_changed.emit(self.danmaku_preferences())

    def danmaku_preferences(self):
        return {"enabled": self.danmaku_enabled, "room": self.room,
                "source": self.danmaku_source, "interval_seconds": self.danmaku_interval_seconds,
                "validation_room": self.danmaku_validation_room,
                "repair_enabled": self.danmaku_repair_enabled}

    def set_danmaku_repair_enabled(self, enabled: bool):
        """Persist a deliberate repair action; only starting opens its test room."""
        if type(enabled) is not bool:
            raise ValueError("invalid_danmaku_settings")
        if enabled and not self.danmaku_validation_room:
            raise ValueError("validation_room_required")
        self.save_preferences(self.room, self.auto_check, danmaku_repair_enabled=enabled)
        if enabled:
            from .i18n import get_language
            self.open(room_url(self.danmaku_validation_room), get_language())

    @property
    def is_browser_ready(self):
        return self._ready and not self._stopping and not self._closed

    def live_source_snapshot(self):
        return dict(self._live_source)

    def _clear_live_source(self):
        self._live_source = {"state": "unavailable", "room": "", "process_id": 0,
                             "generation": 0, "browser_generation": self._generation}
        self.live_source_changed.emit(dict(self._live_source))

    def _read_live_source(self, event):
        basic_keys = {"kind", "state", "room", "generation"}
        pid_keys = basic_keys | {"browser_process_id"}
        reference_keys = pid_keys | {"source_kind", "source_url"}
        if (set(event) not in (basic_keys, pid_keys, reference_keys)
                or not isinstance(event["state"], str)
                or event["state"] not in {"ready", "navigating", "unavailable"}
                or type(event["generation"]) is not int or event["generation"] < 0
                or not isinstance(event["room"], str) or not self.is_browser_ready):
            return
        room = event["room"]
        if event["state"] == "ready":
            # Only the local helper's native WebView2 browser identity is usable.
            # A helper PID fallback may miss a reused WebView2 runtime entirely.
            pid = event.get("browser_process_id")
            if set(event) not in (pid_keys, reference_keys) or type(pid) is not int or not 0 < pid <= 0xFFFFFFFF:
                return
            if set(event) == reference_keys:
                if event["source_kind"] != "video" or room:
                    return
                try:
                    video_url = normalize_reference_video_url(event["source_url"])
                except ValueError:
                    return
                if video_url != event["source_url"]:
                    return
            else:
                try:
                    room = room_url(room).rsplit("/", 1)[-1]
                except (ValueError, TypeError):
                    return
                if room != event["room"]:
                    return
        else:
            if set(event) == reference_keys or room or ("browser_process_id" in event
                        and (type(event["browser_process_id"]) is not int or event["browser_process_id"] != 0)):
                return
            pid = 0
        if (self._live_source["browser_generation"] == self._generation
                and event["generation"] < self._live_source["generation"]):
            return
        self._live_source = {"state": event["state"], "room": room, "process_id": pid,
                             "generation": event["generation"], "browser_generation": self._generation}
        if set(event) == reference_keys:
            self._live_source.update(source_kind="video", source_url=video_url)
        if event["state"] == "ready":
            self._current_url = video_url if set(event) == reference_keys else room_url(room)
        self.live_source_changed.emit(dict(self._live_source))

    def _cache_danmaku_status(self, value):
        self.danmaku_snapshot = dict(value)

    def resume_danmaku(self):
        self.danmaku_resume_requested.emit()

    def clear_danmaku_queue(self):
        if self.danmaku_available and not self._closed:
            self.danmaku_clear_requested.emit()

    def request_content_filter(self):
        if self.danmaku_available and not self._closed:
            self.content_filter_requested.emit()

    def request_danmaku(self, operation, request_id, room, text="") -> bool:
        from .danmaku_protocol import validate_danmaku_command
        if operation not in ("send", "check", "login") or not self.danmaku_available:
            return False
        command = {"op": "danmaku_" + operation, "request_id": request_id, "room": room}
        if operation != "login":
            command["text"] = text
        try:
            validate_danmaku_command(command)
        except (ValueError, TypeError):
            return False
        if not self.is_browser_ready or request_id in self._danmaku_requests:
            return False
        if operation == "send" and ((room, text) in self._unknown_submissions
                                     or (room, text) in self._retired_submissions):
            # An unknown old payload may still appear later. Never POST it again
            # merely because its failed FIFO slot was handed to rule management.
            self.danmaku_result.emit({"kind": "danmaku_result", "request_id": request_id,
                "operation": "send", "result": {"outcome": "not_sent", "error": "script_unavailable"}})
            return True
        if operation == "send" and any(key not in self._retired_submissions
                and value["generation"] == self._generation for key, value in self._unknown_submissions.items()):
            return False
        self._danmaku_requests[request_id] = operation
        self._danmaku_commands[request_id] = {"operation": operation, "room": room, "text": text,
                                               "generation": self._generation}
        if not self._send(command):
            self._danmaku_requests.pop(request_id, None)
            self._danmaku_commands.pop(request_id, None)
            return False
        return True

    def retire_danmaku_submission(self, room, text):
        """After durable handoff, fence the old text and retire a tainted helper."""
        key = (room, text)
        record = self._unknown_submissions.get(key)
        pending = next((value for value in self._danmaku_commands.values()
                        if value["operation"] == "send" and (value["room"], value["text"]) == key), None)
        if record is None and pending is None:
            return False
        self._retired_submissions.add(key)
        generation = (pending or record)["generation"]
        unsettled = pending is not None or record["unsettled"]
        if (unsettled and generation == self._generation and not self._closed
                and not self._stopping and self.process.state() != QProcess.ProcessState.NotRunning
                and self._current_url):
            # Reuse the existing profile and current owned page. .finished is
            # the sole restart point, after the tainted old helper has exited.
            self._pending_url = self._current_url
            self.event.emit({"kind": "transport_recovering"})
            self._begin_stop()
        return True

    def _observe_danmaku_submission(self, event, command):
        if command is None:
            return
        key = (command["room"], command["text"])
        result = event["result"]
        if command["operation"] == "send" and result.get("outcome") == "result_unknown":
            self._unknown_submissions[key] = {"generation": command["generation"],
                "before": result.get("before_count"),
                "unsettled": result.get("error") in {"script_timeout", "script_unavailable", "navigation_changed", "window_closed"}}
        elif command["operation"] == "check":
            record = self._unknown_submissions.get(key)
            before, count = (record or {}).get("before"), result.get("occurrence_count")
            if (record is not None and type(before) is int and type(count) is int and count > before
                    and result.get("room_matches") is True and result.get("chat_ready") is True
                    and result.get("present") is True and result.get("error") in {"none", "", None}):
                self._unknown_submissions.pop(key, None)
                self._retired_submissions.discard(key)

    def _lose_danmaku_requests(self):
        requests, self._danmaku_requests = self._danmaku_requests, {}
        for request_id, operation in requests.items():
            command = self._danmaku_commands.pop(request_id, None)
            if command is not None and operation == "send":
                self._unknown_submissions[(command["room"], command["text"])] = {
                    "generation": command["generation"], "before": None, "unsettled": True}
            self.danmaku_result.emit({"kind": "danmaku_result", "request_id": request_id,
                "operation": operation, "result": {"outcome": "result_unknown", "error": "browser_closed"}})

    def activate_checks(self):
        if self.auto_check and not self._closed:
            self.timer.start()
            self.check_updates()

    def check_updates(self):
        if self._closed or self._checking:
            return
        self._checking = True
        self.event.emit({"kind": "checking"})
        active = self.active_version
        self._send({"op": "version"})
        future = self._executor.submit(runtime_snapshot, active, True)
        def finished(result):
            try:
                snapshot = result.result()
            except Exception:
                snapshot = {"state": "check_unavailable"}
            if not self._closed:
                self._check_finished.emit(snapshot)
        future.add_done_callback(finished)

    def _on_check_finished(self, snapshot):
        self._checking = False
        if not self._closed:
            # A helper can close/restart while the network check is in flight.
            # Apply downloaded metadata to the current process and local install.
            self.snapshot.emit(runtime_snapshot(self.active_version, False,
                catalog_version=snapshot.get("catalog_version"),
                check_error=snapshot.get("check_error") or (
                    "检查未完成" if snapshot.get("state") == "check_unavailable" else None)))

    @Slot(str, str)
    def open(self, url: str, language="zh-CN"):
        if QThread.currentThread() != self.thread():
            self._open_requested.emit(url, language)
            return
        if self._closed:
            return
        if url != "https://passport.bilibili.com/login":
            try:
                url = normalize_reference_video_url(url)
            except ValueError:
                url = room_url(url)
        self._pending_url = url
        self._current_url = url
        self._pending_language = language
        # A close is asynchronous: retain the latest open request until the old
        # helper has actually exited, rather than navigating its dying window.
        if self._stopping:
            return
        if self._launching or self.process.state() != QProcess.ProcessState.NotRunning:
            if self._ready:
                self._send({"op": "navigate", "url": url})
                self._send({"op": "show"})
                self._pending_url = None
            return
        self._start_browser()

    def open_reference_video(self, url: str, language: str = "zh-CN"):
        # Reuse the current browser and its profile; this never changes the
        # saved live-room or formal danmaku destination preferences.
        self.open(normalize_reference_video_url(url), language)

    def _start_browser(self):
        if (self._closed or self._stopping or self._launching or not self._pending_url
                or self.process.state() != QProcess.ProcessState.NotRunning):
            return
        self._ready, self._buffer = False, b""
        try:
            self.profile.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.event.emit({"kind": "error", "code": "profile_unavailable"})
            return
        language = self._pending_language
        if getattr(sys, "frozen", False):
            arguments = ["--browser-helper", "--profile", str(self.profile), "--language", language]
        else:
            arguments = ["-m", "fanyiji.browser_helper", "--profile", str(self.profile), "--language", language]
            environment = QProcessEnvironment.systemEnvironment()
            source = str(Path(__file__).resolve().parents[1])
            environment.insert("PYTHONPATH", source + os.pathsep + environment.value("PYTHONPATH"))
            self.process.setProcessEnvironment(environment)
        self.process.setProgram(sys.executable)
        self.process.setArguments(arguments)
        self._generation += 1
        self._launching = True
        self.process.start()
        if self._launching:
            self.startup_timer.start()
            self.event.emit({"kind": "starting"})

    def refresh(self):
        if not self._stopping:
            self._send({"op": "refresh"})

    def stop_browser(self):
        self._pending_url = None
        self._begin_stop()

    def _begin_stop(self):
        if self._stopping or (not self._launching
                and self.process.state() == QProcess.ProcessState.NotRunning):
            return
        self._stopping = True
        self._clear_live_source()
        self._ready = False
        self.startup_timer.stop()
        self._send({"op": "close"})
        generation = self._generation
        def still_stopping():
            return (not self._closed and self._stopping and self._generation == generation
                    and self.process.state() != QProcess.ProcessState.NotRunning)
        def force_close():
            if still_stopping():
                self.process.kill()
        def terminate():
            if still_stopping():
                self.process.terminate()
                QTimer.singleShot(1000, force_close)
        QTimer.singleShot(3000, terminate)

    def _process_started(self):
        if self._stopping:
            self._send({"op": "close"})

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.startup_timer.stop()
            self._ready = self._launching = self._stopping = False
            self._pending_url = None
            self.active_version = None
            self._clear_live_source()
        self.event.emit({"kind": "error", "code": "process_failed"})

    def _send(self, command):
        if self.process.state() == QProcess.ProcessState.Running and (self._ready or command.get("op") == "close"):
            payload = (json.dumps(command, ensure_ascii=True) + "\n").encode("utf-8")
            return self.process.write(payload) == len(payload)
        return False

    def _read_events(self):
        self._buffer += bytes(self.process.readAllStandardOutput())
        if len(self._buffer) > 65536:
            self._buffer = b""
            self.event.emit({"kind": "error", "code": "invalid_protocol"})
            return
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(event, dict) or event.get("kind") not in {
                "ready", "version", "closed", "error", "update_available", "danmaku_result", "live_source"}:
                continue
            if event["kind"] == "live_source":
                self._read_live_source(event)
                continue
            if event["kind"] == "danmaku_result":
                request_id = event.get("request_id")
                if (request_id not in self._danmaku_requests
                        or event.get("operation") != self._danmaku_requests[request_id]
                        or not isinstance(event.get("result"), dict)):
                    continue
                self._danmaku_requests.pop(request_id, None)
                command = self._danmaku_commands.pop(request_id, None)
                self._observe_danmaku_submission(event, command)
                self.danmaku_result.emit(event)
                continue
            if self._stopping and event["kind"] in ("ready", "version", "update_available"):
                self._send({"op": "close"})
                continue
            if event["kind"] in ("ready", "version"):
                self.active_version = event.get("active_version")
            elif event["kind"] == "closed":
                if not self._stopping:
                    self._pending_url = None
                self._begin_stop()
                self.active_version = None
            if event["kind"] == "ready":
                self.startup_timer.stop()
                self._launching = False
                self._ready = True
                if self._pending_url:
                    self._send({"op": "navigate", "url": self._pending_url})
                    self._pending_url = None
            self.event.emit(event)

    def _finished(self, exit_code, _status):
        was_ready, was_stopping = self._ready, self._stopping
        self.startup_timer.stop()
        self._ready = self._launching = self._stopping = False
        self.active_version = None
        self._buffer = b""
        self._clear_live_source()
        self._lose_danmaku_requests()
        if not was_stopping:
            self._pending_url = None
        self.event.emit({"kind": "closed"} if was_ready or was_stopping or exit_code == 0 else
                        {"kind": "error", "code": "startup_failed"})
        # QProcess.finished is emitted only after the old process exits. This is
        # the sole restart point, so rapid close/open clicks cannot overlap hosts.
        self._start_browser()

    def _startup_timeout(self):
        if not self._ready and self.process.state() != QProcess.ProcessState.NotRunning:
            self.event.emit({"kind": "error", "code": "startup_timeout"})
            self._pending_url = None
            self._stopping = True
            self._clear_live_source()
            timed_out_pid = self.process.processId()
            generation = self._generation
            self.process.terminate()
            def force_close():
                if (not self._closed and self._stopping and self._generation == generation
                        and self.process.processId() == timed_out_pid
                        and self.process.state() != QProcess.ProcessState.NotRunning):
                    self.process.kill()
            QTimer.singleShot(1000, force_close)

    def close(self):
        self._closed = True
        self._clear_live_source()
        self._pending_url = None
        self.timer.stop()
        self.startup_timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self._send({"op": "close"})
            self.process.closeWriteChannel()
            if not self.process.waitForFinished(3000):
                self.process.terminate()
                if not self.process.waitForFinished(1000):
                    self.process.kill()
                    self.process.waitForFinished(1000)
        self._lose_danmaku_requests()
