"""Isolated WebView2 host with a restricted, credential-free danmaku pipe."""
from __future__ import annotations

import argparse
import ctypes
import io
import json
import logging
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from .danmaku_protocol import (
    MIN_SEND_INTERVAL_SECONDS, login_script, presence_script, room_url,
    sanitize_check, sanitize_login, sanitize_receipt, send_script, validate_danmaku_command,
)
from .live_audio_protocol import normalize_reference_video_url

_OVERRIDES = ("WEBVIEW2_BROWSER_EXECUTABLE_FOLDER", "WEBVIEW2_RELEASE_CHANNEL_PREFERENCE",
              "WEBVIEW2_CHANNEL_SEARCH_KIND", "WEBVIEW2_RELEASE_CHANNELS",
              "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS")
_SMOKE_HOST = "fanyiji.local"


def stable_version(value: object) -> bool:
    return (isinstance(value, str) and bool(re.fullmatch(r"\d+\.\d+\.\d+\.\d+", value))
            and any(int(part) for part in value.split(".")))


def allowed_navigation(url: object) -> bool:
    if not isinstance(url, str) or len(url) > 2048 or any(ord(c) < 33 for c in url):
        return False
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == "https" and parsed.port in (None, 443)
                and parsed.username is None and parsed.password is None
                and (parsed.hostname == "bilibili.com" or
                     (parsed.hostname or "").endswith(".bilibili.com")))
    except ValueError:
        return False


def _reference_source_url(url: str) -> str:
    if not allowed_navigation(url):
        return ""
    parsed = urlsplit(url)
    if (parsed.hostname == "live.bilibili.com"
            and re.fullmatch(r"/[1-9][0-9]{0,11}/?", parsed.path)):
        return "https://live.bilibili.com/" + str(int(parsed.path.strip("/")))
    try:
        return normalize_reference_video_url(url)
    except ValueError:
        return ""


def validate_command(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("invalid_command")
    operation = value.get("op")
    if operation in ("danmaku_send", "danmaku_check", "danmaku_login"):
        return validate_danmaku_command(value)
    if operation in ("show", "refresh", "close", "version") and set(value) == {"op"}:
        return value
    if operation == "navigate" and set(value) == {"op", "url"} and allowed_navigation(value["url"]):
        parsed = urlsplit(value["url"])
        if not parsed.query and not parsed.fragment and (
            (parsed.hostname == "passport.bilibili.com" and parsed.path == "/login") or
            (parsed.hostname == "live.bilibili.com" and re.fullmatch(r"/[1-9]\d{0,19}", parsed.path))
        ):
            return value
        try:
            if normalize_reference_video_url(value["url"]) == value["url"]:
                return value
        except ValueError:
            pass
    raise ValueError("invalid_command")


def _pipe(stream, identifier: int, mode: str):
    if stream is not None:
        return stream
    # Windowed PyInstaller executables expose inherited pipe handles without sys.std*.
    import msvcrt
    from ctypes import wintypes
    getter = ctypes.windll.kernel32.GetStdHandle
    getter.argtypes, getter.restype = [wintypes.DWORD], wintypes.HANDLE
    handle = getter(identifier & 0xFFFFFFFF)
    if not handle or handle == ctypes.c_void_p(-1).value:
        raise OSError("missing_pipe")
    descriptor = msvcrt.open_osfhandle(handle, os.O_BINARY | (os.O_RDONLY if mode == "r" else os.O_WRONLY))
    return io.TextIOWrapper(os.fdopen(descriptor, mode + "b", buffering=0), encoding="utf-8")


class _Host:
    def __init__(self, window, source, sink, action, environment_type, smoke: bool):
        self.window, self.source, self.sink = window, source, sink
        self.action, self.environment_type, self.smoke = action, environment_type, smoke
        self.lock, self.native_ready, self.stopped = threading.Lock(), threading.Event(), threading.Event()
        self.initialized = False
        self.smoke_complete = False
        self.failure = False
        self.danmaku_queue = queue.Queue(maxsize=32)
        self.danmaku_thread = None
        self.danmaku_worker_lock = threading.Lock()
        self.next_send_at = 0.0
        self.navigation_generation = 0
        self._source_navigation_id = None
        self._source_document_pending = False
        self._source_url = ""
        self.transport_uncertain = False

    def emit(self, kind: str, **fields) -> None:
        try:
            with self.lock:
                self.sink.write(json.dumps({"kind": kind, **fields}, ensure_ascii=False) + "\n")
                self.sink.flush()
        except (OSError, ValueError):
            self.stopped.set()

    def error(self, code: str) -> None:
        self.failure = True
        self.emit("error", code=code)

    def dispatch(self, operation) -> None:
        if self.native_ready.wait(20) and not self.window.native.IsDisposed:
            try:
                self.window.native.BeginInvoke(self.action(operation))
            except Exception:
                self.error("browser_unavailable")

    def close(self) -> None:
        self.stopped.set()
        self.dispatch(lambda: self.window.native.Close())

    def on_closed(self, *_args) -> None:
        self.stopped.set()

    def join_worker(self) -> None:
        if self.danmaku_thread is not None and self.danmaku_thread is not threading.current_thread():
            self.danmaku_thread.join(timeout=2)

    def danmaku_result(self, command: dict, value: object) -> None:
        operation = command["op"].removeprefix("danmaku_")
        sanitizer = {"send": sanitize_receipt, "login": sanitize_login, "check": sanitize_check}[operation]
        self.emit("danmaku_result", request_id=command["request_id"], operation=operation,
                  result=sanitizer(value))

    def enqueue_danmaku(self, command: dict) -> None:
        from .local_features import danmaku_available
        if not danmaku_available():
            self.danmaku_result(command, {"error": "feature_unavailable"})
            return
        if self.stopped.is_set():
            self.danmaku_result(command, {"error": "window_closed"})
            return
        with self.danmaku_worker_lock:
            if self.danmaku_thread is None:
                self.danmaku_thread = threading.Thread(target=self.danmaku_worker,
                                                       name="embedded-danmaku", daemon=True)
                self.danmaku_thread.start()
        try:
            self.danmaku_queue.put_nowait(command)
        except queue.Full:
            self.danmaku_result(command, {"error": "queue_full"})

    def native_room_matches(self, room: str) -> bool:
        """Only Source access runs on the UI thread; the worker waits for the probe."""
        finished, result = threading.Event(), []
        def probe():
            try:
                native = self.window.native
                result.append(not native.IsDisposed and str(native.webview.CoreWebView2.Source) == room_url(room))
            except Exception:
                result.append(False)
            finally:
                finished.set()
        try:
            if not self.native_ready.wait(2) or self.stopped.is_set():
                return False
            self.window.native.BeginInvoke(self.action(probe))
        except Exception:
            return False
        deadline = time.monotonic() + 2
        while not finished.wait(0.05):
            if self.stopped.is_set() or time.monotonic() >= deadline:
                return False
        return bool(result) and result[0] is True

    def evaluate_danmaku(self, script: str, *, promise: bool, generation: int, is_send: bool = False) -> object:
        finished, result = threading.Event(), []
        def callback(value):
            result.append(value)
            finished.set()
        def invoke():
            try:
                value = self.window.evaluate_js(script, callback=callback) if promise else self.window.evaluate_js(script)
                if not promise:
                    callback(value)
            except Exception:
                if is_send:
                    self.transport_uncertain = True
                callback({"error": "script_unavailable"})
        # pywebview's initial native dispatch can block as a window closes. Keep it
        # separate from the bounded worker; after an ambiguous dispatch we never post again.
        deadline = time.monotonic() + 18
        threading.Thread(target=invoke, name="embedded-danmaku-js", daemon=True).start()
        while not finished.wait(0.05):
            if self.stopped.is_set():
                if is_send:
                    self.transport_uncertain = True
                return {"error": "window_closed"}
            if self.navigation_generation != generation:
                if is_send:
                    self.transport_uncertain = True
                return {"error": "navigation_changed"}
            if time.monotonic() >= deadline:
                if is_send:
                    self.transport_uncertain = True
                return {"error": "script_timeout"}
        if self.navigation_generation != generation:
            if is_send:
                self.transport_uncertain = True
            return {"error": "navigation_changed"}
        return result[0] if result else {"error": "invalid_response"}

    def process_danmaku(self, command: dict) -> None:
        from .local_features import danmaku_available
        if not danmaku_available():
            self.danmaku_result(command, {"error": "feature_unavailable"})
            return
        if self.stopped.is_set() or (self.transport_uncertain and command["op"] == "danmaku_send"):
            self.danmaku_result(command, {"error": "window_closed" if self.stopped.is_set() else "script_unavailable"})
            return
        if command["op"] == "danmaku_send":
            while time.monotonic() < self.next_send_at:
                if self.stopped.wait(min(0.1, self.next_send_at - time.monotonic())):
                    self.danmaku_result(command, {"error": "window_closed"})
                    return
        generation = self.navigation_generation
        if not self.native_room_matches(command["room"]):
            self.danmaku_result(command, {"error": "wrong_room"})
            return
        if command["op"] == "danmaku_send":
            script, promise = send_script(command["room"], command["text"]), True
        elif command["op"] == "danmaku_login":
            script, promise = login_script(command["room"]), True
        else:
            script, promise = presence_script(command["room"], command["text"]), False
        value = self.evaluate_danmaku(script, promise=promise, generation=generation,
                                      is_send=command["op"] == "danmaku_send")
        if command["op"] == "danmaku_send":
            # Waiting from completion is conservative and survives reloads and clock
            # changes; browser-side storage also gates the actual fetch start.
            self.next_send_at = time.monotonic() + MIN_SEND_INTERVAL_SECONDS
        if not self.stopped.is_set() and not self.native_room_matches(command["room"]):
            value = {"error": "navigation_changed"}
            if command["op"] == "danmaku_send":
                self.transport_uncertain = True
        self.danmaku_result(command, value)

    def danmaku_worker(self) -> None:
        while not self.stopped.is_set():
            try:
                command = self.danmaku_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.process_danmaku(command)
            except Exception:
                self.danmaku_result(command, {"error": "script_unavailable"})
            finally:
                self.danmaku_queue.task_done()

    def versions(self, kind: str = "version") -> None:
        current = str(self.window.native.webview.CoreWebView2.Environment.BrowserVersionString)
        available = str(self.environment_type.GetAvailableBrowserVersionString())
        if not stable_version(current) or not stable_version(available):
            self.error("unsupported_runtime_channel")
            self.window.native.Close()
            return
        self.emit(kind, active_version=current, available_version=available)

    def ready(self, *_args) -> None:
        control = self.window.native.webview
        if self.initialized or control.CoreWebView2 is None:
            return
        self.initialized = True
        core = control.CoreWebView2
        self.versions("ready")
        if self.failure:
            return
        core.Environment.NewBrowserVersionAvailable += self.update_available
        core.SourceChanged += self.source_changed
        control.NavigationStarting += self.navigation_starting
        control.NavigationCompleted += self.navigation_completed
        if self.smoke:
            from Microsoft.Web.WebView2.Core import CoreWebView2HostResourceAccessKind
            core.SetVirtualHostNameToFolderMapping(_SMOKE_HOST, self.profile, CoreWebView2HostResourceAccessKind.DenyCors)
            core.Navigate("https://" + _SMOKE_HOST + "/browser-smoke.html")

    def update_available(self, *_args) -> None:
        try:
            self.versions("update_available")
        except Exception:
            self.emit("error", code="version_probe_failed")

    def navigation_starting(self, _sender, args) -> None:
        if self.stopped.is_set():
            return
        url = str(args.Uri)
        if url != "about:blank" and not allowed_navigation(url) and not (
            self.smoke and url == "https://" + _SMOKE_HOST + "/browser-smoke.html"
        ):
            args.Cancel = True
            self.emit("error", code="navigation_blocked")
        else:
            self.navigation_generation += 1
            self._source_navigation_id = getattr(args, "NavigationId", None)
            self._source_document_pending = True
            self._source_url = ""
            if not self.smoke:
                self.emit("live_source", state="navigating", room="", generation=self.navigation_generation)

    def navigation_completed(self, _sender, args) -> None:
        if self.smoke or self.stopped.is_set():
            return
        if (not self._source_document_pending
                or getattr(args, "NavigationId", None) != self._source_navigation_id):
            return
        self._source_document_pending = False
        self._publish_source(bool(args.IsSuccess))

    def source_changed(self, _sender, args) -> None:
        if self.smoke or self.stopped.is_set() or self._source_document_pending:
            return
        try:
            canonical = _reference_source_url(str(self.window.native.webview.CoreWebView2.Source))
        except Exception:
            canonical = ""
        if canonical == self._source_url:
            # Fragment/query-only changes keep the same canonical audio source.
            # No raw query enters the source pipe.
            return
        self.navigation_generation += 1
        self._source_navigation_id = None
        self._source_url = ""
        self.emit("live_source", state="navigating", room="", generation=self.navigation_generation)
        if bool(getattr(args, "IsNewDocument", True)):
            # A full document must still complete its matching navigation;
            # SourceChanged fires before the document finishes loading.
            self._source_document_pending = True
            return
        self._publish_source(True)

    def _publish_source(self, success: bool) -> None:
        room = ""
        video_url = ""
        browser_pid = 0
        try:
            core = self.window.native.webview.CoreWebView2
            url = _reference_source_url(str(core.Source)) if success else ""
            if url:
                if url.startswith("https://live.bilibili.com/"):
                    room = url.rsplit("/", 1)[-1]
                else:
                    video_url = url
                # The runtime can reuse an existing browser process for this
                # profile. The helper's own descendant tree is then incomplete.
                # This native property never comes from page JavaScript.
                raw_pid = core.BrowserProcessId
                if isinstance(raw_pid, (bool, float, str, bytes)):
                    raise ValueError("invalid_native_browser_pid")
                browser_pid = int(raw_pid)
                if not 0 < browser_pid <= 0xFFFFFFFF:
                    raise ValueError("invalid_native_browser_pid")
        except Exception:
            room, video_url, browser_pid = "", "", 0
        self._source_url = "https://live.bilibili.com/" + room if room else video_url
        if room:
            self.emit("live_source", state="ready", room=room, generation=self.navigation_generation,
                      browser_process_id=browser_pid)
        elif video_url:
            # A reference video has no actual live room. Registration chooses
            # its target room separately, rather than pretending the page is live.
            self.emit("live_source", state="ready", room="", generation=self.navigation_generation,
                      browser_process_id=browser_pid, source_kind="video", source_url=video_url)
        else:
            self.emit("live_source", state="unavailable", room="", generation=self.navigation_generation)

    def before_show(self) -> None:
        self.native_ready.set()
        native = self.window.native
        if not hasattr(native.webview, "CoreWebView2"):
            self.error("unsupported_browser_engine")
            native.BeginInvoke(self.action(native.Close))
            return
        native.webview.CoreWebView2InitializationCompleted += self.initialization_completed
        if native.webview.CoreWebView2 is not None:
            self.ready()
        if self.stopped.is_set():
            native.BeginInvoke(self.action(native.Close))

    def initialization_completed(self, _sender, args) -> None:
        try:
            if args.IsSuccess:
                self.ready()
                return
        except Exception:
            pass
        if not self.stopped.is_set():
            self.error("browser_initialization_failed")
            self.window.native.Close()

    def execute(self, command: dict) -> None:
        try:
            native = self.window.native
            if command["op"].startswith("danmaku_"):
                self.enqueue_danmaku(command)
                return
            if command["op"] == "close":
                self.stopped.set()
                native.Close()
            elif self.stopped.is_set():
                return
            elif command["op"] == "show":
                native.Show()
                native.Activate()
            elif command["op"] == "version":
                self.versions()
            elif command["op"] == "refresh":
                native.webview.CoreWebView2.Reload()
            else:
                native.webview.CoreWebView2.Navigate(command["url"])
        except Exception:
            self.emit("error", code="command_failed")

    def read_commands(self) -> None:
        try:
            while not self.stopped.is_set():
                line = self.source.readline(65537)
                if not line:
                    break
                try:
                    command = validate_command(json.loads(line))
                    if command["op"].startswith("danmaku_"):
                        self.enqueue_danmaku(command)
                    else:
                        self.dispatch(lambda command=command: self.execute(command))
                except (ValueError, TypeError):
                    self.emit("error", code="invalid_command")
        except (OSError, ValueError):
            pass
        finally:
            self.close()

    def smoke_loaded(self) -> None:
        if not self.initialized or self.window.get_current_url() != "https://" + _SMOKE_HOST + "/browser-smoke.html":
            return
        try:
            result = self.window.evaluate_js("(() => {const key='fanyiji_browser_smoke';"
                "const previous=Number(localStorage.getItem(key)||0);"
                "localStorage.setItem(key,String(previous+1));return {previous,current:previous+1};})()")
            self.emit("smoke", **result)
            self.smoke_complete = True
        except Exception:
            self.error("browser_storage_failed")
        finally:
            self.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="翻译姬 · 直播浏览器")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--language", choices=("zh-CN", "ja", "en"), default="zh-CN")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    source, sink = _pipe(sys.stdin, -10, "r"), _pipe(sys.stdout, -11, "w")
    def fail(code):
        sink.write(json.dumps({"kind": "error", "code": code}) + "\n")
        sink.flush()
        return 1
    if os.name != "nt":
        return fail("unsupported_platform")
    if any(os.environ.get(name) for name in _OVERRIDES):
        return fail("runtime_override_blocked")
    try:
        import webview
        import clr
        from webview.util import interop_dll_path
        clr.AddReference(interop_dll_path("Microsoft.Web.WebView2.Core.dll"))
        from Microsoft.Web.WebView2.Core import CoreWebView2Environment
        from System import Action
        if not stable_version(str(CoreWebView2Environment.GetAvailableBrowserVersionString())):
            return fail("unsupported_runtime_channel")
    except ImportError:
        return fail("browser_dependency_missing")
    except Exception:
        return fail("browser_runtime_missing")
    # Third-party browser diagnostics never share the structured status pipe.
    logging.getLogger("pywebview").disabled = True
    sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")
    profile = args.profile.resolve()
    try:
        profile.mkdir(parents=True, exist_ok=True)
        if args.smoke_test:
            (profile / "browser-smoke.html").write_text("<!doctype html><title>翻译姬浏览器测试</title>", encoding="utf-8")
        webview.settings["ALLOW_FILE_URLS"] = False
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
        window = webview.create_window("翻译姬 · 直播浏览器", "about:blank", width=1000, height=700,
                                      min_size=(560, 420), hidden=args.smoke_test, text_select=True)
        host = _Host(window, source, sink, Action, CoreWebView2Environment, args.smoke_test)
        host.profile = str(profile)
        window.events.before_show += host.before_show
        window.events.closed += host.on_closed
        if args.smoke_test:
            window.events.loaded += host.smoke_loaded
        else:
            threading.Thread(target=host.read_commands, daemon=True).start()
        def timeout():
            if not host.initialized or (host.smoke and not host.smoke_complete):
                host.error("browser_initialization_timeout")
                host.close()
        timer = threading.Timer(30, timeout)
        timer.daemon = True
        timer.start()
        webview.start(gui="edgechromium", private_mode=False, storage_path=str(profile), debug=False)
        timer.cancel()
        host.stopped.set()
        host.join_worker()
        host.emit("closed")
        return int(host.failure)
    except Exception:
        return fail("browser_start_failed")


if __name__ == "__main__":
    raise SystemExit(main())
