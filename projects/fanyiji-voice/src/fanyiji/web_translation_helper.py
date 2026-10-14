"""A hidden, single-flight WebView2 host for supported translation webpages.

The parent supplies text and language codes, never arbitrary script or URLs.
Only the fixed DOM scripts are evaluated; no private website API is called.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time

from .browser_helper import _OVERRIDES, _pipe, stable_version
from .web_translation_engine import _safe_diagnostic
from .web_translation_protocol import (
    allowed_navigation, poll_script, prepare_script, provider_url, validate_command, validated_result,
)

_SMOKE_HOST = "fanyiji.translation.local"
_SMOKE_URL = "https://" + _SMOKE_HOST + "/translation-smoke.html"
_INTERACTION_CODES = {"interaction_required", "page_changed", "language_mismatch", "input_unavailable",
                      "translation_timeout", "source_changed", "invalid_response"}


def _diagnostic_script(command: dict, provider: str) -> str:
    payload = json.dumps({"text": command["text"], "source_lang": command["source_lang"],
                          "target_lang": command["target_lang"]}, ensure_ascii=True)
    return r"""(() => {
      const c=__COMMAND__, google=__GOOGLE__;
      const norm=s=>String(s||'').replace(/\s+/g,' ').trim();
      const visible=e=>!!e && e.getClientRects().length>0;
      const input=document.querySelector(google?'textarea[jsname="BJE2fc"]':'#js_fanyi_input');
      const output=document.querySelector(google?'[jsname="W297wb"]':'#js_fanyi_output_resultOutput');
      const controls=google?Array.from(document.querySelectorAll('[role="tablist"]')).filter(visible).slice(0,2)
        :[document.querySelector('.lanFrom-container'),document.querySelector('.lanTo-container')];
      const selected=e=>google?e?.querySelector('[aria-selected="true"]')?.getAttribute('data-language-code'):norm(e?.innerText);
      const source=google?(c.source_lang==='zh'?'zh-CN':'ja'):(c.source_lang==='zh'?'中文':'日语');
      const target=google?(c.target_lang==='zh'?'zh-CN':'ja'):(c.target_lang==='zh'?'中文':'日语');
      const phase=window.__fanyijiWebTranslation?.phase;
      return {input_present:visible(input),output_present:visible(output),
        source_matches:!!input && norm(google?input.value:input.innerText)===norm(c.text),
        language_pair_selected:selected(controls[0])===source && selected(controls[1])===target,
        document_hidden:document.hidden===true,
        phase:['clearing','languages','waiting'].includes(phase)?phase:'unknown',
        ready_state:['loading','interactive','complete'].includes(document.readyState)?document.readyState:'unknown'};
    })()""".replace("__GOOGLE__", "true" if provider == "google_web" else "false").replace("__COMMAND__", payload)


class _Host:
    def __init__(self, window, source, sink, action, environment_type, provider: str, smoke: bool = False):
        self.window, self.source, self.sink = window, source, sink
        self.action, self.environment_type = action, environment_type
        self.provider, self.smoke = provider, smoke
        self.output_lock = threading.Lock()
        self.worker_lock = threading.Lock()
        self.native_ready, self.page_ready, self.stopped = threading.Event(), threading.Event(), threading.Event()
        self.initialized = self.failure = self.smoke_complete = False
        self._worker = None
        self._busy = False
        self._translation_url = ""
        self._navigation_failed = False
        self._navigation_id = None
        self._awaiting_navigation_start = False
        self._shown = False
        self._queue = queue.Queue(maxsize=1)

    def emit(self, kind: str, **fields) -> None:
        try:
            with self.output_lock:
                self.sink.write(json.dumps({"kind": kind, **fields}, ensure_ascii=True) + "\n")
                self.sink.flush()
        except (OSError, ValueError):
            self.stopped.set()

    def dispatch(self, operation) -> bool:
        if not self.native_ready.wait(3) or self.stopped.is_set():
            return False
        try:
            if self.window.native.IsDisposed:
                return False
            self.window.native.BeginInvoke(self.action(operation))
            return True
        except Exception:
            return False

    def close(self) -> None:
        self.stopped.set()
        # Closing must still dispatch after setting the stop fence.
        if self.native_ready.is_set():
            try:
                self.window.native.BeginInvoke(self.action(self.window.native.Close))
            except Exception:
                pass

    def on_closed(self, *_args) -> None:
        self.stopped.set()

    def show(self) -> None:
        def show_native():
            self.window.native.Show()
            self.window.native.Activate()
            self._shown = True
        self.dispatch(show_native)

    def show_for_translation(self) -> None:
        if self.provider != "google_web" or self._shown:
            return
        # Google does not reliably translate a hidden WebView2 document. Show
        # once without repeatedly activating it over the user's live window.
        self._shown = True
        if not self.dispatch(self.window.native.Show):
            self._shown = False

    def result(self, command: dict, value: object) -> dict:
        value = validated_result(value, command)
        if value.get("status") == "error" and value.get("code") == "translation_timeout":
            diagnostic = _safe_diagnostic(self.evaluate(_diagnostic_script(command, self.provider)))
            if diagnostic:
                self.emit("diagnostic", id=command["id"], **diagnostic)
        if value.get("status") == "error" and value.get("code") in _INTERACTION_CODES:
            self.show()
        self.emit("result", id=command["id"], result=value)
        return value

    def error_result(self, command: dict, code: str) -> None:
        self.result(command, {"id": command["id"], "status": "error", "code": code})

    def before_show(self) -> None:
        self.native_ready.set()
        native = self.window.native
        if not hasattr(native.webview, "CoreWebView2"):
            self.failure = True
            self.emit("error", code="unsupported_browser_engine")
            self.close()
            return
        native.webview.CoreWebView2InitializationCompleted += self.initialization_completed
        if native.webview.CoreWebView2 is not None:
            self.ready()

    def initialization_completed(self, _sender, args) -> None:
        if bool(getattr(args, "IsSuccess", False)):
            self.ready()
        else:
            self.failure = True
            self.emit("error", code="browser_initialization_failed")
            self.close()

    def ready(self) -> None:
        native = self.window.native
        core = native.webview.CoreWebView2
        if self.initialized or core is None:
            return
        current = str(core.Environment.BrowserVersionString)
        available = str(self.environment_type.GetAvailableBrowserVersionString())
        if not stable_version(current) or not stable_version(available):
            self.failure = True
            self.emit("error", code="unsupported_runtime_channel")
            self.close()
            return
        self.initialized = True
        native.webview.NavigationStarting += self.navigation_starting
        native.webview.NavigationCompleted += self.navigation_completed
        # No translator website is opened until show/translate is requested.
        self.emit("ready", provider=self.provider, active_version=current)
        if self.smoke:
            from Microsoft.Web.WebView2.Core import CoreWebView2HostResourceAccessKind
            core.SetVirtualHostNameToFolderMapping(_SMOKE_HOST, self.profile,
                                                  CoreWebView2HostResourceAccessKind.DenyCors)
            core.Navigate(_SMOKE_URL)

    def navigation_starting(self, _sender, args) -> None:
        url = str(args.Uri)
        if (url != "about:blank" and not allowed_navigation(self.provider, url)
                and not (self.smoke and url == _SMOKE_URL)):
            args.Cancel = True
            self._navigation_failed = True
            self.emit("error", code="navigation_blocked")
            self.show()
            return
        self.page_ready.clear()
        self._navigation_failed = False
        self._navigation_id = getattr(args, "NavigationId", None)
        self._awaiting_navigation_start = False

    def navigation_completed(self, _sender, args) -> None:
        if self.stopped.is_set():
            return
        navigation_id = getattr(args, "NavigationId", None)
        if navigation_id is not None and (self._awaiting_navigation_start
                                         or (self._navigation_id is not None and navigation_id != self._navigation_id)):
            return
        self._navigation_failed = not bool(getattr(args, "IsSuccess", False))
        self._awaiting_navigation_start = False
        self.page_ready.set()

    def navigate(self, url: str) -> bool:
        self.page_ready.clear()
        self._navigation_failed = False
        self._awaiting_navigation_start = True
        self._translation_url = url
        return self.dispatch(lambda: self.window.native.webview.CoreWebView2.Navigate(url))

    def evaluate(self, script: str) -> object:
        completed, values = threading.Event(), []
        def evaluate():
            try:
                values.append(self.window.evaluate_js(script))
            except Exception:
                values.append(None)
            finally:
                completed.set()
        # Native evaluation can hang as a window closes; never block the worker
        # forever or reuse that closed window's eventual result.
        threading.Thread(target=evaluate, name="web-translation-js", daemon=True).start()
        deadline = time.monotonic() + 5
        while not completed.wait(0.05):
            if self.stopped.is_set() or time.monotonic() >= deadline:
                return None
        return None if self.stopped.is_set() else values[0]

    def native_call(self, method: str, parameters: dict) -> bool:
        if self.provider != "google_web" or method not in ("Input.dispatchKeyEvent", "Input.insertText"):
            return False
        dispatched, values = threading.Event(), []
        def invoke():
            try:
                core = self.window.native.webview.CoreWebView2
                if not allowed_navigation(self.provider, str(core.Source)):
                    values.append(None)
                else:
                    values.append(core.CallDevToolsProtocolMethodAsync(method, json.dumps(parameters, ensure_ascii=True)))
            except Exception:
                values.append(None)
            finally:
                dispatched.set()
        if not self.dispatch(invoke):
            return False
        deadline = time.monotonic() + 5
        while not dispatched.wait(0.05):
            if self.stopped.is_set() or time.monotonic() >= deadline:
                return False
        task = values[0] if values else None
        if task is None:
            return False
        try:
            while not bool(task.IsCompleted):
                if self.stopped.wait(0.05) or time.monotonic() >= deadline:
                    return False
            if self.stopped.is_set() or bool(task.IsFaulted) or bool(task.IsCanceled):
                return False
            # Read only completion, never a page/DevTools diagnostic message.
            task.GetAwaiter().GetResult()
            return True
        except Exception:
            return False

    def native_input(self, command: dict, action: str) -> bool:
        if self.provider != "google_web" or action not in ("clear", "insert"):
            return False
        identifier = json.dumps(command["id"])
        focused = self.evaluate("(() => {const e=document.querySelector('textarea[jsname=\"BJE2fc\"]');"
                                "return !!e && document.activeElement===e && window.__fanyijiWebTranslation?.id==="
                                + identifier + ";})()")
        if focused is not True or self.stopped.is_set():
            return False
        if action == "insert":
            return self.native_call("Input.insertText", {"text": command["text"]})
        keys = (
            {"type": "rawKeyDown", "modifiers": 2, "key": "Control", "code": "ControlLeft", "windowsVirtualKeyCode": 17},
            {"type": "rawKeyDown", "modifiers": 2, "key": "a", "code": "KeyA", "windowsVirtualKeyCode": 65},
            {"type": "keyUp", "modifiers": 2, "key": "a", "code": "KeyA", "windowsVirtualKeyCode": 65},
            {"type": "keyUp", "modifiers": 0, "key": "Control", "code": "ControlLeft", "windowsVirtualKeyCode": 17},
            {"type": "rawKeyDown", "modifiers": 0, "key": "Backspace", "code": "Backspace", "windowsVirtualKeyCode": 8},
            {"type": "keyUp", "modifiers": 0, "key": "Backspace", "code": "Backspace", "windowsVirtualKeyCode": 8},
        )
        return all(self.native_call("Input.dispatchKeyEvent", key) for key in keys)

    def process_translation(self, command: dict) -> None:
        self.show_for_translation()
        url = provider_url(self.provider, command["source_lang"], command["target_lang"])
        if self._translation_url != url or self._navigation_failed:
            if not self.navigate(url):
                self.error_result(command, "page_not_ready")
                return
        deadline = time.monotonic() + 30
        while not self.page_ready.wait(0.05):
            if self.stopped.is_set():
                return
            if time.monotonic() >= deadline:
                self.error_result(command, "translation_timeout")
                return
        if self._navigation_failed:
            self.error_result(command, "page_not_ready")
            self.show()
            return
        while not self.stopped.is_set():
            # prepare is idempotent for this request: it clears stale results,
            # progresses language controls, then enters this request's source.
            prepared = validated_result(self.evaluate(prepare_script(command, self.provider)), command)
            if prepared.get("status") == "input":
                if not self.native_input(command, prepared["action"]):
                    self.error_result(command, "input_unavailable")
                    return
            if prepared.get("status") == "error" and prepared.get("code") != "page_not_ready":
                self.result(command, prepared)
                return
            if prepared.get("status") != "error":
                value = validated_result(self.evaluate(poll_script(command, self.provider)), command)
                if value.get("status") == "ok":
                    self.result(command, value)
                    return
                if value.get("status") == "error" and value.get("code") != "page_not_ready":
                    self.result(command, value)
                    return
            if time.monotonic() >= deadline:
                self.error_result(command, "translation_timeout")
                return
            self.stopped.wait(0.2)

    def enqueue(self, command: dict) -> None:
        with self.worker_lock:
            if self._busy:
                self.error_result(command, "page_not_ready")
                return
            self._busy = True
            if self._worker is None:
                self._worker = threading.Thread(target=self.worker, name="web-translation-worker", daemon=True)
                self._worker.start()
        self._queue.put_nowait(command)

    def worker(self) -> None:
        while not self.stopped.is_set():
            try:
                command = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.process_translation(command)
            except Exception:
                if not self.stopped.is_set():
                    self.error_result(command, "invalid_response")
            finally:
                with self.worker_lock:
                    self._busy = False
                self._queue.task_done()

    def execute(self, command: dict) -> None:
        operation = command["op"]
        if operation == "close":
            self.close()
        elif self.stopped.is_set():
            return
        elif operation == "translate":
            self.enqueue(command)
        elif operation == "show":
            if not self._translation_url:
                self.navigate(provider_url(self.provider))
            self.show()
        else:
            self.dispatch(lambda: self.emit(
                "version", active_version=str(self.window.native.webview.CoreWebView2.Environment.BrowserVersionString)))

    def read_commands(self) -> None:
        try:
            while not self.stopped.is_set():
                line = self.source.readline(65537)
                if not line:
                    break
                if len(line) > 65536 or not line.endswith("\n"):
                    self.emit("error", code="invalid_command")
                    break
                try:
                    command = validate_command(json.loads(line))
                    if not self.native_ready.wait(20) or not self.initialized:
                        self.emit("error", code="browser_initialization_timeout")
                        break
                    self.execute(command)
                except (ValueError, TypeError):
                    self.emit("error", code="invalid_command")
        except (OSError, ValueError, UnicodeError):
            pass
        finally:
            self.close()

    def smoke_loaded(self) -> None:
        if not self.initialized or self.stopped.is_set():
            return
        try:
            if self.window.get_current_url() != _SMOKE_URL:
                return
            result = self.window.evaluate_js("(() => ({local_fixture:document.title === '翻译姬翻译窗口测试',"
                                              "external_network:false,audio_capture:false}))()")
            self.smoke_complete = (isinstance(result, dict) and result.get("local_fixture") is True
                                   and result.get("external_network") is False and result.get("audio_capture") is False)
            if not self.smoke_complete:
                self.failure = True
            self.emit("smoke", passed=self.smoke_complete, external_network=False, audio_capture=False)
        except Exception:
            self.failure = True
            self.emit("error", code="browser_smoke_failed")
        finally:
            self.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="翻译姬 · 网页翻译窗口")
    parser.add_argument("--provider", choices=("google_web", "youdao_web"), required=True)
    parser.add_argument("--profile", type=Path, required=True)
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
    logging.getLogger("pywebview").disabled = True
    sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")
    try:
        profile = args.profile.resolve()
        profile.mkdir(parents=True, exist_ok=True)
        if args.smoke_test:
            (profile / "translation-smoke.html").write_text(
                "<!doctype html><meta charset='utf-8'><title>翻译姬翻译窗口测试</title>", encoding="utf-8")
        webview.settings["ALLOW_FILE_URLS"] = False
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
        title = "Google 翻译" if args.provider == "google_web" else "有道翻译"
        window = webview.create_window("翻译姬 · " + title, "about:blank", width=980, height=650,
                                      min_size=(600, 420), hidden=True, text_select=True)
        host = _Host(window, source, sink, Action, CoreWebView2Environment, args.provider, args.smoke_test)
        host.profile = str(profile)
        window.events.before_show += host.before_show
        window.events.closed += host.on_closed
        if args.smoke_test:
            window.events.loaded += host.smoke_loaded
        else:
            threading.Thread(target=host.read_commands, name="web-translation-commands", daemon=True).start()
        def timeout():
            if not host.initialized or (host.smoke and not host.smoke_complete):
                host.failure = True
                host.emit("error", code="browser_initialization_timeout")
                host.close()
        timer = threading.Timer(30, timeout)
        timer.daemon = True
        timer.start()
        webview.start(gui="edgechromium", private_mode=False, storage_path=str(profile), debug=False)
        timer.cancel()
        host.stopped.set()
        if host._worker is not None:
            host._worker.join(timeout=2)
        host.emit("closed")
        return int(host.failure)
    except Exception:
        return fail("browser_start_failed")


if __name__ == "__main__":
    raise SystemExit(main())
