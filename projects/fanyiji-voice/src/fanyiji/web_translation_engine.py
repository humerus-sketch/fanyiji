"""Owned WebView2 translation pages, isolated from the live-browser profile."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import threading
import time
from typing import Any

from .paths import data_root
from .translation_output import ensure_chinese_output, is_chinese_output


_ERROR_MESSAGES = {
    "unsupported_platform": "网页翻译目前需要 Windows 和 WebView2。",
    "runtime_override_blocked": "请移除 WebView2 运行时覆盖设置后再使用网页翻译。",
    "unsupported_runtime_channel": "网页翻译需要稳定版 WebView2。",
    "browser_dependency_missing": "缺少网页翻译的 WebView2 运行组件。",
    "browser_runtime_missing": "未检测到 WebView2，请安装 WebView2 Evergreen Runtime。",
    "profile_unavailable": "网页翻译的独立浏览器资料目录无法访问。",
    "login_required": "翻译网站需要登录，请在已打开的翻译窗口自行完成。",
    "captcha_required": "翻译网站需要验证，请在已打开的翻译窗口自行完成。",
    "page_mismatch": "翻译页面结构不匹配，请检查已打开的翻译窗口。",
    "page_not_ready": "翻译页面尚未准备好，请检查翻译窗口。",
    "page_changed": "翻译页面结构已变化，请检查翻译窗口。",
    "interaction_required": "翻译网站需要交互，请在已打开的翻译窗口自行完成登录或验证。",
    "invalid_response": "翻译网页没有返回可验证的译文。",
    "source_changed": "翻译网页中的原文已变化，请重试本条翻译。",
    "input_unavailable": "翻译网页输入框暂不可用，请检查翻译窗口。",
    "language_mismatch": "翻译页面语言不匹配，请检查翻译窗口的语言设置。",
    "navigation_blocked": "翻译窗口阻止了不支持的页面跳转。",
    "network_error": "翻译网页未能加载，请检查网络或在翻译窗口重试。",
    "translation_timeout": "翻译网页未及时返回译文，请检查网络或翻译窗口。",
    "cancelled": "网页翻译请求已取消。",
    "closed": "网页翻译已关闭。",
}


class WebTranslationError(RuntimeError):
    """Only a fixed code/message crosses the UI/logging boundary."""

    def __init__(self, code: str):
        self.code = code if isinstance(code, str) and code in _ERROR_MESSAGES else "browser_unavailable"
        super().__init__(_ERROR_MESSAGES.get(self.code, "网页翻译暂不可用，请检查翻译窗口。"))


@dataclass
class _Session:
    process: Any
    events: queue.Queue
    generation: int
    job: Any = None


class _Cancellation:
    def __init__(self, owned: threading.Event, external=None):
        self.owned, self.external = owned, external

    def is_set(self) -> bool:
        return self.owned.is_set() or (self.external is not None and self.external.is_set())


def _safe_diagnostic(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    booleans = ("input_present", "output_present", "source_matches", "language_pair_selected")
    if any(type(value.get(key)) is not bool for key in booleans):
        return {}
    return {**{key: value[key] for key in booleans},
            **({"document_hidden": value["document_hidden"]} if type(value.get("document_hidden")) is bool else {}),
            "phase": value.get("phase") if value.get("phase") in ("clearing", "languages", "waiting") else "unknown",
            "ready_state": value.get("ready_state") if value.get("ready_state") in ("loading", "interactive", "complete") else "unknown"}


class WebTranslationTransport:
    """One process and one request at a time; cancellation discards its DOM."""

    STARTUP_TIMEOUT = 35.0
    REQUEST_TIMEOUT = 45.0
    MAX_LINE_CHARACTERS = 65536

    def __init__(self, provider: str, *, profile: Path | None = None, process_factory=None,
                 startup_timeout: float | None = None, request_timeout: float | None = None):
        if provider not in ("google_web", "youdao_web"):
            raise ValueError("网页翻译器必须是 google_web 或 youdao_web")
        self.provider = provider
        self.profile = Path(profile) if profile is not None else data_root() / "web-translation" / provider
        self._factory = process_factory or subprocess.Popen
        self._protect_process = process_factory is None
        self.startup_timeout = self.STARTUP_TIMEOUT if startup_timeout is None else startup_timeout
        self.request_timeout = self.REQUEST_TIMEOUT if request_timeout is None else request_timeout
        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._session: _Session | None = None
        self._closed = False
        self._generation = 0
        self._diagnostic: dict = {}

    @property
    def last_diagnostic(self) -> dict:
        with self._state_lock:
            return dict(self._diagnostic)

    def _check(self, generation: int, cancellation_event=None) -> None:
        if cancellation_event is not None and cancellation_event.is_set():
            raise WebTranslationError("cancelled")
        with self._state_lock:
            if self._closed:
                raise WebTranslationError("closed")
            if generation != self._generation:
                raise WebTranslationError("cancelled")

    def _launch(self, generation: int, cancellation_event=None) -> _Session:
        self._check(generation, cancellation_event)
        with self._state_lock:
            self._check(generation, cancellation_event)
            if self._session is not None and self._session.process.poll() is None:
                return self._session
            stale, self._session = self._session, None
            self._dispose(stale)
            try:
                self.profile.mkdir(parents=True, exist_ok=True)
            except OSError:
                raise WebTranslationError("profile_unavailable") from None
            args = [sys.executable]
            if getattr(sys, "frozen", False):
                args += ["--web-translation-helper"]
            else:
                args += ["-m", "fanyiji.web_translation_helper"]
            args += ["--provider", self.provider, "--profile", str(self.profile)]
            environment = os.environ.copy()
            if not getattr(sys, "frozen", False):
                source = str(Path(__file__).resolve().parents[1])
                environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
            options = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL,
                       "text": True, "encoding": "utf-8", "errors": "strict", "bufsize": 1, "env": environment}
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            job = None
            process = None
            try:
                if self._protect_process and os.name == "nt":
                    from .gguf_engine import _WindowsJob, _external_dll_search
                    job = _WindowsJob()
                    with _external_dll_search():
                        self._check(generation, cancellation_event)
                        process = self._factory(args, **options)
                    job.assign(process)
                else:
                    self._check(generation, cancellation_event)
                    process = self._factory(args, **options)
                if process.stdin is None or process.stdout is None:
                    raise OSError("missing_pipe")
            except Exception as error:
                if process is not None:
                    try:
                        process.kill()
                    except Exception:
                        pass
                if job is not None:
                    job.close()
                if isinstance(error, WebTranslationError):
                    raise
                raise WebTranslationError("browser_unavailable") from None
            session = _Session(process, queue.Queue(maxsize=64), generation, job)
            self._session = session
            threading.Thread(target=self._read, args=(session,), name="web-translation-pipe", daemon=True).start()
        try:
            self._wait(session, "ready", None, self.startup_timeout, cancellation_event)
        except Exception:
            self._discard_session(session)
            raise
        return session

    def _read(self, session: _Session) -> None:
        try:
            while True:
                line = session.process.stdout.readline(self.MAX_LINE_CHARACTERS + 1)
                if not line:
                    break
                if len(line) > self.MAX_LINE_CHARACTERS or not line.endswith("\n"):
                    break
                value = json.loads(line)
                if not isinstance(value, dict):
                    break
                if value.get("kind") == "diagnostic":
                    safe = _safe_diagnostic(value)
                    with self._state_lock:
                        if safe and self._session is session and self._generation == session.generation:
                            self._diagnostic = safe
                    continue
                session.events.put_nowait(value)
        except (OSError, ValueError, UnicodeError, queue.Full):
            pass
        finally:
            try:
                session.events.put_nowait({"kind": "closed"})
            except queue.Full:
                pass

    def _send(self, session: _Session, command: dict, cancellation_event=None) -> None:
        with self._state_lock:
            self._check(session.generation, cancellation_event)
            if self._session is not session or session.generation != self._generation or self._closed:
                raise WebTranslationError("cancelled" if not self._closed else "closed")
            try:
                session.process.stdin.write(json.dumps(command, ensure_ascii=True) + "\n")
                session.process.stdin.flush()
            except (OSError, ValueError, BrokenPipeError):
                raise WebTranslationError("browser_unavailable") from None

    def _wait(self, session: _Session, kind: str, identifier: str | None, timeout: float,
              cancellation_event=None) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            self._check(session.generation, cancellation_event)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._discard_session(session)
                raise WebTranslationError("translation_timeout")
            try:
                value = session.events.get(timeout=min(0.05, remaining))
            except queue.Empty:
                with self._state_lock:
                    self._check(session.generation, cancellation_event)
                    if session.process.poll() is not None:
                        raise WebTranslationError("browser_unavailable")
                continue
            self._check(session.generation, cancellation_event)
            if value.get("kind") == "error":
                self._discard_session(session)
                raise WebTranslationError(value.get("code"))
            if value.get("kind") == "closed":
                with self._state_lock:
                    self._check(session.generation, cancellation_event)
                    raise WebTranslationError("browser_unavailable")
            if value.get("kind") != kind:
                continue
            if identifier is not None and value.get("id") != identifier:
                continue
            self._check(session.generation, cancellation_event)
            return value

    def preload(self, *, cancellation_event=None) -> None:
        with self._state_lock:
            generation = self._generation
        with self._operation_lock:
            self._launch(generation, cancellation_event)

    def request(self, command: dict, *, cancellation_event=None) -> dict:
        with self._state_lock:
            generation = self._generation
        with self._operation_lock:
            session = self._launch(generation, cancellation_event)
            try:
                with self._state_lock:
                    self._diagnostic = {}
                self._check(generation, cancellation_event)
                self._send(session, command, cancellation_event)
                value = self._wait(session, "result", command["id"], self.request_timeout, cancellation_event)
            except WebTranslationError as error:
                if error.code in ("cancelled", "browser_unavailable"):
                    self._discard_session(session)
                raise
            result = value.get("result")
            if not isinstance(result, dict):
                raise WebTranslationError("browser_unavailable")
            return result

    def show_window(self, *, cancellation_event=None) -> None:
        with self._state_lock:
            generation = self._generation
            self._check(generation, cancellation_event)
            session = self._session
            if session is not None and session.process.poll() is None:
                self._send(session, {"op": "show"}, cancellation_event)
                return
        with self._operation_lock:
            session = self._launch(generation, cancellation_event)
            try:
                self._send(session, {"op": "show"}, cancellation_event)
                self._check(generation, cancellation_event)
            except WebTranslationError as error:
                if error.code == "cancelled":
                    self._discard_session(session)
                raise

    @staticmethod
    def _dispose(session: _Session | None) -> None:
        if session is None:
            return
        try:
            session.process.kill()
        except Exception:
            pass
        try:
            session.process.wait(timeout=2)
        except Exception:
            pass
        if session.job is not None:
            try:
                session.job.close()
            except OSError:
                pass
        for stream in (session.process.stdin, session.process.stdout):
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def cancel_pending(self) -> None:
        with self._state_lock:
            self._generation += 1
            session, self._session = self._session, None
        self._dispose(session)

    def _discard_session(self, session: _Session) -> None:
        with self._state_lock:
            if self._session is not session:
                return
            self._generation += 1
            self._session = None
        self._dispose(session)

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
        self.cancel_pending()


class _WebTranslationEngine:
    supports_cancellation = True
    supports_source_context = False
    supports_mixed_sources = False
    supports_batch = False
    provider = ""
    display_name = ""

    def __init__(self, *, transport=None, process_factory=None, profile: Path | None = None):
        self.transport = transport if transport is not None else WebTranslationTransport(
            self.provider, profile=profile, process_factory=process_factory)
        self._closed = False
        self._generation = 0
        self._cancellation = threading.Event()
        self._state_lock = threading.RLock()
        self._translation_lock = threading.Lock()
        self._metrics: dict[str, float] = {}

    @property
    def backend_info(self) -> dict[str, Any]:
        return {"provider": self.provider, "model_id": self.provider, "model_name": self.display_name,
                "display_name": self.display_name, "backend": "webview2", "device": "web",
                "network_required": True, "external_network_required": True,
                "supports_source_context": False, "supports_mixed_sources": False,
                "supports_batch": False, "supports_cancellation": True}

    @property
    def last_request_metrics(self) -> dict[str, float]:
        with self._state_lock:
            return dict(self._metrics)

    @property
    def last_diagnostic(self) -> dict:
        return _safe_diagnostic(getattr(self.transport, "last_diagnostic", {}))

    def _check(self, generation: int, cancellation_event=None) -> None:
        with self._state_lock:
            if self._closed:
                raise WebTranslationError("closed")
            if generation != self._generation or (cancellation_event is not None and cancellation_event.is_set()):
                raise WebTranslationError("cancelled")

    def preload(self, *, cancellation_event=None) -> None:
        with self._state_lock:
            generation = self._generation
            cancellation = _Cancellation(self._cancellation, cancellation_event)
        self._check(generation, cancellation_event)
        self.transport.preload(cancellation_event=cancellation)
        self._check(generation, cancellation_event)

    def translate(self, text: str, source_lang: str, target_lang: str, *, cancellation_event=None) -> str:
        from .web_translation_protocol import validate_command, validated_result
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        if source_lang not in ("zh", "ja") or target_lang not in ("zh", "ja"):
            raise ValueError("网页翻译语言仅支持 zh 或 ja")
        with self._state_lock:
            generation = self._generation
            cancellation = _Cancellation(self._cancellation, cancellation_event)
        with self._translation_lock:
            self._check(generation, cancellation_event)
            text = text.strip()
            if not text:
                return ""
            if source_lang == target_lang and (target_lang != "zh" or is_chinese_output(text)):
                return text
            command = validate_command({"op": "translate", "id": secrets.token_hex(16), "text": text,
                                        "source_lang": source_lang, "target_lang": target_lang})
            started = time.monotonic()
            value = self.transport.request(command, cancellation_event=cancellation)
            self._check(generation, cancellation_event)
            result = validated_result(value, command)
            if result.get("status") != "ok":
                raise WebTranslationError(result.get("code"))
            translation = ensure_chinese_output(result["text"], target_lang)
            with self._state_lock:
                self._metrics = {"translation_seconds": time.monotonic() - started}
            return translation

    def show_window(self, *, cancellation_event=None) -> None:
        with self._state_lock:
            generation = self._generation
            cancellation = _Cancellation(self._cancellation, cancellation_event)
        self._check(generation, cancellation_event)
        self.transport.show_window(cancellation_event=cancellation)
        self._check(generation, cancellation_event)

    def cancel_pending(self) -> None:
        with self._state_lock:
            self._generation += 1
            self._cancellation.set()
            self._cancellation = threading.Event()
        self.transport.cancel_pending()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._cancellation.set()
        self.transport.close()


class GoogleWebTranslationEngine(_WebTranslationEngine):
    provider = "google_web"
    display_name = "Google 翻译"


class YoudaoWebTranslationEngine(_WebTranslationEngine):
    provider = "youdao_web"
    display_name = "有道翻译"
