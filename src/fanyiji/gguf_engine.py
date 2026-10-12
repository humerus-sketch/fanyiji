"""Fixed local GGUF translation through an owned, hidden llama.cpp helper.

No model downloading, proxy-aware client, or shared system server is used.
The Windows Job handle is not inherited, so helper processes die with this app.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import http.client
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from .engines import ModelAssetsError, _language
from .language_policy import validate_translation_source
from .livestream_prompt import PROFILE_ID, chinese_output_rules, hy_translation_prompt, livestream_rules
from .translation_output import ChineseOutputError, ensure_chinese_output, is_chinese_output
from .paths import app_root, path_override
from .text_context import MAX_CONTEXT_CHARACTERS, MAX_CONTEXT_TOKENS, bounded_source_context

LOGGER = logging.getLogger(__name__)
_DLL_LAUNCH_LOCK = threading.Lock()


class _WindowsJob:
    """Only processes owned by this engine are assigned to this kill-on-close job."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("GGUF helper currently requires Windows")
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSet", ctypes.c_size_t),
                        ("MaximumWorkingSet", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperations", "WriteOperations", "OtherOperations", "ReadBytes", "WriteBytes", "OtherBytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "Cannot create helper Job Object")
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise OSError(error, "Cannot set helper Job Object limits")

    def assign(self, process: subprocess.Popen) -> None:
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise OSError(ctypes.get_last_error(), "Cannot protect helper process lifetime")

    def close(self) -> None:
        handle, self.handle = self.handle, None
        if handle and not self.kernel.CloseHandle(handle):
            raise OSError(ctypes.get_last_error(), "Cannot close helper Job Object")


@contextmanager
def _external_dll_search():
    """Do not let a frozen Qt DLL directory contaminate the native helper."""
    if os.name != "nt" or not getattr(sys, "frozen", False):
        yield
        return
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetDllDirectoryW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
    kernel.GetDllDirectoryW.restype = wintypes.DWORD
    kernel.SetDllDirectoryW.argtypes = [wintypes.LPCWSTR]
    kernel.SetDllDirectoryW.restype = wintypes.BOOL
    # This is a process-wide Windows setting: keep the change limited to spawn.
    with _DLL_LAUNCH_LOCK:
        size = kernel.GetDllDirectoryW(0, None)
        buffer = ctypes.create_unicode_buffer(size + 1)
        kernel.GetDllDirectoryW(len(buffer), buffer)
        previous = buffer.value or None
        if not kernel.SetDllDirectoryW(None):
            raise OSError(ctypes.get_last_error(), "Cannot isolate helper DLL search path")
        try:
            yield
        finally:
            if not kernel.SetDllDirectoryW(previous):
                LOGGER.error("Could not restore application DLL search directory")


def _relative_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise ModelAssetsError(f"{name} 必须是本地目录内的相对路径")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ModelAssetsError(f"{name} 不得越出本地目录")
    return path


class GgufTranslationEngine:
    """One serialized helper for both translation directions; no chat history."""

    supports_mixed_sources = True
    supports_cancellation = True
    STARTUP_TIMEOUT = 120.0
    REQUEST_TIMEOUT = 120.0
    CONTEXT_WINDOW_TOKENS = 2048
    MAX_OUTPUT_TOKENS = 512
    TEMPLATE_TOKEN_RESERVE = 128

    def __init__(self, model_dir: str | Path, num_threads: int = 4, *, config: dict[str, Any]) -> None:
        if not 2 <= num_threads <= 4:
            raise ValueError("翻译 num_threads 必须在 2 到 4 之间")
        self.model_dir = Path(model_dir).resolve()
        self.model_file = _relative_path(self.model_dir, config.get("model_file"), "model_file")
        if self.model_file.suffix.lower() != ".gguf" or not self.model_file.is_file() or not self.model_file.stat().st_size:
            raise ModelAssetsError("本地 GGUF 模型缺失或为空，请先准备模型；程序不会联网下载。")
        self.model_kind = config.get("model_kind", "hy")
        self.model_id, self.display_name = config.get("model_id"), config.get("display_name")
        if self.display_name is not None and (not isinstance(self.display_name, str) or not 1 <= len(self.display_name) <= 80):
            raise ModelAssetsError("模型显示名称必须是 1 到 80 个字符")
        if self.model_kind not in ("hy", "qwen"):
            raise ModelAssetsError("model_kind 必须是已验证提示格式的 hy/qwen")
        self.preference = config.get("runtime_preference", "auto")
        if self.preference not in ("auto", "cpu", "cuda"):
            raise ModelAssetsError("runtime_preference 必须是 auto/cpu/cuda")
        self.gpu_layers = config.get("gpu_layers")
        if self.gpu_layers is not None and (type(self.gpu_layers) is not int or self.gpu_layers < 0):
            raise ModelAssetsError("gpu_layers 必须是非负整数或 null")
        self.num_threads = num_threads
        self._config = dict(config)
        self._state_lock = threading.RLock()
        self._translation_lock = threading.Lock()
        self._closed = False
        self._cancel_generation = 0
        self._active_cancellation_event: threading.Event | None = None
        self._process: subprocess.Popen | None = None
        self._job: _WindowsJob | None = None
        self._log = None
        self._connection: http.client.HTTPConnection | None = None
        self._port: int | None = None
        self._device: str | None = None
        self._runtime: Path | None = None
        self._api_key = secrets.token_urlsafe(24)
        self._alias = "fanyiji-" + secrets.token_hex(8)
        self._tokenizer_available: bool | None = None
        self._template_available: bool | None = None

    @property
    def backend_info(self) -> dict[str, Any]:
        with self._state_lock:
            return {"backend": "llama_cpp", "model_kind": self.model_kind, "model_file": str(self.model_file),
                    "model_id": self.model_id, "display_name": self.display_name,
                    "runtime_preference": self.preference, "device": self._device or "not_loaded",
                    "num_threads": self.num_threads, "helper_pid": self._process.pid if self._process else None,
                    "gpu_layers_requested": self.gpu_layers,
                    "runtime_directory": str(self._runtime) if self._runtime else None,
                    "transport": "localhost_http", "external_network_required": False,
                    "supports_source_context": True, "source_context_max_tokens": MAX_CONTEXT_TOKENS,
                    "supports_mixed_sources": self.supports_mixed_sources,
                    "supports_cancellation": self.supports_cancellation,
                    "translation_profile": PROFILE_ID,
                    "source_context_max_characters": MAX_CONTEXT_CHARACTERS,
                    "context_window_tokens": self.CONTEXT_WINDOW_TOKENS,
                    "max_output_tokens": self.MAX_OUTPUT_TOKENS}

    @property
    def local_endpoint(self) -> tuple[str, int] | None:
        with self._state_lock:
            return ("127.0.0.1", self._port) if self._port is not None else None

    def _runtime_roots(self) -> list[Path]:
        override = path_override("RUNTIME")
        if override is not None:
            return [override]
        roots = []
        if self._config.get("runtime_dir") is not None:
            roots.append(_relative_path(self.model_dir.parent / "runtime", self._config["runtime_dir"], "runtime_dir"))
        roots.append(app_root() / "runtime")
        if getattr(sys, "frozen", False):
            roots.append(Path(sys._MEIPASS) / "runtime")
        else:
            roots.append(app_root() / "model-cache" / "translation-evaluation" / "llama-b11429")
        return roots

    def _candidates(self) -> list[tuple[str, Path]]:
        devices = ("cuda", "cpu") if self.preference == "auto" else (self.preference,)
        for root in self._runtime_roots():
            candidates = [(device, root / device / "llama-server.exe") for device in devices
                          if (root / device / "llama-server.exe").is_file()]
            if candidates:
                return candidates
        raise ModelAssetsError("本地 llama.cpp runtime 缺失，请先准备程序运行文件；程序不会联网下载。")

    def _environment(self, executable: Path) -> dict[str, str]:
        environment = {key: value for key, value in os.environ.items()
                       if not key.upper().startswith("LLAMA_") and key.upper() not in (
                           "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        windows = Path(environment.get("SystemRoot", "C:/Windows"))
        environment["PATH"] = os.pathsep.join(map(str, (
            executable.parent, executable.parent / "cuda-runtime", windows / "System32", windows)))
        environment.update(LLAMA_API_KEY=self._api_key, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        return environment

    def _start_runtime(self, device: str, executable: Path, *, generation: int | None = None) -> None:
        # Port selection may race another application. Readiness verifies an
        # authenticated unique alias, so an unrelated localhost server is rejected.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command = [str(executable), "-m", str(self.model_file), "--host", "127.0.0.1", "--port", str(port),
                   "--alias", self._alias, "--offline", "--no-webui", "--jinja", "--parallel", "1",
                   "--ctx-size", "2048", "--threads", str(self.num_threads), "--threads-batch", str(self.num_threads),
                   "--log-verbosity", "2"]
        if device == "cuda":
            command += ["--fit", "on", "--fit-target", "1024"]
            if self.gpu_layers is not None:
                command += ["--gpu-layers", str(self.gpu_layers)]
        else:
            command += ["--gpu-layers", "0"]
        with self._state_lock:
            if self._closed:
                raise RuntimeError("本地翻译引擎已关闭")
            self._check_cancelled()
            if generation is not None and generation != self._cancel_generation:
                raise RuntimeError("本地翻译请求已取消")
            job = log = process = None
            try:
                job = _WindowsJob()
                log = tempfile.TemporaryFile(mode="w+b")
                with _external_dll_search():
                    process = subprocess.Popen(command, cwd=executable.parent, env=self._environment(executable),
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=subprocess.CREATE_NO_WINDOW, close_fds=True)
                job.assign(process)
            except BaseException:
                self._dispose_resources(process, job, log)
                raise
            self._process, self._job, self._log = process, job, log
            self._port, self._device, self._runtime = port, device, executable.parent

    def _request(self, method: str, path: str, body: dict | None = None, *, timeout: float | None = None) -> dict:
        with self._state_lock:
            self._check_cancelled()
            if self._closed or self._port is None:
                raise RuntimeError("本地翻译引擎已关闭或尚未加载")
            connection = http.client.HTTPConnection("127.0.0.1", self._port, timeout=timeout or self.REQUEST_TIMEOUT)
            self._connection = connection
        try:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
            connection.request(method, path, body=payload,
                               headers={"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read(1_048_577)
            if response.status != 200:
                raise RuntimeError(f"本地 GGUF helper 返回 HTTP {response.status}")
            if len(raw) > 1_048_576:
                raise RuntimeError("本地 GGUF helper 返回内容过大")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise RuntimeError("本地 GGUF helper 返回格式无效")
            return result
        finally:
            connection.close()
            with self._state_lock:
                if self._connection is connection:
                    self._connection = None

    def _ensure_loaded(self) -> None:
        with self._state_lock:
            self._check_cancelled()
            if self._closed:
                raise RuntimeError("本地翻译引擎已关闭")
            generation = self._cancel_generation
            if self._process is not None and self._process.poll() is None:
                return
        self._stop_runtime()
        errors = []
        for device, executable in self._candidates():
            try:
                self._start_runtime(device, executable, generation=generation)
                deadline = time.monotonic() + self.STARTUP_TIMEOUT
                while time.monotonic() < deadline:
                    with self._state_lock:
                        self._check_cancelled()
                        if self._closed:
                            raise RuntimeError("本地翻译引擎已关闭")
                        if generation != self._cancel_generation:
                            raise RuntimeError("本地翻译请求已取消")
                        process = self._process
                    if process is None or process.poll() is not None:
                        raise RuntimeError("本地 llama.cpp helper 提前退出")
                    try:
                        health = self._request("GET", "/health", timeout=0.5)
                        if health.get("status") == "ok":
                            models = self._request("GET", "/v1/models", timeout=0.5)
                            if any(item.get("id") == self._alias for item in models.get("data", []) if isinstance(item, dict)):
                                return
                    except (OSError, ValueError, http.client.HTTPException, RuntimeError):
                        pass
                    time.sleep(0.05)
                raise TimeoutError("本地 llama.cpp 模型加载超时")
            except Exception as exc:
                errors.append(type(exc).__name__)
                self._stop_runtime()
                with self._state_lock:
                    self._check_cancelled()
                    if self._closed:
                        raise RuntimeError("本地翻译引擎已关闭") from exc
                    if generation != self._cancel_generation:
                        raise RuntimeError("本地翻译请求已取消") from exc
                LOGGER.warning("Local GGUF %s startup failed (%s)", device, type(exc).__name__)
        raise ModelAssetsError("本地 GGUF runtime 无法启动：" + ", ".join(errors))

    def preload(self, *, cancellation_event: threading.Event | None = None) -> None:
        with self._translation_operation(cancellation_event):
            self._ensure_loaded()

    def _check_cancelled(self) -> None:
        event = self._active_cancellation_event
        if event is not None and event.is_set():
            raise RuntimeError("本地翻译请求已取消")

    @contextmanager
    def _translation_operation(self, cancellation_event):
        with self._translation_lock:
            self._active_cancellation_event = cancellation_event
            try:
                # Check after acquiring the shared engine: an old queue worker
                # must not restart the helper after its run has been stopped.
                self._check_cancelled()
                yield
            finally:
                self._active_cancellation_event = None

    def _translation_prompt(self, text: str, source_lang: str, target_lang: str, context: str = "") -> str:
        if self.model_kind == "qwen" and not context and source_lang != "mixed":
            language = "简体中文" if target_lang == "zh" else "日语"
            profile_rules = livestream_rules(source_lang, target_lang, (text,)) or chinese_output_rules(target_lang)
            profile_rules = "\n" + profile_rules if profile_rules else ""
            source = {"zh": "中文", "ja": "日语", "en": "英语"}[source_lang]
            return (f"将以下{source}文本翻译为{language}。只输出译文，保留人名、数字与否定关系，"
                    f"不要补充原文没有的内容。{profile_rules}\n\n{text}\n/no_think")
        prompt = hy_translation_prompt(text, source_lang, target_lang, context)
        return prompt + ("\n/no_think" if self.model_kind == "qwen" else "")

    @staticmethod
    def _unsupported_endpoint(exc: RuntimeError) -> bool:
        return any(f"HTTP {code}" in str(exc) for code in (404, 405, 501))

    def _token_count(self, text: str) -> int | None:
        if not text:
            return 0
        if self._tokenizer_available is False:
            return None
        try:
            result = self._request("POST", "/tokenize", {"content": text, "add_special": False,
                                                        "parse_special": False, "with_pieces": False})
        except RuntimeError as exc:
            if not self._unsupported_endpoint(exc):
                raise
            self._tokenizer_available = False
            return None
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or not all(type(token) is int for token in tokens):
            self._tokenizer_available = False
            return None
        self._tokenizer_available = True
        return len(tokens)

    def _prompt_token_count(self, prompt: str) -> int:
        rendered = None
        if self._template_available is not False:
            try:
                result = self._request("POST", "/apply-template", {
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                rendered = result.get("prompt")
                self._template_available = isinstance(rendered, str) and bool(rendered)
            except RuntimeError as exc:
                if not self._unsupported_endpoint(exc):
                    raise
                self._template_available = False
        reference = rendered if self._template_available else prompt
        count = self._token_count(reference)
        # GGUF byte-fallback tokenizers cannot require more lexical tokens than
        # input UTF-8 bytes. Reserve template/generation markers in both paths.
        return (len(reference.encode("utf-8")) if count is None else count) + self.TEMPLATE_TOKEN_RESERVE

    @staticmethod
    def _reduced_source_context(context: str) -> str:
        # Drop an entire oldest ASR segment and recount before shortening the
        # newest segment. A midpoint may retain a misleading older fragment.
        boundary = context.find("\n")
        character_budget = len(context) - boundary - 1 if boundary >= 0 else max(1, len(context) // 2)
        if character_budget <= 0 or character_budget >= len(context):
            return ""
        return bounded_source_context(context, max_characters=character_budget)

    def _context_prompt(self, text: str, source_lang: str, target_lang: str, context: str) -> str:
        context = bounded_source_context(context)
        while context:
            count = self._token_count(context)
            if count is None:
                count = len(context.encode("utf-8"))
            if count <= MAX_CONTEXT_TOKENS:
                break
            context = self._reduced_source_context(context)
        budget = self.CONTEXT_WINDOW_TOKENS - self.MAX_OUTPUT_TOKENS
        while True:
            prompt = self._translation_prompt(text, source_lang, target_lang, context)
            if self._prompt_token_count(prompt) <= budget:
                return prompt
            if not context:
                # Never truncate the current utterance to fit reference text.
                raise ValueError("当前待翻译文本超过上下文输入预算，请缩短音频片段")
            context = self._reduced_source_context(context)

    def translate(self, text: str, source_lang: str, target_lang: str, *, context: str = "",
                  cancellation_event: threading.Event | None = None) -> str:
        source_lang, target_lang = validate_translation_source(source_lang), _language(target_lang)
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        if not isinstance(context, str):
            raise TypeError("context 必须是字符串")
        text = text.strip()
        context = context.strip()
        if not text or (source_lang == target_lang and (target_lang != "zh" or is_chinese_output(text))):
            return text
        with self._translation_operation(cancellation_event):
            self._ensure_loaded()
            try:
                # The live translation rules also consume input tokens when
                # no history exists. Preserve current text and trim history first.
                needs_budget = bool(context or livestream_rules(source_lang, target_lang, (text,)) or chinese_output_rules(target_lang))
                prompt = (self._context_prompt(text, source_lang, target_lang, context) if needs_budget
                          else self._translation_prompt(text, source_lang, target_lang))
                for attempt in range(2):
                    response = self._request("POST", "/v1/chat/completions", {
                        "model": self._alias, "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0, "seed": 42, "repeat_penalty": 1.05, "top_p": 1, "top_k": 0,
                        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
                        "max_tokens": self.MAX_OUTPUT_TOKENS, "stream": False,
                    })
                    choices = response.get("choices", [])
                    if not choices or not isinstance(choices[0], dict):
                        raise RuntimeError("本地 GGUF 模型未返回译文")
                    choice = choices[0]
                    if choice.get("finish_reason") == "length":
                        raise RuntimeError("本地 GGUF 译文超过输出上限，请缩短片段")
                    content = choice.get("message", {}).get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise RuntimeError("本地 GGUF 模型返回了空译文")
                    try:
                        return ensure_chinese_output(content.strip(), target_lang)
                    except ChineseOutputError:
                        if attempt:
                            raise
                        # Fits the reserved template budget; no foreign name is
                        # silently removed from the current source or output.
                        prompt += "\n重新完整翻译 必须全部中文 姓名音译为中文 不留外文 不删除内容"
            except ChineseOutputError:
                # A language mismatch is a valid completed inference, not a
                # broken native runtime. Keep the resident model for the next job.
                raise
            except Exception:
                # A timed-out request may still occupy the only server slot.
                # Release this owned helper; the next job can load a fresh one.
                self._stop_runtime()
                raise

    def _stop_runtime(self) -> None:
        with self._state_lock:
            process, job, log = self._process, self._job, self._log
            connection = self._connection
            self._process = self._job = self._log = self._connection = None
            self._port = self._device = self._runtime = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                LOGGER.warning("Could not close local helper HTTP connection")
        self._dispose_resources(process, job, log)

    @staticmethod
    def _dispose_resources(process, job, log) -> None:
        protected = False
        try:
            if job is not None:
                job.close()
                protected = True
        finally:
            try:
                if process is not None:
                    if not protected and process.poll() is None:
                        process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
                if log is not None:
                    log.close()

    def close(self) -> None:
        # Do not wait for a translate/preload lock: app exit must cancel native
        # inference or startup, rather than wait for the HTTP timeout.
        with self._state_lock:
            self._closed = True
        self._stop_runtime()

    def cancel_pending(self) -> None:
        """Interrupt this helper without permanently closing the selected engine."""
        with self._state_lock:
            self._cancel_generation += 1
        # A generation change also stops startup from trying CPU fallback after
        # cancellation. A later preload may create a new owned helper normally.
        self._stop_runtime()
