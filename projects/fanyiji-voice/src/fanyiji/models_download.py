"""Pinned model catalogue and cancellable downloads; no model/audio initialization."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import errno
import hashlib
from http.client import IncompleteRead
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import socket
import shutil
import ssl
import threading
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

LOGGER = logging.getLogger(__name__)
_SELECTION_LOCK = threading.Lock()


class DownloadError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False,
                 http_status: int | None = None, host: str | None = None,
                 source_refresh: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.http_status = http_status
        self.host = host
        self.source_refresh = source_refresh


class DownloadCancelled(DownloadError):
    pass


_LOCAL_FILE_ERRNOS = {errno.ENOSPC, errno.EACCES, errno.EPERM, errno.EROFS,
                      errno.ENOENT, errno.ENOTDIR, errno.EISDIR, errno.EBUSY}
_LOCAL_FILE_WINERRORS = {5, 32, 112}
_TRANSPORT_ERRNOS = {getattr(errno, name) for name in (
    "ECONNRESET", "ECONNABORTED", "ECONNREFUSED", "ETIMEDOUT", "ENETDOWN",
    "ENETUNREACH", "ENETRESET", "EHOSTDOWN", "EHOSTUNREACH", "EPIPE",
    "ESHUTDOWN", "ENOBUFS", "EINTR") if hasattr(errno, name)}
_TRANSPORT_WINERRORS = {10004, 10050, 10051, 10052, 10053, 10054, 10055,
                        10057, 10058, 10060, 10061, 10064, 10065}


def _local_file_error(reason: object) -> bool:
    # SSL uses its own numeric errno (including 1); do not confuse it with EPERM.
    return (isinstance(reason, OSError) and not isinstance(reason, ssl.SSLError)
            and (reason.errno in _LOCAL_FILE_ERRNOS
                 or getattr(reason, "winerror", None) in _LOCAL_FILE_WINERRORS))


def _transport_error(reason: object) -> bool:
    if isinstance(reason, ssl.SSLCertVerificationError) or _local_file_error(reason):
        return False
    return (isinstance(reason, (TimeoutError, socket.gaierror, ConnectionError,
                               IncompleteRead, ssl.SSLError))
            or isinstance(reason, OSError) and (
                reason.errno in _TRANSPORT_ERRNOS | _TRANSPORT_WINERRORS
                or getattr(reason, "winerror", None) in _TRANSPORT_WINERRORS))


def describe_download_error(exc: BaseException) -> str:
    """Return actionable text without exposing signed URLs or proxy credentials."""
    if isinstance(exc, DownloadError):
        message = re.sub(r"https?://[^\s]+", "[下载地址已隐藏]", str(exc), flags=re.I)
        message = re.sub(r"\b(?:token|password|secret|authorization|api[_-]?key)\s*[:=]\s*[^\s,;]+",
                         "[凭证已隐藏]", message, flags=re.I)
        message = "".join(char for char in message if char.isprintable())[:600]
        host = exc.host
        if isinstance(host, str) and re.fullmatch(r"[a-z0-9.-]{1,253}", host):
            try:
                validate_download_url(f"https://{host}/")
            except DownloadError:
                pass
            else:
                if "（下载主机：" not in message:
                    message += f"（下载主机：{host}）"
        return message
    if isinstance(exc, HTTPError):
        explanations = {
            401: "下载源要求身份验证，请换用另一个官方来源。",
            403: "下载源拒绝访问或临时下载链接已失效，请重试或换用另一个官方来源。",
            404: "下载源没有找到此模型文件，请换用另一个官方来源。",
            407: "代理身份验证失败，请检查 Windows 或环境变量中的代理设置。",
            408: "下载源等待超时，保留临时文件，可续传。",
            429: "下载源暂时限流，保留临时文件，请稍后续传。",
        }
        return f"下载 HTTP {exc.code}；" + explanations.get(exc.code, "下载源返回错误，保留临时文件，可重试。")
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "TLS 证书验证失败，请检查系统时间、系统证书或代理证书；不会关闭安全校验。"
    if isinstance(reason, ssl.SSLError):
        return "TLS 安全连接失败，请检查网络或代理设置，保留临时文件，可续传。"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "连接或读取超时，请检查网络或代理设置，保留临时文件，可续传。"
    if isinstance(reason, socket.gaierror):
        return "无法解析下载域名，请检查 DNS、网络或代理设置，保留临时文件，可续传。"
    if _local_file_error(reason):
        if reason.errno == errno.ENOSPC or getattr(reason, "winerror", None) == 112:
            return "模型目录所在磁盘空间不足，请释放空间后续传。"
        if reason.errno in (errno.EACCES, errno.EPERM, errno.EROFS, errno.EBUSY) or getattr(reason, "winerror", None) in (5, 32):
            return "无法写入模型目录，可能没有权限或文件被占用，请选择可写目录后重试。"
        return "本地模型文件读写失败，已有文件保留，请检查磁盘和目录权限。"
    if isinstance(exc, URLError) or _transport_error(reason):
        return "下载网络连接失败或中断，请检查网络或代理设置，保留临时文件，可续传。"
    if isinstance(exc, OSError):
        if exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112:
            return "模型目录所在磁盘空间不足，请释放空间后续传。"
        if exc.errno in (errno.EACCES, errno.EPERM) or getattr(exc, "winerror", None) in (5, 32):
            return "无法写入模型目录，可能没有权限或文件被占用，请选择可写目录后重试。"
        return "本地模型文件读写失败，已有文件保留，请检查磁盘和目录权限。"
    return "模型下载失败，临时文件保留；可续传或通过“导入本地模型”选择已下载的 GGUF 文件。"


def validate_download_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    # Downloads may redirect to the providers' signed delivery URLs. Validate
    # every redirect before following it, not just the final response URL.
    exact = {"huggingface.co", "modelscope.cn", "cdn-lfs-cn-1.modelscope.cn", "github.com", "raw.githubusercontent.com",
             "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
    delivery = host.endswith(".hf.co") or host.endswith(".xethub.hf.co")
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in (None, 443) or not (host in exact or delivery)):
        raise DownloadError("下载地址必须使用官方 HTTPS 域名")


class _OfficialRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        validate_download_url(url)
        return super().redirect_request(request, response, code, message, headers, url)


@dataclass(frozen=True)
class ModelFile:
    relative_path: str
    url: str | None
    size: int
    sha256: str
    fallback_urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}
        if (not self.relative_path or path.is_absolute() or "\\" in self.relative_path
                or any(p in ("", ".", "..") for p in self.relative_path.split("/"))
                or any(p in (".", "..") or ":" in p or p.endswith((" ", "."))
                       or p.split(".")[0].upper() in reserved for p in path.parts)):
            raise ValueError("模型相对路径无效")
        if self.size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("模型大小或 SHA-256 无效")
        if self.url is not None:
            validate_download_url(self.url)
        if not isinstance(self.fallback_urls, tuple) or self.url is None and self.fallback_urls:
            raise ValueError("模型备用下载地址无效")
        for url in self.fallback_urls:
            validate_download_url(url)


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    files: tuple[ModelFile, ...]
    info_url: str
    description: str
    required: bool = False
    model_kind: str | None = None
    recommendation: str = "待评测"
    accuracy_priority: bool = True
    min_memory_description: str = ""
    recommendation_basis: str = ""
    verification: str = "pinned_catalog_sha256"
    source_model_id: str | None = None
    architecture: str | None = None

    @property
    def size(self) -> int:
        return sum(file.size for file in self.files)


def translation_profile(entry: ModelEntry) -> dict:
    if entry.required or entry.model_kind not in ("hy", "qwen") or len(entry.files) != 1:
        raise DownloadError("请选择可启用的翻译模型")
    relative = PurePosixPath(entry.files[0].relative_path)
    if not relative.is_relative_to(PurePosixPath("translation")) or relative.suffix != ".gguf":
        raise DownloadError("翻译权重必须放在 translation 目录")
    raw = {"backend": "llama_cpp", "model_file": str(relative.relative_to("translation")),
           "model_kind": entry.model_kind, "runtime_preference": "auto"}
    if entry.id.startswith("local-"):
        raw.update(model_id=entry.id, display_name=entry.name)
    return raw


@dataclass(frozen=True)
class ModelCatalog:
    entries: tuple[ModelEntry, ...]
    default_model_id: str = "hy-mt2-7b-q6_k"

    def __post_init__(self) -> None:
        ids = [entry.id for entry in self.entries]
        paths = [file.relative_path for entry in self.entries for file in entry.files]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("模型 ID 和目标文件路径必须唯一")
        if self.default_model_id not in ids:
            raise ValueError("默认模型不在目录中")

    def get(self, model_id: str) -> ModelEntry:
        return next(entry for entry in self.entries if entry.id == model_id)

    @property
    def required_entries(self) -> tuple[ModelEntry, ...]:
        return tuple(entry for entry in self.entries if entry.required)

    @classmethod
    def default(cls) -> ModelCatalog:
        asr_repo = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
        asr_rev = "2365baeacb507f821a0c8120fcee3d484dba7a07"
        asr_url = f"https://huggingface.co/{asr_repo}/resolve/{asr_rev}/"
        base = ModelEntry(
            "base-sensevoice-vad", "基础识别 · SenseVoice INT8 + Silero VAD",
            (
                ModelFile("asr/model.int8.onnx", asr_url + "model.int8.onnx", 239233841,
                          "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"),
                ModelFile("asr/tokens.txt", asr_url + "tokens.txt", 315894,
                          "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"),
                ModelFile("vad/silero_vad.onnx",
                          "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
                          643854, "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"),
            ), f"https://huggingface.co/{asr_repo}/tree/{asr_rev}",
            "必需：中日语音识别、词表与静音分句。请保留官方模型许可。", required=True,
            accuracy_priority=False, min_memory_description="识别在 CPU 运行")
        candidates = (
            ("hy-mt2-1.8b-q6_k", "Hy-MT2 1.8B · Q6_K", "tencent/Hy-MT2-1.8B-GGUF",
             "a0c709d9fac510f2c807aa3af52872340dc37a4a", "Hy-MT2-1.8B-Q6_K.gguf",
             1474785120, "d98fe604dec1f28f58f80d7d560f7177e584d3b8e5835862687660e5ff97cb40",
             "hy", "专用翻译备用；体积较小。", "CPU 或 CUDA"),
            ("hy-mt2-7b-q6_k", "Hy-MT2 7B · Q6_K", "tencent/Hy-MT2-7B-GGUF",
             "ab8472660ac61fac25f1af43fac2599d52a8a775", "HY-MT2-7B-Q6_K.gguf",
             6164482720, "88ef0aba59952a4cfe4be36cb5baf797dbb370bc60e9dcbd7297036021e52831",
             "hy", "本机质量与实时使用首选。", "8 GB 显存需控制上下文，必要时 CPU + GPU 混合"),
            ("hy-mt2-30b-a3b-q4_k_m", "Hy-MT2 30B-A3B · Q4_K_M",
             "tencent/Hy-MT2-30B-A3B-GGUF", "fd3dcbb6b31e9e03923ef4f9f42500f74c3851c3",
             "Hy-MT2-30B-A3B-Q4_K_M.gguf", 18236702880,
             "bb44b11bb0f7cd3d1321645b41e911cd3de2e473731227fc6bf37aa18b543f88",
             "hy", "手动质量对照；同组评测未稳定优于7B，非默认选择。", "8 GB 显存需 CPU + GPU 混合；32 GB 本机试跑系统内存余量曾仅36.6 MiB"),
            ("qwen3-1.7b-q8_0", "Qwen3 1.7B · Q8_0", "Qwen/Qwen3-1.7B-GGUF",
             "90862c4b9d2787eaed51d12237eafdfe7c5f6077", "Qwen3-1.7B-Q8_0.gguf",
             1834426016, "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
             "qwen", "通用模型，作为翻译对照候选。", "CPU 或 CUDA；速度需实测"),
        )
        filter_repo = "k2-fsa/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15"
        filter_revision = "c24f9d0fdf7d0b8c0c4f9733aa0cac51fda10c95"
        filter_root = f"https://huggingface.co/{filter_repo}/resolve/{filter_revision}/"
        speech_filter = ModelEntry(
            "speech-filter-zipformer", "讲话筛选 · Zipformer Small INT8",
            (ModelFile("speech_filter/model.int8.onnx", filter_root + "model.int8.onnx",
                       27038066, "69304b8a1b96bbe6b7d16c24079f0732c65faf3568a14cb82a8238c709afe76c"),
             ModelFile("speech_filter/class_labels_indices.csv", filter_root + "class_labels_indices.csv",
                       14675, "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429")),
            "https://k2-fsa.github.io/sherpa/onnx/audio-tagging/pretrained_models.html",
            "本地区分讲话、唱歌和音乐；开启讲话筛选时需要。许可Apache-2.0。",
            required=True, recommendation="声音筛选依赖", accuracy_priority=False,
            min_memory_description="CPU运行；同时重叠的讲话与歌声不能保证完全分离。")
        entries = [base, speech_filter]
        recommendations = {
            "hy-mt2-1.8b-q6_k": ("★★★☆☆", "本机42条固定文本的备用方案；仍有增译和义变。"),
            "hy-mt2-7b-q6_k": ("★★★★★", "本机42条固定文本的当前综合首选；30B两轮未稳定更好且系统内存紧张。"),
            "qwen3-1.7b-q8_0": ("★★☆☆☆", "本机42条固定文本的对照候选；出现数字条件遗漏与日文残留。"),
            "hy-mt2-30b-a3b-q4_k_m": ("★★★☆☆", "两轮各42条本机文本评测未一致优于7B；翻译模型单句耗时中位数1.754/1.848秒，保留手动质量对照。"),
        }
        # These official ModelScope revisions expose the same size and SHA-256
        # as the pinned Hugging Face files. Never substitute an unverified build.
        domestic_sources = {
            "hy-mt2-1.8b-q6_k": "https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-1.8B-GGUF/resolve/ef1d40b8b315575d30d1eb6579996130b8fc0bb2/Hy-MT2-1.8B-Q6_K.gguf",
            "hy-mt2-7b-q6_k": "https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-7B-GGUF/resolve/0ea6c87047cb05b5bc10cd51ee484e2ed0f6cb5f/HY-MT2-7B-Q6_K.gguf",
            "qwen3-1.7b-q8_0": "https://modelscope.cn/models/Qwen/Qwen3-1.7B-GGUF/resolve/dc80e1956e7551cd4aa5309c914e767b69188639/Qwen3-1.7B-Q8_0.gguf",
        }
        for mid, name, repo, rev, filename, size, sha, kind, description, memory in candidates:
            recommendation, basis = recommendations[mid]
            entries.append(ModelEntry(
                mid, name, (ModelFile(f"translation/{filename}",
                    f"https://huggingface.co/{repo}/resolve/{rev}/{filename}", size, sha,
                    (domestic_sources[mid],) if mid in domestic_sources else ()),),
                f"https://huggingface.co/{repo}/blob/{rev}/README.md",
                description + " 权重许可：Apache-2.0，请保留官方 LICENSE。",
                model_kind=kind, min_memory_description=memory, recommendation=recommendation,
                recommendation_basis=basis + " 推荐指数不是准确率或通用排行榜。"))
        return cls(tuple(entries))


@dataclass(frozen=True)
class VerificationResult:
    ready: bool
    state: str
    detail: str


Progress = Callable[[str, int, int], None]


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise DownloadCancelled("已取消；未完成的 .part 文件保留，可继续下载")


class ModelDownloader:
    def __init__(self, models_root: Path, *, opener=None, timeout: float = 60,
                 chunk_size: int = 1024 * 1024, max_attempts: int = 4,
                 retry_delay: float = 1):
        self.models_root = Path(models_root).resolve()
        self.opener = opener or build_opener(_OfficialRedirects())
        if timeout <= 0 or chunk_size <= 0:
            raise ValueError("下载超时和块大小必须大于 0")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 4 or retry_delay < 0:
            raise ValueError("下载最多尝试 1 到 4 次，重试等待不能为负数")
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self._response_lock = threading.Lock()
        self._active_response = None
        self._cache_lock = threading.RLock()
        self._cache_path = self.models_root / ".model-verification.json"
        self._verified = self._read_cache()
        self._verified_parts: dict[str, tuple[str, dict[str, int]]] = {}

    @staticmethod
    def _cache_key(file: ModelFile) -> str:
        return json.dumps([file.relative_path, file.size, file.sha256], ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _identity(stat) -> dict[str, int]:
        return {"dev": stat.st_dev, "ino": stat.st_ino, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _read_cache(self) -> dict[str, dict[str, int]]:
        try:
            if not self._cache_path.resolve().is_relative_to(self.models_root) or self._cache_path.stat().st_size > 1_048_576:
                return {}
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if (not isinstance(raw, dict) or type(raw.get("schema_version")) is not int
                    or raw["schema_version"] != 1 or not isinstance(raw.get("verified_files"), dict)):
                return {}
            return {key: value for key, value in raw["verified_files"].items()
                    if isinstance(key, str) and isinstance(value, dict)
                    and set(value) == {"dev", "ino", "size", "mtime_ns"}
                    and all(type(number) is int and number >= 0 for number in value.values())}
        except (OSError, ValueError):
            # A missing/corrupt cache never certifies a weight. Hash normally.
            return {}

    def _save_cache(self) -> None:
        if not self._cache_path.resolve().is_relative_to(self.models_root):
            return
        try:
            self.models_root.mkdir(parents=True, exist_ok=True)
            temp = self._cache_path.with_name(self._cache_path.name + "." + uuid.uuid4().hex + ".tmp")
            with temp.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump({"schema_version": 1, "verified_files": self._verified}, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self._cache_path)
        except OSError:
            LOGGER.warning("模型校验缓存无法持久保存；下次打开可能需要重新校验")

    def _remember_verified(self, key: str, identity: dict[str, int]) -> None:
        with self._cache_lock:
            self._verified[key] = identity
            self._save_cache()

    def _forget_verified(self, key: str) -> None:
        with self._cache_lock:
            if self._verified.pop(key, None) is not None:
                self._save_cache()

    def _publish_verified_part(self, file: ModelFile, part: Path, final: Path,
                               cancel: threading.Event | None) -> None:
        _check_cancel(cancel)
        certificate = self._verified_parts.pop(str(part.resolve()), None)
        if certificate is None or self._identity(part.stat()) != certificate[1]:
            raise DownloadError("临时模型文件在校验后发生变化，请重新校验")
        os.replace(part, final)
        if self._identity(final.stat()) != certificate[1]:
            raise DownloadError("模型文件在完成下载后发生变化，请重新校验")
        self._remember_verified(certificate[0], certificate[1])

    def cancel_active_download(self) -> None:
        """Interrupt an established HTTP socket without taking its buffered I/O lock."""
        with self._response_lock:
            response = self._active_response
        raw = getattr(getattr(response, "fp", None), "raw", None)
        active_socket = getattr(raw, "_sock", None)
        if active_socket is not None:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    @contextmanager
    def _receiving(self, response):
        with self._response_lock:
            self._active_response = response
        try:
            with response:
                yield response
        finally:
            with self._response_lock:
                if self._active_response is response:
                    self._active_response = None

    def path_for(self, file: ModelFile) -> Path:
        path = self.models_root.joinpath(*PurePosixPath(file.relative_path).parts)
        if not path.resolve().is_relative_to(self.models_root):
            raise DownloadError("模型路径超出指定目录")
        return path

    def verify_file(self, file: ModelFile, progress: Progress | None = None,
                    cancel: threading.Event | None = None, *, path: Path | None = None,
                    force: bool = False) -> bool:
        target = path or self.path_for(file)
        key = self._cache_key(file)
        _check_cancel(cancel)
        if not target.is_file() or target.stat().st_size != file.size:
            if path is None:
                self._forget_verified(key)
            return False
        before = target.stat()
        if path is None and not force:
            with self._cache_lock:
                cached = self._verified.get(key)
            if cached == self._identity(before) and cached == self._identity(target.stat()):
                _check_cancel(cancel)
                if progress:
                    progress("verifying", file.size, file.size)
                return True
        digest, done = hashlib.sha256(), 0
        with target.open("rb") as stream:
            while block := stream.read(self.chunk_size):
                _check_cancel(cancel)
                digest.update(block)
                done += len(block)
                if progress:
                    progress("verifying", done, file.size)
        after = target.stat()
        _check_cancel(cancel)
        valid = digest.hexdigest() == file.sha256 and done == file.size and self._identity(before) == self._identity(after)
        if valid:
            if path is None:
                self._remember_verified(key, self._identity(after))
            else:
                self._verified_parts[str(target.resolve())] = (key, self._identity(after))
        elif path is None:
            self._forget_verified(key)
        return valid

    def verify_entry(self, entry: ModelEntry, progress: Progress | None = None,
                     cancel: threading.Event | None = None, *, force: bool = False) -> VerificationResult:
        for file in entry.files:
            path = self.path_for(file)
            if not path.is_file():
                part = path.with_name(path.name + ".part")
                return VerificationResult(False, "partial" if part.exists() else "missing",
                                          "下载未完成，可续传" if part.exists() else "未下载")
            if not self.verify_file(file, progress, cancel, force=force):
                return VerificationResult(False, "invalid", f"校验失败：{file.relative_path}")
        if entry.verification == "local_sha256":
            from .gguf_inspection import inspect_gguf
            if len(entry.files) != 1 or entry.model_kind not in ("hy", "qwen"):
                return VerificationResult(False, "invalid", "该自定义模型格式尚未适配。")
            path = self.path_for(entry.files[0])
            try:
                with path.open("rb") as stream:
                    metadata = inspect_gguf(stream, path.stat().st_size, cancel=cancel)
                if metadata.architecture != entry.architecture or metadata.model_kind != entry.model_kind:
                    return VerificationResult(False, "invalid", "自定义模型元数据与注册记录不一致。")
            except DownloadCancelled:
                raise
            except (DownloadError, OSError) as exc:
                return VerificationResult(False, "invalid", str(exc))
            return VerificationResult(True, "ready", "本地导入 · 本地 SHA-256 一致；官方来源未经验证")
        return VerificationResult(True, "ready", "已下载 · 校验通过（文件未变时复用记录）")

    @staticmethod
    def _network_failure(exc: BaseException, source_url: str) -> DownloadError:
        status = exc.code if isinstance(exc, HTTPError) else None
        delivery_url = exc.url if isinstance(exc, HTTPError) else source_url
        host = (urlparse(delivery_url).hostname or "").lower()
        if not re.fullmatch(r"[a-z0-9.-]{1,253}", host):
            host = None
        reason = exc.reason if isinstance(exc, URLError) else exc
        if status is not None:
            retryable = status in (408, 429) or 500 <= status <= 599
        elif isinstance(reason, ssl.SSLCertVerificationError) or _local_file_error(reason):
            retryable = False
        else:
            retryable = (_transport_error(reason)
                         or isinstance(exc, URLError) and isinstance(reason, str))
        refresh = (status == 403 and host != urlparse(source_url).hostname
                   and (host or "").endswith((".hf.co", ".modelscope.cn")))
        result = DownloadError(describe_download_error(exc), retryable=retryable,
                               http_status=status, host=host, source_refresh=refresh)
        if isinstance(exc, HTTPError):
            exc.close()
        return result

    def download_file(self, file: ModelFile, progress: Progress | None = None,
                      cancel: threading.Event | None = None, *, restart: bool = False) -> Path:
        if file.url is None:
            raise DownloadError("自定义本地模型没有下载地址，请使用本地导入。")
        # Keep .url stable for links and imported-model provenance. Domestic
        # mirrors are preferred only by the transfer policy, with identical hash.
        sources = tuple(dict.fromkeys((*file.fallback_urls, file.url)))
        for source in sources:
            validate_download_url(source)
        unavailable: set[str] = set()
        refreshed: set[str] = set()
        source_index = 0
        for attempt in range(1, self.max_attempts + 1):
            _check_cancel(cancel)
            source = sources[source_index]
            try:
                return self._download_file_once(file, source, progress, cancel,
                                                restart=restart and attempt == 1)
            except DownloadCancelled:
                raise
            except DownloadError as exc:
                _check_cancel(cancel)
                if not (exc.retryable or exc.source_refresh or exc.http_status in (401, 403, 404)):
                    raise
                if attempt == self.max_attempts:
                    raise DownloadError(
                        f"{describe_download_error(exc)} 已尝试 {attempt} 次；保留进度，可续传或导入已下载文件。",
                        http_status=exc.http_status, host=exc.host) from exc
                if exc.source_refresh and source not in refreshed:
                    # Re-resolve once instead of reusing an expired CDN signature.
                    refreshed.add(source)
                    next_index = source_index
                else:
                    if not exc.retryable:
                        unavailable.add(source)
                    remaining = [(source_index + step) % len(sources)
                                 for step in range(1, len(sources) + 1)
                                 if sources[(source_index + step) % len(sources)] not in unavailable]
                    if not remaining:
                        raise
                    next_index = remaining[0]
                part = self.path_for(file).with_name(self.path_for(file).name + ".part")
                done = part.stat().st_size if part.is_file() else 0
                if progress:
                    progress("source_switching" if next_index != source_index else "retrying", done, file.size)
                delay = min(self.retry_delay * 2 ** (attempt - 1), 8)
                if (cancel or threading.Event()).wait(delay):
                    _check_cancel(cancel)
                _check_cancel(cancel)
                source_index = next_index
            except OSError as exc:
                _check_cancel(cancel)
                # Local disk/permission failures cannot be repaired by networking.
                raise DownloadError(describe_download_error(exc)) from exc
        raise AssertionError("下载尝试次数无效")

    def _download_file_once(self, file: ModelFile, source_url: str,
                            progress: Progress | None, cancel: threading.Event | None,
                            *, restart: bool) -> Path:
        if file.url is None:
            raise DownloadError("自定义本地模型没有下载地址，请使用本地导入。")
        final = self.path_for(file)
        final.parent.mkdir(parents=True, exist_ok=True)
        final = self.path_for(file)
        if final.exists():
            if self.verify_file(file, progress, cancel):
                return final
            raise DownloadError(f"已存在文件校验失败，已保留；请手动移走后重试：{final}")
        part = final.with_name(final.name + ".part")
        if not part.resolve().is_relative_to(self.models_root) or part.is_symlink():
            raise DownloadError("临时文件路径无效")
        if restart and part.exists():
            # An explicit restart preserves the old partial rather than deleting it.
            part.rename(part.with_name(part.name + ".invalid-" + uuid.uuid4().hex))
        offset = part.stat().st_size if part.exists() else 0
        if offset > file.size:
            raise DownloadError("临时文件超过预期大小，请选择重新下载")
        if offset == file.size:
            if not self.verify_file(file, progress, cancel, path=part):
                raise DownloadError("临时文件 SHA-256 校验失败，请选择重新下载")
            self._publish_verified_part(file, part, final, cancel)
            return final
        if shutil.disk_usage(part.parent).free < file.size - offset:
            raise DownloadError("模型目录所在磁盘空间不足，请释放空间后续传。")
        headers = {"User-Agent": "FanyiJi/0.1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        _check_cancel(cancel)
        validate_download_url(source_url)
        try:
            response = self.opener.open(Request(source_url, headers=headers), timeout=self.timeout)
        except (HTTPError, URLError, OSError, IncompleteRead) as exc:
            _check_cancel(cancel)
            raise self._network_failure(exc, source_url) from exc
        with self._receiving(response):
            validate_download_url(response.geturl())
            status = response.status
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise DownloadError("服务器返回压缩内容，无法保证续传校验")
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                     response.headers.get("Content-Range", ""))
                if (not match or int(match[1]) != offset or int(match[3]) != file.size
                        or int(match[2]) != file.size - 1):
                    raise DownloadError("服务器续传范围不匹配，保留临时文件")
                mode, expected = "ab" if offset else "wb", file.size - offset
            elif status == 200:
                # Range was ignored: never append a full response to a partial.
                mode, offset, expected = "wb", 0, file.size
            else:
                raise DownloadError(f"下载状态 {status} 无效，保留临时文件")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) != expected):
                raise DownloadError("服务器文件大小不匹配，保留临时文件")
            _check_cancel(cancel)
            with part.open(mode) as stream:
                done = offset
                # read1 returns after one underlying read, so trickle traffic does
                # not hold cancellation hostage until a whole 1 MiB has arrived.
                read = getattr(response, "read1", response.read)
                if progress:
                    progress("downloading", done, file.size)
                while True:
                    _check_cancel(cancel)
                    try:
                        block = read(self.chunk_size)
                    except (URLError, OSError, IncompleteRead) as exc:
                        _check_cancel(cancel)
                        raise self._network_failure(exc, source_url) from exc
                    if not block:
                        break
                    if done + len(block) > file.size:
                        raise DownloadError("下载内容超出预期大小，保留临时文件")
                    stream.write(block)
                    done += len(block)
                    if progress:
                        progress("downloading", done, file.size)
                stream.flush()
                os.fsync(stream.fileno())
        if done != file.size:
            raise DownloadError("下载文件大小或 SHA-256 校验未完成：网络连接提前结束，保留 .part，可续传。",
                                retryable=True, host=urlparse(source_url).hostname)
        if not self.verify_file(file, progress, cancel, path=part):
            raise DownloadError("下载文件大小或 SHA-256 校验失败；保留 .part，请重试或重新下载")
        self._publish_verified_part(file, part, final, cancel)
        return final

    def download_entry(self, entry: ModelEntry, progress: Progress | None = None,
                       cancel: threading.Event | None = None, *, restart: bool = False) -> None:
        for file in entry.files:
            self.download_file(file, progress, cancel, restart=restart)

    def activate_model(self, catalog: ModelCatalog, model_id: str,
                       progress: Progress | None = None,
                       cancel: threading.Event | None = None) -> Path:
        selected = catalog.get(model_id)
        raw = translation_profile(selected)
        for entry in (*catalog.required_entries, selected):
            result = self.verify_entry(entry, progress, cancel)
            if not result.ready:
                raise DownloadError(f"{entry.name} 尚未就绪：{result.detail}")
        _check_cancel(cancel)
        target = self.models_root / "translation" / "backend.json"
        if not target.resolve().is_relative_to(self.models_root):
            raise DownloadError("backend.json 路径超出模型目录")
        temp = target.with_name(f"backend.json.{uuid.uuid4().hex}.tmp")
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(raw, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _check_cancel(cancel)
        os.replace(temp, target)
        return target

    @staticmethod
    def _stage_configuration(target: Path, raw: bytes) -> Path:
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary

    def activate_models(self, catalog: ModelCatalog, model_id: str | None, asr_catalog, asr_model_id: str,
                        progress: Progress | None = None, cancel: threading.Event | None = None,
                        *, speech_only: bool = True, translation_required: bool = True) -> tuple[Path, ...]:
        """Select local models; an online translator leaves the saved MT profile intact."""
        from .asr_catalog import profile_for_entry, vad_dependencies
        asr_selected = asr_catalog.get(asr_model_id)
        selected = None
        if translation_required:
            selected = catalog.get(model_id)
            translation_selected_profile = translation_profile(selected)
        asr_profile = profile_for_entry(asr_selected)
        entries = (*vad_dependencies(catalog, speech_only=speech_only), asr_selected)
        if selected is not None:
            entries += (selected,)
        for entry in entries:
            result = self.verify_entry(entry, progress, cancel)
            if not result.ready:
                raise DownloadError(f"{entry.name} 尚未就绪：{result.detail}")
        targets = (self.models_root / "asr/backend.json",)
        profiles = (asr_profile,)
        if selected is not None:
            targets += (self.models_root / "translation/backend.json",)
            profiles += (translation_selected_profile,)
        for target in targets:
            if not target.resolve().is_relative_to(self.models_root):
                raise DownloadError("模型配置路径超出模型目录")
            target.parent.mkdir(parents=True, exist_ok=True)
        with _SELECTION_LOCK:
            previous = []
            for target in targets:
                if target.exists() and target.stat().st_size > 65536:
                    raise DownloadError("原模型配置过大，已保留，未切换")
                previous.append(target.read_bytes() if target.exists() else None)
            staged = [self._stage_configuration(target, (json.dumps(profile, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
                      for target, profile in zip(targets, profiles)]
            _check_cancel(cancel)
            committed = []
            try:
                for index, (temporary, target) in enumerate(zip(staged, targets)):
                    _check_cancel(cancel)
                    os.replace(temporary, target)
                    committed.append(index)
            except BaseException as exc:
                failures = []
                defaults = ({"backend": "sense_voice"}, {"backend": "ctranslate2"})
                for index in committed:
                    target = targets[index]
                    old = previous[index]
                    # No deletion: if an implicit default had no file, restore
                    # that same behavior through an explicit legal profile.
                    raw = old if old is not None else (json.dumps(defaults[index]) + "\n").encode("utf-8")
                    try:
                        os.replace(self._stage_configuration(target, raw), target)
                    except OSError as rollback_error:
                        failures.append(type(rollback_error).__name__)
                if failures:
                    raise DownloadError("模型配置保存失败且恢复未完成，未启用；请重新选择模型。恢复错误：" + ", ".join(failures)) from exc
                raise
        return targets
