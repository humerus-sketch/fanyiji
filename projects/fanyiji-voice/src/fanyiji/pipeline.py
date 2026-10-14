"""Bounded, fair two-source pipeline with session-safe UI events."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import AppConfig
from .asr_protocol import AsrProtocolError, contains_asr_protocol, require_subtitle_text
from .language_policy import context_language, continuation_language, normalize_detected_language
from .text_context import (MAX_CONTEXT_CHARACTERS, MAX_CONTEXT_TOKENS, SourceTextContext,
                           join_continuation_text, looks_incomplete)
from .translation_queue import TranslationQueue
from .translation_output import ChineseOutputError, is_chinese_output

LOGGER = logging.getLogger(__name__)
_PROTOCOL_PLACEHOLDER = "[识别控制标记异常，已跳过此片段]"
_WEB_TRANSLATION_NAMES = {"google_web": "Google 网页翻译", "youdao_web": "有道网页翻译"}


def _final_translation_provider(config: AppConfig) -> str:
    return (getattr(config, "hybrid_final_provider", "codex")
            if config.translation_provider == "hybrid" else config.translation_provider)


def _final_cli_name(config: AppConfig) -> str:
    return "国内 CLI" if _final_translation_provider(config) == "domestic_cli" else "Codex"


class FairQueue:
    """Limit pending audio duration, alternating sources when both have work."""

    def __init__(self, max_seconds: float = 30.0):
        self.max_seconds = max_seconds
        self._queues: dict[str, deque[tuple[Any, float]]] = {"mic": deque(), "system": deque()}
        self._pending = 0.0
        self._last = "system"
        self._closed = False
        self._condition = threading.Condition()

    def put(self, channel: str, item: Any, seconds: float) -> bool:
        with self._condition:
            if self._closed or self._pending + seconds > self.max_seconds:
                return False
            self._queues[channel].append((item, seconds))
            self._pending += seconds
            self._condition.notify()
            return True

    def get(self, timeout: float | None = None) -> Any | None:
        with self._condition:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not any(self._queues.values()):
                if self._closed:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("No queued work")
                self._condition.wait(remaining)
            first = "system" if self._last == "mic" else "mic"
            channel = first if self._queues[first] else self._last
            item, seconds = self._queues[channel].popleft()
            self._pending -= seconds
            self._last = channel
            self._condition.notify_all()
            return item

    def close(self, discard: bool = False) -> None:
        with self._condition:
            self._closed = True
            if discard:
                for queue in self._queues.values():
                    queue.clear()
                self._pending = 0.0
            self._condition.notify_all()

    def peek(self, channel: str) -> Any | None:
        with self._condition:
            return self._queues[channel][0][0] if self._queues[channel] else None


@dataclass
class RecognitionJob:
    channel: str
    segment_id: str
    samples: np.ndarray
    ended_at: float
    direction: tuple[str, str]
    start_sample: int | None = None
    end_sample: int | None = None
    generation: int = 0
    ended_by: str = "unknown"
    queued_at: float = field(default_factory=time.monotonic)
    reset_reason: str = ""
    recognition_uncertain: bool = False
    detected_language: str | None = None
    detection_source: str = "unknown"
    context_generation: int | None = None
    timing: dict[str, float] = field(default_factory=dict)
    source_room: str = ""
    source_segment_ids: tuple[str, ...] = ()


@dataclass
class TranslationJob:
    channel: str
    segment_id: str
    text: str
    ended_at: float
    direction: tuple[str, str]
    seconds: float
    context: str = ""
    revision: int = 1
    provisional: bool = False
    source_segment_ids: tuple[str, ...] = ()
    queued_at: float = field(default_factory=time.monotonic)
    context_generation: int = 0
    timing: dict[str, float] = field(default_factory=dict)
    source_room: str = ""
    source_generation: int = 0


@dataclass
class _PendingText:
    job: RecognitionJob
    text: str
    samples: np.ndarray
    ready_at: float
    expires_at: float
    enqueued: bool = False


@dataclass
class _TranslationResult:
    revision: int
    context_generation: int
    draft_succeeded: bool = False
    final_succeeded: bool = False
    draft_failed: bool = False
    final_failed: bool = False


@dataclass
class _Run:
    session: int
    config: AppConfig
    cancel: threading.Event = field(default_factory=threading.Event)
    asr_queue: FairQueue = field(default_factory=FairQueue)
    mt_queue: TranslationQueue = field(default_factory=lambda: TranslationQueue(job_factory=TranslationJob))
    draft_queue: TranslationQueue = field(default_factory=lambda: TranslationQueue(job_factory=TranslationJob))
    threads: list[threading.Thread] = field(default_factory=list)
    capture: Any = None
    segmenters: dict[str, Any] = field(default_factory=dict)
    overloaded: bool = False
    capture_lock: threading.RLock = field(default_factory=threading.RLock)
    source_context: SourceTextContext = field(default_factory=SourceTextContext)
    pending_text: dict[str, _PendingText] = field(default_factory=dict)
    translator: Any = None
    translators: tuple[Any, ...] = ()
    draft_translator: Any = None
    draft_active: int = 0
    translation_results: dict[tuple[str, str], _TranslationResult] = field(default_factory=dict)
    translation_versions: dict[tuple[str, str], int] = field(default_factory=dict)
    translation_lock: threading.RLock = field(default_factory=threading.RLock)
    last_queue_report: float = 0.0
    last_backlog_notice: float = 0.0
    translation_request_count: int = 0
    multilingual_recognition: bool = False
    source_languages: dict[str, str] = field(default_factory=dict)
    context_generations: dict[str, int] = field(default_factory=lambda: {"mic": 0, "system": 0})
    source_directions: dict[str, tuple[str, str]] = field(default_factory=dict)
    source_audio_positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    translation_channels: tuple[str | None, ...] = ()
    audio_clock_anchors: dict[str, dict[int, tuple[int, float]]] = field(default_factory=dict)
    capture_progress: dict[str, Any] = field(default_factory=dict)
    source_room_generations: dict[str, dict[int, str]] = field(default_factory=dict)
    source_identities: dict[str, tuple[str, int]] = field(default_factory=dict)
    audio_processor: Any = None


class SubtitlePipeline:
    def __init__(self, models: Path, on_event: Callable[[dict], None], *,
                 capture_factory=None, source_validator=None):
        self.models = models
        self.on_event = on_event
        self.on_audio = None
        self.on_segment = None
        self.on_filtered = None
        self.capture_factory = capture_factory
        self.source_validator = source_validator
        self._run: _Run | None = None
        self._recognizer = None
        self._translator = None
        self._translators: list[Any] = []
        self._draft_translator = None
        self._parallel_selection: tuple[str, str, str] | None = None
        self._translator_selection = None
        self._speech_filter = None
        self._audio_processors: dict[str, Any] = {}
        self._model_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._translation_window_request: tuple[_Run, Any, object] | None = None
        self._shutdown = False

    def _emit(self, run: _Run, kind: str, **values) -> None:
        if not run.cancel.is_set():
            self.on_event({"session": run.session, "kind": kind, **values})

    def start(self, config: AppConfig, session: int) -> None:
        self.stop()
        with self._lifecycle_lock:
            if self._shutdown:
                return
            run = self._run = _Run(session=session, config=config)
            if not config.enabled_channels:
                self._emit(run, "state", text="待机：两路采集均已关闭")
                return
            preparing = (f"正在准备本地暂译和 {_final_cli_name(config)} 终稿…" if config.translation_provider == "hybrid" else
                "正在准备本地识别和 Codex 翻译…" if config.translation_provider == "codex" else
                "正在准备本地识别和国内 CLI 翻译…" if config.translation_provider == "domestic_cli" else
                f"正在准备本地识别和 {_WEB_TRANSLATION_NAMES[config.translation_provider]}…"
                if config.translation_provider in _WEB_TRANSLATION_NAMES else "正在加载本地模型…")
            self._emit(run, "state", text=preparing)
            self._emit(run, "models_loading")
            loader = threading.Thread(target=self._initialize, args=(run,), name="model-loader", daemon=True)
            run.threads.append(loader)
            loader.start()

    def _initialize(self, run: _Run) -> None:
        models_loaded = False
        try:
            from .audio import AudioCaptureManager
            from .segmentation import SpeechSegmenter, SegmentationConfig
            from .engines import RecognizerEngine, create_translation_engine

            external_asr = (self.models / "asr" / "backend.json").is_file()
            required = ("vad/silero_vad.onnx",) if external_asr else (
                "asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx")
            missing = [name for name in required if not (self.models / name).is_file()]
            if missing:
                raise RuntimeError("本地模型不完整，请先准备本地模型。缺少：" + ", ".join(missing))
            # Rapid settings changes may overlap canceled loader threads. Keep
            # one resident engine set, including while first loading is in flight.
            with self._model_lock:
                if run.cancel.is_set() or self._shutdown:
                    return
                if self._recognizer is None:
                    if external_asr:
                        from .asr_engine import create_recognizer_engine
                        self._recognizer = create_recognizer_engine(self.models / "asr", num_threads=4)
                    else:
                        self._recognizer = RecognizerEngine(self.models / "asr", num_threads=4)
                fixed_languages, codex_channels, batching = self._prepare_translators(run, create_translation_engine)
                if run.cancel.is_set() or self._shutdown:
                    return
                if run.config.speech_only:
                    from .speech_filter import SpeechOnlyFilter
                    if self._speech_filter is None:
                        speech_filter = SpeechOnlyFilter(self.models)
                        with self._lifecycle_lock:
                            if self._shutdown or run.cancel.is_set():
                                speech_filter.close()
                                return
                            self._speech_filter = speech_filter
                    self._speech_filter.preload()
                processing_info = self._prepare_audio_processor(run)
                final_provider = _final_translation_provider(run.config)
                unavailable_translation = ({"backend": "domestic_cli", "provider": "domestic_cli",
                    "cli_provider": getattr(run.config, "domestic_cli_provider", "minimax"),
                    "model_id": getattr(run.config, "domestic_cli_model", "") or "unknown", "available": False}
                    if final_provider == "domestic_cli" else {
                        "backend": "codex_cli", "model_id": run.config.codex_model or "unknown", "available": False})
                model_info = {"asr": self._recognizer.backend_info,
                              "translation": (dict(self._translator.backend_info) if self._translator is not None
                                              else unavailable_translation),
                              "speech_filter": {"enabled": run.config.speech_only,
                                  **(self._speech_filter.backend_info if run.config.speech_only else {})},
                              "audio_processing": processing_info}
                model_info["asr"]["accuracy_processing"] = {
                    "segmentation": asdict(SegmentationConfig()), "speed_recheck": 0.92,
                    "pending_wait_seconds": 1.2, "maximum_phrase_join_parts": 2,
                    "source_context_segments": run.source_context.max_segments,
                    "source_context_ttl_seconds": run.source_context.ttl_seconds,
                    "source_context_max_characters": MAX_CONTEXT_CHARACTERS,
                    "source_context_local_max_tokens": MAX_CONTEXT_TOKENS,
                    "joint_asr_max_seconds": 4, "global_time_stretch": False}
                model_info["asr"].update(
                    recognition_language=("auto" if run.multilingual_recognition else
                        fixed_languages[0] if len(fixed_languages) == 1 else "multiple"),
                    recognition_mode="auto" if run.multilingual_recognition else "fixed",
                    multilingual_requested=run.config.multilingual_asr)
                model_info["asr"].pop("multilingual_fallback", None)
                if run.config.multilingual_asr and not run.multilingual_recognition:
                    model_info["asr"]["multilingual_fallback"] = "translation_engine_requires_fixed_source"
                    self._emit(run, "notice", text="当前翻译模型要求固定源语言，识别沿用通道方向；选择 Codex 或支持混合语言的本地模型可启用自动多语。")
                run.translator = self._translator
                run.translators = tuple(self._translators)
                run.draft_translator = self._draft_translator
                run.translation_channels = (tuple(codex_channels) if final_provider == "codex"
                    and len(run.translators) == len(codex_channels) else (None,) * len(run.translators))
                if final_provider == "codex":
                    model_info["translation"].update(worker_count=len(run.translators), batch_limit=3 if batching else 1,
                        scheduling="ordered_workers_bound_to_audio_channels",
                        worker_channels=list(run.translation_channels))
                elif final_provider == "domestic_cli" or final_provider in _WEB_TRANSLATION_NAMES:
                    model_info["translation"].update(worker_count=len(run.translators), batch_limit=1,
                        scheduling="fair_serial_audio_channels", worker_channels=list(run.translation_channels))
                if run.config.translation_provider == "hybrid":
                    model_info["translation"].update(mode="hybrid", available=self._translator is not None,
                        draft_worker_count=int(self._draft_translator is not None),
                        draft_model=(dict(self._draft_translator.backend_info) if self._draft_translator is not None
                            else {"available": False}))
            if run.cancel.is_set():
                return
            self._emit(run, "models_ready", models=model_info)
            LOGGER.info("Models ready: asr=%s translation=%s provider=%s",
                        model_info["asr"].get("model_id", model_info["asr"].get("model_kind", "unknown")),
                        model_info["translation"].get("model_id", "unknown"), run.config.translation_provider)
            models_loaded = True
            for channel in run.config.enabled_channels:
                run.segmenters[channel] = SpeechSegmenter(channel=channel, vad_model_path=self.models / "vad" / "silero_vad.onnx")
            capture_factory = self.capture_factory or AudioCaptureManager
            run.capture = capture_factory(on_chunk=lambda chunk: self._on_chunk(run, chunk),
                on_status=lambda status: self._emit(run, "source_status", channel=status.channel, text=status.message))
            targets = [("recognition", self._recognize, (run,))]
            targets.extend((f"translation-{index + 1}", self._translate,
                            (run, translator, run.translation_channels[index]))
                           for index, translator in enumerate(run.translators))
            if run.draft_translator is not None:
                targets.append(("translation-draft", self._translate_drafts, (run, run.draft_translator)))
            for name, target, arguments in targets:
                worker = threading.Thread(target=target, args=arguments, name=name, daemon=True)
                run.threads.append(worker)
                worker.start()
            with run.capture_lock:
                if run.cancel.is_set():
                    return
                run.capture.start(enabled_channels=run.config.enabled_channels)
                LOGGER.info("Capture started: channels=%s", ",".join(sorted(run.config.enabled_channels)))
                self._emit(run, "state", text=(f"正在听 · 本地暂译 / {_final_cli_name(run.config)} 终稿" if run.config.translation_provider == "hybrid"
                    else "正在听 · 本地识别 / Codex 订阅翻译" if run.config.translation_provider == "codex" else
                    "正在听 · 本地识别 / 国内 CLI 账号翻译" if final_provider == "domestic_cli" else
                    f"正在听 · 本地识别 / {_WEB_TRANSLATION_NAMES[run.config.translation_provider]}"
                    if run.config.translation_provider in _WEB_TRANSLATION_NAMES else "正在听 · 完全本地"))
        except Exception as exc:
            LOGGER.error("Model initialization failed: %s: %s", type(exc).__name__, str(exc))
            if not models_loaded:
                self._emit(run, "models_failed")
            self._emit(run, "error", text=str(exc))

    def _prepare_audio_processor(self, run: _Run) -> dict:
        mode = run.config.audio_processing
        if mode == "off":
            return {"mode": "off", "enabled": False, "available": True, "model_loaded": False}
        try:
            from .audio_processing import AudioProcessor
            processor = self._audio_processors.get(mode)
            if processor is None:
                processor = AudioProcessor(self.models, mode=mode, num_threads=1)
                with self._lifecycle_lock:
                    if self._shutdown or run.cancel.is_set():
                        processor.close()
                        return {"mode": mode, "enabled": True, "available": False,
                                "model_loaded": False, "reason": "cancelled"}
                    self._audio_processors[mode] = processor
            run.audio_processor = processor
            processor.preload()
            info = dict(processor.backend_info)
            loaded = info.get("ready") is True
            return {"enabled": True, **info, "available": loaded, "model_loaded": loaded}
        except Exception as exc:
            LOGGER.warning("Audio processing unavailable: mode=%s error=%s", mode, type(exc).__name__)
            self._emit(run, "notice", text="语音增强或伴奏分离暂不可用，继续使用原始声音识别。")
            return {"mode": mode, "enabled": True, "available": False,
                    "model_loaded": False, "action": "bypass", "reason": type(exc).__name__}

    def _prepare_translators(self, run: _Run, create_local) -> tuple[tuple[str, ...], list[str], bool]:
        hybrid = run.config.translation_provider == "hybrid"
        final_provider = _final_translation_provider(run.config)
        codex = final_provider == "codex"
        domestic = final_provider == "domestic_cli"
        web = final_provider in _WEB_TRANSLATION_NAMES
        final_name = _final_cli_name(run.config)
        selection = ((run.config.translation_provider, run.config.codex_model,
                      run.config.codex_effort, run.config.codex_speed) if codex else
                     (run.config.translation_provider, "domestic_cli",
                      getattr(run.config, "domestic_cli_provider", "minimax"),
                      getattr(run.config, "domestic_cli_model", "")) if domestic else (run.config.translation_provider,))
        if self._translator_selection is not None and self._translator_selection != selection:
            previous = self._resident_translators()
            with self._lifecycle_lock:
                self._translator, self._translators, self._draft_translator = None, [], None
                self._parallel_selection = None
            for translator in previous:
                close = getattr(translator, "close", None)
                if close is not None:
                    close()
            self._emit(run, "translation_unloaded")
        self._translator_selection = selection

        def publish(translator, *, draft=False):
            with self._lifecycle_lock:
                if self._shutdown or run.cancel.is_set():
                    close = getattr(translator, "close", None)
                    if close is not None:
                        close()
                    return False
                if draft:
                    self._draft_translator = translator
                    run.draft_translator = translator
                else:
                    self._translator = translator
                    self._translators = [translator]
                return True

        if self._translator is None:
            try:
                if codex:
                    from .codex_engine import CodexTranslationEngine
                    translator = CodexTranslationEngine(model=run.config.codex_model,
                        effort=run.config.codex_effort, speed=run.config.codex_speed)
                elif domestic:
                    from .domestic_cli_engine import DomesticCliTranslationEngine
                    translator = DomesticCliTranslationEngine(
                        provider=getattr(run.config, "domestic_cli_provider", "minimax"),
                        model=getattr(run.config, "domestic_cli_model", ""))
                elif web:
                    from .web_translation_engine import GoogleWebTranslationEngine, YoudaoWebTranslationEngine
                    factory = GoogleWebTranslationEngine if run.config.translation_provider == "google_web" else YoudaoWebTranslationEngine
                    translator = factory()
                else:
                    translator = create_local(self.models / "translation", num_threads=4)
                publish(translator)
            except Exception as exc:
                if not hybrid:
                    raise
                LOGGER.warning("Hybrid %s startup failed: %s", final_name, type(exc).__name__)
                self._emit(run, "notice", text=f"{final_name} 终稿暂不可用，本地暂译继续工作。")
        if hybrid and self._draft_translator is None and not run.cancel.is_set() and not self._shutdown:
            try:
                publish(create_local(self.models / "translation", num_threads=4), draft=True)
            except Exception as exc:
                LOGGER.warning("Hybrid local startup failed: %s", type(exc).__name__)
                self._emit(run, "notice", text=f"本地暂译暂不可用，继续等待 {final_name} 终稿。")
        fixed_languages = tuple(sorted({run.config.for_channel(c).source_lang for c in run.config.enabled_channels}))
        channels = sorted(run.config.enabled_channels)
        if run.cancel.is_set() or self._shutdown:
            return fixed_languages, channels, False
        for draft in (False, True) if hybrid else (False,):
            translator = self._draft_translator if draft else self._translator
            if translator is None:
                continue
            if draft:
                with run.translation_lock:
                    if run.cancel.is_set():
                        return fixed_languages, channels, False
                    run.draft_translator = translator
                    run.draft_active += 1
            try:
                if (draft or web or domestic) and getattr(translator, "supports_cancellation", False) is True:
                    translator.preload(cancellation_event=run.cancel)
                else:
                    translator.preload()
            except Exception as exc:
                if not hybrid:
                    raise
                if run.cancel.is_set() or self._shutdown:
                    return fixed_languages, channels, False
                close = getattr(translator, "close", None)
                if close is not None:
                    close()
                with self._lifecycle_lock:
                    if draft:
                        self._draft_translator = None
                        extras = []
                    else:
                        extras = self._translators[1:]
                        self._translator, self._translators = None, []
                        self._parallel_selection = None
                for extra in extras:
                    extra.close()
                LOGGER.warning("Hybrid %s preload failed: %s", "local" if draft else final_name, type(exc).__name__)
                self._emit(run, "notice", text=(f"本地暂译暂不可用，继续等待 {final_name} 终稿。" if draft else
                    f"{final_name} 终稿暂不可用，本地暂译继续工作。"))
            finally:
                if draft:
                    with run.translation_lock:
                        run.draft_active -= 1
            if run.cancel.is_set() or self._shutdown:
                return fixed_languages, channels, False
        if self._translator is None and self._draft_translator is None:
            raise RuntimeError(f"本地模型和 {final_name} 均无法启动，请检查模型文件、登录状态和网络。")
        available = [engine for engine in (self._translator, self._draft_translator) if engine is not None]
        # The source-language policy follows successfully loaded engines. A
        # failed legacy local engine must not disable Codex's mixed-source ASR.
        run.multilingual_recognition = bool(run.config.multilingual_asr and available and all(
            getattr(engine, "supports_mixed_sources", False) is True for engine in available))
        self._recognizer.preload(("auto",) if run.multilingual_recognition else fixed_languages)
        if run.cancel.is_set() or self._shutdown:
            return fixed_languages, channels, False
        information = self._translator.backend_info if self._translator is not None else {}
        batching = codex and isinstance(information, dict) and information.get("supports_batch") is True
        if codex and self._translator is not None and len(channels) > 1:
            from .codex_engine import CodexTranslationEngine
            resolved_effort = information.get("reasoning_effort") or run.config.codex_effort
            target = (information["model_id"], resolved_effort, run.config.codex_speed)
            extra = self._translators[1] if len(self._translators) > 1 else None
            if extra is not None and self._parallel_selection != target:
                with self._lifecycle_lock:
                    self._translators.remove(extra)
                    self._parallel_selection = None
                extra.close()
                extra = None
            try:
                if extra is None:
                    extra = CodexTranslationEngine(model=target[0], effort=target[1], speed=target[2])
                    with self._lifecycle_lock:
                        if self._shutdown or run.cancel.is_set():
                            extra.close()
                            return fixed_languages, channels, batching
                        self._translators.append(extra)
                        self._parallel_selection = target
                extra.preload()
                extra_info = extra.backend_info
                if extra_info.get("model_id") != target[0] or (resolved_effort and
                    extra_info.get("reasoning_effort", resolved_effort) != resolved_effort):
                    raise RuntimeError("第二路翻译的实际模型或推理强度与主路不一致")
            except Exception as exc:
                if run.cancel.is_set() or self._shutdown:
                    return fixed_languages, channels, batching
                if extra is not None:
                    extra.close()
                with self._lifecycle_lock:
                    if extra in self._translators:
                        self._translators.remove(extra)
                    self._parallel_selection = None
                LOGGER.warning("Second Codex worker unavailable: %s", type(exc).__name__)
                self._emit(run, "notice", text="第二路 Codex 翻译暂不可用，继续使用当前模型单路翻译。")
        elif len(self._translators) > 1:
            extras = self._translators[1:]
            with self._lifecycle_lock:
                self._translators = self._translators[:1]
                self._parallel_selection = None
            for extra in extras:
                extra.close()
        return fixed_languages, channels, batching

    def _on_chunk(self, run: _Run, chunk) -> None:
        if run.cancel.is_set() or run.overloaded or not self._source_job_current(run, chunk):
            return
        if self.on_audio is not None:
            self.on_audio(chunk)
        self._emit(run, "level", channel=chunk.channel, level=float(min(1.0, np.sqrt(np.mean(chunk.samples ** 2)) * 6)))
        segmenter = run.segmenters.get(chunk.channel)
        if segmenter is None:
            return
        anchors = run.audio_clock_anchors.setdefault(chunk.channel, {})
        anchors[chunk.generation] = (chunk.end_sample, chunk.captured_at)
        rooms = run.source_room_generations.setdefault(chunk.channel, {})
        rooms[chunk.generation] = getattr(chunk, "source_room", "")
        # A reset may release a held final from the preceding source generation
        # before emitting its reset marker. Its old clock must remain available.
        while len(anchors) > 2:
            anchors.pop(next(iter(anchors)))
        while len(rooms) > 2:
            rooms.pop(next(iter(rooms)))
        for segment in segmenter.feed(chunk):
            if segment.kind == "reset":
                source = run.config.for_channel(chunk.channel)
                marker = RecognitionJob(chunk.channel, "", np.empty(0, np.float32), chunk.captured_at,
                    (source.source_lang, source.target_lang), generation=chunk.generation,
                    reset_reason=segment.reason, source_room=rooms.get(chunk.generation, ""))
                run.asr_queue.put(chunk.channel, marker, 0)
                self._emit(run, "segmentation_reset", channel=chunk.channel, reason=segment.reason)
                continue
            if segment.kind != "final" or not segment.samples.size:
                continue
            source = run.config.for_channel(chunk.channel)
            # Capture-to-display latency includes endpointing: final audio end is
            # reconstructed from the capture sample clock, not worker start time.
            anchor = anchors.get(segment.generation)
            anchor_end, anchor_time = anchor or (chunk.end_sample, chunk.captured_at)
            ended_at = anchor_time - max(0, anchor_end - segment.end_sample) / 16000
            started_at = ended_at - (segment.end_sample - segment.start_sample) / 16000
            # Held hard cuts update emitted_at when actually released. Using
            # queued_at here would conceal that endpointing/hold interval.
            segment_ready_at = getattr(segment, "emitted_at", None)
            if segment_ready_at is None:
                segment_ready_at = time.monotonic()
            timing = {"segment_ready_at": segment_ready_at, "source_segment_count": 1}
            if anchor is not None:
                timing.update(audio_started_at=started_at, audio_ended_at=ended_at,
                    segmentation_seconds=max(0.0, segment_ready_at - ended_at))
            job = RecognitionJob(chunk.channel, f"{run.session}:{chunk.channel}:{segment.utterance_id}",
                segment.samples, ended_at, (source.source_lang, source.target_lang),
                segment.start_sample, segment.end_sample, segment.generation, segment.reason,
                queued_at=time.monotonic(),
                timing=timing, source_room=rooms.get(segment.generation, ""))
            accepted = run.asr_queue.put(chunk.channel, job, segment.samples.size / 16000)
            if self.on_segment is not None:
                self.on_segment(job, accepted)
            if not accepted:
                self._overload(run)
                return
        snapshot = getattr(segmenter, "continuation_snapshot", None)
        if callable(snapshot):
            # Publish only after every completed segment is in the ASR queue.
            # The frozen value is safe to read on the recognizer's worker.
            run.capture_progress[chunk.channel] = snapshot()

    def _overload(self, run: _Run) -> None:
        if run.overloaded or run.cancel.is_set():
            return
        run.overloaded = True
        self._emit(run, "notice", text="字幕处理跟不上，已暂停接收新片段；当前片段未接收。已接收片段继续处理，点击暂停后再继续可恢复。")
        self._emit(run, "state", text="处理积压 · 已暂停采集")
        # Do not join an audio thread from its own callback.
        if run.capture is not None:
            threading.Thread(target=run.capture.stop, name="overload-stop", daemon=True).start()

    def _source_job_current(self, run: _Run, job) -> bool:
        if run.cancel.is_set():
            return False
        if self.source_validator is None:
            return True
        try:
            return bool(self.source_validator(job))
        except Exception as exc:
            LOGGER.warning("Audio source validation failed: %s", type(exc).__name__)
            return False

    def _process_audio(self, run: _Run, job: RecognitionJob, samples: np.ndarray) -> np.ndarray:
        # The callback/capture threads only queue PCM. Native processing stays
        # on the recognition worker and retains the original sample clock.
        started = time.monotonic()
        mode = run.config.audio_processing
        metadata = {"mode": mode, "action": "off" if mode == "off" else "bypass",
                    "reason": "disabled" if mode == "off" else "processor_unavailable"}
        processed = samples
        if mode != "off" and run.audio_processor is not None:
            try:
                result = run.audio_processor.process(samples.copy())
                if (not isinstance(result.samples, np.ndarray) or result.samples.dtype != np.float32
                        or result.samples.shape != samples.shape or not np.isfinite(result.samples).all()):
                    raise ValueError("Invalid processed audio")
                if not isinstance(result.metadata, dict):
                    raise ValueError("Invalid processing metadata")
                processed = result.samples
                metadata = result.metadata
            except Exception as exc:
                # A model failure must never erase captured speech or alter its
                # boundaries; the original filtered PCM remains the fallback.
                metadata.update(action="bypass", reason=type(exc).__name__)
                LOGGER.warning("Audio processing bypass: mode=%s error=%s", mode, type(exc).__name__)
        finished = started if mode == "off" else time.monotonic()
        elapsed = finished - started
        job.timing.update(audio_processing_started_at=started, audio_processing_finished_at=finished,
                          audio_processing_seconds=elapsed)
        rtf = elapsed / (samples.size / 16000) if samples.size else 0.0
        LOGGER.info("AUDIO_PROCESSING channel=%s id=%s mode=%s action=%s duration=%.3f processing=%.3f rtf=%.3f reason=%s",
                    job.channel, job.segment_id, mode, metadata.get("action", "bypass"),
                    samples.size / 16000, elapsed, rtf, metadata.get("reason", "unknown"))
        if mode != "off" and self._source_job_current(run, job):
            backend = metadata.get("backend_info", {})
            loaded = isinstance(backend, dict) and backend.get("ready") is True
            self._emit(run, "audio_processing", channel=job.channel, segment_id=job.segment_id,
                       source_room=job.source_room, source_generation=job.generation,
                       mode=mode, action=metadata.get("action", "bypass"),
                       reason=metadata.get("reason", "unknown"), model_id=metadata.get("model_id", "unknown"),
                       model_loaded=loaded, processing_seconds=elapsed, rtf=rtf, timing=dict(job.timing))
        return processed

    def _recognize(self, run: _Run) -> None:
        from .recognition_quality import QualityRecognizer, RecognitionCancelled

        quality = QualityRecognizer(self._recognizer, cancel_event=run.cancel)
        try:
            while not run.cancel.is_set():
                self._flush_pending(run)
                try:
                    job = run.asr_queue.get(timeout=0.1)
                except TimeoutError:
                    continue
                if job is None:
                    break
                if job.reset_reason:
                    self._break_context(run, job.channel)
                    run.source_audio_positions.pop(job.channel, None)
                    continue
                try:
                    if not self._source_job_current(run, job):
                        self._break_context(run, job.channel)
                        continue
                    samples = job.samples
                    dequeued_at = time.monotonic()
                    job.timing = dict(job.timing)
                    job.timing["recognition_dequeued_at"] = dequeued_at
                    if "segment_ready_at" in job.timing:
                        job.timing["asr_queue_seconds"] = max(0.0, dequeued_at - job.timing["segment_ready_at"])
                    filter_seconds = 0.0
                    job.timing["filter_started_at"] = dequeued_at
                    if run.config.speech_only:
                        filter_started = time.monotonic()
                        job.timing["filter_started_at"] = filter_started
                        decision = self._speech_filter.process(samples)
                        job.timing["filter_finished_at"] = time.monotonic()
                        filter_seconds = job.timing["filter_finished_at"] - filter_started
                        job.timing["speech_filter_seconds"] = filter_seconds
                        if run.cancel.is_set():
                            continue
                        if self.on_filtered is not None:
                            self.on_filtered(job, decision)
                        self._emit(run, "speech_filter", channel=job.channel,
                                   segment_id=job.segment_id, timing=dict(job.timing), **decision.metadata)
                        if decision.skip:
                            self._break_context(run, job.channel)
                            continue
                        samples = decision.samples
                    else:
                        job.timing["filter_finished_at"] = dequeued_at
                        job.timing["speech_filter_seconds"] = 0.0
                    if not self._source_job_current(run, job):
                        self._break_context(run, job.channel)
                        continue
                    samples = self._process_audio(run, job, samples)
                    if not self._source_job_current(run, job):
                        self._break_context(run, job.channel)
                        continue
                    started = time.monotonic()
                    job.timing["asr_started_at"] = started
                    job.timing["recognition_dispatch_seconds"] = (
                        max(0.0, job.timing["filter_started_at"] - dequeued_at)
                        + max(0.0, job.timing["audio_processing_started_at"] - job.timing["filter_finished_at"])
                        + max(0.0, started - job.timing["audio_processing_finished_at"]))
                    selected_asr_language = "auto" if run.multilingual_recognition else job.direction[0]
                    result = quality.recognize(samples, selected_asr_language)
                    if not self._source_job_current(run, job):
                        self._break_context(run, job.channel)
                        continue
                    text = result.text
                    job.recognition_uncertain = bool(result.metadata.get("uncertain", False))
                    job.detected_language = normalize_detected_language(result.metadata.get("detected_language"))
                    job.detection_source = result.metadata.get("detection_source", "unknown")
                    job.timing["asr_finished_at"] = time.monotonic()
                    processing_seconds = job.timing["asr_finished_at"] - started
                    job.timing["asr_seconds"] = processing_seconds
                    LOGGER.info("ASR channel=%s id=%s duration=%.3f processing=%.3f filter=%.3f queue_wait=%.3f selected_asr_language=%s detected_language=%s detection_source=%s",
                        job.channel, job.segment_id, job.samples.size / 16000, processing_seconds,
                        filter_seconds, dequeued_at - job.queued_at, selected_asr_language,
                        job.detected_language or "unknown", job.detection_source)
                    self._emit(run, "recognition", channel=job.channel, segment_id=job.segment_id,
                               text=text, quality=result.metadata, queue_wait_seconds=dequeued_at - job.queued_at,
                               selected_asr_language=selected_asr_language,
                               detected_language=job.detected_language, detection_source=job.detection_source,
                               processing_seconds=processing_seconds, speech_filter_seconds=filter_seconds,
                               audio_processing_seconds=job.timing.get("audio_processing_seconds", 0.0),
                               source_room=job.source_room, source_generation=job.generation,
                               ended_by=job.ended_by, start_sample=job.start_sample, end_sample=job.end_sample,
                               timing=dict(job.timing))
                    if not text or run.cancel.is_set():
                        if not run.cancel.is_set():
                            self._break_context(run, job.channel)
                        continue
                    self._accept_text(run, job, text, samples, quality)
                except RecognitionCancelled:
                    LOGGER.debug("ASR cancelled: session=%s channel=%s", run.session, job.channel)
                    if run.cancel.is_set():
                        break
                    self._break_context(run, job.channel)
                except Exception as exc:
                    if not run.cancel.is_set():
                        self._break_context(run, job.channel)
                    LOGGER.error("ASR job failed: %s", type(exc).__name__)
                    self._emit(run, "notice", text="一个片段的讲话筛选或语音识别失败，已跳过；后续片段继续处理。")
        finally:
            if not run.cancel.is_set():
                self._flush_pending(run, force=True)
            run.mt_queue.close()
            run.draft_queue.close()

    def _enqueue_translation(self, run: _Run, job: RecognitionJob, text: str, *,
                             revision: int = 1, provisional: bool = False,
                             source_ids: tuple[str, ...] = (), seconds: float | None = None,
                             ended_at: float | None = None) -> None:
        if not self._source_text_safe(run, job, text):
            return
        self._prepare_source_context(run, job, text)
        ids = source_ids or job.source_segment_ids or (job.segment_id,)
        queued_at = time.monotonic()
        timing = dict(job.timing)
        timing["translation_queued_at"] = queued_at
        timing["source_segment_count"] = len(ids)
        translated = TranslationJob(job.channel, job.segment_id, text,
            job.ended_at if ended_at is None else ended_at, job.direction,
            job.samples.size / 16000 if seconds is None else seconds,
            context=run.source_context.get(job.channel, job.direction,
                now=job.ended_at if ended_at is None else ended_at), revision=revision,
            provisional=provisional, source_segment_ids=ids,
            context_generation=job.context_generation, queued_at=queued_at, timing=timing,
            source_room=job.source_room, source_generation=job.generation)
        with run.translation_lock:
            run.translation_versions[(job.channel, job.segment_id)] = revision
            if run.config.translation_provider == "hybrid":
                run.translation_results[(job.channel, job.segment_id)] = _TranslationResult(
                    revision, translated.context_generation)
            # Revisions can only follow nearby audio; retain ample history for
            # slow in-flight requests without growing metadata for an entire day.
            if run.config.translation_provider == "hybrid":
                if len(run.translation_results) > 8192 and len(run.translation_results) % 128 == 1:
                    # Pending accepted jobs retain their source identity until
                    # both branches terminate, even during a large disk backlog.
                    finished = [key for key, state in run.translation_results.items()
                        if (state.final_succeeded or state.final_failed)
                        and (state.draft_succeeded or state.draft_failed)]
                    for key in finished[:max(0, len(run.translation_results) - 8192)]:
                        run.translation_results.pop(key)
                        run.translation_versions.pop(key, None)
            else:
                while len(run.translation_versions) > 8192:
                    run.translation_versions.pop(next(iter(run.translation_versions)))
            self._emit(run, "original", channel=job.channel, segment_id=job.segment_id, text=text,
                       revision=revision, provisional=provisional, source_segment_ids=ids,
                       uncertain=job.recognition_uncertain, context_generation=translated.context_generation,
                       source_room=job.source_room, source_generation=job.generation,
                       timing=dict(translated.timing))
        if run.cancel.is_set():
            return
        branches = [(run.mt_queue, "final", run.translator)]
        if run.config.translation_provider == "hybrid":
            branches.append((run.draft_queue, "draft", run.draft_translator))
        for queue, stage, translator in branches:
            # Both resident workers mutate request timestamps independently.
            # Sharing the same job corrupts timings when the slower branch ends.
            branch_queued_at = time.monotonic()
            branch_job = replace(translated, queued_at=branch_queued_at,
                timing={**translated.timing, "translation_queued_at": branch_queued_at})
            if run.config.translation_provider == "hybrid" and translator is None:
                self._publish_translation(run, branch_job, "", failed=True, error="engine_unavailable",
                    **self._translation_identity(run, translator, stage))
                continue
            try:
                queue.put(job.channel, branch_job, branch_job.seconds)
            except Exception as exc:
                self._publish_translation(run, branch_job, "[翻译缓存失败，请检查本地磁盘]", failed=True,
                    error=type(exc).__name__, **self._translation_identity(run, translator, stage))
                self._translation_cache_failed(run, exc)
        self._report_translation_queue(run)

    def _source_text_safe(self, run: _Run, job: RecognitionJob, text: str) -> bool:
        if not self._source_job_current(run, job):
            return False
        try:
            require_subtitle_text(text)
        except AsrProtocolError:
            self._clear_source_context(run, job.channel)
            LOGGER.warning("ASR_PROTOCOL_REJECTED session=%s channel=%s", run.session, job.channel)
            self._emit(run, "notice", text="识别控制标记混入原文，已跳过此片段；后续字幕继续处理。")
            return False
        return True

    def _finish_pending(self, run: _Run, pending: _PendingText) -> None:
        if not self._source_text_safe(run, pending.job, pending.text):
            return
        if pending.enqueued:
            self._emit(run, "subtitle_final", channel=pending.job.channel,
                       segment_id=pending.job.segment_id, revision=1,
                       source_room=pending.job.source_room, source_generation=pending.job.generation,
                       timing=dict(pending.job.timing))
        else:
            self._enqueue_translation(run, pending.job, pending.text)
        self._remember_source(run, pending.job, pending.text)

    @staticmethod
    def _remember_source(run: _Run, job: RecognitionJob, text: str,
                         *, ended_at: float | None = None) -> None:
        # Unverified rechecks must not become trusted background for the next
        # subtitle. TTL follows capture time so processing backlog cannot make
        # an old source phrase fresh again.
        if contains_asr_protocol(text):
            SubtitlePipeline._clear_source_context(run, job.channel)
            return
        language = SubtitlePipeline._context_language(run, job, text)
        if job.recognition_uncertain or (run.multilingual_recognition and language is None):
            SubtitlePipeline._clear_source_context(run, job.channel)
        else:
            if run.multilingual_recognition and run.source_languages.get(job.channel) not in (None, language):
                SubtitlePipeline._clear_source_context(run, job.channel)
            run.source_context.remember(job.channel, job.direction, text,
                now=job.ended_at if ended_at is None else ended_at)
            run.source_languages[job.channel] = language

    @staticmethod
    def _continuation_language(run: _Run, job: RecognitionJob, text: str) -> str | None:
        return continuation_language(text, job.direction[0], job.detected_language,
                                     automatic=run.multilingual_recognition)

    @staticmethod
    def _context_language(run: _Run, job: RecognitionJob, text: str) -> str | None:
        return context_language(text, job.direction[0], job.detected_language,
                                automatic=run.multilingual_recognition)

    @staticmethod
    def _clear_source_context(run: _Run, channel: str) -> None:
        # Only local metadata changes on ASR/capture boundaries. The ordered MT
        # worker passes this generation later, so older queued jobs keep theirs.
        with run.translation_lock:
            run.source_context.clear(channel)
            run.source_languages.pop(channel, None)
            run.context_generations[channel] = run.context_generations.get(channel, 0) + 1

    def _prepare_source_context(self, run: _Run, job: RecognitionJob, text: str, *, check_audio: bool = False) -> None:
        if job.context_generation is not None:
            return
        previous_audio = run.source_audio_positions.get(job.channel)
        previous_direction = run.source_directions.get(job.channel)
        source_identity = (job.source_room, job.generation)
        previous_source = run.source_identities.get(job.channel)
        language = self._context_language(run, job, text)
        pending = run.pending_text.get(job.channel)
        pending_language_changed = pending is not None and run.multilingual_recognition and (
            language is None or language != self._context_language(run, pending.job, pending.text))
        # A pause that is too long for joining a half-sentence still belongs to
        # the same conversation. Use source-context TTL here, keeping the much
        # shorter PCM/text continuation limit exclusively in _audio_contiguous.
        # The TTL is evaluated at the new audio end, matching context.get(now).
        context_samples = round(run.source_context.ttl_seconds * 16000)
        context_end = job.end_sample if job.end_sample is not None else job.start_sample
        discontinuity = check_audio and previous_audio is not None and (
            previous_audio[0] != job.generation
            or (job.start_sample is not None and job.start_sample < previous_audio[1])
            or (context_end is not None and not 0 <= context_end - previous_audio[1] < context_samples))
        if (discontinuity or previous_direction not in (None, job.direction) or pending_language_changed
                or previous_source not in (None, source_identity)):
            self._break_context(run, job.channel)
        if job.recognition_uncertain or (run.multilingual_recognition and (
            language is None or run.source_languages.get(job.channel) not in (None, language))):
            self._clear_source_context(run, job.channel)
        run.source_directions[job.channel] = job.direction
        run.source_identities[job.channel] = source_identity
        if check_audio and job.end_sample is not None:
            run.source_audio_positions[job.channel] = (job.generation, job.end_sample)
        with run.translation_lock:
            job.context_generation = run.context_generations[job.channel]

    def _break_context(self, run: _Run, channel: str) -> None:
        pending = run.pending_text.pop(channel, None)
        if pending is not None:
            self._finish_pending(run, pending)
        self._clear_source_context(run, channel)

    def _flush_pending(self, run: _Run, *, force: bool = False) -> None:
        now = time.monotonic()
        for channel, pending in list(run.pending_text.items()):
            # Read capture progress before peeking the queue. A newer snapshot
            # guarantees its completed segments have already been enqueued.
            progress = run.capture_progress.get(channel)
            following = run.asr_queue.peek(channel)
            continuation_queued = (following is not None and not following.reset_reason
                and self._audio_contiguous(pending.job, following)
                and now < pending.expires_at + 30)
            continuation_impossible = (not continuation_queued
                and self._continuation_window_closed(pending.job, progress))
            if force or continuation_impossible or (now >= pending.expires_at and not continuation_queued):
                self._finish_pending(run, pending)
                run.pending_text.pop(channel, None)
            elif not pending.enqueued and now >= pending.ready_at and not continuation_queued:
                # A recorded continuation is already waiting for ASR. Avoid
                # translating the fragment before its complete revision.
                self._enqueue_translation(run, pending.job, pending.text, provisional=True)
                pending.enqueued = True

    @staticmethod
    def _continuation_window_closed(job: RecognitionJob, progress) -> bool:
        if (progress is None or job.end_sample is None or progress.generation != job.generation
                or progress.speech_active or progress.held_start_sample is not None):
            return False
        return progress.earliest_future_start_sample > job.end_sample + round(1.8 * 16000)

    @staticmethod
    def _audio_contiguous(old: RecognitionJob, following: RecognitionJob) -> bool:
        return (old.end_sample is not None and following.start_sample is not None
            and 0 <= following.start_sample - old.end_sample <= round(1.8 * 16000)
            and old.generation == following.generation and old.direction == following.direction
            and old.source_room == following.source_room
            and (old.start_sample is None or following.end_sample is None
                 or following.end_sample - old.start_sample <= 22 * 16000))

    @staticmethod
    def _joined_timing(old: RecognitionJob, following: RecognitionJob, *,
                       started_at: float, finished_at: float, joint_seconds: float) -> dict[str, float]:
        """Keep the full source envelope and separately total actual CPU work.

        The ASR timestamp envelope can include the next utterance being spoken;
        its duration must never be advertised as continuous recognizer work.
        """
        timing: dict[str, float] = {}
        starts = {"audio_started_at", "recognition_dequeued_at", "filter_started_at",
                  "filter_finished_at", "asr_started_at", "audio_processing_started_at"}
        ends = {"audio_ended_at", "segment_ready_at", "asr_finished_at", "audio_processing_finished_at"}
        totals = {"segmentation_seconds", "asr_queue_seconds", "speech_filter_seconds", "asr_seconds",
                  "joint_asr_seconds", "assembly_seconds", "assembly_processing_seconds",
                  "recognition_dispatch_seconds"}
        totals.add("audio_processing_seconds")
        for key in starts | ends | totals:
            values = [item.timing[key] for item in (old, following) if key in item.timing]
            if values:
                timing[key] = min(values) if key in starts else max(values) if key in ends else sum(values)
        timing["assembly_started_at"] = started_at
        timing["assembly_finished_at"] = finished_at
        timing["joint_asr_seconds"] = timing.get("joint_asr_seconds", 0.0) + joint_seconds
        timing["assembly_seconds"] = timing.get("assembly_seconds", 0.0) + max(
            0.0, finished_at - started_at - joint_seconds)
        timing["assembly_processing_seconds"] = timing.get("assembly_processing_seconds", 0.0) + max(
            0.0, finished_at - started_at - joint_seconds)
        timing["source_segment_count"] = sum(item.timing.get("source_segment_count", 1)
            for item in (old, following))
        return timing

    def _accept_text(self, run: _Run, job: RecognitionJob, text: str,
                     samples: np.ndarray, quality) -> None:
        from .recognition_quality import RecognitionCancelled

        if not self._source_text_safe(run, job, text):
            self._break_context(run, job.channel)
            return
        self._prepare_source_context(run, job, text, check_audio=True)
        now = time.monotonic()
        language = self._continuation_language(run, job, text)
        previous = run.pending_text.pop(job.channel, None)
        if previous is not None:
            old = previous.job
            old_language = self._continuation_language(run, old, previous.text)
            contiguous = self._audio_contiguous(old, job)
            can_join = (language is not None and language == old_language
                and now < previous.expires_at + 30 and contiguous
                and job.ended_by != "natural_pause" and old.ended_by != "natural_pause"
                and looks_incomplete(previous.text, old_language)
                and not previous.text.rstrip().endswith(("?", "？", "!", "！"))
                and text.strip().rstrip("。.!！?？") not in ("はい", "いいえ", "是", "不是", "好", "好的", "谢谢")
                and len(previous.text) + len(text) <= 512)
            if can_join:
                assembly_started_at = time.monotonic()
                joint_seconds = 0.0
                # Re-recognize complete short phrases from the actual filtered
                # PCM. Long phrases use text assembly to keep CPU work bounded.
                gap = job.start_sample - old.end_sample
                joined_audio = np.concatenate((previous.samples, samples))
                raw_joined_text = previous.text + "\n" + text
                joined_text = join_continuation_text(previous.text, text, language)
                assembly_text = joined_text
                joint_metadata = {"performed": False,
                    "reason": "audio_gap_not_recorded" if gap else "audio_over_4_seconds"}
                # A VAD gap is not available as PCM: never replace it with
                # invented silence or claim a text join is continuous audio.
                if gap == 0 and joined_audio.size <= 4 * 16000 and not run.cancel.is_set():
                    joint_started_at = time.monotonic()
                    try:
                        joint = quality.recognize(joined_audio,
                            "auto" if run.multilingual_recognition else job.direction[0])
                        joint_metadata = {"performed": True, "quality": joint.metadata,
                                          "text": joint.text}
                        if joint.text and not joint.metadata.get("uncertain", False):
                            joined_text = joint.text
                            old.detected_language = normalize_detected_language(joint.metadata.get("detected_language"))
                            old.detection_source = joint.metadata.get("detection_source", "unknown")
                    except RecognitionCancelled:
                        LOGGER.debug("ASR joint recheck cancelled: session=%s channel=%s", run.session, job.channel)
                        return
                    except Exception as exc:
                        joint_metadata = {"performed": True, "error": type(exc).__name__}
                    finally:
                        joint_seconds = time.monotonic() - joint_started_at
                if not self._source_job_current(run, job) or not self._source_job_current(run, old):
                    self._clear_source_context(run, job.channel)
                    return
                old.recognition_uncertain = (old.recognition_uncertain or job.recognition_uncertain
                    or bool(joint_metadata.get("quality", {}).get("uncertain", False))
                    or bool(joint_metadata.get("error")))
                if old.recognition_uncertain:
                    self._clear_source_context(run, old.channel)
                    old.context_generation = run.context_generations[old.channel]
                old.timing = self._joined_timing(old, job, started_at=assembly_started_at,
                    finished_at=time.monotonic(), joint_seconds=joint_seconds)
                self._emit(run, "sentence_join", channel=job.channel, segment_id=old.segment_id,
                           revision=2, source_segment_ids=(old.segment_id, job.segment_id),
                           text=joined_text, joint_recognition=joint_metadata,
                           raw_source_texts=(previous.text, text), assembly_text=assembly_text,
                           boundary_normalized=assembly_text != raw_joined_text,
                           uncertain=old.recognition_uncertain,
                           source_start_sample=old.start_sample, source_end_sample=job.end_sample,
                           generation=job.generation, gap_samples=gap, gap_filled=False,
                           source_room=job.source_room, source_generation=job.generation,
                           timing=dict(old.timing))
                self._enqueue_translation(run, old, joined_text, revision=2,
                    source_ids=(old.segment_id, job.segment_id),
                    seconds=joined_audio.size / 16000, ended_at=job.ended_at)
                self._remember_source(run, old, joined_text, ended_at=job.ended_at)
                return
            self._finish_pending(run, previous)
        if (language is not None and job.ended_by != "natural_pause"
                and looks_incomplete(text, language)):
            run.pending_text[job.channel] = _PendingText(job, text, samples.copy(), now + 1.2, now + 15)
            self._emit(run, "original", channel=job.channel, segment_id=job.segment_id, text=text,
                       revision=1, provisional=True, source_segment_ids=(job.segment_id,),
                       uncertain=job.recognition_uncertain, context_generation=job.context_generation,
                       source_room=job.source_room, source_generation=job.generation,
                       timing=dict(job.timing))
        else:
            self._enqueue_translation(run, job, text)
            self._remember_source(run, job, text)

    def _report_translation_queue(self, run: _Run) -> None:
        now = time.monotonic()
        with run.translation_lock:
            if now - run.last_queue_report < 5:
                return
            run.last_queue_report = now
            try:
                snapshot = run.mt_queue.snapshot()
                draft_snapshot = run.draft_queue.snapshot() if run.config.translation_provider == "hybrid" else None
            except Exception as exc:
                self._translation_cache_failed(run, exc)
                return
            LOGGER.info("MT_QUEUE pending=%s oldest=%.3f coalesced=%s spilled=%s",
                snapshot["pending"], snapshot["oldest_age_seconds"], snapshot["coalesced"], snapshot["spilled"])
            self._emit(run, "translation_queue", **snapshot, capture_paused=run.overloaded,
                **({"draft_queue": draft_snapshot} if draft_snapshot is not None else {}))
            if snapshot["oldest_age_seconds"] >= 10 and now - run.last_backlog_notice >= 30:
                run.last_backlog_notice = now
                self._emit(run, "notice", text=f"翻译排队 {snapshot['pending']} 句，最早已等待 {snapshot['oldest_age_seconds']:.0f} 秒；语音识别继续运行。")

    def _translation_cache_failed(self, run: _Run, exc: Exception) -> None:
        LOGGER.error("Translation cache failed: %s", type(exc).__name__)
        with run.translation_lock:
            if run.overloaded or run.cancel.is_set():
                return
            run.overloaded = True
            self._emit(run, "notice", text="翻译缓存读写失败，已暂停采集；已接收的原文仍显示。请检查磁盘空间后暂停再继续；重新开始会取消当前待处理任务。")
            self._emit(run, "state", text="翻译缓存不可用 · 已暂停采集")
        if run.capture is not None:
            threading.Thread(target=run.capture.stop, name="spool-failure-stop", daemon=True).start()

    @staticmethod
    def _translation_identity(run: _Run, translator, stage: str) -> dict[str, str]:
        provider = "local" if stage == "draft" else _final_translation_provider(run.config)
        information = getattr(translator, "backend_info", {}) if translator is not None else {}
        information = information if isinstance(information, dict) else {}
        model = information.get("model_id") or information.get("model_file") or information.get("model_kind")
        if not isinstance(model, str) or not model:
            model = (run.config.codex_model or "unknown" if provider == "codex" else
                     "unknown")
        identity = {"translation_stage": stage, "translation_provider": provider, "translation_model": model}
        if provider == "domestic_cli":
            cli = information.get("cli_provider") or getattr(run.config, "domestic_cli_provider", "minimax")
            identity["translation_cli_provider"] = cli if isinstance(cli, str) else "unknown"
        return identity

    def _publish_translation(self, run: _Run, job: TranslationJob, text: str, **metadata) -> None:
        with run.translation_lock:
            if not self._source_job_current(run, job):
                return
            timing = dict(job.timing)
            timing["event_emitted_at"] = time.monotonic()
            latest = run.translation_versions.get((job.channel, job.segment_id), job.revision)
            if latest != job.revision:
                LOGGER.info("MT_SUPERSEDED channel=%s id=%s revision=%s latest=%s",
                            job.channel, job.segment_id, job.revision, latest)
                self._emit(run, "translation_superseded", channel=job.channel, segment_id=job.segment_id,
                    revision=job.revision, latest_revision=latest, timing=timing, **metadata)
                return
            hybrid = run.config.translation_provider == "hybrid"
            state = None
            if hybrid:
                state = run.translation_results.setdefault((job.channel, job.segment_id),
                    _TranslationResult(job.revision, job.context_generation))
                if (state.revision, state.context_generation) != (job.revision, job.context_generation):
                    self._emit(run, "translation_superseded", channel=job.channel, segment_id=job.segment_id,
                        revision=job.revision, context_generation=job.context_generation,
                        latest_revision=state.revision, latest_context_generation=state.context_generation,
                        timing=timing, **metadata)
                    return
            if contains_asr_protocol(text):
                text = _PROTOCOL_PLACEHOLDER
                metadata.update(failed=True, protocol_rejected=True)
                LOGGER.warning("MT_PROTOCOL_REJECTED session=%s channel=%s", run.session, job.channel)
                self._emit(run, "notice", text="译文包含识别控制标记，已阻止显示；后续字幕继续处理。")
            elif not isinstance(text, str) or (not text.strip() and not metadata.get("failed", False)):
                text = "[此片段翻译失败]"
                metadata.update(failed=True, error="empty_translation")
            elif job.direction[1] == "zh" and not metadata.get("failed", False) and not is_chinese_output(text):
                text = "[译文含非中文字符，已阻止显示]"
                metadata.update(failed=True, error="non_chinese_translation", language_rejected=True)
                LOGGER.warning("MT_LANGUAGE_REJECTED session=%s channel=%s", run.session, job.channel)
                self._emit(run, "notice", text="中文译文仍含外语，已阻止显示和推送；后续字幕继续处理。")
            latency = time.monotonic() - job.ended_at
            LOGGER.info("MT channel=%s id=%s revision=%s processing=%.3f queue_wait=%.3f end_to_end=%.3f batch=%s",
                job.channel, job.segment_id, job.revision, metadata.get("processing_seconds", 0),
                metadata.get("queue_wait_seconds", 0), latency, metadata.get("batch_size", 1))
            kind = "translation"
            if state is not None:
                stage = metadata.get("translation_stage", "final")
                if metadata.get("failed", False):
                    if stage == "draft":
                        state.draft_failed = True
                    else:
                        state.final_failed = True
                    overall_failed = state.draft_failed and state.final_failed and not (
                        state.draft_succeeded or state.final_succeeded)
                    self._emit(run, "translation_status", channel=job.channel, segment_id=job.segment_id,
                        revision=job.revision, context_generation=job.context_generation,
                        provisional=job.provisional, source_segment_ids=job.source_segment_ids,
                        latency=latency, overall_failed=overall_failed,
                        has_translation=state.draft_succeeded or state.final_succeeded,
                        display_stage="final" if state.final_succeeded else "draft" if state.draft_succeeded else "",
                        source_room=job.source_room, source_generation=job.source_generation,
                        final_failed=state.final_failed, timing=timing, **metadata)
                    if not overall_failed:
                        return
                    # Only a failure of both independent branches may replace
                    # the pending label. A valid local draft is never erased.
                    text = f"[本地暂译和 {_final_cli_name(run.config)} 终稿均失败]"
                    metadata.update(translation_stage="final", overall_failed=True)
                elif stage == "draft":
                    state.draft_succeeded, state.draft_failed = True, False
                    if state.final_succeeded:
                        kind = "translation_candidate"
                    metadata["final_failed"] = state.final_failed
                else:
                    state.final_succeeded, state.final_failed = True, False
            self._emit(run, kind, channel=job.channel, segment_id=job.segment_id,
                text=text, latency=latency, context="" if contains_asr_protocol(job.context) else job.context, revision=job.revision,
                provisional=job.provisional and not metadata.get("failed", False),
                source_segment_ids=job.source_segment_ids, context_generation=job.context_generation,
                source_room=job.source_room, source_generation=job.source_generation,
                timing=dict(timing), **metadata)

    def _translation_request(self, run: _Run, translator, jobs: list[TranslationJob], *, fallback=False,
                             stage: str = "final") -> None:
        from .codex_engine import CodexBatchResponseError

        if run.cancel.is_set():
            return
        identity = self._translation_identity(run, translator, stage)
        codex = identity["translation_provider"] == "codex"
        reset = getattr(translator, "reset_conversation", None)
        safe_jobs = []
        for job in jobs:
            if not self._source_job_current(run, job):
                continue
            try:
                require_subtitle_text(job.text)
                require_subtitle_text(job.context)
            except AsrProtocolError:
                self._clear_source_context(run, job.channel)
                if codex and callable(reset):
                    reset(f"{run.session}:{job.channel}")
                LOGGER.warning("MT_SOURCE_PROTOCOL_REJECTED session=%s channel=%s", run.session, job.channel)
                self._emit(run, "notice", text="识别控制标记混入原文或前文，已跳过此片段；后续字幕继续处理。")
                self._publish_translation(run, job, _PROTOCOL_PLACEHOLDER, failed=True,
                    protocol_rejected=True, **identity)
            else:
                safe_jobs.append(job)
        jobs = safe_jobs
        if not jobs or run.cancel.is_set():
            return
        started = time.monotonic()
        for job in jobs:
            job.timing = dict(job.timing)
            job.timing.setdefault("translation_started_at", started)
        with run.translation_lock:
            run.translation_request_count += 1
            batch_id = f"{run.session}:mt:{run.translation_request_count}"
        texts, error = {}, None
        conversation_key = f"{run.session}:{jobs[0].channel}"
        isolated = codex and any(
            job.provisional or job.revision > 1 for job in jobs)
        if stage == "draft":
            with run.translation_lock:
                if run.cancel.is_set():
                    return
                run.draft_active += 1
        try:
            if isolated and callable(reset):
                reset(conversation_key)
            information = translator.backend_info
            source_language = ("mixed" if run.multilingual_recognition
                and getattr(translator, "supports_mixed_sources", False) is True else jobs[0].direction[0])
            if len(jobs) > 1:
                texts = translator.translate_batch([
                    {"id": job.segment_id, "text": job.text, "context": job.context} for job in jobs
                ], source_language, jobs[0].direction[1], cancellation_event=run.cancel,
                    conversation_key=conversation_key, context_generation=jobs[0].context_generation)
            else:
                job = jobs[0]
                if codex:
                    text = translator.translate(job.text, source_language, job.direction[1], context=job.context,
                        cancellation_event=run.cancel, conversation_key=conversation_key,
                        context_generation=job.context_generation)
                elif isinstance(information, dict) and information.get("supports_source_context"):
                    options = {"context": job.context}
                    if getattr(translator, "supports_cancellation", False) is True:
                        options["cancellation_event"] = run.cancel
                    text = translator.translate(job.text, source_language, job.direction[1], **options)
                else:
                    options = {"cancellation_event": run.cancel} if getattr(translator, "supports_cancellation", False) is True else {}
                    text = translator.translate(job.text, source_language, job.direction[1], **options)
                texts = {job.segment_id: text}
            # JSON validation alone does not make each returned translation
            # safe. Reject control-token pollution before the thread is reused.
            if codex and callable(reset) and any(contains_asr_protocol(text) for text in texts.values()):
                reset(conversation_key)
        except Exception as exc:
            error = exc
        finally:
            if isolated and callable(reset):
                try:
                    reset(conversation_key)
                except Exception as exc:
                    error = error or exc
            if stage == "draft":
                with run.translation_lock:
                    run.draft_active -= 1
        finished = time.monotonic()
        duration = finished - started
        if identity["translation_provider"] == "domestic_cli" and error is None and not run.cancel.is_set():
            # Some clients report the executed model only with the completed
            # answer. Never label a saved selection as an observed model.
            identity = self._translation_identity(run, translator, stage)
            backend = getattr(translator, "backend_info", {})
            if isinstance(backend, dict):
                self._emit(run, "translation_backend", information=dict(backend))
        for job in jobs:
            job.timing["translation_finished_at"] = finished
            job.timing["translation_seconds"] = job.timing.get("translation_seconds", 0.0) + duration
        metrics = getattr(translator, "last_request_metrics", {})
        metrics = dict(metrics) if isinstance(metrics, dict) else {}
        request_metadata = {"batch_id": batch_id, "batch_size": len(jobs),
            "processing_seconds": duration, "rpc_metrics": metrics, "fallback": fallback, **identity}
        LOGGER.info("MT_REQUEST id=%s batch=%s processing=%.3f failed=%s rpc=%s", batch_id,
            len(jobs), duration, type(error).__name__ if error else "none", metrics)
        self._emit(run, "translation_request", **request_metadata, failed=error is not None)
        if run.cancel.is_set():
            return
        if len(jobs) > 1 and isinstance(error, CodexBatchResponseError):
            self._emit(run, "notice", text="批次译文格式未通过校验，正在逐句重译；原文顺序保持不变。")
            for job in jobs:
                if run.cancel.is_set():
                    break
                self._translation_request(run, translator, [job], fallback=True, stage=stage)
            return
        if error is not None:
            LOGGER.error("MT job failed: %s", type(error).__name__)
            if isinstance(error, ChineseOutputError):
                self._emit(run, "notice", text="译文经纠正仍含非中文字符，已阻止显示和推送；后续字幕继续处理。")
            elif codex:
                self._emit(run, "notice", text="Codex 翻译暂不可用，请检查联网、ChatGPT 登录或订阅额度；可在更改设置切回本地模型。")
            elif identity["translation_provider"] == "domestic_cli":
                self._emit(run, "notice", text="国内 CLI 翻译暂不可用，请检查官方客户端、账号登录、网络和可用额度；可在更改设置切换翻译源。")
            elif identity["translation_provider"] in _WEB_TRANSLATION_NAMES:
                name = _WEB_TRANSLATION_NAMES[identity["translation_provider"]]
                self._emit(run, "notice", text=f"{name}暂不可用，请检查网络或打开翻译器完成页面验证；可在更改设置切换翻译源。")
        for job in jobs:
            self._publish_translation(run, job, texts.get(job.segment_id, "[此片段翻译失败]"),
                queue_wait_seconds=started - job.queued_at, failed=error is not None,
                **({"error": type(error).__name__} if error is not None else {}), **request_metadata)

    def _translate(self, run: _Run, translator=None, channel: str | None = None, *, stage: str = "final") -> None:
        # Every worker stays bound to its selected engine during settings
        # changes. Web translators use one fair, serial queue for both sources.
        if translator is None:
            translator = run.translator if run.translator is not None else self._translator
        information = translator.backend_info
        batching = (stage == "final" and _final_translation_provider(run.config) == "codex" and isinstance(information, dict)
                    and information.get("supports_batch") is True)
        queue = run.draft_queue if stage == "draft" else run.mt_queue
        while not run.cancel.is_set():
            try:
                if batching:
                    jobs = queue.get_batch(limit=3, max_characters=1200, channel=channel)
                else:
                    job = queue.get(channel=channel)
                    jobs = [] if job is None else [job]
            except Exception as exc:
                self._translation_cache_failed(run, exc)
                return
            if not jobs or run.cancel.is_set():
                return
            self._translation_request(run, translator, jobs, stage=stage)
            self._report_translation_queue(run)

    def _translate_drafts(self, run: _Run, translator) -> None:
        # One local model owns one fair queue for both audio channels. Codex
        # workers use a separate queue, so neither branch blocks the other.
        self._translate(run, translator, stage="draft")

    def _resident_translators(self) -> list[Any]:
        with self._lifecycle_lock:
            owned = list(self._translators)
            if self._translator is not None and not any(item is self._translator for item in owned):
                owned.insert(0, self._translator)
            if self._draft_translator is not None and not any(item is self._draft_translator for item in owned):
                owned.append(self._draft_translator)
            return owned

    def show_translation_window(self) -> bool:
        """Schedule the current web translator window without blocking Qt."""
        with self._lifecycle_lock:
            run = self._run
            if (self._shutdown or run is None or run.cancel.is_set()
                    or run.config.translation_provider not in _WEB_TRANSLATION_NAMES
                    or self._translator_selection != (run.config.translation_provider,)
                    or run.translator is None or run.translator is not self._translator
                    or not callable(getattr(run.translator, "show_window", None))):
                return False
            pending = self._translation_window_request
            if pending is not None and pending[0] is run and pending[1] is run.translator:
                return True
            request = (run, run.translator, object())
            self._translation_window_request = request
            worker = threading.Thread(target=self._show_translation_window, args=request,
                                      name="translation-browser-show", daemon=True)
            run.threads.append(worker)
            worker.start()
            return True

    def _show_translation_window(self, run: _Run, translator, token: object) -> None:
        try:
            with self._lifecycle_lock:
                if (self._shutdown or self._run is not run or run.cancel.is_set()
                        or translator is not self._translator or run.translator is not translator
                        or self._translator_selection != (run.config.translation_provider,)):
                    return
            # The helper may wait for an in-flight translation. It owns its
            # own generation checks; do not keep pipeline/UI locks held here.
            translator.show_window(cancellation_event=run.cancel)
        except Exception as exc:
            LOGGER.warning("Translation browser show failed: provider=%s error=%s",
                           run.config.translation_provider, type(exc).__name__)
            self._emit(run, "notice", text="无法打开翻译器页面，请检查内置浏览器运行状态后重试。")
        finally:
            with self._lifecycle_lock:
                request = self._translation_window_request
                if request is not None and request[2] is token:
                    self._translation_window_request = None

    def stop(self) -> None:
        with self._lifecycle_lock:
            run, self._run = self._run, None
            if run is None:
                return
            run.cancel.set()
            pending = self._translation_window_request
            if pending is not None and pending[0] is run:
                self._translation_window_request = None
        run.asr_queue.close(discard=True)
        run.mt_queue.close(discard=True)
        run.draft_queue.close(discard=True)
        if _final_translation_provider(run.config) in ("codex", "domestic_cli", *_WEB_TRANSLATION_NAMES):
            owned = run.translators or ((run.translator,) if run.translator is not None else self._translators)
            for translator in owned:
                cancel = getattr(translator, "cancel_pending", None)
                if cancel is not None:
                    cancel()
        with run.translation_lock:
            draft = run.draft_translator if run.draft_active else None
        if draft is not None:
            cancel = getattr(draft, "cancel_pending", None)
            if cancel is not None:
                cancel()
        with run.capture_lock:
            if run.capture is not None:
                run.capture.stop()
        # Engines stay resident for resume/settings changes. Native inference is
        # allowed to finish; canceled sessions cannot publish its late results.
        for thread in run.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=0.5)

    def shutdown(self) -> None:
        """Permanently close resident resources; stop() remains resumable."""
        with self._lifecycle_lock:
            if self._shutdown:
                return
            self._shutdown = True
            translators = self._resident_translators()
        try:
            self.stop()
        finally:
            try:
                for translator in translators:
                    close = getattr(translator, "close", None)
                    if close is not None:
                        close()
            finally:
                close_asr = getattr(self._recognizer, "close", None)
                try:
                    if close_asr is not None:
                        close_asr()
                finally:
                    close_filter = getattr(self._speech_filter, "close", None)
                    if close_filter is not None:
                        close_filter()
                    for processor in self._audio_processors.values():
                        processor.close()
                    self._audio_processors.clear()
