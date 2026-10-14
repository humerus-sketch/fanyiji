"""Independent corpus collection and local teacher comparisons; no deployment."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import logging
from pathlib import Path
import queue
import threading
import time
import uuid

from .config import AppConfig
from .models_download import ModelCatalog
from .specialized_store import SpecializedStore

LOGGER = logging.getLogger(__name__)
TEACHER_IDS = ("hy-mt2-1.8b-q6_k", "hy-mt2-7b-q6_k", "qwen3-1.7b-q8_0")
_EVENTS = {"original", "translation", "translation_candidate", "translation_status",
           "subtitle_final", "sentence_join", "recognition", "speech_filter"}


class SpecializedModelService:
    """Disk work is isolated from capture/ASR/translation threads.

    The deployment lock is deliberately a constant, rather than a file flag.
    Neither collected pairs nor an adapter on disk can activate this model.
    """

    def __init__(self, root: Path, models: Path, *, request_compare=None, store=None, engine_factory=None):
        self.root = Path(root).resolve()
        self.models = Path(models).resolve()
        self.store = store if store is not None else SpecializedStore(self.root)
        self._request_compare = request_compare
        self._engine_factory = engine_factory
        self._lock = threading.RLock()
        self._enabled = True
        self._closed = False
        self._busy = False
        self._status = "采集开启 · 等待完整语句"
        self._error = ""
        self._session = None
        self._engine = None
        self._cancel = threading.Event()
        self._queue = queue.Queue(maxsize=128)
        self._preferences = self.root / "collection.json"
        if self._preferences.exists():
            try:
                saved = json.loads(self._preferences.read_text(encoding="utf-8"))
                if type(saved.get("enabled")) is not bool:
                    raise ValueError("采集开关无效")
                self._enabled = saved["enabled"]
            except (OSError, ValueError, AttributeError):
                self._enabled = False
                self._error = "专用模型采集设置读取失败，已暂停采集；可重新开启。"
        self._worker = threading.Thread(target=self._write_loop, name="specialized-corpus-writer", daemon=True)
        self._worker.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def set_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("采集开关必须为布尔值")
        with self._lock:
            if self._closed:
                return
            self.root.mkdir(parents=True, exist_ok=True)
            temp = self._preferences.with_suffix(".tmp")
            temp.write_text(json.dumps({"enabled": enabled}, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self._preferences)
            self._enabled = enabled
            self._error = ""
            self._status = "采集开启 · 等待完整语句" if enabled else "已暂停积累训练数据"

    def fail_collection(self) -> None:
        with self._lock:
            self._enabled = False
            self._error = "专用模型采集失败，已暂停积累；字幕继续运行。"

    def begin_session(self, epoch: int, config: AppConfig) -> None:
        with self._lock:
            self._session = {"epoch": epoch, "id": uuid.uuid4().hex, "config": config,
                             "model": config.codex_model if config.translation_provider in ("codex", "hybrid") else "unknown",
                             "models": {"codex": config.codex_model or "unknown", "local": "unknown"}}

    def _context(self, epoch: int, channel: str, event: dict | None = None):
        with self._lock:
            current = self._session
            if not self._enabled or self._closed or current is None or current["epoch"] != epoch:
                return None
            try:
                source = current["config"].for_channel(channel)
            except ValueError:
                return None
            if not source.enabled:
                return None
            provider = current["config"].translation_provider
            model = current["model"]
            if event and event.get("kind") in ("translation", "translation_candidate", "translation_status"):
                actual_provider = event.get("translation_provider")
                if actual_provider in ("local", "codex"):
                    provider = actual_provider
                elif provider == "hybrid":
                    # Never label a parallel branch with the overall mode name.
                    provider = {"draft": "local", "final": "codex"}.get(event.get("translation_stage"))
                    if provider is None:
                        return None
                model = event.get("translation_model") or current["models"].get(provider, model)
            return current["id"], source.direction, provider, str(model)

    def on_event(self, event: dict) -> None:
        if event.get("kind") == "models_ready":
            with self._lock:
                current = self._session
                if current is not None and event.get("session") == current["epoch"]:
                    info = event.get("models", {}).get("translation", {})
                    current["model"] = str(info.get("model_id") or info.get("model")
                                           or info.get("model_file") or current["model"])
                    provider = "codex" if current["config"].translation_provider in ("codex", "hybrid") else "local"
                    current["models"][provider] = current["model"]
                    draft_info = info.get("draft_model", {})
                    if isinstance(draft_info, dict):
                        current["models"]["local"] = str(draft_info.get("model_id") or draft_info.get("model")
                            or draft_info.get("model_file") or current["models"]["local"])
            return
        if event.get("kind") not in _EVENTS:
            return
        context = self._context(event.get("session"), event.get("channel", ""), event)
        if context is not None:
            sid, direction, provider, model = context
            self._enqueue(("event", (sid, deepcopy(event)),
                           {"direction": direction, "provider": provider, "model": model}))

    def on_segment(self, job, accepted: bool) -> None:
        # Pipeline IDs include the run epoch; late callbacks must not migrate
        # an old recording into a new settings/direction session.
        try:
            epoch = int(job.segment_id.split(":", 1)[0])
        except (ValueError, AttributeError):
            return
        context = self._context(epoch, job.channel)
        if context is not None and accepted:
            copied = replace(job, samples=job.samples.copy())
            self._enqueue(("audio", (context[0], copied, accepted), {}))

    def _enqueue(self, task) -> None:
        with self._lock:
            if self._closed or not self._enabled:
                return
            try:
                self._queue.put_nowait(task)
            except queue.Full:
                self._enabled = False
                self._error = "专用模型写入跟不上，已暂停数据积累；字幕继续运行。"

    def _write_loop(self):
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                operation, args, kwargs = task
                if operation == "barrier":
                    args[0].set()
                elif operation == "event":
                    self.store.record_event(*args, **kwargs)
                elif operation == "audio":
                    self.store.save_audio(*args, **kwargs)
            except Exception as exc:
                LOGGER.error("专用模型数据写入失败：%s", type(exc).__name__)
                with self._lock:
                    self._enabled = False
                    self._error = "专用模型数据写入失败，已暂停积累；字幕继续运行。"
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 15) -> None:
        marker = threading.Event()
        self._queue.put(("barrier", (marker,), {}), timeout=timeout)
        if not marker.wait(timeout):
            raise TimeoutError("专用模型数据仍在写入，请稍后重试")

    def summary(self) -> dict:
        result = self.store.summary()
        with self._lock:
            result.update(enabled=self._enabled, busy=self._busy, status=self._status,
                          error=self._error or result.get("collection_error", ""), deployment_locked=True)
        return result

    def list_samples(self, limit=100):
        return self.store.list_samples(limit)

    def review_candidate(self, sample_id, candidate_id, *, source_verified, rights_confirmed):
        return self.store.review_candidate(sample_id, candidate_id,
                                          source_verified=source_verified, rights_confirmed=rights_confirmed)

    def export_dataset(self):
        with self._lock:
            if self._busy:
                raise ValueError("本地模型对照进行中，请完成后导出")
        self.flush()
        return self.store.export_dataset()

    def prepare_training(self):
        from .specialized_training import prepare_training
        return prepare_training(self.root)

    def available_teachers(self):
        catalog = ModelCatalog.default()
        available = []
        for mid in TEACHER_IDS:
            entry = catalog.get(mid)
            if all((self.models / f.relative_path).is_file()
                   and (self.models / f.relative_path).stat().st_size == f.size for f in entry.files):
                available.append(entry)
        return available

    def compare_local(self) -> bool:
        with self._lock:
            if self._closed or self._busy:
                return False
            if not self.available_teachers():
                raise ValueError("尚未找到已完整下载的本地对照模型，请先通过模型管理下载")
            if not self.store.pending_comparisons([e.id for e in self.available_teachers()], limit=1):
                raise ValueError("暂无待对照的完整语句；请先开启采集积累数据")
            if self._request_compare is None:
                raise ValueError("当前环境未连接字幕暂停控制，不能启动对照")
            self._busy = True
            self._cancel.clear()
            self._status = "正在暂停字幕并释放模型 · 每批最多30句"
            self._error = ""
        try:
            self._request_compare()
        except Exception:
            with self._lock:
                self._busy = False
            raise
        return True

    def run_comparison(self) -> None:
        """Caller shuts down resident audio/models before this GPU batch."""
        engine = None
        completed = 0
        try:
            self.flush()
            teachers = self.available_teachers()
            rows = self.store.pending_comparisons([e.id for e in teachers], limit=30)
            if self._engine_factory is None:
                from .gguf_engine import GgufTranslationEngine
                factory = GgufTranslationEngine
            else:
                factory = self._engine_factory
            for entry in teachers:
                batch = [row for row in rows if entry.id in row.get("pending_model_ids", [])]
                if not batch or self._cancel.is_set():
                    continue
                with self._lock:
                    self._status = f"正在对照 {entry.name} · 已保存 {completed} 条候选"
                    if self._closed or self._cancel.is_set():
                        break
                    engine = factory(self.models / "translation", config={
                        "model_file": Path(entry.files[0].relative_path).name,
                        "model_kind": entry.model_kind, "runtime_preference": "auto"})
                    self._engine = engine
                try:
                    engine.preload()
                    for row in batch:
                        if self._cancel.is_set():
                            break
                        source, target = ("ja", "zh") if row["direction"] == "ja_to_zh" else ("zh", "ja")
                        started = time.monotonic()
                        text = engine.translate(row["source"], source, target, context=row.get("context", ""))
                        if self._cancel.is_set():
                            break
                        self.store.add_candidate(row["id"], text=text, origin="local", model=entry.id,
                                                 processing_seconds=time.monotonic() - started)
                        completed += 1
                finally:
                    engine.close()
                    with self._lock:
                        self._engine = None
                    engine = None
            with self._lock:
                self._status = f"对照{'已取消' if self._cancel.is_set() else '完成'} · 本次保存 {completed} 条本地候选 · 尚未训练"
        except Exception as exc:
            LOGGER.error("专用模型对照失败：%s", type(exc).__name__)
            with self._lock:
                self._error = f"本地对照未完成（{type(exc).__name__}）；已保存的候选保留。"
                self._status = "对照停止 · 尚未训练"
        finally:
            if engine is not None:
                engine.close()
            with self._lock:
                self._engine = None
                self._busy = False

    def cancel_comparison(self) -> None:
        self._cancel.set()
        with self._lock:
            engine = self._engine
        if engine is not None:
            engine.close()

    def finish_comparison(self, *, failed: bool = False) -> None:
        """Also release the busy gate when shutdown fails before the batch runs."""
        with self._lock:
            self._busy = False
            if failed:
                self._error = "本地对照中断，已有数据保留；可重新开始。"
                self._status = "对照停止 · 尚未训练"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.cancel_comparison()
        self._queue.put(None)
        self._worker.join(timeout=20)
        if self._worker.is_alive():
            # Do not close SQLite underneath an in-progress audio write.
            LOGGER.error("专用模型写入未在退出期限内结束")
        else:
            self.store.close()
