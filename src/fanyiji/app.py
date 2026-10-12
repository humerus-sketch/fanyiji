"""Desktop lifecycle. First-run settings are completed before any capture starts."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from .config import AppConfig, ConfigError, load_config, save_config
from .i18n import set_language, tr
from .identity import APP_NAME, INSTANCE_NAME, LEGACY_APP_ID
from .paths import data_root, models_root


class EventBridge(QObject):
    event = Signal(object)


class AppController(QObject):
    def __init__(self, window, config: AppConfig, config_path: Path, diagnostics_output: Path | None = None,
                 *, specialized_service=None):
        super().__init__(window)
        from .pipeline import SubtitlePipeline
        self.window, self.config, self.config_path = window, config, config_path
        self.epoch, self.paused, self.closed = 0, False, False
        self.active_models: dict[str, dict] = {}
        from .latency import LatencyTracker
        self.latency = LatencyTracker()
        if hasattr(window, "subtitle_presented"):
            window.subtitle_presented.connect(self._on_subtitle_presented)
        self.model_state = "not_loaded"
        self.model_dialog = None
        self.browser_service = None
        self.browser_dialog = None
        self.audio_processing_dialog = None
        self.danmaku = None
        self._danmaku_startup_checked = False
        self._waiting_for_download_close = False
        window.close_guard = self._may_close_window
        self.recorder = None
        self.specialized = specialized_service
        self._collection_error_reported = False
        if self.specialized is None and hasattr(window, "specialized_models_requested"):
            try:
                from .specialized_model import SpecializedModelService
                self.specialized = SpecializedModelService(data_root() / "specialized_model", models_root(),
                                                           request_compare=self._request_local_comparison)
            except Exception as exc:
                logging.error("专用模型数据服务不可用：%s", type(exc).__name__)
                window.append_notice("专用模型数据目录不可用；字幕仍可运行。")
        if hasattr(window, "specialized_models_requested"):
            window.specialized_models_requested.connect(self.change_specialized_model)
        if diagnostics_output is not None:
            from .diagnostics import SessionRecorder
            self.recorder = SessionRecorder(diagnostics_output, config)
        self.bridge = EventBridge(self)
        self.bridge.event.connect(self._on_event)
        from .live_source_service import LiveSourceService
        self.live_source_service = LiveSourceService(self, data_dir=data_root())
        self.pipeline = SubtitlePipeline(models_root(), self._pipeline_event)
        self._attach_recording()
        self.diagnostic_timer = QTimer(self)
        self.diagnostic_timer.setInterval(30000)
        self.diagnostic_timer.timeout.connect(self._diagnostic_snapshot)
        if self.recorder is not None:
            self.diagnostic_timer.start()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="session-control")
        self.live_source_service.configuration_changed.connect(self._source_configuration_changed)
        window.pause_toggled.connect(self.set_paused)
        window.settings_requested.connect(self.change_settings)
        if hasattr(window, "browser_requested"):
            window.browser_requested.connect(self.open_browser)
        if hasattr(window, "audio_processing_requested"):
            window.audio_processing_requested.connect(self.open_audio_processing)
        if hasattr(window, "models_requested"):
            window.models_requested.connect(self.change_models)
        window.closing.connect(self.close)

    def _ensure_browser_service(self):
        from .browser_service import BrowserService
        if self.browser_service is None:
            self.browser_service = BrowserService(self)
            self.browser_service.live_source_service = self.live_source_service
            self.browser_service.audio_processing_mode_getter = lambda: self.config.audio_processing
            self.browser_service.audio_processing_apply_mode = self._apply_audio_processing_mode
            self.browser_service.audio_processing_requested = self.open_audio_processing
            self.browser_service.live_source_changed.connect(self.live_source_service.bind_source)
            self.live_source_service.bind_source(self.browser_service.live_source_snapshot())
            self.browser_service.prefs_changed.connect(self._configure_danmaku)
            self.browser_service.content_filter_requested.connect(self.open_content_filter)
            if self.browser_service.danmaku_available:
                from .danmaku_service import DanmakuService
                try:
                    self.danmaku = DanmakuService(self.browser_service, self,
                        store_path=self.browser_service.preferences_path.parent / "danmaku-queue.sqlite3",
                        reset_on_start=True)
                    self.danmaku.configure_ai(self.config)
                    self.danmaku.status.connect(self.browser_service.danmaku_status.emit)
                    if hasattr(self.window, "set_danmaku_status"):
                        self.danmaku.status.connect(self.window.set_danmaku_status)
                    self.browser_service.danmaku_resume_requested.connect(self.danmaku.resume)
                    self.browser_service.danmaku_clear_requested.connect(self.danmaku.clear_pending)
                    self.danmaku.begin_session(self.epoch)
                    self._configure_danmaku(self.browser_service.danmaku_preferences())
                except Exception as exc:
                    logging.error("DANMAKU initialization_failed type=%s", type(exc).__name__)
                    self.window.append_notice(tr("弹幕队列暂不可用，字幕翻译继续运行。"))
        return self.browser_service

    def _source_configuration_changed(self) -> None:
        # Restart fences recognition and translation queued for the old room.
        if not self.closed:
            self.start()

    def _apply_audio_processing_mode(self, mode: str) -> bool:
        if self.closed:
            return False
        # Read current config at click time; a settings dialog may have changed
        # unrelated preferences since the audio controls were opened.
        return self.apply_config(replace(self.config, audio_processing=mode))

    def open_audio_processing(self) -> None:
        if self.closed:
            return
        from .audio_processing_dialog import AudioProcessingDialog
        browser = self._ensure_browser_service()
        if self.audio_processing_dialog is None:
            self.audio_processing_dialog = AudioProcessingDialog(
                self.live_source_service, self.window, browser_service=browser,
                mode_getter=lambda: self.config.audio_processing,
                apply_mode=self._apply_audio_processing_mode, models_dir=models_root())
        self.audio_processing_dialog.exec()

    def _configure_danmaku(self, preferences):
        if self.danmaku is not None and not self.closed:
            try:
                desired_ai = {name: getattr(self.config, name) for name in ("codex_model", "codex_effort", "codex_speed")}
                if self.danmaku._ai_settings != desired_ai:
                    self.danmaku.configure_ai(self.config)
                self.danmaku.configure(**preferences)
            except Exception as exc:
                logging.error("DANMAKU configuration_failed type=%s", type(exc).__name__)
                self.window.append_notice(tr("弹幕设置未生效，请检查直播间和队列目录。"))

    def open_browser(self, *, modal=False) -> None:
        from .browser_dialog import BrowserDialog
        self._ensure_browser_service()
        if self.browser_dialog is None:
            self.browser_dialog = BrowserDialog(self.browser_service, self.window)
        if modal:
            self.browser_dialog.exec()
            return
        self.browser_dialog.show()
        self.browser_dialog.raise_()
        self.browser_dialog.activateWindow()

    def open_content_filter(self) -> None:
        if self.closed:
            return
        self._ensure_browser_service()
        if self.danmaku is None:
            self.window.append_notice(tr("弹幕队列暂不可用，字幕翻译继续运行。"))
            return
        content_filter = getattr(self.danmaku, "content_filter", None)
        from .danmaku_filter_dialog import DanmakuFilterDialog
        dialog = DanmakuFilterDialog(content_filter, self.browser_service.room,
                                    self.browser_dialog or self.window)
        try:
            dialog.exec()
            # An utterance that has started sending keeps its original version.
            self.danmaku.refresh_content_filter()
        finally:
            dialog.deleteLater()

    def start(self) -> None:
        if self.closed:
            return
        self.epoch += 1
        epoch, config = self.epoch, self.config
        if not self._danmaku_startup_checked:
            self._danmaku_startup_checked = True
            try:
                preferences = json.loads((data_root() / "browser-preferences.json").read_text(encoding="utf-8"))
                if isinstance(preferences, dict):
                    self._ensure_browser_service()
            except (OSError, ValueError, UnicodeError):
                pass
            # Missing preferences must not strand an existing failure library
            # or restore an old queue when the browser is opened later.
            if self.browser_service is None and any((data_root() / filename).exists() for filename in (
                    "danmaku-queue.sqlite3", "danmaku-content-filter.json")):
                self._ensure_browser_service()
        if self.danmaku is not None:
            self.danmaku.begin_session(epoch)
        self.latency.begin_session(epoch)
        if hasattr(self.window, "set_presentation_session"):
            self.window.set_presentation_session(epoch)
        preparing = ("正在准备本地暂译和 Codex 终稿…" if config.translation_provider == "hybrid"
                     else "正在准备本地识别和 Codex 翻译…" if config.translation_provider == "codex"
                     else "正在加载本地模型…")
        self.window.set_session_state(preparing if config.enabled_channels else "待机：两路采集均已关闭")
        self._set_model_state("loading" if not self.paused and config.enabled_channels
                              else ("ready" if self.active_models else "not_loaded"))
        self.executor.submit(self._transition, epoch, config, self.paused)

    def _record(self, operation, *args):
        if self.recorder is None or not self.recorder.active:
            return
        try:
            operation(*args)
        except Exception as exc:
            logging.error("本地采样已停止：%s", type(exc).__name__)
            try:
                self.recorder.fail(exc)
            except Exception:
                self.recorder.active = False

    def _pipeline_event(self, event):
        if self.recorder is not None:
            self._record(self.recorder.event, event)
        if self.specialized is not None and not self.closed and event.get("session") == self.epoch:
            self._collect_specialized(self.specialized.on_event, event)
        self.bridge.event.emit(event)

    def _attach_recording(self):
        self.pipeline.capture_factory = self.live_source_service.create_capture
        self.pipeline.source_validator = self.live_source_service.validate_job
        if self.recorder is not None:
            self.pipeline.on_audio = lambda chunk: self._record(self.recorder.on_chunk, chunk)
            self.pipeline.on_filtered = lambda job, decision: self._record(self.recorder.on_filtered, job, decision)
        if self.recorder is not None or self.specialized is not None:
            self.pipeline.on_segment = self._record_segment

    def _record_segment(self, job, accepted):
        if self.recorder is not None:
            self._record(self.recorder.on_segment, job, accepted)
        if self.specialized is not None and not self.closed:
            self._collect_specialized(self.specialized.on_segment, job, accepted)

    def _collect_specialized(self, operation, *args):
        try:
            operation(*args)
        except Exception as exc:
            logging.error("专用模型采集失败：%s", type(exc).__name__)
            try:
                self.specialized.fail_collection()
            except Exception:
                pass
            if not self._collection_error_reported:
                self._collection_error_reported = True
                self.bridge.event.emit({"session": self.epoch, "kind": "notice",
                                        "text": "专用模型采集失败，已暂停积累；字幕继续运行。"})

    def _diagnostic_snapshot(self):
        if self.recorder is not None and self.recorder.active:
            asr_info = getattr(getattr(self.pipeline, "_recognizer", None), "backend_info", None)
            if isinstance(asr_info, dict):
                previous = self.recorder.model_info.get("asr")
                self.recorder.model_info["asr"] = {**(previous if isinstance(previous, dict) else {}), **asr_info}
            info = getattr(getattr(self.pipeline, "_translator", None), "backend_info", None)
            if isinstance(info, dict):
                previous = self.recorder.model_info.get("translation")
                self.recorder.model_info["translation"] = {**(previous if isinstance(previous, dict) else {}), **info}
            local_info = getattr(getattr(self.pipeline, "_draft_translator", None), "backend_info", None)
            if self.config.translation_provider == "hybrid" and isinstance(local_info, dict):
                self.recorder.model_info.setdefault("translation", {})["draft_model"] = dict(local_info)
            filter_info = getattr(getattr(self.pipeline, "_speech_filter", None), "backend_info", None)
            previous = self.recorder.model_info.get("speech_filter")
            self.recorder.model_info["speech_filter"] = {
                **(previous if isinstance(previous, dict) else {}),
                **(filter_info if isinstance(filter_info, dict) else {}), "enabled": self.config.speech_only}
            processors = getattr(self.pipeline, "_audio_processors", {})
            processor = processors.get(self.config.audio_processing) if isinstance(processors, dict) else None
            processing_info = getattr(processor, "backend_info", None)
            self.recorder.model_info["audio_processing"] = {
                **self.active_models.get("audio_processing", {}),
                **(processing_info if isinstance(processing_info, dict) else {}),
                "mode": self.config.audio_processing, "enabled": self.config.audio_processing != "off"}
            self._record(self.recorder.snapshot)

    def _transition(self, epoch, config, paused) -> None:
        if self.closed or epoch != self.epoch:
            return
        self.pipeline.stop()
        if not paused and not self.closed and epoch == self.epoch:
            if self.specialized is not None:
                self.specialized.begin_session(epoch, config)
            self.pipeline.start(config, epoch)

    def set_paused(self, paused: bool) -> None:
        if self.closed:
            return
        if self.specialized is not None and self.specialized.busy:
            self.window.append_notice("本地模型对照进行中；完成后自动恢复此前的字幕状态。")
            return
        self.paused = paused
        if paused:
            self.window.cancel_pending()
        self.start()
        if paused:
            self.window.set_session_state("已暂停")

    def change_settings(self) -> None:
        if self.closed:
            return
        if self.specialized is not None and self.specialized.busy:
            self.window.append_notice("请先完成或取消专用模型对照，再更改设置。")
            return
        from .ui import SettingsDialog
        from .codex_discovery import wire_codex_picker
        models = self.active_models.get("translation", {}).get("available_models", [])
        dialog = SettingsDialog(self.config, first_run=False, parent=self.window, codex_models=models)
        if hasattr(dialog, "browser_requested"):
            dialog.browser_requested.connect(lambda: self.open_browser(modal=True))
        if hasattr(dialog, "audio_processing_requested"):
            dialog.audio_processing_requested.connect(self.open_audio_processing)
        wire_codex_picker(dialog)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        config = dialog.result_config
        if config is None:
            return
        # The nested audio dialog owns this field and may already have saved a
        # newer mode while the main settings dialog still holds its old base.
        config = replace(config, audio_processing=self.config.audio_processing)
        self.apply_config(config)

    def apply_config(self, config: AppConfig) -> bool:
        if self.closed:
            return False
        try:
            save_config(self.config_path, config)
        except (OSError, ConfigError) as exc:
            QMessageBox.critical(self.window, tr("无法保存"), tr("设置未改变，请检查输入内容和设置目录权限。") + "\n" + str(exc))
            return False
        language_changed = config.ui_language != self.config.ui_language
        previous = self.config
        self.config = config
        if previous == config:
            return True
        if self.recorder is not None:
            self._diagnostic_snapshot()
            self._record(self.recorder.finish, "settings_changed")
        self.window.update_config(config)
        if previous.audio_processing != config.audio_processing:
            self.epoch += 1
            if self.danmaku is not None:
                self.danmaku.begin_session(self.epoch)
            self.latency.begin_session(self.epoch)
            if hasattr(self.window, "set_presentation_session"):
                self.window.set_presentation_session(self.epoch)
            self.window.cancel_pending()
            self.executor.submit(self._replace_pipeline, self.epoch, config, self.paused)
            self._set_model_state("switching")
            self.window.set_session_state("已暂停" if self.paused else "正在加载所选模型…")
        else:
            self.start()
        if self.paused:
            self.window.set_session_state("设置已保存 · 当前已暂停")
        if language_changed:
            self.window.append_notice(tr("界面语言已保存，重新启动软件后生效。"))
        return True

    def _on_event(self, event: dict) -> None:
        if self.closed or event.get("session") != self.epoch:
            return
        kind, channel = event["kind"], event.get("channel", "")
        # Qt delivery is queued: navigation can happen after the pipeline emits
        # a valid result but before the main thread consumes it.
        if kind in {"original", "translation", "translation_status", "subtitle_final", "audio_processing"} and "source_generation" in event:
            if not self.live_source_service.validate_job(SimpleNamespace(**event)):
                return
        if self.danmaku is not None:
            try:
                source = self.live_source_service.snapshot_value()
                selected_system = channel == "system" and source["source_enabled"] and "source_generation" in event
                same_live_room = bool(event.get("source_room") and event.get("source_room") == self.danmaku.room)
                if not selected_system or (source["source_kind"] == "live" and same_live_room):
                    self.danmaku.handle_event(event)
            except Exception as exc:
                logging.error("DANMAKU event_failed type=%s", type(exc).__name__)
        if kind == "state":
            self.window.set_session_state(event["text"])
        elif kind == "specialized_comparison_done":
            self.paused = event["previous_paused"]
            if event.get("failed"):
                self.window.append_notice("专用模型对照中断；可在专用模型窗口查看状态。")
            self.start()
            if self.paused:
                self.window.set_session_state("已暂停")
        elif kind == "models_loading":
            self._set_model_state("loading")
        elif kind == "models_ready":
            self._set_model_state("ready", event["models"])
        elif kind == "audio_processing":
            information = {**self.active_models.get("audio_processing", {}),
                **{key: event[key] for key in ("mode", "action", "reason", "model_id", "model_loaded",
                                              "processing_seconds", "rtf") if key in event}}
            self._set_model_state(self.model_state, {**self.active_models, "audio_processing": information})
        elif kind == "models_failed":
            self._set_model_state("failed")
        elif kind == "translation_unloaded":
            self._set_model_state("loading", {role: info for role, info in self.active_models.items()
                                              if role != "translation"})
        elif kind == "models_unloaded":
            self._set_model_state("not_loaded", {})
        elif kind == "source_status":
            self.window.set_source_status(channel, event["text"])
        elif kind == "translation_queue" and not event.get("capture_paused") and not self.paused:
            pending = event["pending"]
            if pending:
                self.window.set_session_state(tr("正在听 · 翻译排队 {0} 句 · 最早等待 {1:.0f} 秒",
                                                 pending, event["oldest_age_seconds"]))
            else:
                self.window.set_session_state("正在听 · 本地暂译 / Codex 终稿"
                    if self.config.translation_provider == "hybrid" else "正在听 · 本地识别 / Codex 订阅翻译"
                    if self.config.translation_provider == "codex" else "正在听 · 完全本地")
        elif kind == "level":
            self.window.set_level(channel, event["level"])
        elif kind == "original":
            options = {"revision": event.get("revision", 1), "provisional": event.get("provisional", False),
                       "uncertain": event.get("uncertain", False)} if "revision" in event else {}
            if "context_generation" in event:
                options["context_generation"] = event["context_generation"]
            self.window.set_original(channel, event["text"], event["segment_id"], **options)
        elif kind == "translation":
            options = {"revision": event.get("revision", 1), "provisional": event.get("provisional", False)} if "revision" in event else {}
            options.update({key: event[key] for key in (
                "translation_stage", "translation_provider", "translation_model", "context_generation",
                "final_failed", "failed", "overall_failed") if key in event})
            if isinstance(event.get("timing"), dict) and hasattr(self.window, "subtitle_presented"):
                options["presentation"] = {
                    key: event[key] for key in (
                        "session", "channel", "segment_id", "revision", "context_generation",
                        "translation_stage", "translation_provider", "translation_model", "timing",
                        "rpc_metrics", "failed", "overall_failed", "provisional", "source_generation", "source_room") if key in event}
                options["presentation"]["timing"] = dict(event["timing"])
                options["presentation"]["ui_received_at"] = time.monotonic()
            self.window.set_translation(channel, event["text"], event["segment_id"], **options)
        elif kind == "translation_status":
            options = {key: event[key] for key in (
                "revision", "translation_stage", "translation_provider", "translation_model", "context_generation",
                "failed", "overall_failed", "error") if key in event}
            self.window.set_translation_status(channel, event["segment_id"], **options)
        elif kind == "subtitle_final":
            self.window.finalize_subtitle(channel, event["segment_id"], event.get("revision", 1))
        elif kind == "notice":
            self.window.append_notice(event["text"])
        elif kind == "error":
            self.window.set_session_state("无法开始采集")
            self.window.append_notice(event["text"])

    def _on_subtitle_presented(self, event: dict) -> None:
        if self.closed or event.get("session") != self.epoch:
            return
        if "source_generation" in event and not self.live_source_service.validate_job(SimpleNamespace(**event)):
            return
        metrics = self.latency.on_presented(event)
        if metrics is None:
            return
        # Timing telemetry contains identities and durations, never audio or subtitle text.
        logging.info("LATENCY %s", json.dumps(metrics, ensure_ascii=False))
        if hasattr(self.window, "set_latency"):
            self.window.set_latency(metrics)
        if self.recorder is not None:
            self._record(self.recorder.event, {**event, "kind": "subtitle_presented"})

    def _set_model_state(self, state: str, models: dict | None = None) -> None:
        if models is not None:
            self.active_models = {role: dict(info) for role, info in models.items()
                                  if role in ("asr", "translation", "speech_filter", "audio_processing") and isinstance(info, dict)}
        self.model_state = state
        self.window.set_model_state(state, self.active_models)

    def change_models(self) -> None:
        if self.closed:
            return
        if self.specialized is not None and self.specialized.busy:
            self.window.append_notice("请先完成或取消专用模型对照，再切换模型。")
            return
        from .model_manager import ModelManagerDialog
        previous_paused = self.paused
        self.set_paused(True)
        self.window.set_session_state("模型管理 · 字幕已暂停")
        dialog = ModelManagerDialog(models_root(), parent=self.window, active_models=self.active_models,
                                    speech_only=self.config.speech_only,
                                    translation_required=self.config.translation_provider in ("local", "hybrid"))
        self.model_dialog = dialog
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
        finally:
            self.model_dialog = None
        if self.closed:
            return
        self.paused = previous_paused
        selected = dialog.selected_model_id if self.config.translation_provider in ("local", "hybrid") else dialog.selected_asr_model_id
        if accepted and selected:
            if self.recorder is not None:
                self._diagnostic_snapshot()
                self._record(self.recorder.finish, "model_changed")
            self.epoch += 1
            if self.danmaku is not None:
                self.danmaku.begin_session(self.epoch)
            self.window.cancel_pending()
            self.executor.submit(self._replace_pipeline, self.epoch, self.config, self.paused)
            self._set_model_state("switching")
            self.window.set_session_state("已暂停" if self.paused else "正在加载所选模型…")
        else:
            self.start()
            if self.paused:
                self.window.set_session_state("已暂停")

    def _replace_pipeline(self, epoch, config, paused) -> None:
        if self.closed or epoch != self.epoch:
            return
        from .pipeline import SubtitlePipeline
        self.pipeline.shutdown()
        if self.closed:
            return
        self._pipeline_event({"session": epoch, "kind": "models_unloaded"})
        self.pipeline = SubtitlePipeline(models_root(), self._pipeline_event)
        self._attach_recording()
        if not paused and not self.closed and epoch == self.epoch:
            if self.specialized is not None:
                self.specialized.begin_session(epoch, config)
            self.pipeline.start(config, epoch)

    def change_specialized_model(self) -> None:
        if self.closed:
            return
        if self.specialized is None:
            QMessageBox.warning(self.window, tr("专用模型不可用"), tr("请检查本地数据目录是否可写。字幕功能仍可使用。"))
            return
        from .specialized_dialog import SpecializedModelDialog
        SpecializedModelDialog(self.specialized, parent=self.window).exec()

    def _request_local_comparison(self) -> None:
        if self.closed:
            self.specialized.cancel_comparison()
            raise ValueError("字幕程序正在退出")
        previous_paused = self.paused
        self.paused = True
        self.epoch += 1
        self.window.cancel_pending()
        self.window.set_session_state("专用模型对照 · 字幕暂时暂停")
        self._set_model_state("switching")
        self.executor.submit(self._compare_local_models, self.epoch, previous_paused)

    def _compare_local_models(self, epoch: int, previous_paused: bool) -> None:
        from .pipeline import SubtitlePipeline
        failed = False
        try:
            self.pipeline.shutdown()
            if self.recorder is not None:
                self._record(self.recorder.finish, "specialized_comparison")
            if not self.closed and epoch == self.epoch:
                self.specialized.run_comparison()
        except Exception as exc:
            logging.error("专用模型对照任务失败：%s", type(exc).__name__)
            failed = True
        finally:
            self.specialized.finish_comparison(failed=failed)
            if not self.closed and epoch == self.epoch:
                self.pipeline = SubtitlePipeline(models_root(), self._pipeline_event)
                self._attach_recording()
                self._pipeline_event({"session": epoch, "kind": "specialized_comparison_done",
                                      "previous_paused": previous_paused, "failed": failed})

    def _shutdown_current_pipeline(self) -> None:
        # Resolve on the serialized worker: a model replacement may have been
        # queued before the close event, so a bound old shutdown is insufficient.
        try:
            self.pipeline.shutdown()
        finally:
            try:
                if self.recorder is not None:
                    self._diagnostic_snapshot()
                    self._record(self.recorder.finish)
            finally:
                if self.specialized is not None:
                    self.specialized.close()

    def _may_close_window(self) -> bool:
        dialog = next((item for item in (self.model_dialog, self.audio_processing_dialog)
                       if item is not None and item.is_busy), None)
        if dialog is not None:
            # Keep the event loop alive while the downloader cancels. Qt's
            # aboutToQuit is too late to defer destruction of a running QThread.
            if not self._waiting_for_download_close:
                self._waiting_for_download_close = True
                dialog.finished.connect(lambda _result: self.window.close())
                dialog.reject()
            return False
        return True

    def close(self) -> None:
        if self.closed:
            return
        self.closed, self.epoch = True, self.epoch + 1
        self.diagnostic_timer.stop()
        if self.danmaku is not None:
            self.danmaku.close()
        if self.browser_service is not None:
            self.browser_service.close()
        self.live_source_service.close()
        if self.specialized is not None:
            self.specialized.cancel_comparison()
        self.executor.submit(self._shutdown_current_pipeline)
        self.executor.shutdown(wait=False, cancel_futures=False)


def configure_logging() -> None:
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(root / "diagnostic.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(message)s")


def restore_main_window(window) -> None:
    # A launcher's SW_HIDE can hide the native window while Qt considers it
    # visible. Force a visibility transition instead of repeating show().
    window.hide()
    window.showNormal()
    window.raise_()
    window.activateWindow()


def activate_existing_window(app: QApplication) -> None:
    modal = app.activeModalWidget()
    if modal is not None:
        # Hiding a QDialog can end its exec() and discard unfinished settings.
        modal.showNormal()
        modal.raise_()
        modal.activateWindow()
        return
    window = getattr(app, "_fanyiji_main_window", None)
    if window is not None:
        if not getattr(window, "_closing_emitted", False):
            restore_main_window(window)
        return
    window = app.activeWindow()
    if window is None:
        windows = [widget for widget in app.topLevelWidgets() if widget.isVisible()]
        window = windows[0] if windows else None
    if window is not None:
        window.showNormal()
        window.raise_()
        window.activateWindow()


def single_instance(app: QApplication):
    legacy_name = LEGACY_APP_ID + "-v1"
    # Keep the legacy endpoint as an alias while existing installations migrate.
    # This also lets an older executable find and activate the renamed program.
    for name in (INSTANCE_NAME, legacy_name):
        socket = QLocalSocket()
        socket.connectToServer(name)
        if socket.waitForConnected(200):
            socket.write(b"activate")
            socket.waitForBytesWritten(200)
            return None
    server = QLocalServer(app)
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    if not server.listen(INSTANCE_NAME):
        QMessageBox.warning(None, tr("无法启动"), tr("已有一个字幕程序正在启动，请稍后再试。"))
        return None
    legacy_server = QLocalServer(server)
    legacy_server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    if not legacy_server.listen(legacy_name):
        server.close()
        QMessageBox.warning(None, tr("无法启动"), tr("已有一个字幕程序正在启动，请稍后再试。"))
        return None
    server._legacy_server = legacy_server

    def activate(endpoint):
        connection = endpoint.nextPendingConnection()
        if connection:
            connection.close()
            connection.deleteLater()
        activate_existing_window(app)
    server.newConnection.connect(lambda: activate(server))
    legacy_server.newConnection.connect(lambda: activate(legacy_server))
    return server


def configure_application_identity(app: QApplication) -> None:
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_NAME)


def smoke_test(screenshot: Path | None = None, language: str = "zh-CN") -> int:
    """Synthetic UI-only check. It neither reads settings nor opens audio devices."""
    set_language(language)
    from .ui import MainWindow, SettingsDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    configure_application_identity(app)
    if app.platformName() == "offscreen" and sys.platform == "win32":
        # Offscreen Qt does not enumerate Windows fonts; preview still needs CJK.
        from PySide6.QtGui import QFontDatabase
        for name in ("msyh.ttc", "YuGothM.ttc"):
            font = Path("C:/Windows/Fonts") / name
            if font.is_file():
                QFontDatabase.addApplicationFont(str(font))
    config = AppConfig(ui_language=language)
    dialog = SettingsDialog(config, first_run=True)
    window = MainWindow(config)
    window.set_session_state("界面预览 · 未采集声音")
    window.set_source_status("mic", "界面预览")
    window.set_source_status("system", "界面预览")
    window.set_original("mic", "我们明天下午三点开会。", "preview-mic")
    window.set_translation("mic", "明日の午後3時に会議をします。", "preview-mic")
    window.set_original("system", "資料を先に送ってください。", "preview-system")
    window.set_translation("system", "请先把资料发过来。", "preview-system")
    window.show()
    app.processEvents()
    if screenshot:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(screenshot))
        dialog.show()
        app.processEvents()
        dialog.grab().save(str(screenshot.with_name("first-run.png")))
    dialog.close()
    window.close()
    print(json.dumps({"ui_smoke": "passed", "audio_capture": False, "settings_written": False}))
    return 0


def configured_models_available(directory: Path, *, speech_only: bool = True, translation_provider: str = "local") -> bool:
    """A cheap startup gate; downloads and full SHA validation stay in the UI."""
    required = [directory / "vad/silero_vad.onnx"]
    if speech_only:
        from .asr_catalog import vad_dependencies
        from .models_download import ModelCatalog
        for entry in vad_dependencies(ModelCatalog.default(), speech_only=True):
            required.extend(directory / file.relative_path for file in entry.files
                            if not file.relative_path.startswith("vad/"))
    backend_file = directory / "translation" / "backend.json"
    try:
        asr_directory = directory / "asr"
        if (asr_directory / "backend.json").is_file():
            from .asr_engine import create_recognizer_engine
            asr = create_recognizer_engine(asr_directory)
            required.extend(asr_directory / name for name in asr.asset_files)
        else:
            required.extend(asr_directory / name for name in ("model.int8.onnx", "tokens.txt"))
        if translation_provider == "codex":
            return all(path.is_file() and path.stat().st_size > 0 for path in required)
        backend = json.loads(backend_file.read_text(encoding="utf-8-sig")) if backend_file.is_file() else {}
        if not isinstance(backend, dict):
            return False
        if backend.get("backend") == "llama_cpp":
            value = backend.get("model_file")
            if not isinstance(value, str) or not value or Path(value).is_absolute():
                return False
            root = (directory / "translation").resolve()
            model = (root / value).resolve()
            if not model.is_relative_to(root) or model.suffix.lower() != ".gguf":
                return False
            required.append(model)
        else:
            from .engines import TranslationEngine
            required.extend(directory / "translation" / name for name in TranslationEngine.REQUIRED_FILES)
        return all(path.is_file() and path.stat().st_size > 0 for path in required)
    except (OSError, ValueError, RuntimeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=APP_NAME + " · 实时语音翻译与字幕")
    parser.add_argument("--smoke-test", action="store_true", help="只检查界面，不采集声音")
    parser.add_argument("--smoke-language", choices=("zh-CN", "ja", "en"), default="zh-CN",
                        help="仅用于 --smoke-test 的界面预览语言，不修改设置")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--self-test", action="store_true", help="使用本地样本验证模型，不采集声音")
    parser.add_argument("--self-test-output", type=Path)
    parser.add_argument("--diagnostics-output", type=Path,
                        help="显式保存实际采集片段及字幕到新目录，普通启动不录音")
    args = parser.parse_args()
    if args.smoke_test:
        return smoke_test(args.screenshot, args.smoke_language)
    if args.self_test:
        from .self_test import run_self_test
        return run_self_test(models_root(), args.self_test_output)

    app = QApplication(sys.argv[:1])
    configure_application_identity(app)
    server = single_instance(app)
    if server is None:
        return 0
    configure_logging()
    config_path = data_root() / "settings.json"
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        QMessageBox.warning(None, tr("需要重新设置"), str(exc))
        config = None
    set_language(config.ui_language if config is not None else "zh-CN")
    from .ui import MainWindow, SettingsDialog
    if config is None:
        dialog = SettingsDialog(None, first_run=True)
        from .codex_discovery import wire_codex_picker
        wire_codex_picker(dialog)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_config is None:
            return 0
        config = dialog.result_config
        try:
            save_config(config_path, config)
        except (OSError, ConfigError) as exc:
            QMessageBox.critical(None, tr("无法保存"), tr("无法保存设置，尚未开始采集。") + "\n" + str(exc))
            return 1
        set_language(config.ui_language)
    if config.enabled_channels and not configured_models_available(models_root(), speech_only=config.speech_only,
                                                                   translation_provider=config.translation_provider):
        from .model_manager import ModelManagerDialog
        manager = ModelManagerDialog(models_root(), speech_only=config.speech_only,
                                     translation_required=config.translation_provider in ("local", "hybrid"))
        if manager.exec() != QDialog.DialogCode.Accepted:
            return 0
        selected = manager.selected_model_id if config.translation_provider in ("local", "hybrid") else manager.selected_asr_model_id
        if not selected:
            return 0
    window = MainWindow(config)
    if args.diagnostics_output:
        controller = AppController(window, config, config_path, diagnostics_output=args.diagnostics_output)
    else:
        controller = AppController(window, config, config_path)
    app.aboutToQuit.connect(controller.close)
    app._fanyiji_main_window = window
    window.show()
    restore_main_window(window)
    QTimer.singleShot(0, controller.start)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
