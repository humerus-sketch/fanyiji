"""Deterministic lifecycle tests with no real pipeline, audio, or persistence."""

from collections import deque
from dataclasses import replace
from concurrent.futures import Future
from contextlib import ExitStack
import os
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
import tempfile
from unittest.mock import Mock, call, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QDialog, QWidget

from fanyiji import app as app_module
from fanyiji import live_source_service as source_module
# patch.dict(sys.modules) restores its complete snapshot. Keep this lazily
# imported dependency in that snapshot so package attributes cannot refer to a
# removed module while later helper tests import a fresh copy.
from fanyiji import local_features as local_features_module
from fanyiji.config import AppConfig, SourceConfig
from fanyiji.identity import APP_NAME, INSTANCE_NAME


class _FakeWindow(QWidget):
    pause_toggled = Signal(bool)
    settings_requested = Signal()
    closing = Signal()

    def __init__(self):
        super().__init__()
        for name in (
            "set_session_state", "set_source_status", "set_level", "set_original",
            "set_translation", "update_config", "append_notice", "cancel_pending", "set_model_state",
        ):
            setattr(self, name, Mock())


class _ControlledExecutor:
    """Represent queued transitions without nondeterministic thread timing."""

    def __init__(self, **_kwargs):
        self.jobs = deque()
        self.closed = False
        self.shutdown_arguments = None

    def submit(self, function, *args):
        if self.closed:
            raise RuntimeError("cannot schedule new futures after shutdown")
        future = Future()
        self.jobs.append((function, args, future))
        return future

    def run_next(self):
        function, args, future = self.jobs.popleft()
        future.set_running_or_notify_cancel()
        try:
            result = function(*args)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        future.set_result(result)

    def drain(self):
        while self.jobs:
            self.run_next()

    def shutdown(self, *, wait, cancel_futures):
        self.closed = True
        self.shutdown_arguments = (wait, cancel_futures)


class _FakePipeline:
    def __init__(self, models, on_event):
        self.models = models
        self.on_event = on_event
        self.calls = []
        self.on_stop = None

    def stop(self):
        self.calls.append(("stop",))
        if self.on_stop is not None:
            callback, self.on_stop = self.on_stop, None
            callback()

    def start(self, config, session):
        self.calls.append(("start", config, session))

    def shutdown(self):
        self.calls.append(("shutdown",))


class AppLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.stack = ExitStack()
        fake_module = ModuleType("fanyiji.pipeline")
        fake_module.SubtitlePipeline = _FakePipeline
        self.stack.enter_context(patch.dict(sys.modules, {"fanyiji.pipeline": fake_module}))
        self.stack.enter_context(patch.object(app_module, "ThreadPoolExecutor", _ControlledExecutor))
        self.stack.enter_context(patch.object(app_module, "models_root", return_value=Path("unused-models")))
        self.data_directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        # BrowserService resolves its own imported data_root; isolate all modules.
        self.stack.enter_context(patch.dict(os.environ, {"FANYIJI_DATA": self.data_directory}))
        self.stack.enter_context(patch.object(app_module, "data_root", return_value=Path(self.data_directory)))
        self.window = _FakeWindow()
        self.controller = app_module.AppController(
            self.window, AppConfig(), Path("not-written-settings.json")
        )

    def tearDown(self):
        self.controller.close()
        self.controller.executor.drain()
        self.window.deleteLater()
        self.qt_app.processEvents()
        self.stack.close()

    def _dialog(self, accepted, config=None):
        dialog = Mock()
        dialog.exec.return_value = (
            QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
        )
        dialog.result_config = config
        return self.stack.enter_context(
            patch("fanyiji.ui.SettingsDialog", return_value=dialog)
        )

    def test_source_configuration_fences_old_results_and_preserves_pause(self):
        self.controller.start()
        self.controller.executor.drain()
        old_epoch = self.controller.epoch
        self.controller.paused = True
        self.controller.live_source_service.configuration_changed.emit()
        self.assertGreater(self.controller.epoch, old_epoch)
        self.controller.executor.drain()
        self.assertTrue(self.controller.paused)
        self.controller._on_event({"session": old_epoch, "kind": "translation",
                                   "channel": "system", "segment_id": "old-room", "text": "旧房间"})
        self.window.set_translation.assert_not_called()

    def test_replaced_pipeline_keeps_source_hooks(self):
        self.controller._replace_pipeline(self.controller.epoch, self.controller.config, True)
        self.assertEqual(self.controller.pipeline.capture_factory, self.controller.live_source_service.create_capture)
        self.assertEqual(self.controller.pipeline.source_validator, self.controller.live_source_service.validate_job)

    def test_closing_source_service_prevents_late_restart(self):
        self.controller.close()
        epoch = self.controller.epoch
        self.controller.live_source_service.configuration_changed.emit()
        self.assertEqual(self.controller.epoch, epoch)
        with self.assertRaises(RuntimeError):
            self.controller.live_source_service.create_capture(Mock())

    def test_audio_mode_apply_preserves_unrelated_config_and_replaces_models(self):
        self.controller.config = replace(self.controller.config, ui_language="en", speech_only=False)
        previous = self.controller.config
        with patch.object(app_module, "save_config") as save:
            self.assertTrue(self.controller._apply_audio_processing_mode("uvr"))
        self.assertEqual(self.controller.config, replace(previous, audio_processing="uvr"))
        save.assert_called_once_with(self.controller.config_path, self.controller.config)
        old = self.controller.pipeline
        self.controller.executor.drain()
        self.assertIn(("shutdown",), old.calls)
        self.assertIsNot(self.controller.pipeline, old)
        self.assertEqual(self.controller.pipeline.source_validator, self.controller.live_source_service.validate_job)

    def test_source_navigation_after_emission_blocks_ui_and_push_consumption(self):
        self.controller.danmaku = Mock()
        event = {"session": self.controller.epoch, "kind": "translation", "channel": "system",
                 "segment_id": "old-video", "text": "旧视频", "source_room": "", "source_generation": 99}
        with patch.object(self.controller.live_source_service, "validate_job", return_value=False):
            self.controller._on_event(event)
        self.window.set_translation.assert_not_called()
        self.controller.danmaku.handle_event.assert_not_called()

    def test_valid_video_and_wrong_live_room_never_enter_live_push(self):
        self.controller.danmaku = Mock(room="1698225")
        event = {"session": self.controller.epoch, "kind": "translation", "channel": "system",
                 "segment_id": "video", "text": "视频译文", "source_room": "", "source_generation": 3}
        with patch.object(self.controller.live_source_service, "validate_job", return_value=True), patch.object(
                self.controller.live_source_service, "snapshot_value",
                return_value={"source_enabled": True, "source_kind": "video"}):
            self.controller._on_event(event)
        self.window.set_translation.assert_called_once()
        self.controller.danmaku.handle_event.assert_not_called()
        self.window.set_translation.reset_mock()
        with patch.object(self.controller.live_source_service, "validate_job", return_value=True), patch.object(
                self.controller.live_source_service, "snapshot_value",
                return_value={"source_enabled": True, "source_kind": "live"}):
            self.controller._on_event({**event, "source_room": "21399689"})
            self.controller.danmaku.handle_event.assert_not_called()
            self.controller._on_event({**event, "source_room": "1698225"})
        self.controller.danmaku.handle_event.assert_called_once()

    def test_main_settings_save_preserves_mode_saved_by_nested_audio_controls(self):
        dialog = self._dialog(True, replace(self.controller.config, speech_only=False))
        def nested_change():
            self.controller.config = replace(self.controller.config, audio_processing="uvr")
            return QDialog.DialogCode.Accepted
        dialog.return_value.exec.side_effect = nested_change
        with patch.object(app_module, "save_config"):
            self.controller.change_settings()
        self.assertEqual(self.controller.config.audio_processing, "uvr")
        self.assertFalse(self.controller.config.speech_only)

    def test_processing_runtime_fallback_and_recovery_update_current_model_summary(self):
        self.controller.active_models = {"asr": {"model_id": "sensevoice"},
                                         "audio_processing": {"mode": "gtcrn", "model_loaded": True}}
        self.controller._on_event({"session": self.controller.epoch, "kind": "audio_processing",
            "mode": "gtcrn", "action": "bypass", "reason": "processing_error", "model_loaded": True})
        self.assertEqual(self.controller.active_models["audio_processing"]["action"], "bypass")
        self.controller._on_event({"session": self.controller.epoch, "kind": "audio_processing",
            "mode": "gtcrn", "action": "processed", "reason": "enhanced", "model_loaded": True})
        self.assertEqual(self.controller.active_models["audio_processing"]["action"], "processed")
        self.assertEqual(self.controller.active_models["asr"]["model_id"], "sensevoice")

    def test_queue_status_reports_backlog_without_overriding_pause_or_overload(self):
        self.controller.start()
        epoch = self.controller.epoch
        event = {"session": epoch, "kind": "translation_queue", "pending": 4,
                 "oldest_age_seconds": 12.8, "capture_paused": False}
        self.controller._on_event(event)
        self.window.set_session_state.assert_called_with("正在听 · 翻译排队 4 句 · 最早等待 13 秒")
        self.window.set_session_state.reset_mock()
        self.controller._on_event({**event, "capture_paused": True})
        self.window.set_session_state.assert_not_called()
        self.controller.paused = True
        self.controller._on_event(event)
        self.window.set_session_state.assert_not_called()
        self.controller.paused = False
        self.controller._on_event({**event, "pending": 0})
        self.window.set_session_state.assert_called_with("正在听 · 完全本地")

    def test_saved_push_setting_initializes_browser_without_opening_for_default_start(self):
        with patch.object(self.controller, "_ensure_browser_service") as ensure:
            self.controller.start()
            ensure.assert_not_called()
            self.controller._danmaku_startup_checked = False
            (Path(self.data_directory) / "browser-preferences.json").write_text(
                json.dumps({"danmaku_enabled": True, "room": "1698225"}), encoding="utf-8")
            self.controller.start()
            ensure.assert_called_once()

    def test_enable_push_or_unchanged_save_never_opens_room_automatically(self):
        self.controller.danmaku = Mock(enabled=False, room="")
        self.controller.browser_service = Mock(is_browser_ready=True)
        preferences = {"enabled": True, "room": "1698225", "source": "system", "interval_seconds": 3.5}
        self.controller._configure_danmaku(preferences)
        self.controller.browser_service.open.assert_not_called()
        self.controller.danmaku.enabled, self.controller.danmaku.room = True, "1698225"
        self.controller._configure_danmaku(preferences)
        self.controller.browser_service.open.assert_not_called()

    def test_existing_failure_library_initializes_review_even_without_browser_preferences(self):
        (Path(self.data_directory) / "danmaku-content-filter.json").write_text("{}", encoding="utf-8")
        with patch.object(self.controller, "_ensure_browser_service") as ensure:
            self.controller.start()
            ensure.assert_called_once()

    def test_clear_queue_button_request_reaches_queue_without_reinitializing_it(self):
        with patch("fanyiji.browser_service.data_root", return_value=Path(self.data_directory)), patch(
                "fanyiji.local_features.danmaku_available", return_value=True), patch(
                "fanyiji.danmaku_service.DanmakuService") as queue_class:
            queue = queue_class.return_value
            with patch.object(queue, "enabled", False), patch.object(queue, "room", ""):
                self.controller._ensure_browser_service()
                self.controller.browser_service.clear_danmaku_queue()
                self.controller._ensure_browser_service()
            queue.clear_pending.assert_called_once_with()
            assert queue_class.call_count == 1
            assert queue_class.call_args.kwargs["reset_on_start"] is True
            self.controller.browser_service.open = Mock()
            self.controller.start()
            self.controller.browser_service.open.assert_not_called()

    def test_filter_manager_request_uses_shared_filter_and_refreshes_after_close(self):
        content_filter = object()
        self.controller.danmaku = Mock(content_filter=content_filter)
        self.controller.browser_service = Mock(room="1698225")
        with patch.object(self.controller, "_ensure_browser_service"), patch(
                "fanyiji.danmaku_filter_dialog.DanmakuFilterDialog") as dialog_class:
            self.controller.open_content_filter()
        dialog_class.assert_called_once_with(content_filter, "1698225", self.window,
                                             browser_service=self.controller.browser_service)
        dialog_class.return_value.exec.assert_called_once_with()
        self.controller.danmaku.refresh_content_filter.assert_called_once_with()
        dialog_class.return_value.deleteLater.assert_called_once_with()

    def test_filter_manager_does_not_open_after_controller_closes(self):
        self.controller.closed = True
        with patch.object(self.controller, "_ensure_browser_service") as ensure:
            self.controller.open_content_filter()
        ensure.assert_not_called()
        self.controller.closed = False

    def test_push_receives_current_events_only_and_failure_keeps_subtitles_running(self):
        self.controller.start()
        self.controller.danmaku = Mock()
        event = {"session": self.controller.epoch, "kind": "translation", "channel": "system",
                 "segment_id": "a", "text": "当前正确的字幕终稿译文"}
        self.controller._on_event({**event, "session": self.controller.epoch - 1})
        self.controller.danmaku.handle_event.assert_not_called()
        self.controller.danmaku.handle_event.side_effect = OSError("isolated test")
        self.controller._on_event(event)
        self.window.set_translation.assert_called_once_with("system", event["text"], "a")

    def test_legacy_settings_start_captions_without_removed_integrations(self):
        removed_modules = {f"fanyiji.{name}": None for name in (
            "bilibili_bridge", "danmaku_browser", "admin_access",
        )}
        legacy_settings = {
            "schema_version": 1, "setup_complete": True,
            "microphone": {"enabled": False, "direction": "zh_to_ja"},
            "playback": {"enabled": True, "direction": "ja_to_zh"},
            "danmaku_enabled": True, "danmaku_room": "1698225",
            "danmaku_accounts": [{"id": "default", "name": "旧账号", "enabled": True}],
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, removed_modules):
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps(legacy_settings), encoding="utf-8")
            config = app_module.load_config(path)
            window = _FakeWindow()
            controller = app_module.AppController(window, config, path)
            try:
                controller.start()
                controller.executor.drain()
                self.assertEqual(controller.pipeline.calls[-1], ("start", config, controller.epoch))
                controller._on_event({"session": controller.epoch, "kind": "translation",
                    "channel": "system", "segment_id": "current", "text": "有效译文"})
                window.set_translation.assert_called_once_with("system", "有效译文", "current")
            finally:
                controller.close()
                controller.executor.drain()
                window.deleteLater()
                self.qt_app.processEvents()

    def test_queued_obsolete_start_cannot_activate_capture(self):
        self.controller.start()
        old_epoch = self.controller.epoch
        self.controller.set_paused(True)
        self.controller.executor.drain()
        self.assertEqual(self.controller.pipeline.calls, [("stop",)])
        self.assertGreater(self.controller.epoch, old_epoch)
        self.assertTrue(self.controller.paused)

    def test_new_pause_during_stop_prevents_obsolete_start(self):
        self.controller.start()
        self.controller.pipeline.on_stop = lambda: self.controller.set_paused(True)
        self.controller.executor.drain()
        self.assertFalse(any(call[0] == "start" for call in self.controller.pipeline.calls))

    def test_settings_change_while_paused_never_resumes_capture(self):
        self.controller.set_paused(True)
        self.controller.executor.drain()
        updated = AppConfig(microphone=SourceConfig(direction="ja_to_zh"))
        self._dialog(True, updated)
        saved = self.stack.enter_context(patch.object(app_module, "save_config"))
        self.controller.change_settings()
        self.controller.executor.drain()
        saved.assert_called_once_with(self.controller.config_path, updated)
        self.assertEqual(self.controller.config, updated)
        self.assertTrue(self.controller.paused)
        self.assertFalse(any(call[0] == "start" for call in self.controller.pipeline.calls))
        self.window.update_config.assert_called_once_with(updated)
        self.controller.set_paused(False)
        self.controller.executor.drain()
        self.assertEqual(
            self.controller.pipeline.calls[-1], ("start", updated, self.controller.epoch)
        )

    def test_settings_cancel_preserves_active_session_and_configuration(self):
        self.controller.start()
        self.controller.executor.drain()
        epoch, original = self.controller.epoch, self.controller.config
        calls = list(self.controller.pipeline.calls)
        self._dialog(False)
        saved = self.stack.enter_context(patch.object(app_module, "save_config"))
        self.controller.change_settings()
        self.assertEqual(self.controller.epoch, epoch)
        self.assertIs(self.controller.config, original)
        self.assertEqual(self.controller.pipeline.calls, calls)
        saved.assert_not_called()
        self.window.update_config.assert_not_called()

    def test_save_failure_does_not_apply_or_restart(self):
        original, epoch = self.controller.config, self.controller.epoch
        self._dialog(True, AppConfig(playback=SourceConfig(enabled=False)))
        self.stack.enter_context(patch.object(app_module, "save_config", side_effect=OSError("read only")))
        error = self.stack.enter_context(patch.object(app_module.QMessageBox, "critical"))
        self.controller.change_settings()
        error.assert_called_once()
        self.assertIs(self.controller.config, original)
        self.assertEqual(self.controller.epoch, epoch)
        self.assertEqual(len(self.controller.executor.jobs), 0)
        self.window.update_config.assert_not_called()

    def test_saved_interface_language_waits_for_restart_and_keeps_translation_direction(self):
        from fanyiji.i18n import get_language, set_language
        previous_language = get_language()
        try:
            set_language("zh-CN")
            original = self.controller.config
            updated = replace(original, ui_language="en")
            self._dialog(True, updated)
            saved = self.stack.enter_context(patch.object(app_module, "save_config"))
            self.controller.change_settings()
            saved.assert_called_once_with(self.controller.config_path, updated)
            self.assertEqual(self.controller.config.ui_language, "en")
            self.assertEqual(get_language(), "zh-CN")
            self.assertEqual(self.controller.config.microphone, original.microphone)
            self.assertEqual(self.controller.config.playback, original.playback)
            self.window.append_notice.assert_called_with("界面语言已保存，重新启动软件后生效。")
        finally:
            set_language(previous_language)

    def test_rejected_interface_language_does_not_change_saved_or_active_language(self):
        from fanyiji.i18n import get_language
        active_language = get_language()
        original = self.controller.config
        self._dialog(False, replace(original, ui_language="ja"))
        saved = self.stack.enter_context(patch.object(app_module, "save_config"))
        self.controller.change_settings()
        saved.assert_not_called()
        self.assertIs(self.controller.config, original)
        self.assertEqual(get_language(), active_language)
        self.assertEqual(len(self.controller.executor.jobs), 0)
        self.window.update_config.assert_not_called()

    def test_old_session_events_are_ignored_after_switch(self):
        self.controller.start()
        old_epoch = self.controller.epoch
        self.controller.set_paused(True)
        self.controller._on_event({
            "session": old_epoch, "kind": "original", "channel": "mic",
            "text": "old", "segment_id": "old",
        })
        self.window.set_original.assert_not_called()
        self.controller._on_event({
            "session": self.controller.epoch, "kind": "original", "channel": "mic",
            "text": "current", "segment_id": "current",
        })
        self.window.set_original.assert_called_once_with("mic", "current", "current")

    def _loaded_models(self):
        return {"asr": {"model_id": "qwen3-asr-1.7b-int8", "model_kind": "qwen3_asr", "device": "cpu"},
                "translation": {"model_file": "HY-MT2-7B-Q6_K.gguf", "model_kind": "hy", "device": "cuda"}}

    def test_provider_failure_does_not_show_closed_previous_translation_model(self):
        self.controller.start()
        epoch = self.controller.epoch
        loaded = self._loaded_models()
        self.controller._on_event({"session": epoch, "kind": "models_ready", "models": loaded})
        self.controller._on_event({"session": epoch, "kind": "translation_unloaded"})
        self.controller._on_event({"session": epoch, "kind": "models_failed"})
        self.assertEqual(self.controller.model_state, "failed")
        self.assertEqual(self.controller.active_models, {"asr": loaded["asr"]})

    def test_models_ready_epoch_guard_and_pause_preserve_confirmed_load(self):
        self.controller.start()
        epoch = self.controller.epoch
        loaded = self._loaded_models()
        self.controller._on_event({"session": epoch, "kind": "models_ready", "models": loaded})
        self.assertEqual(self.controller.active_models, loaded)
        self.controller.set_paused(True)
        self.assertEqual(self.controller.active_models, loaded)
        self.window.set_model_state.assert_called_with("ready", loaded)
        self.controller._on_event({"session": epoch, "kind": "models_ready", "models": {}})
        self.assertEqual(self.controller.active_models, loaded)

    def test_selected_model_is_not_loaded_until_new_ready_and_failure_keeps_no_new_name(self):
        self.controller.active_models = self._loaded_models()
        self._model_dialog(True, "hy-mt2-1.8b-q6_k")
        self.controller.change_models()
        self.assertEqual(self.controller.model_state, "switching")
        self.assertIn("7B", self.controller.active_models["translation"]["model_file"])
        self.controller.executor.drain()
        self.qt_app.processEvents()
        self.assertEqual(self.controller.active_models, {})
        self.controller._on_event({"session": self.controller.epoch, "kind": "models_failed"})
        self.assertEqual(self.controller.active_models, {})
        self.window.set_model_state.assert_called_with("failed", {})

    def test_model_dialog_receives_loaded_metadata_not_saved_selection(self):
        self.controller.active_models = self._loaded_models()
        self._model_dialog(False)
        with patch("fanyiji.model_manager.ModelManagerDialog", return_value=Mock(
            selected_model_id=None, is_busy=False, exec=Mock(return_value=QDialog.DialogCode.Rejected))) as factory:
            self.controller.change_models()
            self.assertEqual(factory.call_args.kwargs["active_models"], self._loaded_models())
            self.assertTrue(factory.call_args.kwargs["speech_only"])

    def test_filter_metadata_and_recording_hook_preserve_actual_backend(self):
        self.controller.start()
        loaded = self._loaded_models()
        loaded["speech_filter"] = {"enabled": True, "model_loaded": True, "device": "cpu"}
        self.controller._on_event({"session": self.controller.epoch, "kind": "models_ready", "models": loaded})
        self.assertEqual(self.controller.active_models["speech_filter"], loaded["speech_filter"])
        recorder = self.controller.recorder = Mock(active=True, model_info={})
        self.controller.pipeline._speech_filter = Mock(backend_info={"model_loaded": True, "device": "cpu"})
        self.controller._attach_recording()
        job, decision = object(), object()
        self.controller.pipeline.on_filtered(job, decision)
        recorder.on_filtered.assert_called_once_with(job, decision)
        self.controller._diagnostic_snapshot()
        self.assertEqual(recorder.model_info["speech_filter"], loaded["speech_filter"])

    def test_diagnostic_snapshot_preserves_scheduler_metadata_and_refreshes_backend_fields(self):
        from fanyiji.diagnostics import SessionRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory) / "recording", self.controller.config)
            self.controller.recorder = recorder
            loaded = self._loaded_models()
            loaded["translation"].update(worker_count=2, batch_limit=3)
            loaded["asr"]["accuracy_processing"] = {"pending_wait_seconds": 1.2}
            self.controller._pipeline_event({"session": self.controller.epoch, "kind": "models_ready", "models": loaded})
            self.assertEqual(recorder.model_info["translation"]["worker_count"], 2)
            self.controller.pipeline._recognizer = Mock(backend_info={"model_id": "new-asr"})
            self.controller.pipeline._translator = Mock(backend_info={"model_id": "new-translation", "device": "cpu"})
            self.controller._diagnostic_snapshot()
            report = json.loads((recorder.directory / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["models"]["translation"]["worker_count"], 2)
            self.assertEqual(report["models"]["translation"]["batch_limit"], 3)
            self.assertEqual(report["models"]["translation"]["model_id"], "new-translation")
            self.assertEqual(report["models"]["translation"]["device"], "cpu")
            self.assertEqual(report["models"]["asr"]["accuracy_processing"]["pending_wait_seconds"], 1.2)
            self.assertEqual(report["models"]["asr"]["model_id"], "new-asr")
            recorder.finish()
            self.controller.recorder = None

    def test_close_is_idempotent_and_queued_start_does_not_run(self):
        self.controller.start()
        self.controller.close()
        self.controller.close()
        self.controller.executor.drain()
        self.assertEqual(self.controller.pipeline.calls, [("shutdown",)])
        self.assertEqual(self.controller.executor.shutdown_arguments, (False, False))
        self.assertTrue(self.controller.closed)

    def test_close_optional_service_failure_still_releases_capture_and_other_services(self):
        self.controller.danmaku = Mock()
        self.controller.danmaku.close.side_effect = OSError("fixture storage failed")
        self.controller.browser_service = Mock()
        self.controller.specialized = Mock()
        self.controller.live_source_service.close = Mock()
        self.controller.close()
        self.controller.executor.drain()
        self.controller.browser_service.close.assert_called_once()
        self.controller.live_source_service.close.assert_called_once()
        self.controller.specialized.cancel_comparison.assert_called_once()
        self.controller.specialized.close.assert_called_once()
        self.assertEqual(self.controller.pipeline.calls, [("shutdown",)])
        self.assertEqual(self.controller.executor.shutdown_arguments, (False, False))

    def test_closed_controller_ignores_late_control_calls_and_events(self):
        self.controller.close()
        epoch = self.controller.epoch
        self.controller.start()
        self.controller.set_paused(False)
        dialog_factory = self._dialog(True, AppConfig())
        self.controller.change_settings()
        dialog_factory.assert_not_called()
        self.assertEqual(self.controller.epoch, epoch)
        self.window.set_session_state.reset_mock()
        self.controller._on_event({"session": epoch, "kind": "state", "text": "late"})
        self.window.set_session_state.assert_not_called()

    def _model_dialog(self, accepted, model_id=None):
        dialog = Mock()
        dialog.exec.return_value = QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
        dialog.selected_model_id = model_id
        dialog.is_busy = False
        self.stack.enter_context(patch("fanyiji.model_manager.ModelManagerDialog", return_value=dialog))
        return dialog

    def test_model_cancel_restores_running_session(self):
        self._model_dialog(False)
        old = self.controller.pipeline
        self.controller.change_models()
        self.controller.executor.drain()
        self.assertFalse(self.controller.paused)
        self.assertIs(self.controller.pipeline, old)
        self.assertEqual(old.calls, [("stop",), ("start", self.controller.config, self.controller.epoch)])
        self.assertIsNone(self.controller.model_dialog)

    def test_model_selection_shuts_old_pipeline_before_starting_new(self):
        self._model_dialog(True, "hy-mt2-7b-q6_k")
        old = self.controller.pipeline
        self.controller.change_models()
        self.controller.executor.drain()
        self.assertEqual(old.calls, [("shutdown",)])
        self.assertIsNot(old, self.controller.pipeline)
        self.assertEqual(self.controller.pipeline.calls, [("start", self.controller.config, self.controller.epoch)])

    def test_model_selection_updates_danmaku_session_to_the_replacement_pipeline(self):
        self._model_dialog(True, "hy-mt2-7b-q6_k")
        self.controller.danmaku = Mock()
        self.controller.change_models()
        self.controller.executor.drain()
        assert self.controller.danmaku.begin_session.call_args.args == (self.controller.epoch,)
        assert self.controller.pipeline.calls[-1][-1] == self.controller.epoch

    def test_model_selection_preserves_pause(self):
        self.controller.paused = True
        self._model_dialog(True, "hy-mt2-7b-q6_k")
        self.controller.change_models()
        self.controller.executor.drain()
        self.assertTrue(self.controller.paused)
        self.assertEqual(self.controller.pipeline.calls, [])

    def test_close_before_queued_model_replace_never_creates_new_pipeline(self):
        self._model_dialog(True, "hy-mt2-7b-q6_k")
        old = self.controller.pipeline
        self.controller.change_models()
        self.controller.close()
        self.controller.executor.drain()
        self.assertIs(self.controller.pipeline, old)
        self.assertEqual(old.calls, [("shutdown",)])

    def test_close_during_model_dialog_never_resumes_capture(self):
        dialog = self._model_dialog(True, "hy-mt2-7b-q6_k")
        dialog.exec.side_effect = lambda: (self.controller.close(), QDialog.DialogCode.Accepted)[1]
        self.controller.change_models()
        self.controller.executor.drain()
        self.assertEqual(self.controller.pipeline.calls, [("shutdown",)])

    def test_window_close_waits_for_download_cancellation_once(self):
        dialog = Mock(is_busy=True)
        self.controller.model_dialog = dialog
        self.window.close = Mock()
        self.assertFalse(self.controller._may_close_window())
        self.assertFalse(self.controller._may_close_window())
        dialog.reject.assert_called_once()
        dialog.finished.connect.assert_called_once()
        callback = dialog.finished.connect.call_args.args[0]
        dialog.is_busy = False
        callback(QDialog.DialogCode.Rejected)
        self.window.close.assert_called_once()
        self.assertTrue(self.controller._may_close_window())


class ApplicationIdentityTests(unittest.TestCase):
    def test_application_identity_uses_the_product_name(self):
        app = Mock()
        app_module.configure_application_identity(app)
        app.setApplicationName.assert_called_once_with(APP_NAME)
        app.setApplicationDisplayName.assert_called_once_with(APP_NAME)
        app.setOrganizationName.assert_called_once_with(APP_NAME)

    def test_new_instance_activates_only_its_voice_endpoint(self):
        socket = Mock()
        socket.waitForConnected.return_value = True
        with patch.object(app_module, "QLocalSocket", return_value=socket), \
                patch.object(app_module, "QLocalServer") as factory:
            self.assertIsNone(app_module.single_instance(Mock()))
        socket.connectToServer.assert_called_once_with(INSTANCE_NAME)
        socket.write.assert_called_once_with(b"activate")
        factory.assert_not_called()

    def test_new_instance_listens_only_on_its_voice_endpoint(self):
        socket, server = Mock(), Mock()
        socket.waitForConnected.return_value = False
        with patch.object(app_module, "QLocalSocket", return_value=socket) as sockets, \
                patch.object(app_module, "QLocalServer", return_value=server) as servers:
            self.assertIs(app_module.single_instance(Mock()), server)
        sockets.assert_called_once_with()
        server.listen.assert_called_once_with(INSTANCE_NAME)
        servers.assert_called_once()
        self.assertNotIn("_legacy_server", vars(server))
        server.newConnection.connect.assert_called_once()


class WindowActivationTests(unittest.TestCase):
    def _app(self, window=None):
        app = Mock()
        app.activeModalWidget.return_value = None
        app.activeWindow.return_value = None
        app.topLevelWidgets.return_value = []
        app._fanyiji_main_window = window
        return app

    def test_registered_main_window_is_restored_even_without_an_active_window(self):
        # Windows may hide the native HWND while Qt still reports it as visible.
        for qt_visible in (False, True):
            with self.subTest(qt_visible=qt_visible):
                window = Mock(_closing_emitted=False)
                window.isVisible.return_value = qt_visible
                app = self._app(window)

                app_module.activate_existing_window(app)

                self.assertEqual(window.method_calls, [
                    call.hide(), call.showNormal(), call.raise_(), call.activateWindow(),
                ])
                app.activeWindow.assert_not_called()
                app.topLevelWidgets.assert_not_called()

    def test_modal_dialog_takes_priority_without_hiding_or_changing_main_window(self):
        window = Mock(_closing_emitted=False)
        modal = Mock()
        app = self._app(window)
        app.activeModalWidget.return_value = modal

        app_module.activate_existing_window(app)

        self.assertEqual(modal.method_calls, [
            call.showNormal(), call.raise_(), call.activateWindow(),
        ])
        modal.hide.assert_not_called()
        self.assertEqual(window.method_calls, [])
        app.activeWindow.assert_not_called()
        app.topLevelWidgets.assert_not_called()

    def test_closing_main_window_is_not_reopened_or_replaced_by_fallback(self):
        window = Mock(_closing_emitted=True)
        fallback = Mock()
        app = self._app(window)
        app.activeWindow.return_value = fallback
        app.topLevelWidgets.return_value = [fallback]

        app_module.activate_existing_window(app)

        self.assertEqual(window.method_calls, [])
        self.assertEqual(fallback.method_calls, [])
        app.activeWindow.assert_not_called()
        app.topLevelWidgets.assert_not_called()

    def test_without_registered_main_window_existing_active_window_is_used(self):
        window = Mock(_closing_emitted=False)
        app = self._app()
        app.activeWindow.return_value = window

        app_module.activate_existing_window(app)

        window.showNormal.assert_called_once_with()
        window.raise_.assert_called_once_with()
        window.activateWindow.assert_called_once_with()
        app.topLevelWidgets.assert_not_called()

    def test_without_active_window_visible_top_level_fallback_is_preserved(self):
        hidden, visible = Mock(), Mock(_closing_emitted=False)
        hidden.isVisible.return_value = False
        visible.isVisible.return_value = True
        app = self._app()
        app.topLevelWidgets.return_value = [hidden, visible]

        app_module.activate_existing_window(app)

        hidden.showNormal.assert_not_called()
        hidden.activateWindow.assert_not_called()
        visible.showNormal.assert_called_once_with()
        visible.raise_.assert_called_once_with()
        visible.activateWindow.assert_called_once_with()

    def test_no_windows_can_be_activated_safely(self):
        app_module.activate_existing_window(self._app())


class FirstRunLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def _run_main(self, accepted, save_error=None):
        dialog = Mock()
        dialog.exec.return_value = (
            QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
        )
        dialog.result_config = AppConfig() if accepted else None
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["fanyiji"]))
            stack.enter_context(patch.object(app_module, "QApplication", return_value=self.qt_app))
            stack.enter_context(patch.object(app_module, "single_instance", return_value=object()))
            stack.enter_context(patch.object(app_module, "configure_logging"))
            stack.enter_context(patch("fanyiji.migration.ensure_initial_migration"))
            stack.enter_context(patch.object(app_module, "data_root", return_value=Path("unused-data")))
            stack.enter_context(patch.object(app_module, "load_config", return_value=None))
            stack.enter_context(patch("fanyiji.ui.SettingsDialog", return_value=dialog))
            window = stack.enter_context(patch("fanyiji.ui.MainWindow"))
            controller = stack.enter_context(patch.object(app_module, "AppController"))
            saved = stack.enter_context(patch.object(app_module, "save_config", side_effect=save_error))
            stack.enter_context(patch.object(app_module.QMessageBox, "critical"))
            result = app_module.main()
            return result, saved.call_count, window.call_count, controller.call_count

    def test_first_run_cancel_never_saves_or_initializes_pipeline(self):
        self.assertEqual(self._run_main(False), (0, 0, 0, 0))

    def test_first_run_save_failure_never_initializes_pipeline(self):
        self.assertEqual(self._run_main(True, OSError("read only")), (1, 1, 0, 0))

    def test_saved_settings_start_without_first_run_dialog(self):
        # The persistent config is the only input to later launches; source
        # settings must arrive at the controller without being defaulted again.
        saved_config = AppConfig(
            microphone=SourceConfig(enabled=False, direction="ja_to_zh"),
            playback=SourceConfig(enabled=True, direction="zh_to_ja"),
        )
        fake_app = Mock()
        fake_app.exec.return_value = 0
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["fanyiji"]))
            stack.enter_context(patch.object(app_module, "QApplication", return_value=fake_app))
            stack.enter_context(patch.object(app_module, "single_instance", return_value=object()))
            stack.enter_context(patch.object(app_module, "configure_logging"))
            stack.enter_context(patch("fanyiji.migration.ensure_initial_migration"))
            stack.enter_context(patch.object(app_module, "data_root", return_value=Path("unused-data")))
            stack.enter_context(patch.object(app_module, "load_config", return_value=saved_config))
            stack.enter_context(patch.object(app_module, "configured_models_available", return_value=True))
            dialog = stack.enter_context(patch("fanyiji.ui.SettingsDialog"))
            window = stack.enter_context(patch("fanyiji.ui.MainWindow"))
            controller = stack.enter_context(patch.object(app_module, "AppController"))
            restore = stack.enter_context(patch.object(app_module, "restore_main_window"))
            timer = stack.enter_context(patch.object(app_module.QTimer, "singleShot"))
            timer.side_effect = lambda _delay, callback: callback()
            self.assertEqual(app_module.main(), 0)
            fake_app.setApplicationName.assert_called_once_with(APP_NAME)
            fake_app.setApplicationDisplayName.assert_called_once_with(APP_NAME)
            fake_app.setOrganizationName.assert_called_once_with(APP_NAME)
            dialog.assert_not_called()
            window.assert_called_once_with(saved_config)
            self.assertIs(fake_app._fanyiji_main_window, window.return_value)
            window.return_value.show.assert_called_once_with()
            restore.assert_called_once_with(window.return_value)
            controller.assert_called_once_with(window.return_value, saved_config, Path("unused-data/settings.json"))
            controller.return_value.start.assert_called_once_with()

    def _run_with_models_missing(self, config):
        fake_app = Mock()
        fake_app.exec.return_value = 0
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["fanyiji"]))
            stack.enter_context(patch.object(app_module, "QApplication", return_value=fake_app))
            stack.enter_context(patch.object(app_module, "single_instance", return_value=object()))
            stack.enter_context(patch.object(app_module, "configure_logging"))
            stack.enter_context(patch("fanyiji.migration.ensure_initial_migration"))
            stack.enter_context(patch.object(app_module, "load_config", return_value=config))
            available = stack.enter_context(patch.object(app_module, "configured_models_available", return_value=False))
            manager = Mock(selected_model_id=None)
            manager.exec.return_value = QDialog.DialogCode.Rejected
            factory = stack.enter_context(patch("fanyiji.model_manager.ModelManagerDialog", return_value=manager))
            controller = stack.enter_context(patch.object(app_module, "AppController"))
            stack.enter_context(patch("fanyiji.ui.MainWindow"))
            stack.enter_context(patch.object(app_module.QTimer, "singleShot"))
            result = app_module.main()
            return result, available.call_count, factory.call_count, controller.call_count

    def test_missing_models_cancel_never_initializes_capture(self):
        self.assertEqual(self._run_with_models_missing(AppConfig()), (0, 1, 1, 0))

    def test_both_channels_disabled_need_no_models(self):
        config = AppConfig(microphone=SourceConfig(enabled=False), playback=SourceConfig(enabled=False))
        self.assertEqual(self._run_with_models_missing(config), (0, 0, 0, 1))


class ModelAvailabilityTests(unittest.TestCase):
    def test_complete_gguf_gate_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx", "translation/test.gguf"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            backend = root / "translation/backend.json"
            backend.write_text(json.dumps({"backend": "llama_cpp", "model_file": "test.gguf"}), encoding="utf-8")
            self.assertTrue(app_module.configured_models_available(root, speech_only=False))
            self.assertFalse(app_module.configured_models_available(root, speech_only=True))
            from fanyiji.asr_catalog import vad_dependencies
            from fanyiji.models_download import ModelCatalog
            for dependency in vad_dependencies(ModelCatalog.default(), speech_only=True):
                for file in dependency.files:
                    target = root / file.relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"synthetic dependency")
            self.assertTrue(app_module.configured_models_available(root, speech_only=True))
            for value in ("../../test.gguf", str(root / "translation/test.gguf"), "test.bin", ""):
                backend.write_text(json.dumps({"backend": "llama_cpp", "model_file": value}), encoding="utf-8")
                self.assertFalse(app_module.configured_models_available(root, speech_only=False))

    def test_malformed_or_empty_models_are_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(app_module.configured_models_available(root))
            (root / "translation").mkdir()
            (root / "translation/backend.json").write_text("[]", encoding="utf-8")
            self.assertFalse(app_module.configured_models_available(root))


if __name__ == "__main__":
    unittest.main()
