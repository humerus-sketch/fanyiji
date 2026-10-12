"""Account picker lifecycle tests without CLI, network, audio, or generation."""

import os
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fanyiji.codex_discovery import wire_codex_picker
from fanyiji.config import AppConfig
from fanyiji.ui import SettingsDialog


MODELS = [{"model": "gpt-test-model", "displayName": "Test Model",
           "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]


class _FakeEngine:
    def __init__(self, *, slow_preload=False, slow_cleanup=False, failure=None, cleanup_failure=None):
        self.started = threading.Event()
        self.release_preload = threading.Event()
        self.cleanup_started = threading.Event()
        self.release_cleanup = threading.Event()
        self.closed = threading.Event()
        self.failure = failure
        self.cleanup_failure = cleanup_failure
        if not slow_preload:
            self.release_preload.set()
        if not slow_cleanup:
            self.release_cleanup.set()

    def preload(self):
        self.started.set()
        if not self.release_preload.wait(3):
            raise RuntimeError("test preload timeout")
        if self.closed.is_set():
            raise RuntimeError("test helper cancelled")
        if self.failure:
            raise self.failure

    @property
    def backend_info(self):
        return {"available_models": MODELS}

    def close(self):
        self.cleanup_started.set()
        self.closed.set()
        self.release_preload.set()
        self.release_cleanup.wait(3)
        if self.cleanup_failure:
            raise self.cleanup_failure


class _TrackingDialog(SettingsDialog):
    def __init__(self, config):
        self.model_updates = []
        super().__init__(config)

    def set_codex_models(self, models, error=""):
        self.model_updates.append((threading.get_ident(), models, error))
        super().set_codex_models(models, error)


class CodexDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.dialogs = []
        self.engines = []
        self.gui_thread = threading.get_ident()

    def tearDown(self):
        for engine in self.engines:
            engine.release_preload.set()
            engine.release_cleanup.set()
        for dialog in self.dialogs:
            dialog.reject()
            if hasattr(dialog, "codex_discovery"):
                dialog.codex_discovery.close()
        self._wait(lambda: all(not dialog.codex_discovery._busy for dialog in self.dialogs))
        for dialog in self.dialogs:
            dialog.deleteLater()
        self.app.processEvents()

    def _wait(self, condition, timeout=1.5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if condition():
                return True
            threading.Event().wait(0.001)
        self.app.processEvents()
        return bool(condition())

    def _dialog(self, config):
        dialog = _TrackingDialog(config)
        self.dialogs.append(dialog)
        return dialog

    def _engine(self, **options):
        engine = _FakeEngine(**options)
        self.engines.append(engine)
        return engine

    def test_local_picker_does_not_start_helper_and_switch_to_codex_updates_on_gui_thread(self):
        engine = self._engine(slow_preload=True)
        with patch("fanyiji.codex_engine.CodexTranslationEngine", return_value=engine) as factory:
            dialog = self._dialog(AppConfig(translation_provider="local"))
            wire_codex_picker(dialog)
            self.app.processEvents()
            factory.assert_not_called()
            dialog.translation_provider.setCurrentIndex(dialog.translation_provider.findData("codex"))
            self.assertTrue(engine.started.wait(1))
            self.assertFalse(dialog.codex_models_refresh.isEnabled())
            engine.release_preload.set()
            self.assertTrue(self._wait(lambda: bool(dialog.model_updates)))
            self.assertEqual(dialog.model_updates[0][0], self.gui_thread)
            self.assertGreaterEqual(dialog.codex_model.findData("gpt-test-model"), 0)
            self.assertTrue(dialog.codex_models_refresh.isEnabled())
            self.assertTrue(self._wait(lambda: not dialog.codex_discovery._busy))
            self.assertTrue(engine.closed.is_set())

    def test_saved_unknown_codex_model_still_triggers_account_discovery(self):
        engine = self._engine()
        with patch("fanyiji.codex_engine.CodexTranslationEngine", return_value=engine) as factory:
            dialog = self._dialog(AppConfig(translation_provider="codex", codex_model="saved-model"))
            wire_codex_picker(dialog)
            self.assertTrue(self._wait(lambda: bool(dialog.model_updates)))
            factory.assert_called_once()

    def test_hybrid_mode_queries_cloud_models_without_a_translation(self):
        engine = self._engine()
        with patch("fanyiji.codex_engine.CodexTranslationEngine", return_value=engine) as factory:
            dialog = self._dialog(AppConfig(translation_provider="hybrid", codex_model="saved-model"))
            wire_codex_picker(dialog)
            self.assertTrue(self._wait(lambda: bool(dialog.model_updates)))
            factory.assert_called_once()
            self.assertGreaterEqual(dialog.codex_model.findData("gpt-test-model"), 0)
            self.assertTrue(engine.closed.is_set())
            self.assertEqual(dialog.codex_model.currentData(), "saved-model")
            self.assertIn("暂未验证", dialog.codex_model.currentText())

    def test_closing_during_slow_preload_stops_helper_without_publishing_models(self):
        engine = self._engine(slow_preload=True, slow_cleanup=True)
        with patch("fanyiji.codex_engine.CodexTranslationEngine", return_value=engine) as factory:
            dialog = self._dialog(AppConfig(translation_provider="codex"))
            wire_codex_picker(dialog)
            self.assertTrue(engine.started.wait(1))
            before_cancel = time.monotonic()
            dialog.reject()
            self.assertLess(time.monotonic() - before_cancel, 0.5)
            self.assertTrue(self._wait(engine.closed.is_set))
            engine.release_cleanup.set()
            self.assertTrue(self._wait(lambda: not dialog.codex_discovery._busy))
            self.assertEqual(dialog.model_updates, [])
            dialog.codex_discovery.request()
            factory.assert_called_once()

    def test_refresh_failure_preserves_saved_model_and_effort_on_gui_thread(self):
        engine = self._engine(failure=RuntimeError("模拟登录失效"))
        with patch("fanyiji.codex_engine.CodexTranslationEngine", return_value=engine):
            dialog = self._dialog(AppConfig(translation_provider="codex", codex_model="saved-model",
                                            codex_effort="high", codex_speed="fast"))
            wire_codex_picker(dialog)
            dialog.codex_models_refresh.click()
            self.assertTrue(self._wait(lambda: bool(dialog.model_updates)))
            self.assertEqual(dialog.model_updates[0][0], self.gui_thread)
            self.assertIn("模拟登录失效", dialog.codex_models_status.text())
            self.assertEqual(dialog.codex_model.currentData(), "saved-model")
            self.assertEqual(dialog.codex_effort.currentData(), "high")
            self.assertEqual(dialog.codex_speed.currentData(), "fast")
            self.assertTrue(dialog.codex_models_refresh.isEnabled())

    def test_cleanup_failure_is_reported_and_releases_loading_state(self):
        engine = self._engine(cleanup_failure=OSError("模拟 helper 清理失败"))
        with patch("fanyiji.codex_engine.CodexTranslationEngine", return_value=engine):
            dialog = self._dialog(AppConfig(translation_provider="codex"))
            wire_codex_picker(dialog)
            self.assertTrue(self._wait(lambda: bool(dialog.model_updates)))
            self.assertEqual(dialog.model_updates[0][0], self.gui_thread)
            self.assertIn("清理失败", dialog.codex_models_status.text())
            self.assertTrue(dialog.codex_models_refresh.isEnabled())

    def test_refresh_during_old_helper_cleanup_cannot_leave_picker_loading_forever(self):
        first = self._engine(slow_cleanup=True)
        second = self._engine()
        with patch("fanyiji.codex_engine.CodexTranslationEngine", side_effect=[first, second]) as factory:
            dialog = self._dialog(AppConfig(translation_provider="codex"))
            wire_codex_picker(dialog)
            self.assertTrue(first.cleanup_started.wait(1))
            self.app.processEvents()
            # A result delivered before old-helper cleanup must not re-enable an
            # unusable refresh button, or the second request needs to be handled.
            clicked_before_cleanup = dialog.codex_models_refresh.isEnabled()
            if clicked_before_cleanup:
                dialog.codex_models_refresh.click()
            first.release_cleanup.set()
            self.assertTrue(self._wait(lambda: not dialog.codex_discovery._busy))
            self.assertTrue(self._wait(lambda: dialog.codex_models_refresh.isEnabled()))
            if not clicked_before_cleanup:
                dialog.codex_models_refresh.click()
            self.assertTrue(self._wait(lambda: factory.call_count == 2 and len(dialog.model_updates) >= 2))

    def test_queued_results_after_dialog_cancel_are_ignored(self):
        with patch("fanyiji.codex_engine.CodexTranslationEngine") as factory:
            dialog = self._dialog(AppConfig(translation_provider="local"))
            wire_codex_picker(dialog)

            def emit_queued():
                dialog.codex_discovery.models_ready.emit(MODELS)
                dialog.codex_discovery.failed.emit("queued failure")

            worker = threading.Thread(target=emit_queued)
            worker.start()
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
            dialog.reject()
            self.app.processEvents()
            self.assertEqual(dialog.model_updates, [])
            factory.assert_not_called()
