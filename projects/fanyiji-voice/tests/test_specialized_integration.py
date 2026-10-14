"""Dedicated-data failures and GPU batches must preserve the live lifecycle."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QApplication

from fanyiji import app as app_module
from fanyiji.config import AppConfig
from test_app import _ControlledExecutor, _FakePipeline, _FakeWindow


class SpecializedLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.stack = ExitStack()
        module = ModuleType("fanyiji.pipeline")
        module.SubtitlePipeline = _FakePipeline
        self.stack.enter_context(patch.dict(sys.modules, {"fanyiji.pipeline": module}))
        self.stack.enter_context(patch.object(app_module, "ThreadPoolExecutor", _ControlledExecutor))
        self.stack.enter_context(patch.object(app_module, "models_root", return_value=Path("unused-models")))
        # start() may initialize the saved browser and queue through other modules.
        self.data_directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.dict(os.environ, {"FANYIJI_DATA": self.data_directory}))
        self.window = _FakeWindow()
        self.service = Mock()
        self.service.busy = False
        self.controller = app_module.AppController(self.window, AppConfig(), Path("unused-settings"),
                                                   specialized_service=self.service)
        self.controller.start()
        self.controller.executor.drain()

    def tearDown(self):
        self.controller.close()
        self.controller.executor.drain()
        self.window.deleteLater()
        self.qt_app.processEvents()
        self.stack.close()

    def test_start_session_before_capture_and_ignore_old_session_outputs(self):
        self.service.begin_session.assert_called_once_with(1, self.controller.config)
        self.controller._pipeline_event({"session": 0, "kind": "notice", "text": "old"})
        self.service.on_event.assert_not_called()
        event = {"session": 1, "kind": "original", "channel": "system", "text": "source", "segment_id": "1:system:1"}
        self.controller._pipeline_event(event)
        self.service.on_event.assert_called_once_with(event)
        self.window.set_original.assert_called_once()

    def test_collection_exception_does_not_prevent_subtitle_or_emit_repeated_warnings(self):
        self.service.on_event.side_effect = MemoryError("copy failed")
        event = {"session": 1, "kind": "original", "channel": "system", "text": "source", "segment_id": "1:system:1"}
        self.controller._pipeline_event(event)
        self.controller._pipeline_event(event)
        self.assertEqual(self.window.set_original.call_count, 2)
        self.assertEqual(self.window.append_notice.call_count, 1)
        self.assertEqual(self.service.fail_collection.call_count, 2)

    def test_diagnostics_and_dedicated_audio_both_receive_same_segment(self):
        recorder = Mock()
        recorder.active = True
        recorder.model_info = {}
        self.controller.recorder = recorder
        self.controller._attach_recording()
        job = Mock()
        self.controller.pipeline.on_segment(job, True)
        recorder.on_segment.assert_called_once_with(job, True)
        self.service.on_segment.assert_called_once_with(job, True)

    def test_comparison_shutdown_failure_releases_busy_and_recreates_pipeline(self):
        self.controller.pipeline.shutdown = Mock(side_effect=OSError("unload failed"))
        self.service.busy = True
        self.service.finish_comparison.side_effect = lambda **kwargs: setattr(self.service, "busy", False)
        self.controller._request_local_comparison()
        self.controller.executor.drain()
        self.service.run_comparison.assert_not_called()
        self.service.finish_comparison.assert_called_once_with(failed=True)
        self.assertFalse(self.service.busy)
        self.assertFalse(self.controller.paused)
        self.assertIn(("start", self.controller.config, self.controller.epoch), self.controller.pipeline.calls)

    def test_close_cancels_comparison_and_never_resumes_capture(self):
        previous = self.controller.pipeline
        self.service.busy = True
        self.controller._request_local_comparison()
        self.controller.close()
        self.controller.executor.drain()
        self.service.cancel_comparison.assert_called_once()
        self.service.run_comparison.assert_not_called()
        self.service.close.assert_called_once()
        self.assertIs(self.controller.pipeline, previous)
        self.assertEqual([call for call in previous.calls if call[0] == "start"],
                         [("start", self.controller.config, 1)])


if __name__ == "__main__":
    unittest.main()
