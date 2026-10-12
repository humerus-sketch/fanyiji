"""Audio controls never record a voice profile or automatically open browsers."""
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from fanyiji.audio_processing_dialog import AudioProcessingDialog
from fanyiji.i18n import set_language


class SourceStub(QObject):
    snapshot = Signal(dict)

    def __init__(self):
        super().__init__()
        self.value = {"source_enabled": True, "source_ready": False, "state": "waiting_browser",
                      "room": "", "source_kind": "", "source_url": ""}
        self.calls = []

    def snapshot_value(self):
        return dict(self.value)

    def set_source_enabled(self, enabled):
        self.calls.append(enabled)
        self.value["source_enabled"] = enabled
        self.snapshot.emit(dict(self.value))


@pytest.fixture
def qt_app():
    app = QApplication.instance() or QApplication([])
    set_language("zh-CN")
    yield app
    set_language("zh-CN")
    app.processEvents()


def test_dialog_open_reads_only_and_has_no_voice_enrollment_controls(qt_app, tmp_path):
    source, browser = SourceStub(), Mock(room="1698225")
    dialog = AudioProcessingDialog(source, browser_service=browser, models_dir=tmp_path)
    assert source.calls == []
    browser.open.assert_not_called()
    browser.open_reference_video.assert_not_called()
    assert not hasattr(dialog, "begin_button") and not hasattr(dialog, "lock_enabled")
    assert dialog.mode.currentData() == "gtcrn"
    dialog.deleteLater()


def test_video_open_preserves_actual_source_identity_and_live_push_preferences(qt_app, tmp_path):
    browser = Mock(room="1698225")
    dialog = AudioProcessingDialog(SourceStub(), browser_service=browser, models_dir=tmp_path)
    dialog.video_url.setText("https://www.bilibili.com/video/BV1xx411c7mD?token=private")
    dialog.video_button.click()
    browser.open_reference_video.assert_called_once_with("https://www.bilibili.com/video/BV1xx411c7mD", "zh-CN")
    browser.save_preferences.assert_not_called()
    browser.configure_danmaku.assert_not_called()
    dialog.deleteLater()


def test_snapshot_never_changes_source_switch_or_persisted_config(qt_app):
    source, save = SourceStub(), Mock()
    dialog = AudioProcessingDialog(source, apply_mode=save)
    dialog.on_snapshot({"source_enabled": False, "source_ready": False})
    assert source.calls == []
    save.assert_not_called()
    dialog.source_enabled.click()
    assert source.calls == [True]
    dialog.deleteLater()


def test_mode_apply_uses_current_callback_and_can_select_off_without_download(qt_app):
    current = ["gtcrn"]
    def save(mode):
        current[0] = mode
        return True
    dialog = AudioProcessingDialog(SourceStub(), mode_getter=lambda: current[0], apply_mode=save)
    dialog.mode.setCurrentIndex(dialog.mode.findData("off"))
    assert not dialog.download_button.isEnabled()
    assert not dialog.verify_button.isEnabled()
    dialog.save_button.click()
    assert current[0] == "off"
    assert "关闭" in dialog.active_mode.text()
    dialog.deleteLater()


@pytest.mark.parametrize("language, title, mode", [("ja", "音声ソース", "リアルタイム音声"), ("en", "Audio source", "Real-time speech")])
def test_new_controls_are_localized(qt_app, language, title, mode):
    set_language(language)
    dialog = AudioProcessingDialog(SourceStub())
    assert title in dialog.windowTitle()
    assert mode in dialog.mode.currentText()
    assert "主播" not in dialog.source_enabled.text()
    dialog.deleteLater()


def wait_for_task(qt_app, dialog):
    deadline = time.monotonic() + 3
    while dialog.is_busy and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(.005)
    qt_app.processEvents()
    assert not dialog.is_busy


def test_download_and_verification_run_off_gui_thread_and_do_not_auto_activate(qt_app, tmp_path):
    main_thread = threading.get_ident()
    threads = []
    save = Mock()
    def download(*args, **kwargs):
        threads.append(threading.get_ident())
        kwargs["progress"]("downloading", 2**35, 2**36)
    def verify(*args, **kwargs):
        threads.append(threading.get_ident())
        return SimpleNamespace(ready=True)
    dialog = AudioProcessingDialog(SourceStub(), models_dir=tmp_path, apply_mode=save)
    with patch("fanyiji.audio_processing_models.download_processing_model", side_effect=download), \
         patch("fanyiji.audio_processing_models.verify_processing_model", side_effect=verify):
        dialog.download_button.click()
        wait_for_task(qt_app, dialog)
    assert len(threads) == 2 and all(thread != main_thread for thread in threads)
    assert dialog._verified["gtcrn"] is True
    assert dialog.progress.value() == 50
    save.assert_not_called()
    dialog.deleteLater()


def test_close_cancels_worker_without_destroying_a_running_qthread_and_can_reopen(qt_app, tmp_path):
    started = threading.Event()
    def download(*args, **kwargs):
        started.set()
        kwargs["cancel"].wait(1)
        raise RuntimeError("token=private")
    dialog = AudioProcessingDialog(SourceStub(), models_dir=tmp_path)
    dialog.show()
    with patch("fanyiji.audio_processing_models.download_processing_model", side_effect=download):
        dialog.download_button.click()
        assert started.wait(1)
        dialog.reject()
        assert dialog.is_busy
        wait_for_task(qt_app, dialog)
    assert "private" not in dialog.status.text()
    assert "取消" in dialog.status.text()
    dialog.show()
    assert not dialog._close_requested
    dialog.reject()
    dialog.deleteLater()


def test_off_has_no_model_link_action(qt_app):
    dialog = AudioProcessingDialog(SourceStub(), mode_getter=lambda: "off")
    with patch("fanyiji.audio_processing_dialog.QDesktopServices.openUrl") as open_url:
        assert not dialog.model_link_button.isEnabled()
        dialog.model_link_button.click()
        open_url.assert_not_called()
    dialog.deleteLater()
