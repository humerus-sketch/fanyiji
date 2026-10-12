"""Audio controls reuse the application-owned dialog and current config."""
from unittest.mock import Mock

from test_browser_dialog_danmaku import BrowserStub, make_dialog, qt_app


def test_browser_audio_entry_reuses_controller_callback_and_does_not_send(qt_app):
    service = BrowserStub()
    service.audio_processing_requested = Mock()
    dialog = make_dialog(service)
    assert dialog.audio_processing_button.text() == "音源与语音处理"
    dialog.audio_processing_button.click()
    dialog.audio_processing_button.click()
    assert service.audio_processing_requested.call_count == 2
    assert service.opened == service.saved == []
    assert not hasattr(dialog, "speaker_button")


def test_standalone_browser_without_audio_controller_has_no_broken_entry(qt_app):
    dialog = make_dialog(BrowserStub())
    assert dialog.audio_processing_button is None
