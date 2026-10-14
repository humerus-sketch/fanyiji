"""Audio controls reuse the application-owned dialog and current config."""
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QLabel

from fanyiji.i18n import set_language, tr

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


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_sending_guide_uses_character_floor_and_weighted_cap_without_side_effects(qt_app, language):
    set_language(language)
    service = BrowserStub()
    dialog = make_dialog(service)
    source = "正文中的汉字、数字各计 1 个字符，空格和【】不计入短片段阈值。满 2 字直接独立发送；不足 2 字保留原位，仅与紧接下一句合并。每条含【】最多加权 40 单位，账号上限更低时取较小值；长句直接切分，不搬字均衡尾段。"
    labels = [label.text() for label in dialog.findChildren(QLabel)]
    assert tr(source) in labels
    if language != "zh-CN":
        assert tr(source) != source
    assert not any("11 到 40" in label or "少于 11" in label for label in labels)
    assert service.opened == service.saved == []
