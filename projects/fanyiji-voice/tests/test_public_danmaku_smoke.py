"""Packaged UI checks expose the sender without starting browser or capture."""
import json
import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fanyiji import app, local_features
from fanyiji.i18n import get_language, set_language


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_frozen_public_smoke_exposes_sender_without_external_grant(language, capsys, monkeypatch):
    qt_app = QApplication.instance() or QApplication([])
    previous = get_language()
    monkeypatch.setattr(local_features.sys, "frozen", True, raising=False)
    monkeypatch.setattr(local_features, "data_root", lambda: pytest.fail("must not inspect legacy grants"))
    try:
        with patch("fanyiji.browser_service.BrowserService", side_effect=AssertionError("must not create browser service")), \
                patch.object(app, "AppController", side_effect=AssertionError("must not start capture")):
            assert app.smoke_test(language=language) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["ui_smoke"] == "passed"
        assert result["danmaku_available"] is True
        assert result["danmaku_controls_visible"] is True
        assert result["audio_capture"] is False
        assert result["settings_written"] is False
    finally:
        set_language(previous)
        for widget in qt_app.topLevelWidgets():
            widget.close()
            widget.deleteLater()
        qt_app.processEvents()
