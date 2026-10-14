"""The voice edition retains its pipeline while rejecting caption-mode dispatch."""
import ast
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QComboBox, QPushButton

from fanyiji import app as app_module
from fanyiji.config import AppConfig, ConfigError, SourceConfig, load_config, save_config
from fanyiji.ui import MainWindow, SettingsDialog


@pytest.fixture
def qt_app():
    instance = QApplication.instance() or QApplication([])
    yield instance
    for widget in instance.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    instance.processEvents()


def test_legacy_caption_settings_restore_native_channel_preferences(tmp_path):
    raw = asdict(AppConfig(microphone=SourceConfig(False), playback=SourceConfig(True, "ja_to_zh"),
                          translation_provider="hybrid", audio_processing="uvr"))
    raw["processing_mode"] = "chrome_caption"
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(path)
    assert config.processing_mode == "standard"
    assert config.enabled_channels == {"system"}
    assert config.translation_provider == "hybrid" and config.audio_processing == "uvr"
    save_config(path, config)
    assert json.loads(path.read_text(encoding="utf-8"))["processing_mode"] == "standard"


def test_dataclass_replace_cannot_activate_caption_mode():
    config = AppConfig(microphone=SourceConfig(False), playback=SourceConfig(False))
    assert replace(config, processing_mode="chrome_caption").enabled_channels == set()
    with pytest.raises(ConfigError):
        replace(config, processing_mode="unknown")


@pytest.mark.parametrize("first_run", [False, True])
def test_settings_expose_all_native_providers_without_a_mode_switch(qt_app, first_run):
    dialog = SettingsDialog(AppConfig(), first_run=first_run)
    assert not dialog.findChildren(QComboBox, "processingModeCombo")
    assert not hasattr(dialog, "processing_mode_changed")
    assert {dialog.translation_provider.itemData(index) for index in range(dialog.translation_provider.count())} == {
        "local", "codex", "hybrid", "google_web", "youdao_web", "domestic_cli"}
    for provider in ("local", "codex", "hybrid", "google_web", "youdao_web", "domestic_cli"):
        dialog.translation_provider.setCurrentIndex(dialog.translation_provider.findData(provider))
        dialog.accept()
        assert dialog.result_config.translation_provider == provider
        assert dialog.result_config.processing_mode == "standard"


def test_main_ui_has_only_native_source_panels_and_features(qt_app):
    window = MainWindow(AppConfig())
    assert not hasattr(window, "chrome_caption_requested")
    assert not hasattr(window, "processing_mode_requested")
    assert not any(button.objectName().startswith("processingMode_") for button in window.findChildren(QPushButton))
    assert not any("Chrome" in button.text() for button in window.findChildren(QPushButton))
    assert {window._source_panels[key].title_label.text() for key in ("microphone", "playback")} == {"麦克风", "电脑声音"}
    assert all(hasattr(window, name) for name in (
        "models_requested", "browser_requested", "audio_processing_requested", "specialized_models_requested",
        "translation_page_requested", "domestic_cli_requested", "subtitle_presented"))


def test_app_and_entrypoint_have_no_caption_module_import_or_dispatch():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "src/fanyiji/app.py", root / "src/fanyiji/ui.py", root / "run.py"):
        text = path.read_text(encoding="utf-8")
        assert "chrome_caption" not in text
        assert "ChromeCaption" not in text and "ChromeBrowser" not in text
        tree = ast.parse(text)
        assert not any(isinstance(node, ast.ImportFrom) and (node.module or "").startswith("chrome_")
                       for node in ast.walk(tree))
    assert not hasattr(app_module.AppController, "switch_processing_mode")


def test_smoke_cli_rejects_caption_mode_without_initializing_helpers(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src"), "QT_QPA_PLATFORM": "offscreen",
                   "FANYIJI_VOICE_DATA": str(tmp_path)}
    result = subprocess.run([sys.executable, "run.py", "--smoke-test", "--smoke-processing-mode", "chrome_caption"],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert not (tmp_path / "settings.json").exists()

